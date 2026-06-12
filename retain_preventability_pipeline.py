"""
=============================================================================
RETAIN-Based Readmission Preventability Pipeline
=============================================================================
Input columns expected:
    claim_id        - unique claim identifier
    member_id       - patient/beneficiary identifier
    drg_code        - DRG code for the claim
    diag_codes      - pipe-delimited ICD-10 diagnosis codes  e.g. "I50.43|E11.65|I10"
    proc_codes      - pipe-delimited procedure codes         e.g. "99232|93010"
    los             - length of stay in days
    provider_tax_id - provider tax identification number
    admit_date      - admission date  (YYYY-MM-DD)
    discharge_date  - discharge date  (YYYY-MM-DD)

The script expects PAIRS of claims already identified:
    pair_id         - links index claim to readmission claim
    claim_type      - 'index' or 'readmission'

Or alternatively a flat paired format:
    claimA_*        - index claim fields
    claimB_*        - readmission claim fields

Both formats are handled below.
=============================================================================
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import os
import warnings
import logging
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import networkx as nx

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# 1. CONFIGURATION — EDIT THESE TO MATCH YOUR SETUP
# =============================================================================
CONFIG = {
    # --- Data ---
    "data_path":         "readmission_pairs.csv",   # path to your CSV
    "data_format":       "paired",                   # "paired" or "long"
    # If "long": expects pair_id + claim_type columns
    # If "paired": expects claimA_* and claimB_* columns

    # --- Column names (paired format) ---
    "col_pair_id":       "pair_id",
    "col_member_id":     "member_id",
    "col_claimA_prefix": "claimA_",
    "col_claimB_prefix": "claimB_",
    "col_claim_id":      "claim_id",
    "col_drg":           "drg_code",
    "col_diag":          "diag_codes",
    "col_proc":          "proc_codes",
    "col_los":           "los",
    "col_provider":      "provider_tax_id",
    "col_admit":         "admit_date",
    "col_discharge":     "discharge_date",

    # --- Label ---
    # Set to None to auto-generate using CCS relatedness rules
    # Set to column name if you have manual/expert labels
    "label_column":      None,

    # --- Graph embeddings ---
    "use_graph_embeddings":   True,
    "embedding_dim":          128,
    "node2vec_walk_length":   20,
    "node2vec_num_walks":     80,
    "node2vec_p":             1.0,
    "node2vec_q":             0.5,
    "min_cooccurrence":       5,    # minimum co-occurrence count for edge

    # --- RETAIN model ---
    "hidden_dim":        128,
    "dropout":           0.3,
    "num_epochs":        40,
    "batch_size":        32,
    "learning_rate":     5e-4,
    "freeze_epochs":     5,         # epochs to train with frozen embeddings
    "pos_class_weight":  4.0,       # weight for preventable=1 class

    # --- Output ---
    "checkpoint_dir":    "./checkpoints",
    "output_dir":        "./outputs",
    "top_n_providers":   20,        # top N providers to flag in audit queue
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["output_dir"],     exist_ok=True)


# =============================================================================
# 2. DATA LOADING AND PARSING
# =============================================================================

def parse_codes(code_string):
    """
    Parse a delimited string of codes into a clean list.
    Handles pipe, comma, semicolon delimiters.
    Returns empty list for null/empty values.
    """
    if pd.isna(code_string) or str(code_string).strip() == "":
        return []
    raw = str(code_string).replace("|", ",").replace(";", ",")
    return [c.strip().upper() for c in raw.split(",") if c.strip()]


def encode_los(los_days):
    """Bin LOS into categorical tokens RETAIN can embed."""
    try:
        los = float(los_days)
    except (ValueError, TypeError):
        return ["LOS_UNKNOWN"]
    if los <= 1:   return ["LOS_1DAY"]
    elif los <= 3: return ["LOS_2_3DAYS"]
    elif los <= 7: return ["LOS_4_7DAYS"]
    elif los <= 14: return ["LOS_8_14DAYS"]
    else:           return ["LOS_OVER14"]


def encode_discharge_status(status):
    """Map discharge status codes to readable tokens."""
    mapping = {
        "01": "DISCH_HOME",
        "02": "DISCH_SHORT_TERM_HOSP",
        "03": "DISCH_SNF",
        "04": "DISCH_ICF",
        "05": "DISCH_CANCER_CHILDRENS",
        "06": "DISCH_HOME_HEALTH",
        "07": "DISCH_AMA",          # Against Medical Advice — high risk signal
        "20": "DISCH_EXPIRED",
        "30": "DISCH_STILL_PATIENT",
        "43": "DISCH_FEDERAL_HOSP",
        "50": "DISCH_HOSPICE_HOME",
        "51": "DISCH_HOSPICE_MEDICAL",
        "61": "DISCH_SWING_BED",
        "62": "DISCH_REHAB",
        "63": "DISCH_LTC",
        "65": "DISCH_PSYCH",
        "66": "DISCH_CRITICAL_ACCESS",
        "69": "DISCH_DISASTER",
        "70": "DISCH_ANOTHER_FACILITY",
        "71": "DISCH_ANOTHER_ACUTE",
        "72": "DISCH_ANOTHER_ACUTE_SWING",
    }
    key = str(status).strip().zfill(2) if pd.notna(status) else "99"
    return [mapping.get(key, f"DISCH_{key}")]


def load_paired_data(cfg):
    """
    Load data and return a DataFrame of paired claims.
    Supports both 'paired' format (claimA_*/claimB_* columns)
    and 'long' format (pair_id + claim_type columns).
    """
    log.info(f"Loading data from {cfg['data_path']}")
    df = pd.read_csv(cfg["data_path"])
    log.info(f"Raw shape: {df.shape}")

    if cfg["data_format"] == "long":
        # Pivot long format into paired format
        a_prefix = cfg["col_claimA_prefix"]
        b_prefix = cfg["col_claimB_prefix"]

        index_df = df[df["claim_type"] == "index"].copy()
        readmit_df = df[df["claim_type"] == "readmission"].copy()

        index_df   = index_df.add_prefix(a_prefix).rename(
            columns={f"{a_prefix}{cfg['col_pair_id']}": cfg["col_pair_id"],
                     f"{a_prefix}{cfg['col_member_id']}": cfg["col_member_id"]}
        )
        readmit_df = readmit_df.add_prefix(b_prefix).rename(
            columns={f"{b_prefix}{cfg['col_pair_id']}": cfg["col_pair_id"]}
        )
        df = pd.merge(index_df, readmit_df, on=cfg["col_pair_id"])

    log.info(f"Paired records: {len(df)}")
    return df


# =============================================================================
# 3. PREVENTABILITY LABELING
# =============================================================================

# CCS categories considered ambulatory care sensitive /
# clinically preventable readmission conditions
PREVENTABLE_CCS = {
    "2",    # Septicemia
    "55",   # Fluid and electrolyte disorders
    "95",   # Other nervous system disorders
    "96",   # Heart valve disorders
    "97",   # Peri-endo-myocarditis
    "100",  # Acute MI
    "101",  # Coronary atherosclerosis
    "105",  # Conduction disorders
    "106",  # Cardiac dysrhythmias
    "108",  # CHF / Heart failure
    "114",  # Peripheral vascular disease
    "122",  # Pneumonia
    "127",  # COPD
    "128",  # Asthma
    "131",  # Respiratory failure
    "149",  # Biliary tract disease
    "153",  # Hip/Knee arthroplasty complications
    "157",  # Acute renal failure
    "158",  # Chronic kidney disease
    "159",  # Urinary tract infections
    "197",  # Skin and subcutaneous tissue infections
    "202",  # Superficial injury / contusion
    "237",  # Complication of device
    "238",  # Complication of medical care
    "250",  # Septicemia
    "259",  # Residual codes
}

# Short-stay threshold — readmissions within this many days
# are almost always preventable regardless of diagnosis
SHORT_READMISSION_DAYS = 7


def build_preventability_label(row, cfg):
    """
    Rule-based preventability labeling using CCS relatedness.

    A readmission is labeled preventable if ANY of:
    1. Primary diagnosis on Claim B maps to same CCS as Claim A
    2. Primary diagnosis on Claim B is an ambulatory care sensitive condition
    3. Readmission occurred within SHORT_READMISSION_DAYS
    4. DRG severity escalated AND same clinical family

    Returns 1 (preventable) or 0 (not preventable).
    """
    try:
        from pyhealth.medcode import CrossMap
        ccs_map = CrossMap.load("ICD10CM", "CCSCM")
    except Exception:
        ccs_map = None

    a_pre = cfg["col_claimA_prefix"]
    b_pre = cfg["col_claimB_prefix"]

    diag_a = parse_codes(row.get(f"{a_pre}{cfg['col_diag']}", ""))
    diag_b = parse_codes(row.get(f"{b_pre}{cfg['col_diag']}", ""))

    primary_a = diag_a[0] if diag_a else None
    primary_b = diag_b[0] if diag_b else None

    # Rule 3: Very short readmission window
    try:
        admit_b    = pd.to_datetime(row.get(f"{b_pre}{cfg['col_admit']}", pd.NaT))
        discharge_a = pd.to_datetime(row.get(f"{a_pre}{cfg['col_discharge']}", pd.NaT))
        days_between = (admit_b - discharge_a).days
        if 0 <= days_between <= SHORT_READMISSION_DAYS:
            return 1
    except Exception:
        days_between = None

    # CCS-based rules
    if ccs_map and primary_a and primary_b:
        try:
            ccs_a = ccs_map.map(primary_a)
            ccs_b = ccs_map.map(primary_b)

            # Rule 1: Same CCS category — clinically related
            if ccs_a and ccs_b and ccs_a == ccs_b:
                return 1

            # Rule 2: Readmission is an ambulatory care sensitive condition
            if ccs_b and str(ccs_b) in PREVENTABLE_CCS:
                return 1
        except Exception:
            pass

    # Rule 4: DRG severity escalation within same MDC
    drg_a = str(row.get(f"{a_pre}{cfg['col_drg']}", "")).strip()
    drg_b = str(row.get(f"{b_pre}{cfg['col_drg']}", "")).strip()
    if drg_a and drg_b and len(drg_a) == 3 and len(drg_b) == 3:
        # Same first digit = same MDC (Major Diagnostic Category)
        same_mdc = drg_a[0] == drg_b[0]
        try:
            severity_escalated = int(drg_b) < int(drg_a)  # lower DRG = higher severity
        except ValueError:
            severity_escalated = False
        if same_mdc and severity_escalated:
            return 1

    return 0


def label_dataset(df, cfg):
    """Apply preventability labels to all pairs."""
    if cfg["label_column"] and cfg["label_column"] in df.columns:
        log.info(f"Using existing label column: {cfg['label_column']}")
        df["preventable"] = df[cfg["label_column"]].astype(int)
    else:
        log.info("Auto-generating preventability labels using CCS rules...")
        df["preventable"] = df.apply(
            lambda r: build_preventability_label(r, cfg), axis=1
        )

    rate = df["preventable"].mean()
    log.info(f"Preventability rate: {rate:.1%}  "
             f"({df['preventable'].sum():,} preventable / {len(df):,} total)")
    return df


# =============================================================================
# 4. FEATURE ENGINEERING FOR RETAIN
# =============================================================================

def build_samples(df, cfg):
    """
    Convert paired claim DataFrame into RETAIN-compatible samples.
    Each sample = one pair, represented as a 2-step sequence:
        Step 1: Claim A (index admission)
        Step 2: Claim B (readmission)
    """
    a_pre = cfg["col_claimA_prefix"]
    b_pre = cfg["col_claimB_prefix"]

    samples = []

    for _, row in df.iterrows():

        # ---- Claim A features ----
        diag_a = parse_codes(row.get(f"{a_pre}{cfg['col_diag']}", ""))
        proc_a = parse_codes(row.get(f"{a_pre}{cfg['col_proc']}", ""))
        drg_a  = [f"DRG_{row.get(f'{a_pre}{cfg[\"col_drg\"]}', 'UNK')}".strip()]
        los_a  = encode_los(row.get(f"{a_pre}{cfg['col_los']}", None))
        disch_a = encode_discharge_status(row.get(f"{a_pre}discharge_status", None))

        # ---- Claim B features ----
        diag_b = parse_codes(row.get(f"{b_pre}{cfg['col_diag']}", ""))
        proc_b = parse_codes(row.get(f"{b_pre}{cfg['col_proc']}", ""))
        drg_b  = [f"DRG_{row.get(f'{b_pre}{cfg[\"col_drg\"]}', 'UNK')}".strip()]
        los_b  = encode_los(row.get(f"{b_pre}{cfg['col_los']}", None))
        disch_b = encode_discharge_status(row.get(f"{b_pre}discharge_status", None))

        # ---- Days between claims ----
        try:
            admit_b     = pd.to_datetime(row.get(f"{b_pre}{cfg['col_admit']}", pd.NaT))
            discharge_a = pd.to_datetime(row.get(f"{a_pre}{cfg['col_discharge']}", pd.NaT))
            days_between = max(0, (admit_b - discharge_a).days)
            days_token = [f"DAYS_{min(days_between, 30)}"]  # cap at 30
        except Exception:
            days_token = ["DAYS_UNKNOWN"]

        # ---- Combine all code types per claim ----
        # Each visit is a flat list of all code tokens
        visit_a = diag_a + proc_a + drg_a + los_a + disch_a
        visit_b = diag_b + proc_b + drg_b + los_b + disch_b + days_token

        # Remove empty tokens
        visit_a = [c for c in visit_a if c]
        visit_b = [c for c in visit_b if c]

        if not visit_a or not visit_b:
            continue

        sample = {
            "pair_id":         str(row.get(cfg["col_pair_id"], "")),
            "member_id":       str(row.get(cfg["col_member_id"], "")),
            "provider_tax_id": str(row.get(f"{a_pre}{cfg['col_provider']}", "")),

            # RETAIN input: list of visits, each visit is list of codes
            # Shape: [num_visits=2, num_codes_per_visit (variable)]
            "conditions":      [visit_a, visit_b],

            # Individual feature lists kept for analysis
            "diag_A":          diag_a,
            "diag_B":          diag_b,
            "drg_A":           drg_a,
            "drg_B":           drg_b,
            "los_A":           los_a,
            "los_B":           los_b,
            "disch_A":         disch_a,
            "disch_B":         disch_b,

            # Label
            "preventable":     int(row["preventable"]),
        }
        samples.append(sample)

    log.info(f"Built {len(samples):,} samples")
    return samples


# =============================================================================
# 5. VOCABULARY AND TOKENIZER
# =============================================================================

class Vocabulary:
    """Maps code strings to integer indices and back."""

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self.token2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2token = {0: self.PAD, 1: self.UNK}

    def fit(self, samples):
        """Build vocabulary from all codes seen in samples."""
        for sample in samples:
            for visit in sample["conditions"]:
                for token in visit:
                    if token not in self.token2idx:
                        idx = len(self.token2idx)
                        self.token2idx[token] = idx
                        self.idx2token[idx] = token
        log.info(f"Vocabulary size: {len(self.token2idx):,} tokens")
        return self

    def encode(self, tokens):
        """Convert list of token strings to list of integer indices."""
        return [self.token2idx.get(t, self.token2idx[self.UNK]) for t in tokens]

    def __len__(self):
        return len(self.token2idx)


# =============================================================================
# 6. GRAPH EMBEDDING CONSTRUCTION
# =============================================================================

def build_medical_graph(samples, cfg):
    """
    Build a medical code graph with:
    - Hierarchical edges from ICD-10 ontology (via PyHealth)
    - Co-occurrence edges from your claims data
    """
    log.info("Building medical code graph...")

    G = nx.Graph()

    # ---- Collect all codes ----
    all_codes = set()
    for sample in samples:
        for visit in sample["conditions"]:
            all_codes.update(visit)

    # Filter to ICD-10 codes only for hierarchy (not DRG/LOS tokens)
    icd_codes = {c for c in all_codes
                 if not c.startswith(("DRG_", "LOS_", "DISCH_", "DAYS_", "99"))}

    log.info(f"Total unique tokens: {len(all_codes)}")
    log.info(f"ICD codes for hierarchy: {len(icd_codes)}")

    # Add all tokens as nodes
    for code in all_codes:
        G.add_node(code)

    # ---- Hierarchical edges from ICD-10 tree ----
    try:
        from pyhealth.medcode import InnerMap
        icd10_map = InnerMap.load("ICD10CM")
        hierarchy_edges = 0

        for code in icd_codes:
            try:
                ancestors = icd10_map.get_ancestors(code)
                if ancestors:
                    # Direct parent — strongest connection
                    parent = ancestors[0]
                    G.add_node(parent)
                    if not G.has_edge(code, parent):
                        G.add_edge(code, parent,
                                   weight=1.0,
                                   edge_type="hierarchical_parent")
                        hierarchy_edges += 1

                    # Grandparent — weaker connection
                    if len(ancestors) > 1:
                        grandparent = ancestors[1]
                        G.add_node(grandparent)
                        if not G.has_edge(code, grandparent):
                            G.add_edge(code, grandparent,
                                       weight=0.5,
                                       edge_type="hierarchical_grandparent")
                            hierarchy_edges += 1

                    # Sibling connection via shared parent
                    # (handled implicitly through parent node)

            except Exception:
                continue

        log.info(f"Added {hierarchy_edges:,} hierarchical edges")

    except ImportError:
        log.warning("PyHealth not available — skipping ICD hierarchy edges")

    # ---- Co-occurrence edges from claims data ----
    co_occur = defaultdict(int)
    total_pairs = len(samples)

    for sample in samples:
        # Get all unique codes across both visits for this pair
        pair_codes = list(set(
            c for visit in sample["conditions"] for c in visit
        ))
        for code1, code2 in combinations(pair_codes, 2):
            key = tuple(sorted([code1, code2]))
            co_occur[key] += 1

    cooccur_edges = 0
    for (c1, c2), count in co_occur.items():
        if count >= cfg["min_cooccurrence"]:
            weight = count / total_pairs
            if G.has_edge(c1, c2):
                G[c1][c2]["weight"] = G[c1][c2].get("weight", 0) + weight
                G[c1][c2]["edge_type"] = "both"
            else:
                G.add_edge(c1, c2,
                           weight=weight,
                           edge_type="co_occurrence")
                cooccur_edges += 1

    log.info(f"Added {cooccur_edges:,} co-occurrence edges")
    log.info(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    return G


def generate_graph_embeddings(G, vocab, cfg):
    """
    Run Node2Vec on the medical code graph to produce
    embedding vectors for each code in the vocabulary.
    """
    log.info("Running Node2Vec to generate graph embeddings...")

    try:
        from node2vec import Node2Vec
    except ImportError:
        log.warning("node2vec not installed. Run: pip install node2vec")
        log.warning("Falling back to random embeddings.")
        return None

    # Only embed nodes that are in the graph
    if G.number_of_nodes() == 0:
        return None

    node2vec = Node2Vec(
        G,
        dimensions=cfg["embedding_dim"],
        walk_length=cfg["node2vec_walk_length"],
        num_walks=cfg["node2vec_num_walks"],
        p=cfg["node2vec_p"],
        q=cfg["node2vec_q"],
        workers=4,
        quiet=True,
        weight_key="weight"
    )

    emb_model = node2vec.fit(
        window=5,
        min_count=1,
        batch_words=4,
        epochs=10
    )

    # Build embedding matrix aligned to vocabulary indices
    vocab_size = len(vocab)
    emb_dim    = cfg["embedding_dim"]
    emb_matrix = np.random.normal(scale=0.01, size=(vocab_size, emb_dim))

    found = 0
    for token, idx in vocab.token2idx.items():
        if token in emb_model.wv:
            emb_matrix[idx] = emb_model.wv[token]
            found += 1
        else:
            # Try rolling up: N17.9 → N17 → N1
            for rollup_len in [3, 2, 1]:
                parent = token[:rollup_len]
                if parent in emb_model.wv:
                    emb_matrix[idx] = emb_model.wv[parent]
                    found += 1
                    break

    log.info(f"Graph embeddings assigned: {found:,} / {vocab_size:,} tokens "
             f"({found/vocab_size:.1%})")

    # Verify similarity for renal codes
    for pair in [("N17.9", "N18.3"), ("I50.43", "N18.3")]:
        if all(c in emb_model.wv for c in pair):
            sim = emb_model.wv.similarity(*pair)
            log.info(f"  Similarity {pair[0]} — {pair[1]}: {sim:.3f}")

    return emb_matrix


# =============================================================================
# 7. PYTORCH DATASET
# =============================================================================

class ReadmissionDataset(Dataset):
    """PyTorch Dataset for RETAIN readmission pairs."""

    def __init__(self, samples, vocab, max_codes_per_visit=50):
        self.samples = samples
        self.vocab   = vocab
        self.max_codes = max_codes_per_visit

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Encode each visit's codes to integers
        encoded_visits = []
        for visit in sample["conditions"]:
            encoded = self.vocab.encode(visit)
            # Truncate or pad to max_codes_per_visit
            if len(encoded) > self.max_codes:
                encoded = encoded[:self.max_codes]
            encoded_visits.append(encoded)

        return {
            "pair_id":         sample["pair_id"],
            "member_id":       sample["member_id"],
            "provider_tax_id": sample["provider_tax_id"],
            "visits":          encoded_visits,   # list of 2 lists of ints
            "diag_A":          sample["diag_A"],
            "diag_B":          sample["diag_B"],
            "drg_A":           sample["drg_A"],
            "drg_B":           sample["drg_B"],
            "los_A":           sample["los_A"],
            "disch_A":         sample["disch_A"],
            "preventable":     torch.tensor(sample["preventable"], dtype=torch.float),
        }


def collate_fn(batch):
    """
    Custom collate function.
    Pads variable-length code lists within each visit.
    """
    pair_ids    = [b["pair_id"]         for b in batch]
    member_ids  = [b["member_id"]       for b in batch]
    providers   = [b["provider_tax_id"] for b in batch]
    labels      = torch.stack([b["preventable"] for b in batch])

    # Each sample has 2 visits; each visit has variable number of codes
    # Pad codes within each visit position across the batch
    num_visits = 2
    padded_visits = []

    for v in range(num_visits):
        visit_codes = [b["visits"][v] for b in batch]
        max_len = max(len(vc) for vc in visit_codes)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        masks  = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, vc in enumerate(visit_codes):
            padded[i, :len(vc)] = torch.tensor(vc, dtype=torch.long)
            masks[i,  :len(vc)] = True
        padded_visits.append((padded, masks))

    return {
        "pair_ids":      pair_ids,
        "member_ids":    member_ids,
        "providers":     providers,
        "visits":        padded_visits,   # list of 2 tuples (codes, mask)
        "labels":        labels,
        # Pass through for audit reporting
        "diag_A":        [b["diag_A"]   for b in batch],
        "diag_B":        [b["diag_B"]   for b in batch],
        "drg_A":         [b["drg_A"]    for b in batch],
        "drg_B":         [b["drg_B"]    for b in batch],
        "los_A":         [b["los_A"]    for b in batch],
        "disch_A":       [b["disch_A"]  for b in batch],
    }


# =============================================================================
# 8. RETAIN MODEL
# =============================================================================

class RETAIN(nn.Module):
    """
    RETAIN: Reverse Time Attention Model
    Adapted for paired claim preventability classification.

    Architecture:
        For each visit v in [Claim_A, Claim_B]:
            1. Sum-pool code embeddings → visit embedding h_v
        Alpha attention: which visit drives preventability?
        Beta attention: which codes within each visit matter?
        Final: weighted sum of visit embeddings → sigmoid → preventable prob
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 dropout=0.3, pretrained_embeddings=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim

        # --- Code embedding layer ---
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        if pretrained_embeddings is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(
                    torch.FloatTensor(pretrained_embeddings)
                )
            log.info("Graph embeddings loaded into RETAIN embedding layer")

        self.dropout = nn.Dropout(dropout)

        # --- Alpha network: visit-level attention ---
        # Runs forward over visits; produces one attention weight per visit
        self.alpha_gru  = nn.GRU(embedding_dim, hidden_dim,
                                  batch_first=True, bidirectional=False)
        self.alpha_proj = nn.Linear(hidden_dim, 1)

        # --- Beta network: code-level attention within each visit ---
        # Produces attention weight per code per visit
        self.beta_gru   = nn.GRU(embedding_dim, hidden_dim,
                                  batch_first=True, bidirectional=False)
        self.beta_proj  = nn.Linear(hidden_dim, embedding_dim)

        # --- Output layer ---
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, visits):
        """
        Args:
            visits: list of 2 tuples, each (codes_tensor, mask_tensor)
                    codes_tensor: [batch, max_codes]  (padded)
                    mask_tensor:  [batch, max_codes]  (True where real code)

        Returns:
            logit:  [batch]        raw prediction score
            alpha:  [batch, 2]     visit-level attention weights
            beta:   list of 2      each [batch, max_codes] code attention
        """
        batch_size  = visits[0][0].shape[0]
        num_visits  = len(visits)  # always 2: Claim A and Claim B

        # Step 1: Embed and pool codes for each visit
        # visit_embeddings: [batch, num_visits, embedding_dim]
        visit_embs = []
        code_embs_list = []

        for codes, mask in visits:
            # codes: [batch, max_codes]
            emb = self.embedding(codes)          # [batch, max_codes, emb_dim]
            emb = self.dropout(emb)

            # Mean pooling over real codes (ignore padding)
            mask_f = mask.unsqueeze(-1).float()  # [batch, max_codes, 1]
            pooled = (emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            # pooled: [batch, emb_dim]

            visit_embs.append(pooled)
            code_embs_list.append(emb)           # keep raw for beta attention

        # Stack visits: [batch, num_visits, emb_dim]
        visit_embs_stacked = torch.stack(visit_embs, dim=1)

        # Step 2: Alpha GRU — visit-level attention
        # Input: [batch, num_visits, emb_dim]
        alpha_out, _ = self.alpha_gru(visit_embs_stacked)
        # alpha_out: [batch, num_visits, hidden_dim]

        alpha_scores = self.alpha_proj(alpha_out).squeeze(-1)
        # alpha_scores: [batch, num_visits]

        alpha = torch.softmax(alpha_scores, dim=1)
        # alpha: [batch, num_visits]  — sums to 1 across visits

        # Step 3: Beta GRU — code-level attention within each visit
        beta_out_all, _ = self.beta_gru(visit_embs_stacked)
        # beta_out_all: [batch, num_visits, hidden_dim]

        beta_list = []    # attention weights per code per visit
        weighted_visits = []

        for v_idx, (codes, mask) in enumerate(visits):
            # Beta projection for this visit
            beta_proj = torch.tanh(
                self.beta_proj(beta_out_all[:, v_idx, :])
            )
            # beta_proj: [batch, emb_dim]

            # Code-level attention scores
            code_embs = code_embs_list[v_idx]    # [batch, max_codes, emb_dim]
            beta_scores = (code_embs * beta_proj.unsqueeze(1)).sum(dim=-1)
            # beta_scores: [batch, max_codes]

            # Mask out padding before softmax
            beta_scores = beta_scores.masked_fill(~mask, -1e9)
            beta = torch.softmax(beta_scores, dim=1)
            # beta: [batch, max_codes]

            beta_list.append(beta)

            # Weighted code embeddings for this visit
            weighted_code_emb = (code_embs * beta.unsqueeze(-1)).sum(dim=1)
            # weighted_code_emb: [batch, emb_dim]

            # Scale by visit-level alpha
            visit_alpha = alpha[:, v_idx].unsqueeze(-1)
            weighted_visits.append(visit_alpha * weighted_code_emb)

        # Step 4: Sum weighted visit representations
        # context: [batch, emb_dim]
        context = sum(weighted_visits)
        context = self.dropout(context)

        # Step 5: Output
        logit = self.output_layer(context).squeeze(-1)  # [batch]

        return logit, alpha, beta_list


# =============================================================================
# 9. TRAINING LOOP
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_probs, all_labels = [], []

    for batch in loader:
        visits = [(v[0].to(device), v[1].to(device)) for v in batch["visits"]]
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logit, alpha, beta = model(visits)
        loss = criterion(logit, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logit).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    try:
        auprc = average_precision_score(all_labels, all_probs)
    except Exception:
        auprc = 0.0

    return avg_loss, auprc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            visits = [(v[0].to(device), v[1].to(device)) for v in batch["visits"]]
            labels = batch["labels"].to(device)

            logit, alpha, beta = model(visits)
            loss = criterion(logit, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logit).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except Exception:
        auroc = auprc = 0.0

    return avg_loss, auroc, auprc, all_probs, all_labels


# =============================================================================
# 10. INFERENCE AND AUDIT REPORT GENERATION
# =============================================================================

def run_inference(model, loader, device):
    """
    Run full inference and collect:
    - Per-pair preventability scores
    - Alpha (visit-level) attention weights
    - Beta (code-level) attention weights
    """
    model.eval()
    records = []

    with torch.no_grad():
        for batch in loader:
            visits = [(v[0].to(device), v[1].to(device)) for v in batch["visits"]]
            labels = batch["labels"]

            logit, alpha, beta_list = model(visits)
            probs = torch.sigmoid(logit).cpu().numpy()

            # alpha: [batch, 2]
            alpha_np = alpha.cpu().numpy()

            # beta_list: [visit_A_beta, visit_B_beta]
            # each: [batch, max_codes]
            beta_A = beta_list[0].cpu().numpy()
            beta_B = beta_list[1].cpu().numpy()

            for i in range(len(batch["pair_ids"])):
                # Top attended codes for Claim A
                diag_a    = batch["diag_A"][i]
                beta_a_i  = beta_A[i][:len(diag_a)]
                top_a_idx = np.argsort(beta_a_i)[::-1][:5]
                top_codes_A = [(diag_a[j], float(beta_a_i[j]))
                               for j in top_a_idx if j < len(diag_a)]

                # Top attended codes for Claim B
                diag_b    = batch["diag_B"][i]
                beta_b_i  = beta_B[i][:len(diag_b)]
                top_b_idx = np.argsort(beta_b_i)[::-1][:5]
                top_codes_B = [(diag_b[j], float(beta_b_i[j]))
                               for j in top_b_idx if j < len(diag_b)]

                records.append({
                    "pair_id":            batch["pair_ids"][i],
                    "member_id":          batch["member_ids"][i],
                    "provider_tax_id":    batch["providers"][i],
                    "preventable_prob":   float(probs[i]),
                    "predicted_preventable": int(probs[i] >= 0.5),
                    "actual_preventable": int(labels[i].item()),

                    # Which claim drove the decision?
                    "alpha_claimA":       float(alpha_np[i, 0]),
                    "alpha_claimB":       float(alpha_np[i, 1]),
                    "index_claim_driven": bool(alpha_np[i, 0] > alpha_np[i, 1]),

                    # Top diagnosis codes driving the score
                    "top_codes_claimA":   str(top_codes_A),
                    "top_codes_claimB":   str(top_codes_B),

                    # Features for context
                    "drg_A":              batch["drg_A"][i],
                    "drg_B":              batch["drg_B"][i],
                    "los_A":              batch["los_A"][i],
                    "disch_A":            batch["disch_A"][i],
                })

    return pd.DataFrame(records)


def build_provider_audit_queue(results_df, cfg):
    """
    Aggregate inference results by provider to build
    a ranked audit queue for overpayment investigation.
    """
    from scipy import stats

    provider_stats = results_df.groupby("provider_tax_id").agg(
        total_pairs=("pair_id", "count"),
        actual_preventable_rate=("actual_preventable", "mean"),
        mean_predicted_prob=("preventable_prob", "mean"),
        high_confidence_preventable=("preventable_prob", lambda x: (x >= 0.8).sum()),
        low_confidence_preventable=("preventable_prob", lambda x: (x >= 0.5).sum()),
        index_driven_count=("index_claim_driven", "sum"),
        mean_alpha_claimA=("alpha_claimA", "mean"),
    ).reset_index()

    # Performance gap: actual − predicted (positive = worse than expected)
    provider_stats["performance_gap"] = (
        provider_stats["actual_preventable_rate"] -
        provider_stats["mean_predicted_prob"]
    )

    # Z-score gap across all providers
    if len(provider_stats) > 1:
        provider_stats["gap_zscore"] = stats.zscore(
            provider_stats["performance_gap"].fillna(0)
        )
    else:
        provider_stats["gap_zscore"] = 0.0

    # Estimated dollar exposure
    # Use DRG-weighted average readmission payment from your data
    # Default: $12,000 per readmission (adjust to your payer)
    AVG_PAYMENT = 12_000
    provider_stats["estimated_exposure_usd"] = (
        provider_stats["high_confidence_preventable"] * AVG_PAYMENT
    )

    # Priority score: gap * volume * confidence
    provider_stats["priority_score"] = (
        provider_stats["gap_zscore"].clip(lower=0) *
        np.log1p(provider_stats["total_pairs"]) *
        provider_stats["mean_predicted_prob"]
    )

    # Filter to providers with sufficient volume
    MIN_PAIRS = 10
    audit_queue = (
        provider_stats[provider_stats["total_pairs"] >= MIN_PAIRS]
        .sort_values("priority_score", ascending=False)
        .head(cfg["top_n_providers"])
        .reset_index(drop=True)
    )

    audit_queue["rank"] = audit_queue.index + 1

    return audit_queue


# =============================================================================
# 11. MAIN PIPELINE
# =============================================================================

def main():
    cfg    = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ---- Step 1: Load and label data ----
    df = load_paired_data(cfg)
    df = label_dataset(df, cfg)

    # ---- Step 2: Build samples ----
    samples = build_samples(df, cfg)
    if not samples:
        log.error("No valid samples built. Check column names in CONFIG.")
        return

    # ---- Step 3: Train / val / test split (split by member_id) ----
    member_ids = np.array([s["member_id"] for s in samples])
    unique_members = np.unique(member_ids)
    np.random.seed(42)
    np.random.shuffle(unique_members)

    n = len(unique_members)
    train_members = set(unique_members[:int(0.70 * n)])
    val_members   = set(unique_members[int(0.70 * n):int(0.85 * n)])
    test_members  = set(unique_members[int(0.85 * n):])

    train_samples = [s for s in samples if s["member_id"] in train_members]
    val_samples   = [s for s in samples if s["member_id"] in val_members]
    test_samples  = [s for s in samples if s["member_id"] in test_members]

    log.info(f"Train: {len(train_samples):,} | Val: {len(val_samples):,} | "
             f"Test: {len(test_samples):,}")

    # ---- Step 4: Build vocabulary from training data only ----
    vocab = Vocabulary()
    vocab.fit(train_samples)

    # ---- Step 5: Build graph and generate embeddings ----
    pretrained_emb = None
    if cfg["use_graph_embeddings"]:
        G = build_medical_graph(train_samples, cfg)
        pretrained_emb = generate_graph_embeddings(G, vocab, cfg)

    # ---- Step 6: Create datasets and loaders ----
    train_ds = ReadmissionDataset(train_samples, vocab)
    val_ds   = ReadmissionDataset(val_samples,   vocab)
    test_ds  = ReadmissionDataset(test_samples,  vocab)
    all_ds   = ReadmissionDataset(samples,        vocab)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=collate_fn)
    all_loader   = DataLoader(all_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=collate_fn)

    # ---- Step 7: Initialize RETAIN ----
    model = RETAIN(
        vocab_size=len(vocab),
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        pretrained_embeddings=pretrained_emb
    ).to(device)

    # Weighted loss for class imbalance
    pos_weight = torch.tensor([cfg["pos_class_weight"]]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- Step 8: Phase 1 — Train with frozen embeddings ----
    if cfg["use_graph_embeddings"] and pretrained_emb is not None:
        log.info("Phase 1: Training with frozen graph embeddings...")
        model.embedding.weight.requires_grad = False
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["learning_rate"]
        )
        for epoch in range(cfg["freeze_epochs"]):
            train_loss, train_prc = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_auroc, val_prc, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            log.info(f"[Frozen Epoch {epoch+1:02d}] "
                     f"Train Loss: {train_loss:.4f} AUPRC: {train_prc:.4f} | "
                     f"Val AUROC: {val_auroc:.4f} AUPRC: {val_prc:.4f}")

        # Unfreeze embeddings
        model.embedding.weight.requires_grad = True
        log.info("Phase 2: Fine-tuning with unfrozen embeddings...")
    else:
        log.info("Training RETAIN from scratch...")

    # ---- Step 9: Phase 2 — Full fine-tuning ----
    optimizer   = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    best_val_prc = 0.0
    best_ckpt    = os.path.join(cfg["checkpoint_dir"], "retain_best.pt")

    for epoch in range(cfg["num_epochs"]):
        train_loss, train_prc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auroc, val_prc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        log.info(f"[Epoch {epoch+1:02d}/{cfg['num_epochs']}] "
                 f"Train Loss: {train_loss:.4f} AUPRC: {train_prc:.4f} | "
                 f"Val AUROC: {val_auroc:.4f} AUPRC: {val_prc:.4f}")

        if val_prc > best_val_prc:
            best_val_prc = val_prc
            torch.save(model.state_dict(), best_ckpt)
            log.info(f"  ✓ Best model saved (Val AUPRC: {val_prc:.4f})")

    # ---- Step 10: Test evaluation ----
    log.info("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    _, test_auroc, test_prc, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    preds = [1 if p >= 0.5 else 0 for p in test_probs]

    log.info("\n" + "="*60)
    log.info("TEST SET RESULTS")
    log.info("="*60)
    log.info(f"AUROC:  {test_auroc:.4f}")
    log.info(f"AUPRC:  {test_prc:.4f}")
    log.info("\nClassification Report:")
    log.info("\n" + classification_report(
        test_labels, preds,
        target_names=["Not Preventable", "Preventable"]
    ))

    # ---- Step 11: Full inference on all pairs ----
    log.info("\nRunning inference on full dataset for audit reporting...")
    results_df = run_inference(model, all_loader, device)

    # ---- Step 12: Provider audit queue ----
    log.info("\nBuilding provider audit queue...")
    audit_queue = build_provider_audit_queue(results_df, cfg)

    # ---- Step 13: Save outputs ----
    results_path = os.path.join(cfg["output_dir"], "pair_level_scores.csv")
    audit_path   = os.path.join(cfg["output_dir"], "provider_audit_queue.csv")

    results_df.to_csv(results_path, index=False)
    audit_queue.to_csv(audit_path,  index=False)

    log.info(f"\nPair-level scores saved to:    {results_path}")
    log.info(f"Provider audit queue saved to: {audit_path}")

    # ---- Step 14: Print top flagged providers ----
    print("\n" + "="*60)
    print("TOP PROVIDERS FLAGGED FOR OVERPAYMENT REVIEW")
    print("="*60)
    display_cols = [
        "rank", "provider_tax_id", "total_pairs",
        "actual_preventable_rate", "mean_predicted_prob",
        "performance_gap", "high_confidence_preventable",
        "estimated_exposure_usd", "priority_score"
    ]
    existing_cols = [c for c in display_cols if c in audit_queue.columns]
    print(audit_queue[existing_cols].to_string(index=False))

    # ---- Step 15: Print sample audit records for top provider ----
    if not audit_queue.empty:
        top_provider = audit_queue.iloc[0]["provider_tax_id"]
        top_cases = results_df[
            (results_df["provider_tax_id"] == top_provider) &
            (results_df["preventable_prob"] >= 0.8)
        ].head(5)

        print(f"\n{'='*60}")
        print(f"SAMPLE HIGH-CONFIDENCE CASES — Provider: {top_provider}")
        print("="*60)
        for _, case in top_cases.iterrows():
            print(f"\n  Pair ID:          {case['pair_id']}")
            print(f"  Member ID:        {case['member_id']}")
            print(f"  Preventable Prob: {case['preventable_prob']:.3f}")
            print(f"  Claim A Alpha:    {case['alpha_claimA']:.3f}  "
                  f"(index claim drove decision: {case['index_claim_driven']})")
            print(f"  DRG A → B:        {case['drg_A']} → {case['drg_B']}")
            print(f"  LOS A:            {case['los_A']}")
            print(f"  Discharge A:      {case['disch_A']}")
            print(f"  Top codes Claim A: {case['top_codes_claimA']}")
            print(f"  Top codes Claim B: {case['top_codes_claimB']}")

    log.info("\nPipeline complete.")


# =============================================================================
# 12. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
