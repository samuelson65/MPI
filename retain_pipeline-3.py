"""
=============================================================================
RETAIN Readmission Preventability Pipeline
=============================================================================
USAGE:
    python retain_pipeline.py

INPUTS  (set paths in SECTION 0):
    TRAIN_PATH   — labeled paired claims CSV  (includes 'preventable' column)
    SCORE_PATH   — unlabeled paired claims CSV (no 'preventable' column)

OUTPUTS (written to OUTPUT_DIR):
    scored_with_explainability.csv  — every scoring pair scored + explained
    provider_audit_queue.csv        — providers ranked by overpayment risk
    retain_model.pkl                — saved model (reusable)

COLUMN FORMAT REQUIRED IN BOTH CSVs:
    pair_id                     unique identifier for each index+readmission pair
    member_id                   patient / beneficiary identifier
    provider_tax_id             single provider tax ID shared by both claims
    claimA_claim_id             index claim ID
    claimA_drg_code             DRG code on index claim
    claimA_diag_codes           pipe-delimited ICD-10 codes e.g. I50.43|E11.65|I10
                                (script uses only first 3 chars: I50|E11|I10)
    claimA_los                  length of stay in days (integer)
    claimA_discharge_status     discharge status code e.g. 01=home 03=SNF 07=AMA
    claimA_admit_date           YYYY-MM-DD
    claimA_discharge_date       YYYY-MM-DD
    claimB_claim_id             readmission claim ID
    claimB_drg_code             DRG code on readmission claim
    claimB_diag_codes           pipe-delimited ICD-10 codes
                                (script uses only first 3 chars: I50|N18 etc)
    claimB_los                  length of stay in days
    claimB_discharge_status     discharge status code
    claimB_admit_date           YYYY-MM-DD
    claimB_discharge_date       YYYY-MM-DD
    preventable                 0 or 1  --- TRAINING CSV ONLY ---

REMOVED COLUMNS (no longer needed):
    claimA_proc_codes / claimB_proc_codes    procedure codes dropped from features
    claimA_provider_tax_id / claimB_*        replaced by single provider_tax_id

DEPENDENCIES (auto-installed if missing):
    pandas  numpy  scipy  scikit-learn  networkx
=============================================================================
"""

# =============================================================================
# SECTION 0 — CONFIGURATION  ← edit these paths and settings
# =============================================================================

TRAIN_PATH    = "training_data.csv"    # labeled CSV with 'preventable' column
SCORE_PATH    = "scoring_data.csv"     # unlabeled CSV to score
LABEL_COL     = "preventable"          # label column name in training CSV

OUTPUT_DIR    = "outputs"              # folder for all output files
MODEL_PATH    = "outputs/retain_model.pkl"  # where trained model is saved

# Model hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIM    = 64
LEARNING_RATE = 0.01
NUM_EPOCHS    = 60
FREEZE_EPOCHS = 10
BATCH_SIZE    = 32
DROPOUT_RATE  = 0.2
POS_WEIGHT    = 3.0    # upweight preventable class (handles imbalance)

# Graph embedding settings
MIN_COOCCUR   = 3      # min co-occurrence count for graph edge
N2V_WALKS     = 40     # random walks per node
N2V_WALK_LEN  = 15     # steps per walk
N2V_WINDOW    = 5      # Word2Vec context window

# Audit settings
AVG_PAYMENT   = 12_000  # average readmission payment $ — adjust to your payer
TOP_N_PROVS   = 15      # how many providers to include in audit queue

# =============================================================================
# SECTION 1 — AUTO-INSTALL + IMPORTS
# =============================================================================

import subprocess, sys, os, warnings, logging, random, pickle
from collections import defaultdict
from itertools import combinations

for pkg in ["pandas","numpy","scipy","scikit-learn","networkx"]:
    imp = pkg.replace("-","_").replace("scikit_learn","sklearn")
    try:
        __import__(imp)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,
                               "--break-system-packages","--quiet"])

import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import expit
from scipy.stats import zscore as sp_zscore
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)
np.random.seed(42); random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 2 — CLINICAL CODE LOOKUP TABLES
# (add your own codes here to get richer explainability descriptions)
# =============================================================================

ICD_DESC = {
    "I50.43":"Acute-on-chronic systolic heart failure",
    "I50.42":"Chronic systolic heart failure",
    "I50.32":"Chronic diastolic heart failure",
    "I50.9":"Heart failure unspecified",
    "I50.1":"Left ventricular failure",
    "E11.65":"Type 2 diabetes with hyperglycemia",
    "E11.9":"Type 2 diabetes without complications",
    "E11.649":"Type 2 diabetes with hypoglycemia",
    "I10":"Essential hypertension",
    "I11.9":"Hypertensive heart disease",
    "I12.9":"Hypertensive chronic kidney disease",
    "I13.10":"Hypertensive heart and CKD",
    "N17.9":"Acute kidney failure unspecified",
    "N18.3":"Chronic kidney disease stage 3",
    "N18.4":"Chronic kidney disease stage 4",
    "N18.5":"Chronic kidney disease stage 5",
    "N19":"Unspecified kidney failure",
    "J18.9":"Pneumonia unspecified organism",
    "J18.1":"Lobar pneumonia",
    "J15.9":"Unspecified bacterial pneumonia",
    "J13":"Pneumonia due to Streptococcus pneumoniae",
    "J44.1":"COPD with acute exacerbation",
    "J44.0":"COPD with acute lower respiratory infection",
    "J44.9":"COPD unspecified",
    "A41.9":"Sepsis unspecified organism",
    "A41.51":"Sepsis due to Escherichia coli",
    "A40.9":"Streptococcal sepsis unspecified",
    "Z87.891":"Personal history of nicotine dependence",
    "I25.10":"Atherosclerotic heart disease of native artery",
    "I48.91":"Unspecified atrial fibrillation",
    "E78.5":"Hyperlipidemia unspecified",
    "S72.001A":"Fracture of neck of right femur",
    "W19.XXXA":"Unspecified fall",
}

PROC_DESC = {
    "99231":"Hospital visit low complexity",
    "99232":"Hospital visit moderate complexity",
    "99233":"Hospital visit high complexity",
    "99291":"Critical care first 30-74 minutes",
    "93010":"Electrocardiogram interpretation",
    "93306":"Echocardiography with Doppler",
    "93458":"Left heart catheterization",
    "31622":"Bronchoscopy with brushings",
    "94002":"Ventilation management inpatient",
    "90935":"Hemodialysis one evaluation",
    "90937":"Hemodialysis repeated evaluation",
    "99214":"Office visit moderate complexity",
    "36415":"Venous blood collection",
}

DRG_DESC = {
    "291":"Heart failure with MCC",
    "292":"Heart failure with CC",
    "293":"Heart failure without CC/MCC",
    "194":"Simple pneumonia with MCC",
    "195":"Simple pneumonia with CC",
    "690":"COPD with MCC",
    "638":"Diabetes with MCC",
    "641":"Misc nutrition/metabolism disorders with MCC",
    "682":"Renal failure with MCC",
    "683":"Renal failure with CC",
    "872":"Septicemia without mechanical ventilation",
    "902":"Wound debridements for injuries",
}

DISCH_DESC = {
    "01":"Discharged home — no services arranged",
    "02":"Transferred to short-term hospital",
    "03":"Discharged to skilled nursing facility (SNF)",
    "06":"Discharged home with home health services",
    "07":"Left against medical advice (AMA)",
    "20":"Expired during admission",
    "50":"Discharged to hospice — home",
    "62":"Discharged to inpatient rehabilitation",
}

# ICD-10 to CCS category mapping (add codes as needed)
ICD_CCS = {
    "I50.43":"108","I50.42":"108","I50.32":"108","I50.9":"108","I50.1":"108",
    "E11.65":"50","E11.9":"50","E11.649":"50",
    "I10":"98","I11.9":"98","I12.9":"98","I13.10":"98",
    "N17.9":"157","N18.3":"158","N18.4":"158","N18.5":"158","N19":"157",
    "J18.9":"122","J18.1":"122","J15.9":"122","J13":"122",
    "J44.1":"127","J44.0":"127","J44.9":"127",
    "A41.9":"2","A41.51":"2","A40.9":"2",
    "I25.10":"101","I48.91":"106","E78.5":"53",
}
CCS_NAME = {
    "2":"Septicemia","50":"Diabetes mellitus","53":"Hyperlipidemia",
    "98":"Hypertension","101":"Coronary artery disease","106":"Cardiac dysrhythmias",
    "108":"Congestive heart failure","122":"Pneumonia","127":"COPD",
    "157":"Acute renal failure","158":"Chronic kidney disease",
}
PREV_CCS = {"108","100","122","127","638","157","158","2","101","106","641","683"}

# =============================================================================
# SECTION 3 — DATA UTILITIES
# =============================================================================

def parse_codes(s):
    if pd.isna(s) or str(s).strip() == "": return []
    return [c.strip().upper()
            for c in str(s).replace("|",",").replace(";",",").split(",")
            if c.strip()]

def encode_los(v):
    try:
        d = float(v)
        if d <= 1:    return "LOS_1D"
        elif d <= 3:  return "LOS_2_3D"
        elif d <= 7:  return "LOS_4_7D"
        elif d <= 14: return "LOS_8_14D"
        else:         return "LOS_15PD"
    except:           return "LOS_UNK"

DISCH_TOK = {
    "01":"DISCH_HOME","02":"DISCH_SHORT_HOSP","03":"DISCH_SNF",
    "06":"DISCH_HOME_HEALTH","07":"DISCH_AMA","20":"DISCH_EXPIRED",
    "50":"DISCH_HOSPICE","62":"DISCH_REHAB",
}
def encode_disch(v):
    return DISCH_TOK.get(str(v).strip().zfill(2), f"DISCH_{v}")

def gap_days(row):
    try:
        return max(0,(pd.to_datetime(row["claimB_admit_date"]) -
                      pd.to_datetime(row["claimA_discharge_date"])).days)
    except: return -1

def truncate_diag(code):
    """
    Keep only the first 3 characters of an ICD-10 code.
    I50.43 → I50 | E11.65 → E11 | N18.3 → N18
    This groups related codes at the category level rather
    than the specific subcategory level.
    """
    clean = code.replace(".", "")   # remove decimal: I5043 → strip to 3
    return clean[:3]                # I50, E11, N18, J44 etc

def encode_days_gap(days):
    """
    Bin days between discharge and readmission into named categories.
    Used as a token on BOTH visits so the model sees the gap as a feature
    of the pair rather than only of the readmission visit.
    """
    try:
        d = int(days)
        if d <= 3:   return "GAP_0_3D"    # very rapid — strong preventability signal
        elif d <= 7:  return "GAP_4_7D"   # rapid
        elif d <= 14: return "GAP_8_14D"  # moderate
        elif d <= 21: return "GAP_15_21D" # borderline
        elif d <= 30: return "GAP_22_30D" # within window
        else:         return "GAP_OVER30" # outside 30-day window
    except:           return "GAP_UNK"

# =============================================================================
# SECTION 4 — PREVENTABILITY LABELING  (training data only)
# =============================================================================

def auto_label(row):
    """
    Rule-based preventability label using CCS relatedness + timing.
    Used only when training data does not have a label column.
    """
    da = parse_codes(row.get("claimA_diag_codes",""))
    db = parse_codes(row.get("claimB_diag_codes",""))
    pa = da[0] if da else None
    pb = db[0] if db else None
    try:
        gap = (pd.to_datetime(row["claimB_admit_date"]) -
               pd.to_datetime(row["claimA_discharge_date"])).days
        if 0 <= gap <= 7: return 1
    except: pass
    if pa and pb:
        ca, cb = ICD_CCS.get(pa), ICD_CCS.get(pb)
        if ca and cb and ca == cb:    return 1
        if cb and cb in PREV_CCS:     return 1
    try:
        da2, db2 = str(row.get("claimA_drg_code","")), str(row.get("claimB_drg_code",""))
        if len(da2)==3 and len(db2)==3 and da2[0]==db2[0] and int(db2)<int(da2):
            return 1
    except: pass
    return 0

# =============================================================================
# SECTION 5 — FEATURE ENGINEERING
# =============================================================================

def visit_tokens(row, prefix, days_gap_token):
    """
    Build token list for one visit.

    Changes from previous version:
      - Procedure codes REMOVED entirely
      - Diagnosis codes truncated to first 3 characters (category level)
      - days_gap_token appended to BOTH visits (not just visit B)
      - provider_tax_id read from single shared column
    """
    # Diagnosis codes — first 3 chars only
    # I50.43|E11.65|I10  →  [I50, E11, I10]
    raw_diag = parse_codes(row.get(f"{prefix}diag_codes",""))
    diag     = [f"DX_{truncate_diag(c)}" for c in raw_diag if c]

    # DRG, LOS, discharge status
    drg   = [f"DRG_{row.get(f'{prefix}drg_code','UNK')}"]
    los   = [encode_los(row.get(f"{prefix}los", None))]
    disch = [encode_disch(row.get(f"{prefix}discharge_status","99"))]

    # Days gap appended to both visits so model sees it at both time steps
    gap   = [days_gap_token]

    return [t for t in diag + drg + los + disch + gap if t]

def build_samples(df, has_label=True):
    samples = []
    for _, row in df.iterrows():
        db            = gap_days(row)
        gap_tok       = encode_days_gap(db)   # single categorical token

        vA = visit_tokens(row, "claimA_", gap_tok)
        vB = visit_tokens(row, "claimB_", gap_tok)
        if not vA or not vB: continue

        # Raw diagnosis codes kept in full for output display only
        raw_diag_A = parse_codes(row.get("claimA_diag_codes",""))
        raw_diag_B = parse_codes(row.get("claimB_diag_codes",""))

        # Truncated (3-char) diagnosis codes used by the model
        trunc_diag_A = [truncate_diag(c) for c in raw_diag_A if c]
        trunc_diag_B = [truncate_diag(c) for c in raw_diag_B if c]

        s = {
            "pair_id":         str(row.get("pair_id","")),
            "member_id":       str(row.get("member_id","")),
            # Single shared provider_tax_id column
            "provider_tax_id": str(row.get("provider_tax_id","")),
            "claimA_id":       str(row.get("claimA_claim_id","")),
            "claimB_id":       str(row.get("claimB_claim_id","")),
            "visit_A":         vA,
            "visit_B":         vB,
            # Full codes for display
            "diag_A":          raw_diag_A,
            "diag_B":          raw_diag_B,
            # Truncated codes used by model
            "trunc_diag_A":    trunc_diag_A,
            "trunc_diag_B":    trunc_diag_B,
            "drg_A":           str(row.get("claimA_drg_code","")),
            "drg_B":           str(row.get("claimB_drg_code","")),
            "los_A":           str(row.get("claimA_los","")),
            "los_B":           str(row.get("claimB_los","")),
            "disch_A":         str(row.get("claimA_discharge_status","")),
            "disch_B":         str(row.get("claimB_discharge_status","")),
            "admit_A":         str(row.get("claimA_admit_date","")),
            "admit_B":         str(row.get("claimB_admit_date","")),
            "days_between":    db,
            "days_gap_token":  gap_tok,
            "preventable":     int(row[LABEL_COL]) if has_label else -1,
        }
        samples.append(s)
    return samples

# =============================================================================
# SECTION 6 — VOCABULARY
# =============================================================================

class Vocab:
    PAD, UNK = "<PAD>", "<UNK>"
    def __init__(self):
        self.t2i = {self.PAD:0, self.UNK:1}
        self.i2t = {0:self.PAD, 1:self.UNK}
    def fit(self, samples):
        for s in samples:
            for t in s["visit_A"] + s["visit_B"]:
                if t not in self.t2i:
                    i = len(self.t2i)
                    self.t2i[t] = i; self.i2t[i] = t
        log.info(f"Vocabulary: {len(self.t2i):,} tokens")
        return self
    def enc(self, tokens):
        return [self.t2i.get(t, self.t2i[self.UNK]) for t in tokens]
    def __len__(self): return len(self.t2i)

# =============================================================================
# SECTION 7 — GRAPH EMBEDDINGS (pure NumPy — no torch/GPU needed)
# =============================================================================

def build_graph(samples):
    log.info("Building medical code graph...")
    G = nx.Graph()
    all_toks = set()
    for s in samples: all_toks.update(s["visit_A"] + s["visit_B"])
    for t in all_toks: G.add_node(t)

    # Hierarchical edges: ICD codes sharing the same 3-char prefix are siblings
    icd = [t for t in all_toks if not t.startswith(("DRG_","LOS_","DISCH_","DAYS_"))]
    by_parent = defaultdict(list)
    for t in icd:
        p = t.split(".")[0]; by_parent[p].append(t)
    for parent, children in by_parent.items():
        G.add_node(parent)
        for c in children:
            G.add_edge(c, parent, weight=1.0)      # direct parent = strongest
        for c1,c2 in combinations(children,2):
            if not G.has_edge(c1,c2):
                G.add_edge(c1, c2, weight=0.5)     # siblings = moderate
        gp = parent[0]; G.add_node(gp)
        G.add_edge(parent, gp, weight=0.3)         # grandparent = weakest

    # Co-occurrence edges from training data
    co = defaultdict(int)
    for s in samples:
        toks = list(set(s["visit_A"] + s["visit_B"]))
        for t1,t2 in combinations(toks,2): co[tuple(sorted([t1,t2]))] += 1
    added = 0
    for (t1,t2),cnt in co.items():
        if cnt >= MIN_COOCCUR:
            w = cnt / len(samples)
            if G.has_edge(t1,t2): G[t1][t2]["weight"] += w
            else: G.add_edge(t1,t2,weight=w); added+=1
    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
             f"({added} co-occurrence edges)")
    return G

def _random_walks(G, nodes, nw, wl):
    nl = list(nodes); walks = []
    for _ in range(nw):
        random.shuffle(nl)
        for start in nl:
            walk = [start]
            for _ in range(wl-1):
                cur = walk[-1]; nb = list(G.neighbors(cur))
                if not nb: break
                wts = np.array([G[cur][n].get("weight",1.0) for n in nb])
                wts /= wts.sum()
                walk.append(np.random.choice(nb, p=wts))
            walks.append(walk)
    return walks

def _skip_gram(walks, vocab_size, t2i, dim, window, epochs=5, lr=0.025):
    W  = np.random.normal(0,0.01,(vocab_size,dim)).astype(np.float32)
    W2 = np.random.normal(0,0.01,(vocab_size,dim)).astype(np.float32)
    for ep in range(epochs):
        tl=0.0; random.shuffle(walks)
        for walk in walks:
            ids = [t2i.get(t,1) for t in walk]
            for i,center in enumerate(ids):
                ctx_range = range(max(0,i-window), min(len(ids),i+window+1))
                for j in ctx_range:
                    if j==i: continue
                    ctx = ids[j]
                    sp = expit(np.dot(W[center],W2[ctx])); ep_ = sp-1.0
                    tl += -np.log(sp+1e-9)
                    gc_p = ep_*W2[ctx]; gc_ctx = ep_*W[center]
                    negs = np.random.randint(2,vocab_size,5); gc_n = np.zeros(dim)
                    for neg in negs:
                        sn=expit(np.dot(W[center],W2[neg])); en=sn
                        tl+=-np.log(1-sn+1e-9); gc_n+=en*W2[neg]; W2[neg]-=lr*en*W[center]
                    W[center]-=lr*(gc_p+gc_n); W2[ctx]-=lr*gc_ctx
        log.info(f"  Embedding epoch {ep+1}/{epochs}  loss={tl:.0f}")
    return W

def generate_embeddings(G, vocab):
    log.info("Generating graph embeddings via Node2Vec skip-gram...")
    nodes = set(vocab.t2i.keys()) & set(G.nodes())
    walks = _random_walks(G, nodes, N2V_WALKS, N2V_WALK_LEN)
    log.info(f"  {len(walks)} random walks")
    emb = _skip_gram(walks, len(vocab), vocab.t2i, EMBEDDING_DIM, N2V_WINDOW)
    # Sanity check — renal codes should be similar
    for p in [("N17.9","N18.3"),("I50.43","N18.3")]:
        i1,i2 = vocab.t2i.get(p[0],0), vocab.t2i.get(p[1],0)
        if i1>1 and i2>1:
            v1,v2 = emb[i1],emb[i2]
            sim = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
            log.info(f"  Embedding similarity {p[0]}—{p[1]}: {sim:.3f}")
    return emb

# =============================================================================
# SECTION 8 — RETAIN MODEL (pure NumPy)
# =============================================================================

class RETAIN:
    """
    RETAIN: Reverse Time Attention Model — pure NumPy implementation.

    For each pair [Claim_A, Claim_B]:
      1. Embed + mean-pool each visit's codes  →  h_A, h_B
      2. Alpha GRU over [h_A, h_B]            →  alpha_A, alpha_B  (visit weights)
      3. Beta GRU + projection per visit       →  beta per code     (code weights)
      4. Context = alpha_A*(beta-weighted h_A) + alpha_B*(beta-weighted h_B)
      5. Sigmoid(W_out @ context)              →  preventable probability
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim,
                 dropout=0.2, pretrained=None):
        self.emb_dim    = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout    = dropout
        self.training   = True

        self.E = (pretrained.copy().astype(np.float32)
                  if pretrained is not None
                  else np.random.randn(vocab_size, emb_dim).astype(np.float32)*0.01)

        d, h = emb_dim, hidden_dim
        def W(r,c): return (np.random.randn(r,c)*np.sqrt(2.0/r)).astype(np.float32)

        # Alpha GRU weights
        self.Wza=W(d,h); self.Uza=W(h,h); self.bza=np.zeros(h)
        self.Wra=W(d,h); self.Ura=W(h,h); self.bra=np.zeros(h)
        self.Wha=W(d,h); self.Uha=W(h,h); self.bha=np.zeros(h)
        self.Wal=W(h,1); self.bal=np.zeros(1)

        # Beta GRU weights
        self.Wzb=W(d,h); self.Uzb=W(h,h); self.bzb=np.zeros(h)
        self.Wrb=W(d,h); self.Urb=W(h,h); self.brb=np.zeros(h)
        self.Whb=W(d,h); self.Uhb=W(h,h); self.bhb=np.zeros(h)
        self.Wbt=W(h,d); self.bbt=np.zeros(d)

        # Output
        self.Wo=W(d,1); self.bo=np.zeros(1)

        self._pn = ["E","Wza","Uza","bza","Wra","Ura","bra","Wha","Uha","bha",
                    "Wal","bal","Wzb","Uzb","bzb","Wrb","Urb","brb","Whb","Uhb","bhb",
                    "Wbt","bbt","Wo","bo"]
        self._g  = {n: np.zeros_like(getattr(self,n)) for n in self._pn}
        self._m  = {n: np.zeros_like(getattr(self,n)) for n in self._pn}
        self._v  = {n: np.zeros_like(getattr(self,n)) for n in self._pn}
        self._t  = 0

    def _drop(self, x):
        if not self.training or self.dropout == 0: return x
        m = np.random.binomial(1, 1-self.dropout, x.shape) / (1-self.dropout)
        return x * m

    def _gru(self, x, h, Wz,Uz,bz, Wr,Ur,br, Wh,Uh,bh):
        z = expit(x@Wz + h@Uz + bz)
        r = expit(x@Wr + h@Ur + br)
        g = np.tanh(x@Wh + (r*h)@Uh + bh)
        return (1-z)*h + z*g

    def _pool(self, idx):
        if not idx: return np.zeros(self.emb_dim, dtype=np.float32)
        return self.E[idx].mean(axis=0)

    def forward(self, idxA, idxB):
        hA = self._drop(self._pool(idxA))
        hB = self._drop(self._pool(idxB))

        # Alpha GRU
        h0   = np.zeros(self.hidden_dim, dtype=np.float32)
        hidA = self._gru(hA,h0, self.Wza,self.Uza,self.bza,
                         self.Wra,self.Ura,self.bra, self.Wha,self.Uha,self.bha)
        hidB = self._gru(hB,hidA, self.Wza,self.Uza,self.bza,
                         self.Wra,self.Ura,self.bra, self.Wha,self.Uha,self.bha)
        sA = float(np.dot(hidA, self.Wal.squeeze()) + self.bal[0])
        sB = float(np.dot(hidB, self.Wal.squeeze()) + self.bal[0])
        eA,eB = np.exp(sA),np.exp(sB); dn = eA+eB+1e-9
        aA,aB = eA/dn, eB/dn

        # Beta GRU
        h2   = np.zeros(self.hidden_dim, dtype=np.float32)
        h2A  = self._gru(hA,h2, self.Wzb,self.Uzb,self.bzb,
                         self.Wrb,self.Urb,self.brb, self.Whb,self.Uhb,self.bhb)
        h2B  = self._gru(hB,h2A, self.Wzb,self.Uzb,self.bzb,
                         self.Wrb,self.Urb,self.brb, self.Whb,self.Uhb,self.bhb)
        bpA  = np.tanh(h2A@self.Wbt + self.bbt)
        bpB  = np.tanh(h2B@self.Wbt + self.bbt)

        def beta_attn(idx, bp):
            if not idx: return np.array([]), np.zeros(self.emb_dim)
            cv = self.E[idx]; sc = cv@bp; sc -= sc.max()
            b  = np.exp(sc)/(np.exp(sc).sum()+1e-9)
            return b, (cv*b[:,None]).sum(axis=0)

        betA, aeA = beta_attn(idxA, bpA)
        betB, aeB = beta_attn(idxB, bpB)

        ctx   = self._drop(aA*aeA + aB*aeB)
        logit = float(np.dot(ctx, self.Wo.squeeze()) + self.bo[0])
        prob  = float(expit(logit))

        cache = dict(hA=hA,hB=hB,hidA=hidA,hidB=hidB,h2A=h2A,h2B=h2B,
                     bpA=bpA,bpB=bpB,aA=aA,aB=aB,aeA=aeA,aeB=aeB,
                     ctx=ctx,logit=logit,prob=prob,
                     idxA=idxA,idxB=idxB,betA=betA,betB=betB)
        return prob, aA, aB, betA, betB, cache

    def backward(self, cache, label, pw=3.0):
        prob=cache["prob"]; y=float(label); w=pw if y==1 else 1.0
        dl = w*(prob-y)
        self._g["Wo"] += np.outer(cache["ctx"],[dl]); self._g["bo"] += [dl]
        dc = dl*self.Wo.squeeze()
        daA = float((dc*cache["aeA"]).sum()); daB = float((dc*cache["aeB"]).sum())
        aA,aB = cache["aA"],cache["aB"]
        dsA = daA*aA*(1-aA) - daB*aA*aB
        dsB = daB*aB*(1-aB) - daA*aA*aB
        self._g["Wal"] += np.outer(cache["hidA"],[dsA]) + np.outer(cache["hidB"],[dsB])
        self._g["bal"] += [dsA+dsB]
        dA = dc*aA; dB = dc*aB
        self._g["Wbt"] += (np.outer(cache["h2A"],dA*(1-cache["bpA"]**2)) +
                           np.outer(cache["h2B"],dB*(1-cache["bpB"]**2)))
        self._g["bbt"] += (dA*(1-cache["bpA"]**2) + dB*(1-cache["bpB"]**2))
        sc = 0.1
        for ci,idx in enumerate(cache["idxA"]):
            if ci < len(cache["betA"]): self._g["E"][idx] += sc*cache["betA"][ci]*dA
        for ci,idx in enumerate(cache["idxB"]):
            if ci < len(cache["betB"]): self._g["E"][idx] += sc*cache["betB"][ci]*dB

    def adam_step(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for n in self._pn:
            g = self._g[n]
            if np.all(g==0): continue
            self._m[n] = b1*self._m[n] + (1-b1)*g
            self._v[n] = b2*self._v[n] + (1-b2)*g**2
            mh = self._m[n]/(1-b1**self._t)
            vh = self._v[n]/(1-b2**self._t)
            setattr(self, n, getattr(self,n) - lr*mh/(np.sqrt(vh)+eps))
            self._g[n][:] = 0

    def freeze_emb(self):
        self._pn = [n for n in self._pn if n != "E"]
        log.info("Embeddings frozen")

    def unfreeze_emb(self):
        if "E" not in self._pn: self._pn = ["E"] + self._pn
        log.info("Embeddings unfrozen")

    def train(self): self.training = True
    def eval(self):  self.training = False

    def save(self, path):
        with open(path,"wb") as f: pickle.dump({"model":self}, f)
        log.info(f"Model saved → {path}")

    @staticmethod
    def load(path):
        with open(path,"rb") as f: return pickle.load(f)["model"]

# =============================================================================
# SECTION 9 — TRAINING LOOP
# =============================================================================

def bce(p, y, pw=1.0):
    w = pw if y==1 else 1.0; p = np.clip(p, 1e-7, 1-1e-7)
    return -w*(y*np.log(p) + (1-y)*np.log(1-p))

def run_epoch(model, samples, vocab, lr, pw, bs=32, train=True):
    model.train() if train else model.eval()
    if train: random.shuffle(samples)
    tl, probs, labs = 0.0, [], []
    for i in range(0, len(samples), bs):
        batch = samples[i:i+bs]
        for s in batch:
            idxA = vocab.enc(s["visit_A"]); idxB = vocab.enc(s["visit_B"])
            y    = s["preventable"]
            prob,aA,aB,bA,bB,cache = model.forward(idxA, idxB)
            tl  += bce(prob, y, pw); probs.append(prob); labs.append(y)
            if train: model.backward(cache, y, pw)
        if train: model.adam_step(lr)
    try:
        auroc = roc_auc_score(labs, probs)
        auprc = average_precision_score(labs, probs)
    except: auroc = auprc = 0.0
    return tl/max(len(samples),1), auroc, auprc

def train_model(tr_samples, va_samples, vocab, pretrained_emb):
    model = RETAIN(len(vocab), EMBEDDING_DIM, HIDDEN_DIM,
                   DROPOUT_RATE, pretrained_emb)
    best_prc = 0.0; best_params = None; lr = LEARNING_RATE

    # Phase 1 — frozen embeddings
    log.info(f"\nPhase 1: {FREEZE_EPOCHS} epochs — embeddings frozen")
    model.freeze_emb()
    for ep in range(FREEZE_EPOCHS):
        tl,_,tp = run_epoch(model,tr_samples,vocab,lr,POS_WEIGHT,BATCH_SIZE,True)
        vl,va,vp= run_epoch(model,va_samples,vocab,lr,POS_WEIGHT,BATCH_SIZE,False)
        log.info(f"  [F{ep+1:02d}] Loss={tl:.3f} TrPRC={tp:.3f} | "
                 f"ValAUROC={va:.3f} ValPRC={vp:.3f}")
        if vp > best_prc:
            best_prc = vp
            best_params = {n: getattr(model,n).copy() for n in model._pn+["E"]}

    # Phase 2 — full fine-tuning
    log.info(f"\nPhase 2: {NUM_EPOCHS} epochs — full fine-tuning")
    model.unfreeze_emb()
    for ep in range(NUM_EPOCHS):
        tl,_,tp = run_epoch(model,tr_samples,vocab,lr,POS_WEIGHT,BATCH_SIZE,True)
        vl,va,vp= run_epoch(model,va_samples,vocab,lr,POS_WEIGHT,BATCH_SIZE,False)
        log.info(f"  [E{ep+1:02d}/{NUM_EPOCHS}] Loss={tl:.3f} TrPRC={tp:.3f} | "
                 f"ValAUROC={va:.3f} ValPRC={vp:.3f}")
        if vp > best_prc:
            best_prc = vp
            best_params = {n: getattr(model,n).copy() for n in model._pn+["E"]}
        if (ep+1) % 15 == 0: lr *= 0.5

    if best_params:
        for n,v in best_params.items(): setattr(model,n,v)
        log.info(f"\nBest model restored — Val AUPRC={best_prc:.4f}")
    return model

# =============================================================================
# SECTION 10 — EXPLAINABILITY ENGINE
# =============================================================================

def describe(code):
    """Human-readable description for any code token."""
    if code.startswith("DRG_"):
        d=code[4:]; return f"DRG {d}: {DRG_DESC.get(d,'Unknown DRG')}"
    if code.startswith("LOS_"):
        m={"LOS_1D":"LOS 1 day — very short stay",
           "LOS_2_3D":"LOS 2-3 days — short stay",
           "LOS_4_7D":"LOS 4-7 days — moderate stay",
           "LOS_8_14D":"LOS 8-14 days — extended stay",
           "LOS_15PD":"LOS 15+ days — prolonged stay"}
        return m.get(code, code)
    if code.startswith("DISCH_"):
        m={"DISCH_HOME":"Discharged home no services",
           "DISCH_SNF":"Discharged to SNF",
           "DISCH_HOME_HEALTH":"Discharged home with home health",
           "DISCH_AMA":"Left against medical advice",
           "DISCH_SHORT_HOSP":"Transferred short-term hospital",
           "DISCH_EXPIRED":"Expired during admission",
           "DISCH_HOSPICE":"Discharged to hospice",
           "DISCH_REHAB":"Discharged to rehab"}
        return m.get(code, code)
    if code.startswith("DAYS_"):
        return f"Days to readmission: {code[5:]}"
    d = ICD_DESC.get(code) or PROC_DESC.get(code)
    return f"{code}: {d}" if d else code

def top_attended(tokens, betas, n=5):
    """Top-N codes with descriptions and attention weights — human readable."""
    if len(betas) == 0: return ""
    n = min(n, len(tokens), len(betas))
    idx = np.argsort(betas)[::-1][:n]
    return " | ".join(
        f"{describe(tokens[i])} [attn={float(betas[i]):.3f}]"
        for i in idx if i < len(tokens)
    )

def audit_flags(s):
    """Rule-based audit flags — pipe-delimited."""
    flags = []
    if s["disch_A"] == "07":
        flags.append("AMA_DISCHARGE")
    try:
        if s["disch_A"] == "01" and float(s["los_A"]) <= 2:
            flags.append("VERY_SHORT_STAY_HOME")
    except: pass
    db = s["days_between"]
    if   0 <= db <= 7:  flags.append("RAPID_READMIT_7D")
    elif 0 <= db <= 14: flags.append("READMIT_8_14D")
    da = s["diag_A"]; db2 = s["diag_B"]
    if da and db2:
        ca, cb = ICD_CCS.get(da[0]), ICD_CCS.get(db2[0])
        if ca and cb and ca == cb:
            flags.append("SAME_PRIMARY_DX_CATEGORY")
    try:
        dA, dB = str(s["drg_A"]), str(s["drg_B"])
        if len(dA)==3 and len(dB)==3 and dA[0]==dB[0] and int(dB)<int(dA):
            flags.append("DRG_SEVERITY_ESCALATION")
    except: pass
    return "|".join(flags) if flags else "NONE"

def overpayment_priority(prob, flags_str, aA):
    score = 0
    if prob >= 0.80:   score += 3
    elif prob >= 0.65: score += 2
    elif prob >= 0.50: score += 1
    if "AMA_DISCHARGE"          in flags_str: score += 2
    if "RAPID_READMIT_7D"       in flags_str: score += 2
    elif "READMIT_8_14D"        in flags_str: score += 1
    if "SAME_PRIMARY_DX"        in flags_str: score += 2
    if "DRG_SEVERITY_ESCALATION"in flags_str: score += 1
    if "VERY_SHORT_STAY"        in flags_str: score += 1
    if aA > 0.5:                               score += 1
    if score >= 7: return "HIGH"
    elif score >= 4: return "MEDIUM"
    return "LOW"

def audit_narrative(s, prob, aA, aB, betA, betB):
    """One-sentence human-readable audit finding."""
    verdict  = "PREVENTABLE" if prob >= 0.5 else "NOT PREVENTABLE"
    conf     = ("High" if abs(prob-0.5)>0.3 else
                "Moderate" if abs(prob-0.5)>0.15 else "Low")
    driver   = ("INDEX CLAIM A" if aA > aB else "READMISSION CLAIM B")
    impl     = ("Provider discharge behavior implicated." if aA > aB
                else "Readmission diagnosis pattern drove decision.")

    # Top code on each claim
    def top1(toks, betas):
        if len(betas)==0 or not toks: return "N/A"
        i = int(np.argmax(betas))
        return describe(toks[i]) if i < len(toks) else "N/A"

    top_a = top1(s["visit_A"], betA)
    top_b = top1(s["visit_B"], betB)

    # Clinical flags
    flags = []
    if s["disch_A"] == "07": flags.append("AMA discharge on index claim")
    try:
        if s["disch_A"]=="01" and float(s["los_A"])<=2:
            flags.append(f"discharged home after only {s['los_A']} day(s)")
    except: pass
    if 0 <= s["days_between"] <= 7:
        flags.append(f"returned in {s['days_between']} days")
    da, db2 = s["diag_A"], s["diag_B"]
    if da and db2:
        ca, cb = ICD_CCS.get(da[0]), ICD_CCS.get(db2[0])
        if ca and cb and ca == cb:
            cn = CCS_NAME.get(ca, f"CCS {ca}")
            flags.append(f"same diagnosis category ({cn}) on both claims")

    flag_str = ("; ".join(flags)) if flags else "no specific rule-based flags"

    return (f"RETAIN verdict: {verdict} (prob={prob:.3f}, confidence={conf}). "
            f"Decision driven by {driver}. {impl} "
            f"Top signal on index claim: [{top_a}]. "
            f"Top signal on readmission: [{top_b}]. "
            f"Clinical flags: {flag_str}.")

# =============================================================================
# SECTION 11 — SCORING WITH EXPLAINABILITY
# =============================================================================

def score_and_explain(model, samples, vocab):
    """Run inference on scoring samples and return enriched DataFrame."""
    model.eval()
    rows = []
    for s in samples:
        idxA = vocab.enc(s["visit_A"])
        idxB = vocab.enc(s["visit_B"])
        prob, aA, aB, betA, betB, _ = model.forward(idxA, idxB)

        flags    = audit_flags(s)
        priority = overpayment_priority(prob, flags, aA)
        narrative= audit_narrative(s, prob, aA, aB, betA, betB)

        # Primary diagnoses
        pa = s["diag_A"][0] if s["diag_A"] else ""
        pb = s["diag_B"][0] if s["diag_B"] else ""
        ca = ICD_CCS.get(pa,""); cb = ICD_CCS.get(pb,"")

        rows.append({
            # ── IDENTIFIERS ────────────────────────────────────────────────
            "pair_id":                        s["pair_id"],
            "member_id":                      s["member_id"],
            "provider_tax_id":                s["provider_tax_id"],
            "claimA_id":                      s["claimA_id"],
            "claimB_id":                      s["claimB_id"],

            # ── MODEL OUTPUT ───────────────────────────────────────────────
            "preventable_probability":        round(prob, 4),
            "predicted_preventable":          "YES" if prob >= 0.5 else "NO",
            "overpayment_priority":           priority,
            "estimated_exposure_usd":         (AVG_PAYMENT if prob >= 0.75
                                               else int(AVG_PAYMENT * prob)
                                               if prob >= 0.50 else 0),

            # ── RETAIN VISIT-LEVEL ATTENTION ───────────────────────────────
            "retain_alpha_claimA":            round(float(aA), 4),
            "retain_alpha_claimB":            round(float(aB), 4),
            "decision_driven_by":             ("INDEX_CLAIM_A" if aA > aB
                                               else "READMISSION_CLAIM_B"),
            "provider_behavior_implicated":   "YES" if aA > aB else "NO",

            # ── RETAIN CODE-LEVEL ATTENTION (HUMAN READABLE) ──────────────
            "top_attended_codes_claimA":      top_attended(s["visit_A"], betA, 5),
            "top_attended_codes_claimB":      top_attended(s["visit_B"], betB, 5),

            # ── RETAIN CODE-LEVEL ATTENTION (RAW NUMERIC) ─────────────────
            "raw_attention_claimA":           "; ".join(
                f"{s['visit_A'][i]}={float(betA[i]):.3f}"
                for i in np.argsort(betA)[::-1][:5]
                if i < len(s["visit_A"])
            ) if len(betA) > 0 else "",
            "raw_attention_claimB":           "; ".join(
                f"{s['visit_B'][i]}={float(betB[i]):.3f}"
                for i in np.argsort(betB)[::-1][:5]
                if i < len(s["visit_B"])
            ) if len(betB) > 0 else "",

            # ── AUDIT NARRATIVE ────────────────────────────────────────────
            "audit_narrative":                narrative,

            # ── AUDIT FLAGS ────────────────────────────────────────────────
            "audit_flags":                    flags,

            # ── CLAIM A CLINICAL DETAIL ────────────────────────────────────
            "claimA_admit_date":              s["admit_A"],
            "claimA_primary_dx":              pa,
            "claimA_primary_dx_desc":         ICD_DESC.get(pa, ""),
            "claimA_all_diagnoses_full":      "|".join(s["diag_A"]),
            "claimA_diag_3char_used_by_model":"|".join(s["trunc_diag_A"]),
            "claimA_drg":                     s["drg_A"],
            "claimA_drg_desc":                DRG_DESC.get(str(s["drg_A"]),""),
            "claimA_los_days":                s["los_A"],
            "claimA_discharge_status":        s["disch_A"],
            "claimA_discharge_desc":          DISCH_DESC.get(str(s["disch_A"]).zfill(2),""),
            "claimA_ccs_category":            ca,
            "claimA_ccs_name":                CCS_NAME.get(ca,""),

            # ── CLAIM B CLINICAL DETAIL ────────────────────────────────────
            "claimB_admit_date":              s["admit_B"],
            "claimB_primary_dx":              pb,
            "claimB_primary_dx_desc":         ICD_DESC.get(pb, ""),
            "claimB_all_diagnoses_full":      "|".join(s["diag_B"]),
            "claimB_diag_3char_used_by_model":"|".join(s["trunc_diag_B"]),
            "claimB_drg":                     s["drg_B"],
            "claimB_drg_desc":                DRG_DESC.get(str(s["drg_B"]),""),
            "claimB_los_days":                s["los_B"],
            "claimB_discharge_status":        s["disch_B"],
            "claimB_discharge_desc":          DISCH_DESC.get(str(s["disch_B"]).zfill(2),""),
            "claimB_ccs_category":            cb,
            "claimB_ccs_name":                CCS_NAME.get(cb,""),

            # ── PAIR COMPARISON ────────────────────────────────────────────
            "days_index_to_readmission":      s["days_between"],
            "days_gap_category":              s["days_gap_token"],
            "same_ccs_category":              ("YES" if ca and cb and ca==cb else "NO"),
            "drg_change":                     (f"{s['drg_A']}→{s['drg_B']}"
                                               if s["drg_A"]!=s["drg_B"] else "SAME"),
        })

    df = pd.DataFrame(rows)
    # Sort: HIGH first, then by score descending
    order = {"HIGH":0,"MEDIUM":1,"LOW":2}
    df["_sort"] = df["overpayment_priority"].map(order)
    df = (df.sort_values(["_sort","preventable_probability"], ascending=[True,False])
            .drop(columns=["_sort"]).reset_index(drop=True))
    return df

# =============================================================================
# SECTION 12 — PROVIDER AUDIT QUEUE
# =============================================================================

def build_audit_queue(df):
    df2 = df.copy()
    df2["_high"] = (df2["preventable_probability"] >= 0.75).astype(int)
    df2["_exp"]  = df2["estimated_exposure_usd"]

    agg = df2.groupby("provider_tax_id").agg(
        total_scoring_pairs       = ("pair_id","count"),
        mean_preventable_prob     = ("preventable_probability","mean"),
        high_conf_preventable     = ("_high","sum"),
        total_exposure_usd        = ("_exp","sum"),
        high_priority_cases       = ("overpayment_priority",
                                     lambda x:(x=="HIGH").sum()),
        index_driven_cases        = ("provider_behavior_implicated",
                                     lambda x:(x=="YES").sum()),
        rapid_readmit_cases       = ("audit_flags",
                                     lambda x:x.str.contains("RAPID_READMIT_7D").sum()),
        ama_discharge_cases       = ("audit_flags",
                                     lambda x:x.str.contains("AMA_DISCHARGE").sum()),
        same_dx_cases             = ("audit_flags",
                                     lambda x:x.str.contains("SAME_PRIMARY_DX").sum()),
    ).reset_index()

    if len(agg) > 1:
        agg["prob_zscore"] = sp_zscore(agg["mean_preventable_prob"].fillna(0))
    else:
        agg["prob_zscore"] = 0.0

    agg["priority_score"] = (
        agg["prob_zscore"].clip(lower=0) *
        np.log1p(agg["total_scoring_pairs"]) *
        agg["mean_preventable_prob"]
    )
    queue = (agg.sort_values("priority_score", ascending=False)
               .reset_index(drop=True))
    queue["rank"] = queue.index + 1
    for c in ["mean_preventable_prob","prob_zscore","priority_score"]:
        queue[c] = queue[c].round(4)
    return queue

# =============================================================================
# SECTION 13 — MAIN
# =============================================================================

def main():
    log.info("="*62)
    log.info("RETAIN Readmission Preventability Pipeline")
    log.info("="*62)

    # ── STEP 1: Load training data ────────────────────────────
    log.info(f"\n[STEP 1] Loading training data: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    log.info(f"  Rows: {len(train_df):,}")

    if LABEL_COL in train_df.columns:
        train_df["preventable"] = train_df[LABEL_COL].astype(int)
        log.info(f"  Label column '{LABEL_COL}' found — "
                 f"preventable rate: {train_df['preventable'].mean():.1%}")
    else:
        log.info(f"  Label column '{LABEL_COL}' not found — "
                 f"auto-generating via CCS rules...")
        train_df["preventable"] = train_df.apply(auto_label, axis=1)
        log.info(f"  Auto-labeled — preventable rate: {train_df['preventable'].mean():.1%}")

    # ── STEP 2: Build training samples ────────────────────────
    log.info("\n[STEP 2] Building training samples...")
    all_samples = build_samples(train_df, has_label=True)
    log.info(f"  {len(all_samples):,} samples built")

    # ── STEP 3: Train/val split by member_id ─────────────────
    log.info("\n[STEP 3] Splitting train/val by member_id...")
    mbrs = np.array([s["member_id"] for s in all_samples])
    um   = np.unique(mbrs); np.random.shuffle(um); n = len(um)
    tr_m = set(um[:int(0.85*n)]); va_m = set(um[int(0.85*n):])
    tr   = [s for s in all_samples if s["member_id"] in tr_m]
    va   = [s for s in all_samples if s["member_id"] in va_m]
    log.info(f"  Train: {len(tr):,} | Val: {len(va):,}")

    # ── STEP 4: Vocabulary ────────────────────────────────────
    log.info("\n[STEP 4] Building vocabulary from training data...")
    vocab = Vocab().fit(tr)

    # ── STEP 5: Graph + embeddings ────────────────────────────
    log.info("\n[STEP 5] Building graph and generating embeddings...")
    G   = build_graph(tr)
    emb = generate_embeddings(G, vocab)

    # ── STEP 6: Train RETAIN ──────────────────────────────────
    log.info("\n[STEP 6] Training RETAIN model...")
    model = train_model(tr, va, vocab, emb)

    # ── STEP 7: Evaluate on validation set ───────────────────
    log.info("\n[STEP 7] Validation set evaluation...")
    _,va_auroc,va_prc = run_epoch(model,va,vocab,LEARNING_RATE,POS_WEIGHT,BATCH_SIZE,False)
    model.eval()
    va_probs,va_labs=[],[]
    for s in va:
        p,*_=model.forward(vocab.enc(s["visit_A"]),vocab.enc(s["visit_B"]))
        va_probs.append(p); va_labs.append(s["preventable"])
    preds=[1 if p>=0.5 else 0 for p in va_probs]
    print("\n"+"="*62)
    print("VALIDATION SET PERFORMANCE")
    print("="*62)
    print(f"AUROC: {va_auroc:.4f}  |  AUPRC: {va_prc:.4f}")
    print(classification_report(va_labs, preds,
                                target_names=["Not Preventable","Preventable"],
                                zero_division=0))

    # ── STEP 8: Save model ────────────────────────────────────
    log.info(f"\n[STEP 8] Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)

    # ── STEP 9: Load scoring data ────────────────────────────
    log.info(f"\n[STEP 9] Loading scoring data: {SCORE_PATH}")
    score_df = pd.read_csv(SCORE_PATH)
    log.info(f"  Rows: {len(score_df):,}")

    if "preventable" not in score_df.columns:
        score_df["preventable"] = -1  # placeholder — not used in scoring
    score_samples = build_samples(score_df, has_label=False)
    log.info(f"  {len(score_samples):,} scoring samples built")

    # ── STEP 10: Score + explain ──────────────────────────────
    log.info("\n[STEP 10] Scoring and generating explainability...")
    scored_df = score_and_explain(model, score_samples, vocab)

    # ── STEP 11: Provider audit queue ────────────────────────
    log.info("\n[STEP 11] Building provider audit queue...")
    audit_df = build_audit_queue(scored_df)

    # ── STEP 12: Save outputs ─────────────────────────────────
    score_path = os.path.join(OUTPUT_DIR, "scored_with_explainability.csv")
    audit_path = os.path.join(OUTPUT_DIR, "provider_audit_queue.csv")
    scored_df.to_csv(score_path, index=False)
    audit_df.to_csv(audit_path,  index=False)

    log.info(f"\n[STEP 12] Outputs written:")
    log.info(f"  Scoring CSV  : {score_path}  "
             f"({len(scored_df)} rows × {len(scored_df.columns)} columns)")
    log.info(f"  Audit queue  : {audit_path}  ({len(audit_df)} providers)")
    log.info(f"  Model        : {MODEL_PATH}")

    # ── Print audit queue ─────────────────────────────────────
    print("\n"+"="*62)
    print("PROVIDER AUDIT QUEUE")
    print("="*62)
    pd.set_option("display.float_format","{:.3f}".format)
    pd.set_option("display.max_columns",20)
    pd.set_option("display.width",140)
    cols = ["rank","provider_tax_id","total_scoring_pairs",
            "mean_preventable_prob","high_priority_cases",
            "total_exposure_usd","rapid_readmit_cases",
            "ama_discharge_cases","priority_score"]
    print(audit_df[[c for c in cols if c in audit_df.columns]].to_string(index=False))

    # ── Print 3 sample high-priority cases ───────────────────
    high = scored_df[scored_df["overpayment_priority"]=="HIGH"].head(3)
    print(f"\n{'='*62}")
    print("SAMPLE HIGH-PRIORITY CASES FROM SCORING DATA")
    print("="*62)
    for _, r in high.iterrows():
        print(f"\n  Pair     : {r['pair_id']}  |  Member: {r['member_id']}")
        print(f"  Provider : {r['provider_tax_id']}")
        print(f"  Score    : {r['preventable_probability']:.3f}  "
              f"Priority: {r['overpayment_priority']}  "
              f"Exposure: ${r['estimated_exposure_usd']:,}")
        print(f"  Flags    : {r['audit_flags']}")
        print(f"  Narrative: {str(r['audit_narrative'])[:250]}...")

    print("\n\nPipeline complete.")

if __name__ == "__main__":
    main()
