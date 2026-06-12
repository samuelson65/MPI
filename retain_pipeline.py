"""
=============================================================================
RETAIN-Based Readmission Preventability Pipeline
=============================================================================
Zero external ML dependencies — runs on: pandas, numpy, scipy,
scikit-learn, networkx (all standard / pre-installed).

RETAIN is implemented from scratch in pure NumPy so the script runs
in a single click without torch, pyhealth, or GPU.

Expected CSV columns (claimA_* = index, claimB_* = readmission):
  pair_id | member_id
  claimA_claim_id | claimA_provider_tax_id | claimA_drg_code
  claimA_diag_codes | claimA_proc_codes | claimA_los
  claimA_discharge_status | claimA_admit_date | claimA_discharge_date
  claimB_* (same fields)
  preventable  (optional — auto-generated if missing)

Outputs (written to ./outputs/):
  pair_level_scores.csv     — per-pair preventability score + attention
  provider_audit_queue.csv  — ranked providers by overpayment risk
=============================================================================
"""

# ── 0. AUTO-INSTALL MISSING PACKAGES ────────────────────────────────────────
import subprocess, sys

REQUIRED = ["pandas","numpy","scipy","scikit-learn","networkx"]
for pkg in REQUIRED:
    import_name = pkg.replace("-","_").replace("scikit_learn","sklearn")
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,
                               "--break-system-packages","--quiet"])

# ── 1. IMPORTS ───────────────────────────────────────────────────────────────
import os, warnings, logging, random
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import expit                       # sigmoid
from scipy.stats import zscore
from sklearn.metrics import (roc_auc_score,
                              average_precision_score,
                              classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)
np.random.seed(42)
random.seed(42)

# ── 2. CONFIGURATION ─────────────────────────────────────────────────────────
CFG = {
    "data_path":          "synthetic_readmission_pairs.csv",
    "label_column":       "preventable",   # set None to auto-generate
    "embedding_dim":      64,
    "hidden_dim":         64,
    "learning_rate":      0.01,
    "num_epochs":         60,
    "batch_size":         32,
    "dropout_rate":       0.2,
    "pos_class_weight":   3.0,
    "min_cooccurrence":   3,
    "node2vec_walks":     40,
    "node2vec_walk_len":  15,
    "node2vec_window":    5,
    "freeze_epochs":      10,
    "top_n_providers":    15,
    "output_dir":         "outputs",
}
os.makedirs(CFG["output_dir"], exist_ok=True)

# ── 3. HELPER UTILITIES ──────────────────────────────────────────────────────

def parse_codes(s):
    if pd.isna(s) or str(s).strip() == "": return []
    return [c.strip().upper() for c in str(s).replace("|",",").replace(";",",").split(",") if c.strip()]

def encode_los(v):
    try:
        d = float(v)
        if d <= 1:    return "LOS_1D"
        elif d <= 3:  return "LOS_2_3D"
        elif d <= 7:  return "LOS_4_7D"
        elif d <= 14: return "LOS_8_14D"
        else:         return "LOS_15PD"
    except:           return "LOS_UNK"

DISCH_MAP = {"01":"DISCH_HOME","02":"DISCH_SHORT_HOSP","03":"DISCH_SNF",
             "06":"DISCH_HOME_HEALTH","07":"DISCH_AMA","20":"DISCH_EXPIRED",
             "03":"DISCH_SNF","50":"DISCH_HOSPICE","62":"DISCH_REHAB"}

def encode_disch(v):
    return DISCH_MAP.get(str(v).strip().zfill(2), f"DISCH_{v}")

# ── 4. PREVENTABILITY LABELS ─────────────────────────────────────────────────

# CCS categories that are ambulatory-care-sensitive / HRRP-tracked
PREV_CCS = {"108","100","122","127","690","638","157","158","55","2",
            "96","97","101","105","106","114","128","131","159","197",
            "237","238","250","641","683","682"}

# Simplified ICD-10 → CCS mapping for the codes in our synthetic data
# (In production, load the full CMS CCS crosswalk CSV)
ICD_TO_CCS = {
    # Heart failure
    "I50.43":"108","I50.42":"108","I50.32":"108","I50.9":"108","I50.1":"108",
    # Diabetes
    "E11.65":"50","E11.9":"50","E11.649":"50","E11.40":"50",
    # Hypertension
    "I10":"98","I11.9":"98","I12.9":"98","I13.10":"98",
    # Renal
    "N17.9":"157","N18.3":"158","N18.4":"158","N18.5":"158","N19":"157",
    # Pneumonia
    "J18.9":"122","J18.1":"122","J15.9":"122","J15.211":"122","J13":"122",
    # COPD
    "J44.1":"127","J44.0":"127","J44.9":"127","J43.9":"127",
    # Sepsis
    "A41.9":"2","A41.51":"2","A41.01":"2","A40.9":"2",
    # Other
    "Z87.891":"259","Z82.49":"259","I25.10":"101","I48.91":"106",
    "E78.5":"53","M79.3":"212",
}

def get_ccs(icd):
    return ICD_TO_CCS.get(icd, None)

def label_preventable(row):
    da = parse_codes(row.get("claimA_diag_codes",""))
    db = parse_codes(row.get("claimB_diag_codes",""))
    prim_a = da[0] if da else None
    prim_b = db[0] if db else None

    # Rule 1: very short gap
    try:
        gap = (pd.to_datetime(row["claimB_admit_date"]) -
               pd.to_datetime(row["claimA_discharge_date"])).days
        if 0 <= gap <= 7: return 1
    except: gap = 99

    # Rule 2: same CCS category
    if prim_a and prim_b:
        ccs_a, ccs_b = get_ccs(prim_a), get_ccs(prim_b)
        if ccs_a and ccs_b and ccs_a == ccs_b: return 1
        if ccs_b and ccs_b in PREV_CCS:        return 1

    # Rule 3: DRG severity escalated within same MDC
    try:
        da_drg = str(row.get("claimA_drg_code",""))
        db_drg = str(row.get("claimB_drg_code",""))
        if (len(da_drg)==3 and len(db_drg)==3 and
            da_drg[0]==db_drg[0] and int(db_drg) < int(da_drg)):
            return 1
    except: pass

    return 0

# ── 5. FEATURE ENGINEERING ───────────────────────────────────────────────────

def build_visit_tokens(row, prefix):
    diag  = parse_codes(row.get(f"{prefix}diag_codes",""))
    proc  = parse_codes(row.get(f"{prefix}proc_codes",""))
    drg   = [f"DRG_{row.get(f'{prefix}drg_code','UNK')}"]
    los   = [encode_los(row.get(f"{prefix}los", None))]
    disch = [encode_disch(row.get(f"{prefix}discharge_status", "99"))]
    return [t for t in diag+proc+drg+los+disch if t]

def compute_days_between(row):
    try:
        return max(0,(pd.to_datetime(row["claimB_admit_date"]) -
                      pd.to_datetime(row["claimA_discharge_date"])).days)
    except: return -1

def build_samples(df):
    samples = []
    for _, row in df.iterrows():
        va = build_visit_tokens(row, "claimA_")
        vb = build_visit_tokens(row, "claimB_")
        days = compute_days_between(row)
        vb  += [f"DAYS_{min(days,30)}"] if days >= 0 else ["DAYS_UNK"]
        if not va or not vb: continue
        samples.append({
            "pair_id":         str(row.get("pair_id","")),
            "member_id":       str(row.get("member_id","")),
            "provider_tax_id": str(row.get("claimA_provider_tax_id","")),
            "visit_A":         va,
            "visit_B":         vb,
            "diag_A":          parse_codes(row.get("claimA_diag_codes","")),
            "diag_B":          parse_codes(row.get("claimB_diag_codes","")),
            "drg_A":           str(row.get("claimA_drg_code","")),
            "drg_B":           str(row.get("claimB_drg_code","")),
            "los_A":           str(row.get("claimA_los","")),
            "disch_A":         encode_disch(row.get("claimA_discharge_status","99")),
            "days_between":    days,
            "preventable":     int(row["preventable"]),
        })
    return samples

# ── 6. VOCABULARY ────────────────────────────────────────────────────────────

class Vocab:
    PAD, UNK = "<PAD>", "<UNK>"
    def __init__(self):
        self.t2i = {self.PAD:0, self.UNK:1}
        self.i2t = {0:self.PAD, 1:self.UNK}
    def fit(self, samples):
        for s in samples:
            for tok in s["visit_A"] + s["visit_B"]:
                if tok not in self.t2i:
                    i = len(self.t2i)
                    self.t2i[tok] = i; self.i2t[i] = tok
        log.info(f"Vocabulary: {len(self.t2i):,} tokens")
        return self
    def enc(self, tokens):
        return [self.t2i.get(t, self.t2i[self.UNK]) for t in tokens]
    def __len__(self): return len(self.t2i)

# ── 7. GRAPH EMBEDDINGS (pure networkx + numpy random walks) ─────────────────

def build_graph(samples, cfg):
    log.info("Building medical code co-occurrence graph...")
    G = nx.Graph()
    all_toks = set()
    for s in samples:
        all_toks.update(s["visit_A"] + s["visit_B"])
    for t in all_toks: G.add_node(t)

    # ICD hierarchical edges: codes sharing the first 3 chars are siblings
    icd_toks = [t for t in all_toks if not t.startswith(
                ("DRG_","LOS_","DISCH_","DAYS_","992","993","994",
                 "936","938","903","318","314","999","000"))]
    by_parent = defaultdict(list)
    for t in icd_toks:
        parent = t.split(".")[0]          # e.g. N18.3 → N18
        by_parent[parent].append(t)
    for parent, children in by_parent.items():
        G.add_node(parent)
        for c in children:
            G.add_edge(c, parent, weight=1.0, etype="hier_parent")
        # connect siblings through parent (weaker)
        for c1, c2 in combinations(children, 2):
            if not G.has_edge(c1,c2):
                G.add_edge(c1, c2, weight=0.5, etype="sibling")
        # grandparent: first char block  e.g. N18 → N
        gp = parent[0]
        G.add_node(gp)
        G.add_edge(parent, gp, weight=0.3, etype="hier_grand")

    # Co-occurrence edges
    co = defaultdict(int)
    for s in samples:
        pair_toks = list(set(s["visit_A"] + s["visit_B"]))
        for t1,t2 in combinations(pair_toks, 2):
            co[tuple(sorted([t1,t2]))] += 1
    added = 0
    for (t1,t2),cnt in co.items():
        if cnt >= cfg["min_cooccurrence"]:
            w = cnt / len(samples)
            if G.has_edge(t1,t2):
                G[t1][t2]["weight"] += w
            else:
                G.add_edge(t1,t2,weight=w,etype="cooccur")
                added += 1
    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
             f"(+{added} co-occurrence)")
    return G

def random_walks(G, nodes, num_walks, walk_len):
    """Biased random walk respecting edge weights."""
    node_list = list(nodes)
    walks = []
    for _ in range(num_walks):
        random.shuffle(node_list)
        for start in node_list:
            walk = [start]
            for _ in range(walk_len - 1):
                cur = walk[-1]
                nbrs = list(G.neighbors(cur))
                if not nbrs: break
                wts  = np.array([G[cur][n].get("weight",1.0) for n in nbrs])
                wts  = wts / wts.sum()
                walk.append(np.random.choice(nbrs, p=wts))
            walks.append(walk)
    return walks

def train_word2vec_numpy(walks, vocab_size, t2i, emb_dim, window, epochs=5, lr=0.025):
    """
    Simplified skip-gram Word2Vec in pure NumPy.
    Trains embeddings so codes appearing in the same walk
    end up with similar vectors.
    """
    W  = np.random.normal(0, 0.01, (vocab_size, emb_dim))   # input embeddings
    W2 = np.random.normal(0, 0.01, (vocab_size, emb_dim))   # output embeddings

    for ep in range(epochs):
        total_loss = 0.0
        random.shuffle(walks)
        for walk in walks:
            idxs = [t2i.get(t, 1) for t in walk]
            for i, center in enumerate(idxs):
                ctx_range = range(max(0, i-window), min(len(idxs), i+window+1))
                for j in ctx_range:
                    if j == i: continue
                    ctx = idxs[j]
                    # Positive sample
                    score_pos = W[center] @ W2[ctx]
                    sig_pos   = expit(score_pos)
                    err_pos   = sig_pos - 1.0
                    total_loss += -np.log(sig_pos + 1e-9)
                    grad_center_pos = err_pos * W2[ctx]
                    grad_ctx_pos    = err_pos * W[center]
                    # Negative samples (5 random)
                    negs = np.random.randint(2, vocab_size, 5)
                    grad_center_neg = np.zeros(emb_dim)
                    for neg in negs:
                        score_neg = W[center] @ W2[neg]
                        sig_neg   = expit(score_neg)
                        err_neg   = sig_neg             # target is 0
                        total_loss += -np.log(1 - sig_neg + 1e-9)
                        grad_center_neg += err_neg * W2[neg]
                        W2[neg] -= lr * err_neg * W[center]
                    W[center] -= lr * (grad_center_pos + grad_center_neg)
                    W2[ctx]   -= lr * grad_ctx_pos
        log.info(f"  Word2Vec epoch {ep+1}/{epochs} loss={total_loss:.1f}")
    return W   # return input embeddings

def generate_embeddings(G, vocab, cfg):
    log.info("Generating graph embeddings via random walks + skip-gram...")
    nodes = set(vocab.t2i.keys()) & set(G.nodes())
    walks = random_walks(G, nodes,
                         cfg["node2vec_walks"],
                         cfg["node2vec_walk_len"])
    log.info(f"  Generated {len(walks)} walks")

    emb = train_word2vec_numpy(
        walks, len(vocab), vocab.t2i,
        cfg["embedding_dim"],
        cfg["node2vec_window"],
        epochs=5, lr=0.025
    )

    # Check similarity for renal codes as sanity check
    for pair in [("N17.9","N18.3"), ("I50.43","N18.3")]:
        i1, i2 = vocab.t2i.get(pair[0],0), vocab.t2i.get(pair[1],0)
        if i1 > 1 and i2 > 1:
            v1, v2 = emb[i1], emb[i2]
            sim = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
            log.info(f"  Embedding similarity {pair[0]}—{pair[1]}: {sim:.3f}")
    return emb

# ── 8. RETAIN MODEL (pure NumPy) ─────────────────────────────────────────────

class RETAINNumpy:
    """
    RETAIN implemented in pure NumPy.

    For each pair [visit_A, visit_B]:
      1. Embed + mean-pool each visit  →  h_A, h_B   [emb_dim]
      2. Alpha GRU over [h_A, h_B]    →  alpha_A, alpha_B  (visit attention)
      3. Beta projection per visit     →  beta per code     (code attention)
      4. Weighted context vector       →  c  [emb_dim]
      5. Output layer                  →  logit  →  sigmoid  →  prob
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim,
                 dropout_rate=0.2, pretrained_emb=None):
        self.vocab_size   = vocab_size
        self.emb_dim      = emb_dim
        self.hidden_dim   = hidden_dim
        self.dropout_rate = dropout_rate
        self.training     = True

        # Embedding table
        if pretrained_emb is not None:
            self.E = pretrained_emb.copy().astype(np.float32)
        else:
            self.E = np.random.randn(vocab_size, emb_dim).astype(np.float32) * 0.01

        # Alpha GRU weights  (input: emb_dim, hidden: hidden_dim)
        d, h = emb_dim, hidden_dim
        self.Wz_a = self._init(d, h); self.Uz_a = self._init(h, h); self.bz_a = np.zeros(h)
        self.Wr_a = self._init(d, h); self.Ur_a = self._init(h, h); self.br_a = np.zeros(h)
        self.Wh_a = self._init(d, h); self.Uh_a = self._init(h, h); self.bh_a = np.zeros(h)

        # Alpha projection: hidden → scalar
        self.W_alpha = self._init(h, 1); self.b_alpha = np.zeros(1)

        # Beta GRU weights (same dims)
        self.Wz_b = self._init(d, h); self.Uz_b = self._init(h, h); self.bz_b = np.zeros(h)
        self.Wr_b = self._init(d, h); self.Ur_b = self._init(h, h); self.br_b = np.zeros(h)
        self.Wh_b = self._init(d, h); self.Uh_b = self._init(h, h); self.bh_b = np.zeros(h)

        # Beta projection: hidden → emb_dim
        self.W_beta = self._init(h, d); self.b_beta = np.zeros(d)

        # Output layer: emb_dim → 1
        self.W_out = self._init(d, 1); self.b_out = np.zeros(1)

        # Store parameter names for gradient updates
        self._param_names = [
            "E","Wz_a","Uz_a","bz_a","Wr_a","Ur_a","br_a","Wh_a","Uh_a","bh_a",
            "W_alpha","b_alpha","Wz_b","Uz_b","bz_b","Wr_b","Ur_b","br_b",
            "Wh_b","Uh_b","bh_b","W_beta","b_beta","W_out","b_out"
        ]
        # Gradient accumulators
        self._grads = {n: np.zeros_like(getattr(self,n)) for n in self._param_names}
        # Adam state
        self._m = {n: np.zeros_like(getattr(self,n)) for n in self._param_names}
        self._v = {n: np.zeros_like(getattr(self,n)) for n in self._param_names}
        self._t = 0

    def _init(self, r, c):
        return (np.random.randn(r, c) * np.sqrt(2.0 / r)).astype(np.float32)

    def _dropout(self, x):
        if not self.training or self.dropout_rate == 0: return x
        mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
        return x * mask

    def _gru_step(self, x, h, Wz,Uz,bz, Wr,Ur,br, Wh,Uh,bh):
        z = expit(x @ Wz + h @ Uz + bz)
        r = expit(x @ Wr + h @ Ur + br)
        g = np.tanh(x @ Wh + (r * h) @ Uh + bh)
        return (1 - z) * h + z * g

    def _embed_visit(self, code_indices):
        """Mean-pool embeddings for a list of code indices."""
        if not code_indices: return np.zeros(self.emb_dim, dtype=np.float32)
        vecs = self.E[code_indices]           # [n_codes, emb_dim]
        return vecs.mean(axis=0)

    def forward(self, idx_A, idx_B):
        """
        Forward pass for a single pair.
        idx_A, idx_B: lists of integer token indices for visit A and B.

        Returns: prob, alpha, beta_A, beta_B, cache
        """
        # Step 1: Visit embeddings
        h_A = self._embed_visit(idx_A)         # [emb_dim]
        h_B = self._embed_visit(idx_B)         # [emb_dim]
        h_A = self._dropout(h_A)
        h_B = self._dropout(h_B)

        # Step 2: Alpha GRU over [h_A, h_B] (sequentially)
        hid = np.zeros(self.hidden_dim, dtype=np.float32)
        hid_A = self._gru_step(h_A, hid,
                                self.Wz_a,self.Uz_a,self.bz_a,
                                self.Wr_a,self.Ur_a,self.br_a,
                                self.Wh_a,self.Uh_a,self.bh_a)
        hid_B = self._gru_step(h_B, hid_A,
                                self.Wz_a,self.Uz_a,self.bz_a,
                                self.Wr_a,self.Ur_a,self.br_a,
                                self.Wh_a,self.Uh_a,self.bh_a)

        score_A = float(hid_A @ self.W_alpha + self.b_alpha)
        score_B = float(hid_B @ self.W_alpha + self.b_alpha)
        exp_A, exp_B = np.exp(score_A), np.exp(score_B)
        denom = exp_A + exp_B + 1e-9
        alpha_A = exp_A / denom
        alpha_B = exp_B / denom

        # Step 3: Beta GRU (same input sequence)
        hid2 = np.zeros(self.hidden_dim, dtype=np.float32)
        hid2_A = self._gru_step(h_A, hid2,
                                 self.Wz_b,self.Uz_b,self.bz_b,
                                 self.Wr_b,self.Ur_b,self.br_b,
                                 self.Wh_b,self.Uh_b,self.bh_b)
        hid2_B = self._gru_step(h_B, hid2_A,
                                 self.Wz_b,self.Uz_b,self.bz_b,
                                 self.Wr_b,self.Ur_b,self.br_b,
                                 self.Wh_b,self.Uh_b,self.bh_b)

        beta_proj_A = np.tanh(hid2_A @ self.W_beta + self.b_beta)  # [emb_dim]
        beta_proj_B = np.tanh(hid2_B @ self.W_beta + self.b_beta)  # [emb_dim]

        # Code-level beta attention for visit A
        if idx_A:
            code_vecs_A = self.E[idx_A]                             # [nA, emb_dim]
            beta_scores_A = code_vecs_A @ beta_proj_A               # [nA]
            beta_scores_A -= beta_scores_A.max()
            beta_A = np.exp(beta_scores_A) / (np.exp(beta_scores_A).sum() + 1e-9)
            attn_emb_A = (code_vecs_A * beta_A[:,None]).sum(axis=0) # [emb_dim]
        else:
            beta_A = np.array([]); attn_emb_A = np.zeros(self.emb_dim)

        # Code-level beta attention for visit B
        if idx_B:
            code_vecs_B = self.E[idx_B]
            beta_scores_B = code_vecs_B @ beta_proj_B
            beta_scores_B -= beta_scores_B.max()
            beta_B = np.exp(beta_scores_B) / (np.exp(beta_scores_B).sum() + 1e-9)
            attn_emb_B = (code_vecs_B * beta_B[:,None]).sum(axis=0)
        else:
            beta_B = np.array([]); attn_emb_B = np.zeros(self.emb_dim)

        # Step 4: Context = alpha-weighted sum of attended visit embeddings
        context = alpha_A * attn_emb_A + alpha_B * attn_emb_B     # [emb_dim]
        context = self._dropout(context)

        # Step 5: Output
        logit = float(context @ self.W_out + self.b_out)
        prob  = float(expit(logit))

        cache = dict(h_A=h_A, h_B=h_B, hid_A=hid_A, hid_B=hid_B,
                     hid2_A=hid2_A, hid2_B=hid2_B,
                     beta_proj_A=beta_proj_A, beta_proj_B=beta_proj_B,
                     alpha_A=alpha_A, alpha_B=alpha_B,
                     attn_emb_A=attn_emb_A, attn_emb_B=attn_emb_B,
                     context=context, logit=logit, prob=prob,
                     idx_A=idx_A, idx_B=idx_B,
                     beta_A=beta_A, beta_B=beta_B)
        return prob, alpha_A, alpha_B, beta_A, beta_B, cache

    def backward(self, cache, label, pos_weight=3.0):
        """
        Compute gradients via manual backprop and accumulate into self._grads.
        Uses weighted BCE loss.
        """
        prob  = cache["prob"]
        y     = float(label)
        w     = pos_weight if y == 1 else 1.0

        # dL/d_logit
        d_logit = w * (prob - y)

        # Output layer gradients
        self._grads["W_out"] += np.outer(cache["context"], [d_logit])
        self._grads["b_out"] += np.array([d_logit])

        # d_context
        d_ctx = d_logit * self.W_out.squeeze()                     # [emb_dim]

        # Alpha gradients (through context = αA*eA + αB*eB)
        d_alpha_A_emb = d_ctx * cache["attn_emb_A"]
        d_alpha_B_emb = d_ctx * cache["attn_emb_B"]
        d_alpha_A = float(d_alpha_A_emb.sum())
        d_alpha_B = float(d_alpha_B_emb.sum())

        # Alpha softmax backward: d_score_A, d_score_B
        aA, aB = cache["alpha_A"], cache["alpha_B"]
        d_score_A = d_alpha_A * aA * (1 - aA) - d_alpha_B * aA * aB
        d_score_B = d_alpha_B * aB * (1 - aB) - d_alpha_A * aA * aB

        # W_alpha gradient
        self._grads["W_alpha"] += np.outer(cache["hid_A"], [d_score_A])
        self._grads["W_alpha"] += np.outer(cache["hid_B"], [d_score_B])
        self._grads["b_alpha"] += np.array([d_score_A + d_score_B])

        # Beta projection gradients (simplified: treat beta as attention over embeddings)
        d_attn_A = d_ctx * cache["alpha_A"]
        d_attn_B = d_ctx * cache["alpha_B"]

        # W_beta gradients
        self._grads["W_beta"] += np.outer(cache["hid2_A"], d_attn_A * (1 - cache["beta_proj_A"]**2))
        self._grads["W_beta"] += np.outer(cache["hid2_B"], d_attn_B * (1 - cache["beta_proj_B"]**2))
        self._grads["b_beta"] += d_attn_A * (1 - cache["beta_proj_A"]**2)
        self._grads["b_beta"] += d_attn_B * (1 - cache["beta_proj_B"]**2)

        # Embedding gradients for visited codes (simplified)
        d_emb_scale = 0.1  # dampening to stabilise embedding updates
        if cache["idx_A"]:
            for ci, idx in enumerate(cache["idx_A"]):
                if ci < len(cache["beta_A"]):
                    self._grads["E"][idx] += d_emb_scale * cache["beta_A"][ci] * d_attn_A
        if cache["idx_B"]:
            for ci, idx in enumerate(cache["idx_B"]):
                if ci < len(cache["beta_B"]):
                    self._grads["E"][idx] += d_emb_scale * cache["beta_B"][ci] * d_attn_B

    def adam_step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for n in self._param_names:
            g = self._grads[n]
            if np.all(g == 0): continue
            self._m[n] = beta1 * self._m[n] + (1 - beta1) * g
            self._v[n] = beta2 * self._v[n] + (1 - beta2) * g**2
            m_hat = self._m[n] / (1 - beta1**self._t)
            v_hat = self._v[n] / (1 - beta2**self._t)
            setattr(self, n, getattr(self, n) - lr * m_hat / (np.sqrt(v_hat) + eps))
            self._grads[n][:] = 0

    def freeze_embeddings(self):
        self._param_names = [n for n in self._param_names if n != "E"]
        log.info("Embeddings frozen")

    def unfreeze_embeddings(self):
        if "E" not in self._param_names:
            self._param_names = ["E"] + self._param_names
        log.info("Embeddings unfrozen")

    def train(self): self.training = True
    def eval(self):  self.training = False

# ── 9. TRAINING UTILITIES ────────────────────────────────────────────────────

def bce_loss(prob, label, pos_weight=1.0):
    w = pos_weight if label == 1 else 1.0
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return -w * (label * np.log(p) + (1 - label) * np.log(1 - p))

def run_epoch(model, samples, vocab, lr, pos_weight,
              batch_size=32, train=True):
    model.train() if train else model.eval()
    if train: random.shuffle(samples)

    total_loss, probs, labels = 0.0, [], []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        for s in batch:
            idxA = vocab.enc(s["visit_A"])
            idxB = vocab.enc(s["visit_B"])
            y    = s["preventable"]

            prob, aA, aB, betA, betB, cache = model.forward(idxA, idxB)
            loss = bce_loss(prob, y, pos_weight)
            total_loss += loss
            probs.append(prob); labels.append(y)

            if train:
                model.backward(cache, y, pos_weight)

        if train:
            model.adam_step(lr)

    avg_loss = total_loss / max(len(samples), 1)
    try:
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
    except: auroc = auprc = 0.0
    return avg_loss, auroc, auprc

# ── 10. INFERENCE & AUDIT REPORT ─────────────────────────────────────────────

def run_inference(model, samples, vocab):
    model.eval()
    records = []
    for s in samples:
        idxA = vocab.enc(s["visit_A"])
        idxB = vocab.enc(s["visit_B"])
        prob, aA, aB, betA, betB, _ = model.forward(idxA, idxB)

        # Top-5 attended codes for each visit
        def top_codes(visit_tokens, beta_weights, n=5):
            if len(beta_weights) == 0: return []
            n = min(n, len(visit_tokens), len(beta_weights))
            idx = np.argsort(beta_weights)[::-1][:n]
            return [(visit_tokens[i], round(float(beta_weights[i]),3))
                    for i in idx if i < len(visit_tokens)]

        top_A = top_codes(s["visit_A"], betA)
        top_B = top_codes(s["visit_B"], betB)

        records.append({
            "pair_id":              s["pair_id"],
            "member_id":            s["member_id"],
            "provider_tax_id":      s["provider_tax_id"],
            "preventable_prob":     round(prob, 4),
            "predicted_preventable":int(prob >= 0.5),
            "actual_preventable":   s["preventable"],
            "alpha_claimA":         round(float(aA), 4),
            "alpha_claimB":         round(float(aB), 4),
            "index_claim_driven":   bool(aA > aB),
            "top_codes_claimA":     str(top_A),
            "top_codes_claimB":     str(top_B),
            "drg_A":                s["drg_A"],
            "drg_B":                s["drg_B"],
            "los_A":                s["los_A"],
            "disch_A":              s["disch_A"],
            "days_between":         s["days_between"],
        })
    return pd.DataFrame(records)

def build_audit_queue(results_df, top_n):
    pvd = results_df.groupby("provider_tax_id").agg(
        total_pairs              =("pair_id","count"),
        actual_preventable_rate  =("actual_preventable","mean"),
        mean_predicted_prob      =("preventable_prob","mean"),
        high_conf_preventable    =("preventable_prob", lambda x:(x>=0.75).sum()),
        index_driven_count       =("index_claim_driven","sum"),
        mean_alpha_A             =("alpha_claimA","mean"),
    ).reset_index()

    pvd["performance_gap"] = (pvd["actual_preventable_rate"] -
                               pvd["mean_predicted_prob"])
    if len(pvd) > 1:
        pvd["gap_zscore"] = zscore(pvd["performance_gap"].fillna(0))
    else:
        pvd["gap_zscore"] = 0.0

    AVG_PAYMENT = 12_000
    pvd["estimated_exposure_usd"] = pvd["high_conf_preventable"] * AVG_PAYMENT

    pvd["priority_score"] = (
        pvd["gap_zscore"].clip(lower=0) *
        np.log1p(pvd["total_pairs"]) *
        pvd["mean_predicted_prob"]
    )
    queue = (pvd[pvd["total_pairs"] >= 10]
             .sort_values("priority_score", ascending=False)
             .head(top_n)
             .reset_index(drop=True))
    queue["rank"] = queue.index + 1
    return queue

# ── 11. MAIN ─────────────────────────────────────────────────────────────────

def main():
    cfg = CFG
    log.info("="*60)
    log.info("RETAIN Preventability Pipeline — Starting")
    log.info("="*60)

    # ── Load data ──────────────────────────────────────────────
    log.info(f"Loading {cfg['data_path']}...")
    df = pd.read_csv(cfg["data_path"])
    log.info(f"Loaded {len(df):,} rows")

    # ── Label ──────────────────────────────────────────────────
    if cfg["label_column"] and cfg["label_column"] in df.columns:
        df["preventable"] = df[cfg["label_column"]].astype(int)
        log.info(f"Using existing labels — preventable rate: {df['preventable'].mean():.1%}")
    else:
        log.info("Auto-generating labels via CCS rules...")
        df["preventable"] = df.apply(label_preventable, axis=1)
        log.info(f"Auto-labeled — preventable rate: {df['preventable'].mean():.1%}")

    # ── Build samples ──────────────────────────────────────────
    samples = build_samples(df)
    log.info(f"Built {len(samples):,} samples")

    # ── Split by member_id ────────────────────────────────────
    members = np.array([s["member_id"] for s in samples])
    unique_mbr = np.unique(members)
    np.random.shuffle(unique_mbr)
    n = len(unique_mbr)
    tr_m = set(unique_mbr[:int(0.70*n)])
    va_m = set(unique_mbr[int(0.70*n):int(0.85*n)])
    te_m = set(unique_mbr[int(0.85*n):])

    tr = [s for s in samples if s["member_id"] in tr_m]
    va = [s for s in samples if s["member_id"] in va_m]
    te = [s for s in samples if s["member_id"] in te_m]
    log.info(f"Split — Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")

    # ── Vocabulary ────────────────────────────────────────────
    vocab = Vocab().fit(tr)

    # ── Graph + embeddings ────────────────────────────────────
    G   = build_graph(tr, cfg)
    emb = generate_embeddings(G, vocab, cfg)

    # ── Model ─────────────────────────────────────────────────
    model = RETAINNumpy(
        vocab_size=len(vocab),
        emb_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout_rate=cfg["dropout_rate"],
        pretrained_emb=emb
    )

    # ── Phase 1: Frozen embeddings ───────────────────────────
    log.info(f"\nPhase 1: {cfg['freeze_epochs']} epochs with frozen embeddings")
    model.freeze_embeddings()
    best_val_prc = 0.0; best_params = None

    for ep in range(cfg["freeze_epochs"]):
        tl, ta, tp = run_epoch(model, tr, vocab, cfg["learning_rate"],
                                cfg["pos_class_weight"], cfg["batch_size"], train=True)
        vl, va_, vp = run_epoch(model, va, vocab, cfg["learning_rate"],
                                 cfg["pos_class_weight"], cfg["batch_size"], train=False)
        log.info(f"  [Frozen {ep+1:02d}] TrainLoss={tl:.3f} AUPRC={tp:.3f} | "
                 f"ValAUROC={va_:.3f} ValAUPRC={vp:.3f}")
        if vp > best_val_prc:
            best_val_prc = vp
            best_params  = {n: getattr(model,n).copy() for n in model._param_names + ["E"]}

    # ── Phase 2: Full fine-tuning ─────────────────────────────
    log.info(f"\nPhase 2: {cfg['num_epochs']} epochs full fine-tuning")
    model.unfreeze_embeddings()
    lr = cfg["learning_rate"]

    for ep in range(cfg["num_epochs"]):
        tl, ta, tp = run_epoch(model, tr, vocab, lr,
                                cfg["pos_class_weight"], cfg["batch_size"], train=True)
        vl, va_, vp = run_epoch(model, va, vocab, lr,
                                 cfg["pos_class_weight"], cfg["batch_size"], train=False)
        log.info(f"  [Epoch {ep+1:02d}/{cfg['num_epochs']}] "
                 f"TrainLoss={tl:.3f} AUPRC={tp:.3f} | "
                 f"ValAUROC={va_:.3f} ValAUPRC={vp:.3f}")
        if vp > best_val_prc:
            best_val_prc = vp
            best_params  = {n: getattr(model,n).copy() for n in model._param_names + ["E"]}
        # Simple LR decay
        if (ep+1) % 15 == 0: lr *= 0.5

    # ── Restore best model ────────────────────────────────────
    if best_params:
        for n, v in best_params.items():
            setattr(model, n, v)
        log.info(f"\nRestored best model (Val AUPRC={best_val_prc:.4f})")

    # ── Test evaluation ───────────────────────────────────────
    log.info("\nTest set evaluation...")
    _, te_auroc, te_prc = run_epoch(model, te, vocab, lr,
                                     cfg["pos_class_weight"],
                                     cfg["batch_size"], train=False)

    model.eval()
    te_probs  = []
    te_labels = []
    for s in te:
        idxA = vocab.enc(s["visit_A"])
        idxB = vocab.enc(s["visit_B"])
        prob, *_ = model.forward(idxA, idxB)
        te_probs.append(prob); te_labels.append(s["preventable"])

    preds = [1 if p >= 0.5 else 0 for p in te_probs]
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"AUROC : {te_auroc:.4f}")
    print(f"AUPRC : {te_prc:.4f}")
    print(classification_report(te_labels, preds,
                                 target_names=["Not Preventable","Preventable"],
                                 zero_division=0))

    # ── Full inference ────────────────────────────────────────
    log.info("Running inference on all pairs...")
    results_df = run_inference(model, samples, vocab)

    # ── Provider audit queue ──────────────────────────────────
    audit_df = build_audit_queue(results_df, cfg["top_n_providers"])

    # ── Save outputs ──────────────────────────────────────────
    r_path = os.path.join(cfg["output_dir"], "pair_level_scores.csv")
    a_path = os.path.join(cfg["output_dir"], "provider_audit_queue.csv")
    results_df.to_csv(r_path, index=False)
    audit_df.to_csv(a_path,   index=False)
    log.info(f"\nSaved: {r_path}")
    log.info(f"Saved: {a_path}")

    # ── Print audit queue ─────────────────────────────────────
    print("\n" + "="*60)
    print("PROVIDER AUDIT QUEUE — TOP FLAGGED PROVIDERS")
    print("="*60)
    cols = ["rank","provider_tax_id","total_pairs","actual_preventable_rate",
            "mean_predicted_prob","performance_gap","high_conf_preventable",
            "estimated_exposure_usd","priority_score"]
    ecols = [c for c in cols if c in audit_df.columns]
    pd.set_option("display.float_format","{:.3f}".format)
    pd.set_option("display.max_columns",20)
    pd.set_option("display.width",120)
    print(audit_df[ecols].to_string(index=False))

    # ── Print sample high-confidence cases for top provider ──
    if not audit_df.empty:
        top_prov = audit_df.iloc[0]["provider_tax_id"]
        cases = results_df[
            (results_df["provider_tax_id"] == top_prov) &
            (results_df["preventable_prob"] >= 0.75)
        ].head(5)

        print(f"\n{'='*60}")
        print(f"SAMPLE HIGH-CONFIDENCE CASES — Provider: {top_prov}")
        print("="*60)
        for _, r in cases.iterrows():
            print(f"\n  Pair ID          : {r['pair_id']}")
            print(f"  Member ID        : {r['member_id']}")
            print(f"  Preventable Prob : {r['preventable_prob']:.3f}")
            print(f"  Actual Label     : {r['actual_preventable']}")
            print(f"  Alpha Claim A    : {r['alpha_claimA']:.3f}  "
                  f"(index driven={r['index_claim_driven']})")
            print(f"  Alpha Claim B    : {r['alpha_claimB']:.3f}")
            print(f"  DRG  A → B       : {r['drg_A']} → {r['drg_B']}")
            print(f"  LOS  A           : {r['los_A']}")
            print(f"  Discharge A      : {r['disch_A']}")
            print(f"  Days Between     : {r['days_between']}")
            print(f"  Top Codes A      : {r['top_codes_claimA']}")
            print(f"  Top Codes B      : {r['top_codes_claimB']}")

    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
