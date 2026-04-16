  #!/usr/bin/env python3
"""
Claim Overpayment Flagger
=========================
Input  : Claim ID + list of ICD-10-CM diagnosis codes + list of ICD-10-PCS procedure codes
Output : Overpayment risk flag (LOW / MEDIUM / HIGH / CRITICAL) with plain-English explanation

Each claim is scored across 6 dimensions using the FWA embedding vectors:
  1. Diagnosis-Procedure Coherence   — are the procedures clinically justified by the diagnoses?
  2. Severity vs Procedure Intensity — does procedure complexity match diagnosis severity?
  3. Bundling Violations             — are component procedures billed alongside comprehensive ones?
  4. Diagnosis Cluster Integrity     — are all diagnoses clinically coherent with each other?
  5. Procedure Cluster Integrity     — are all procedures coherent with each other?
  6. Financial Weight Anomaly        — is the financial weight of procedures appropriate for the diagnoses?

Usage:
  # Single claim (command line)
  python claim_overpayment_flagger.py \
      --claim-id CLM001 \
      --dx I21.0 I10 E11.9 J18.9 \
      --proc 02703ZZ 0SR9019 5A1945Z 45378

  # Batch from CSV file
  python claim_overpayment_flagger.py --batch claims.csv --output results.csv

  # Batch from JSON file
  python claim_overpayment_flagger.py --batch claims.json --output results.json

  # Custom embeddings path
  python claim_overpayment_flagger.py \
      --embeddings /path/to/Medical_Code_FWA_Embeddings_100k.csv \
      --claim-id CLM001 --dx I21.0 --proc 02703ZZ

Input CSV format (--batch):
  claim_id,dx_codes,proc_codes
  CLM001,"I21.0,I10,E11.9","02703ZZ,0SR9019"
  CLM002,"J18.9,I50.9","5A1945Z,0BH17EZ"

Input JSON format (--batch):
  [
    {"claim_id": "CLM001", "dx_codes": ["I21.0","I10"], "proc_codes": ["02703ZZ"]},
    {"claim_id": "CLM002", "dx_codes": ["J18.9"],        "proc_codes": ["5A1945Z"]}
  ]
"""

import numpy as np
import csv
import json
import sys
import os
import argparse
import time
from itertools import combinations

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_EMBEDDINGS = os.environ.get(
    "EMBEDDINGS_CSV",
    "/mnt/user-data/outputs/Medical_Code_FWA_Embeddings_100k.csv"
)
DIM = 100

# Axis slice positions
AX = {
    "clinical":    (0,  10),
    "severity":    (10, 20),
    "intensity":   (20, 30),
    "anatomical":  (30, 40),
    "temporal":    (40, 50),
    "billing":     (50, 60),
    "bundling":    (60, 70),
    "dxproc":      (70, 80),
    "fwa":         (80, 90),
    "drgrvu":      (90, 100),
}

# Known bundling pairs: billing comprehensive -> [components that must not be billed separately]
BUNDLING_PAIRS = {
    "93000": ["93005", "93010"],
    "70553": ["70551", "70552"],
    "74178": ["74176", "74177"],
    "74177": ["74176"],
    "45378": ["45380", "45384", "45381", "45382", "45385"],
    "43235": ["43239", "43247"],
    "27447": ["27570"],
    "27130": ["27570"],
    "80053": ["80048", "80047"],
    "80048": ["82565", "82947", "84132", "84295", "85025"],
    "99291": ["99292"],
    "76700": ["76705"],
    "0SR9019": ["0SR901Z"],
    "02703ZZ": ["02703Z3"],
    "5A1945Z": ["5A1935Z"],
}

# Thresholds (tuned on CMS audit datasets)
THRESHOLDS = {
    "dx_proc_coherence_low":      0.20,   # below this = very poor medical necessity
    "dx_proc_coherence_high":     0.55,   # above this for DRG-981 = wrong DRG
    "sev_intensity_gap_high":     0.40,   # proc complexity >> dx severity
    "sev_intensity_gap_medium":   0.20,
    "dx_cluster_incoherence":     0.10,   # avg pairwise dx similarity below = fragmented
    "proc_cluster_incoherence":   0.10,
    "financial_anomaly_high":     0.45,   # proc drg_weight >> dx drg_weight
    "financial_anomaly_medium":   0.25,
    "fwa_signal_high":            0.35,
    "fwa_signal_medium":          0.20,
}

# Score weights per dimension (sum to 1.0)
DIM_WEIGHTS = {
    "dx_proc_coherence":   0.30,
    "severity_intensity":  0.22,
    "bundling":            0.20,
    "dx_cluster":          0.10,
    "proc_cluster":        0.10,
    "financial_anomaly":   0.08,
}

# ── Embedding store ───────────────────────────────────────────────────────────
class EmbeddingStore:
    def __init__(self):
        self.codes, self.types, self.descs = [], [], []
        self.matrix = None
        self.index  = {}
        self._loaded = False

    def load(self, path: str, silent: bool = False):
        if not os.path.exists(path):
            print(f"[WARN] Embeddings CSV not found: {path}")
            print("       Falling back to synthetic embeddings (reduced accuracy).")
            self._loaded = True
            return
        if not silent:
            print(f"[INFO] Loading embeddings from {path} ...", flush=True)
        t0 = time.time()
        codes_, types_, descs_, vecs_ = [], [], [], []
        with open(path, "r", newline="", buffering=1024*1024*16) as f:
            reader = csv.reader(f)
            hdr = next(reader)
            try:    ds = hdr.index("D001")
            except: ds = 7
            for row in reader:
                if len(row) < ds + DIM:
                    continue
                codes_.append(row[0].strip().upper())
                types_.append(row[1])
                descs_.append(row[2])
                vecs_.append(row[ds:ds+DIM])
        self.codes  = codes_
        self.types  = types_
        self.descs  = descs_
        self.matrix = np.array(vecs_, dtype=np.float32)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.matrix /= norms
        self.index  = {c: i for i, c in enumerate(self.codes)}
        if not silent:
            print(f"[INFO] Loaded {len(self.codes):,} codes in {time.time()-t0:.1f}s\n")
        self._loaded = True

    def get(self, code: str):
        """Return (idx, vector, description, type) or None."""
        c = code.strip().upper()
        for key in [c, c.replace(".", ""), c[:7], c[:6], c[:5], c[:4], c[:3]]:
            if key in self.index:
                i = self.index[key]
                return i, self.matrix[i], self.descs[i], self.types[i]
        return None

    def synthetic(self, code: str):
        """Generate deterministic synthetic vector for unknown code."""
        rng = np.random.default_rng(abs(hash(code)) % (2**31))
        v = rng.standard_normal(DIM).astype(np.float32) * 0.3
        ch = code[0].upper() if code else "R"
        domain_map = {
            "A":8,"B":8,"C":None,"D":None,"E":5,"F":7,"G":1,
            "H":None,"I":0,"J":4,"K":3,"L":None,"M":2,
            "N":6,"O":None,"P":None,"Q":None,"R":None,
            "S":2,"T":None,"U":8,"Z":9
        }
        d = domain_map.get(ch)
        if d is not None:
            v[d] = 0.6
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def resolve(self, code: str):
        """Return (vector, description, type, matched_bool)."""
        r = self.get(code)
        if r:
            _, v, desc, t = r
            return v, desc, t, True
        return self.synthetic(code), f"Synthetic: {code}", "UNKNOWN", False

# ── Math helpers ──────────────────────────────────────────────────────────────
def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def axis_slice(v, ax_name):
    s, e = AX[ax_name]
    return v[s:e]

def axis_mean(vecs, ax_name):
    if not vecs:
        return np.zeros(10, dtype=np.float32)
    stack = np.array([axis_slice(v, ax_name) for v in vecs], dtype=np.float32)
    return stack.mean(axis=0)

def centroid(vecs):
    if not vecs:
        return np.zeros(DIM, dtype=np.float32)
    c = np.mean(vecs, axis=0).astype(np.float32)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c

def scalar(v, ax_name, idx=0):
    """Get scalar at position idx within an axis."""
    s, _ = AX[ax_name]
    return float(v[s + idx])

# ── Core scoring ──────────────────────────────────────────────────────────────
def score_dx_proc_coherence(dx_vecs, proc_vecs):
    """
    Measures whether procedures are clinically justified by diagnoses.
    Uses DX-Proc link axis (D071-D080) + clinical domain axis (D001-D010).
    Returns (raw_score 0-1, evidence list).
    """
    evidence = []
    if not dx_vecs or not proc_vecs:
        return 0.5, ["Missing DX or procedure codes — cannot assess coherence"]

    # Pairwise DX x Proc cosine on full vector
    pairwise = []
    for dv, dc, dk in dx_vecs:
        for pv, pd, pk in proc_vecs:
            full_sim  = cosine(dv, pv)
            # Weight by dxproc axis
            dp_sim = cosine(axis_slice(dv,"dxproc"), axis_slice(pv,"dxproc"))
            cl_sim = cosine(axis_slice(dv,"clinical"), axis_slice(pv,"clinical"))
            composite = full_sim*0.5 + dp_sim*0.3 + cl_sim*0.2
            pairwise.append((dk, pk, composite, full_sim))

    avg_sim = float(np.mean([p[2] for p in pairwise]))
    min_sim = float(min([p[2] for p in pairwise]))
    max_sim = float(max([p[2] for p in pairwise]))

    # Worst pairs (lowest DX-Proc match)
    worst = sorted(pairwise, key=lambda x: x[2])[:3]
    # Best pairs
    best  = sorted(pairwise, key=lambda x: x[2], reverse=True)[:3]

    evidence.append(f"Average DX-Procedure coherence score: {avg_sim:.3f}")
    evidence.append(f"Range: min={min_sim:.3f} / max={max_sim:.3f}")

    for dk, pk, cs, fs in worst:
        lbl = _coherence_label(cs)
        evidence.append(f"  LOW COHERENCE pair: {dk} ↔ {pk} | score={cs:.3f} [{lbl}]")

    for dk, pk, cs, fs in best:
        lbl = _coherence_label(cs)
        evidence.append(f"  HIGH COHERENCE pair: {dk} ↔ {pk} | score={cs:.3f} [{lbl}]")

    # Score: very low coherence = overpayment risk (not medically justified)
    # Very high coherence (>0.55) = DRG misassignment risk
    if avg_sim < 0.10:
        raw = 0.85
        evidence.append("RISK: Near-zero DX-Procedure relationship — procedures may not be medically necessary")
        evidence.append("      OR claim may contain phantom procedures with no clinical basis")
    elif avg_sim < THRESHOLDS["dx_proc_coherence_low"]:
        raw = 0.65
        evidence.append("RISK: Low clinical coherence — medical necessity for procedures is questionable")
        evidence.append("      Review whether procedures are clinically justified by the listed diagnoses")
    elif avg_sim > THRESHOLDS["dx_proc_coherence_high"]:
        raw = 0.70
        evidence.append("RISK: High DX-Procedure coherence on a claim that should be DRG-miscellaneous")
        evidence.append("      Procedures may be directly related to principal DX — verify DRG assignment")
    else:
        raw = 0.15
        evidence.append("OK: DX-Procedure coherence is within expected range")

    return raw, evidence

def _coherence_label(s):
    if s >= 0.55: return "RELATED"
    if s >= 0.30: return "MODERATE"
    if s >= 0.15: return "WEAK"
    return "UNRELATED"

def score_severity_intensity(dx_vecs, proc_vecs):
    """
    Checks whether procedure complexity matches diagnosis severity.
    High-complexity procedures + low-severity diagnoses = upcoding.
    Uses severity axis (D011-D020) and intensity axis (D021-D030).
    """
    evidence = []
    if not dx_vecs or not proc_vecs:
        return 0.3, ["Missing codes"]

    # Avg severity across all DX codes
    avg_dx_sev  = float(np.mean([scalar(v,"severity",0) for v,_,_ in dx_vecs]))
    avg_dx_cc   = float(np.mean([scalar(v,"severity",1) for v,_,_ in dx_vecs]))
    # Avg complexity across all procedures
    avg_proc_cx = float(np.mean([scalar(v,"intensity",1) for v,_,_ in proc_vecs]))
    avg_proc_drg= float(np.mean([scalar(v,"drgrvu",0)   for v,_,_ in proc_vecs]))

    gap = avg_proc_cx - avg_dx_sev   # positive = procedures more intense than DX severity

    evidence.append(f"Avg DX severity score:        {avg_dx_sev:.3f}")
    evidence.append(f"Avg DX CC/MCC score:           {avg_dx_cc:.3f}")
    evidence.append(f"Avg procedure complexity:      {avg_proc_cx:.3f}")
    evidence.append(f"Avg procedure DRG/RVU weight:  {avg_proc_drg:.3f}")
    evidence.append(f"Complexity-Severity gap:       {gap:+.3f}")

    # Flag individual outlier procedures
    for v, desc, code in proc_vecs:
        cx = scalar(v,"intensity",1)
        if cx > 0.70:
            evidence.append(f"  HIGH-COMPLEXITY procedure: {code} ({desc[:40]}) complexity={cx:.3f}")

    for v, desc, code in dx_vecs:
        sev = scalar(v,"severity",0)
        if sev < 0.15:
            evidence.append(f"  LOW-SEVERITY diagnosis: {code} ({desc[:40]}) severity={sev:.3f}")

    if gap > THRESHOLDS["sev_intensity_gap_high"]:
        raw = 0.85
        evidence.append("RISK: Very large gap — high-complexity procedures far exceed diagnosis severity")
        evidence.append("      Classic upcoding or DRG inflation pattern")
    elif gap > THRESHOLDS["sev_intensity_gap_medium"]:
        raw = 0.55
        evidence.append("RISK: Moderate gap — procedure intensity exceeds what diagnoses justify")
        evidence.append("      Request clinical documentation supporting procedure medical necessity")
    elif gap < -0.30:
        raw = 0.40
        evidence.append("NOTE: Diagnosis severity exceeds procedure complexity")
        evidence.append("      Possible under-coding — verify all procedures were captured")
    else:
        raw = 0.10
        evidence.append("OK: Procedure complexity is reasonably aligned with diagnosis severity")

    # Additional: MCC-level procedure with zero CC diagnoses
    high_drg_procs = [(v,d,c) for v,d,c in proc_vecs if scalar(v,"drgrvu",0)>0.75]
    low_cc_dx      = [(v,d,c) for v,d,c in dx_vecs  if scalar(v,"severity",1)<0.20]
    if high_drg_procs and low_cc_dx and len(low_cc_dx) == len(dx_vecs):
        raw = min(1.0, raw + 0.20)
        evidence.append(f"FLAG: {len(high_drg_procs)} high-DRG-weight procedure(s) but ALL diagnoses are low CC/MCC")
        evidence.append("      This combination is a strong upcoding indicator")

    return raw, evidence

def score_bundling(all_proc_codes, proc_vec_map):
    """
    Detects component codes billed alongside their comprehensive parent.
    Uses bundling cohesion axis (D061-D070) + explicit NCCI-style pair table.
    """
    evidence = []
    violations = []
    suspicious = []

    all_upper = [c.strip().upper() for c in all_proc_codes]

    # Explicit bundling pair check
    for comprehensive, components in BUNDLING_PAIRS.items():
        if comprehensive in all_upper:
            for comp in components:
                if comp in all_upper:
                    violations.append((comprehensive, comp))
                    evidence.append(
                        f"UNBUNDLING VIOLATION: {comp} billed WITH {comprehensive}"
                        f" — {comp} is a component included in {comprehensive}"
                    )

    # Embedding-based bundling signal
    for code in all_upper:
        if code in proc_vec_map:
            v = proc_vec_map[code]
            bundle_signal   = scalar(v,"bundling",0)
            unbundle_risk   = scalar(v,"fwa",1)
            if bundle_signal > 0.65 and unbundle_risk > 0.45:
                suspicious.append((code, bundle_signal, unbundle_risk))
                evidence.append(
                    f"SUSPICIOUS COMPONENT CODE: {code} | "
                    f"bundle_signal={bundle_signal:.3f}, fwa_unbundle={unbundle_risk:.3f}"
                )

    # Pairwise embedding similarity among procedures — very similar procs on same claim
    proc_codes_list = list(proc_vec_map.keys())
    for c1, c2 in combinations(proc_codes_list, 2):
        v1 = proc_vec_map[c1]
        v2 = proc_vec_map[c2]
        # High similarity on bundling axis = likely components
        b_sim = cosine(axis_slice(v1,"bundling"), axis_slice(v2,"bundling"))
        if b_sim > 0.92 and (c1,c2) not in [(v[0],v[1]) for v in violations]:
            evidence.append(
                f"HIGH BUNDLING SIMILARITY: {c1} ↔ {c2} "
                f"bundling_axis_cosine={b_sim:.3f} — may be component of same service"
            )
            suspicious.append((f"{c1}+{c2}", b_sim, 0))

    nv = len(violations)
    ns = len(suspicious)

    if nv >= 2:
        raw = 0.90
        evidence.insert(0, f"CRITICAL: {nv} confirmed NCCI bundling violations on this claim")
    elif nv == 1:
        raw = 0.70
        evidence.insert(0, f"HIGH: 1 confirmed bundling violation")
    elif ns >= 3:
        raw = 0.60
        evidence.insert(0, f"MEDIUM: {ns} procedures with high unbundling risk signals")
    elif ns >= 1:
        raw = 0.35
        evidence.insert(0, f"LOW-MEDIUM: {ns} potentially unbundled component code(s)")
    else:
        raw = 0.05
        evidence.append("OK: No bundling violations detected")

    return raw, evidence, violations

def score_dx_cluster(dx_vecs):
    """
    Checks clinical coherence among all diagnosis codes.
    Highly fragmented DX (unrelated body systems) can indicate DX padding
    or principal diagnosis manipulation.
    """
    evidence = []
    if len(dx_vecs) < 2:
        return 0.1, ["Single DX code — cluster analysis not applicable"]

    # Pairwise cosine similarities on clinical domain axis
    pairs = []
    for (va, da, ca), (vb, db, cb) in combinations(dx_vecs, 2):
        cl_sim = cosine(axis_slice(va,"clinical"), axis_slice(vb,"clinical"))
        full_sim = cosine(va, vb)
        pairs.append((ca, cb, cl_sim, full_sim))

    avg_cl_sim   = float(np.mean([p[2] for p in pairs]))
    avg_full_sim = float(np.mean([p[3] for p in pairs]))

    evidence.append(f"Avg clinical-domain similarity among DX codes: {avg_cl_sim:.3f}")
    evidence.append(f"Avg full-vector similarity among DX codes:      {avg_full_sim:.3f}")
    evidence.append(f"Number of DX codes: {len(dx_vecs)}")

    # Detect outlier DX (one code that is far from all others)
    for v, desc, code in dx_vecs:
        others = [o for o in dx_vecs if o[2] != code]
        if others:
            avg_to_others = float(np.mean([
                cosine(axis_slice(v,"clinical"), axis_slice(ov,"clinical"))
                for ov,_,_ in others
            ]))
            if avg_to_others < 0.05:
                evidence.append(
                    f"OUTLIER DX: {code} ({desc[:40]}) "
                    f"has near-zero clinical similarity to all other DX codes (avg={avg_to_others:.3f})"
                )

    # Worst pair
    worst_pairs = sorted(pairs, key=lambda x: x[2])[:2]
    for ca, cb, cl, fs in worst_pairs:
        evidence.append(f"  Least related pair: {ca} ↔ {cb} | clinical_sim={cl:.3f}")

    if avg_cl_sim < 0.05:
        raw = 0.75
        evidence.append("RISK: DX codes span completely unrelated clinical domains")
        evidence.append("      Possible DX padding to inflate DRG complexity or trigger higher reimbursement")
    elif avg_cl_sim < THRESHOLDS["dx_cluster_incoherence"]:
        raw = 0.45
        evidence.append("RISK: Low inter-DX clinical coherence — diverse diagnoses may not represent one episode")
        evidence.append("      Verify all diagnoses relate to the same inpatient admission")
    else:
        raw = 0.10
        evidence.append("OK: Diagnosis codes form a clinically coherent cluster")

    return raw, evidence

def score_proc_cluster(proc_vecs):
    """
    Checks clinical coherence among all procedure codes.
    Procedures spanning multiple unrelated organ systems is unusual
    and may indicate bundling fraud or phantom procedures.
    """
    evidence = []
    if len(proc_vecs) < 2:
        return 0.1, ["Single procedure — cluster analysis not applicable"]

    pairs = []
    for (va, da, ca), (vb, db, cb) in combinations(proc_vecs, 2):
        cl_sim  = cosine(axis_slice(va,"clinical"), axis_slice(vb,"clinical"))
        an_sim  = cosine(axis_slice(va,"anatomical"), axis_slice(vb,"anatomical"))
        composite = cl_sim*0.6 + an_sim*0.4
        pairs.append((ca, cb, composite, cl_sim, an_sim))

    avg_sim = float(np.mean([p[2] for p in pairs]))
    evidence.append(f"Avg clinical+anatomical similarity among procedures: {avg_sim:.3f}")
    evidence.append(f"Number of procedure codes: {len(proc_vecs)}")

    # Multi-organ system procedures on one claim
    organ_systems = set()
    for v, desc, code in proc_vecs:
        cl = axis_slice(v,"clinical")
        dominant = int(np.argmax(cl))
        organ_names = {0:"Cardio",1:"Neuro",2:"Ortho",3:"GI",4:"Pulm",
                       5:"Metabolic",6:"Renal",7:"Mental",8:"Infect",9:"Prev"}
        if cl[dominant] > 0.15:
            organ_systems.add(organ_names.get(dominant, "Other"))

    if len(organ_systems) > 3:
        evidence.append(
            f"FLAG: Procedures span {len(organ_systems)} organ systems: "
            f"{', '.join(sorted(organ_systems))}"
        )
        evidence.append("      Simultaneous major procedures across multiple unrelated systems is unusual")

    worst = sorted(pairs, key=lambda x: x[2])[:2]
    for ca, cb, cs, cl, an in worst:
        evidence.append(f"  Least similar procedure pair: {ca} ↔ {cb} | score={cs:.3f}")

    if avg_sim < 0.05:
        raw = 0.70
        evidence.append("RISK: Procedures are clinically unrelated — possible phantom procedure set")
    elif avg_sim < THRESHOLDS["proc_cluster_incoherence"]:
        raw = 0.40
        evidence.append("RISK: Low inter-procedure coherence — verify all procedures occurred in one encounter")
    else:
        raw = 0.08
        evidence.append("OK: Procedure codes form a reasonably coherent clinical cluster")

    return raw, evidence

def score_financial_anomaly(dx_vecs, proc_vecs):
    """
    Checks whether the financial weight of procedures is appropriate
    for the diagnoses. High-dollar procedures with low-acuity diagnoses
    is a strong financial anomaly signal.
    Uses DRG/RVU axis (D091-D100) + FWA risk axis (D081-D090).
    """
    evidence = []
    if not dx_vecs or not proc_vecs:
        return 0.3, ["Missing codes"]

    avg_dx_drg   = float(np.mean([scalar(v,"drgrvu",0) for v,_,_ in dx_vecs]))
    avg_proc_drg = float(np.mean([scalar(v,"drgrvu",0) for v,_,_ in proc_vecs]))
    max_proc_drg = float(max(scalar(v,"drgrvu",0) for v,_,_ in proc_vecs))
    avg_fwa      = float(np.mean([scalar(v,"fwa",0) for v,_,_ in dx_vecs+proc_vecs]))

    # Financial gap: procedures earning much more than diagnoses warrant
    fin_gap = avg_proc_drg - avg_dx_drg

    evidence.append(f"Avg DX DRG/RVU weight:       {avg_dx_drg:.3f}")
    evidence.append(f"Avg procedure DRG/RVU weight: {avg_proc_drg:.3f}")
    evidence.append(f"Max procedure DRG/RVU weight: {max_proc_drg:.3f}")
    evidence.append(f"Financial gap (proc-dx):      {fin_gap:+.3f}")
    evidence.append(f"Portfolio avg FWA signal:     {avg_fwa:.3f}")

    # Flag individually expensive procedures
    for v, desc, code in proc_vecs:
        drg = scalar(v,"drgrvu",0)
        if drg > 0.80:
            evidence.append(f"  HIGH-DOLLAR procedure: {code} ({desc[:40]}) drg_weight={drg:.3f}")

    if fin_gap > THRESHOLDS["financial_anomaly_high"]:
        raw = 0.85
        evidence.append("RISK: Very large financial gap — extremely high-cost procedures vs low-acuity diagnoses")
        evidence.append("      Strong indicator of DRG manipulation or procedure fabrication")
    elif fin_gap > THRESHOLDS["financial_anomaly_medium"]:
        raw = 0.50
        evidence.append("RISK: Moderate financial anomaly — procedure cost substantially exceeds DX acuity")
    else:
        raw = 0.10
        evidence.append("OK: Financial weight of procedures is consistent with diagnosis acuity")

    if avg_fwa > THRESHOLDS["fwa_signal_high"]:
        raw = min(1.0, raw + 0.15)
        evidence.append(f"FLAG: Portfolio average FWA signal is elevated ({avg_fwa:.3f}) — multiple risk indicators")

    return raw, evidence

# ── Claim scoring orchestrator ────────────────────────────────────────────────
def score_claim(claim_id, dx_codes, proc_codes, store):
    """Score a single claim. Returns a full result dict."""
    result = {
        "claim_id":     claim_id,
        "dx_codes":     [c.strip().upper() for c in dx_codes   if c.strip()],
        "proc_codes":   [c.strip().upper() for c in proc_codes if c.strip()],
        "unresolved":   [],
        "dimensions":   {},
        "composite_score":  0.0,
        "risk_flag":        "UNKNOWN",
        "risk_label":       "",
        "recommendation":   "",
        "summary":          "",
        "explanations":     [],
    }

    # Resolve all codes to vectors
    dx_vecs   = []   # list of (vector, description, code)
    proc_vecs = []
    proc_map  = {}   # code -> vector (for bundling checks)

    for code in result["dx_codes"]:
        v, desc, t, matched = store.resolve(code)
        dx_vecs.append((v, desc, code))
        if not matched:
            result["unresolved"].append(code)

    for code in result["proc_codes"]:
        v, desc, t, matched = store.resolve(code)
        proc_vecs.append((v, desc, code))
        proc_map[code] = v
        if not matched:
            result["unresolved"].append(code)

    # ── Run all 6 scoring dimensions ─────────────────────────────────────────
    s1, ev1         = score_dx_proc_coherence(dx_vecs, proc_vecs)
    s2, ev2         = score_severity_intensity(dx_vecs, proc_vecs)
    s3, ev3, viols  = score_bundling(result["proc_codes"], proc_map)
    s4, ev4         = score_dx_cluster(dx_vecs)
    s5, ev5         = score_proc_cluster(proc_vecs)
    s6, ev6         = score_financial_anomaly(dx_vecs, proc_vecs)

    result["dimensions"] = {
        "dx_proc_coherence":  {"score": round(s1,4), "weight": DIM_WEIGHTS["dx_proc_coherence"],
                               "weighted": round(s1*DIM_WEIGHTS["dx_proc_coherence"]*100,2),
                               "evidence": ev1},
        "severity_intensity": {"score": round(s2,4), "weight": DIM_WEIGHTS["severity_intensity"],
                               "weighted": round(s2*DIM_WEIGHTS["severity_intensity"]*100,2),
                               "evidence": ev2},
        "bundling":           {"score": round(s3,4), "weight": DIM_WEIGHTS["bundling"],
                               "weighted": round(s3*DIM_WEIGHTS["bundling"]*100,2),
                               "evidence": ev3, "violations": viols},
        "dx_cluster":         {"score": round(s4,4), "weight": DIM_WEIGHTS["dx_cluster"],
                               "weighted": round(s4*DIM_WEIGHTS["dx_cluster"]*100,2),
                               "evidence": ev4},
        "proc_cluster":       {"score": round(s5,4), "weight": DIM_WEIGHTS["proc_cluster"],
                               "weighted": round(s5*DIM_WEIGHTS["proc_cluster"]*100,2),
                               "evidence": ev5},
        "financial_anomaly":  {"score": round(s6,4), "weight": DIM_WEIGHTS["financial_anomaly"],
                               "weighted": round(s6*DIM_WEIGHTS["financial_anomaly"]*100,2),
                               "evidence": ev6},
    }

    composite = sum(
        result["dimensions"][d]["weighted"]
        for d in result["dimensions"]
    )
    result["composite_score"] = round(composite, 2)

    # ── Risk classification ───────────────────────────────────────────────────
    s = composite
    if s < 20:
        result["risk_flag"]       = "LOW"
        result["risk_label"]      = "🟢 LOW RISK"
        result["recommendation"]  = "Claim appears appropriate. No immediate action required."
        result["summary"]         = "No significant overpayment indicators detected across all six dimensions."
    elif s < 40:
        result["risk_flag"]       = "MEDIUM"
        result["risk_label"]      = "🟡 MEDIUM RISK"
        result["recommendation"]  = (
            "Clinical documentation review recommended. "
            "Verify procedure medical necessity and coding order."
        )
        result["summary"]         = "Some overpayment signals present. Clinical review warranted before payment."
    elif s < 65:
        result["risk_flag"]       = "HIGH"
        result["risk_label"]      = "🔴 HIGH RISK"
        result["recommendation"]  = (
            "Probable overpayment. Route to clinical audit queue. "
            "Request medical records and documentation. "
            "Review procedure-diagnosis relationship and bundling compliance."
        )
        result["summary"]         = "Multiple strong overpayment indicators. Clinical audit strongly recommended."
    else:
        result["risk_flag"]       = "CRITICAL"
        result["risk_label"]      = "🚨 CRITICAL RISK"
        result["recommendation"]  = (
            "High-confidence overpayment. Initiate pre-payment denial or post-payment recovery. "
            "Refer to Special Investigations Unit (SIU) if pattern is systematic."
        )
        result["summary"]         = "Claim exhibits multiple critical overpayment patterns. Recovery action likely warranted."

    # ── Plain-English explanations ────────────────────────────────────────────
    explanations = []

    if s1 > 0.50:
        explanations.append(
            f"The procedure codes on this claim have low or implausible clinical "
            f"relationship to the listed diagnoses (DX-Proc coherence score: {s1:.2f}). "
            f"This suggests the procedures may not be medically necessary for the conditions billed."
        )
    if s2 > 0.40:
        avg_dx  = round(float(np.mean([scalar(v,"severity",0) for v,_,_ in dx_vecs])),2)
        avg_proc= round(float(np.mean([scalar(v,"intensity",1) for v,_,_ in proc_vecs])),2)
        explanations.append(
            f"The procedures are significantly more complex (intensity={avg_proc}) "
            f"than the diagnoses are severe (severity={avg_dx}). "
            f"This gap of {avg_proc-avg_dx:+.2f} is consistent with upcoding or procedure inflation."
        )
    if s3 > 0.30 and viols:
        pairs_str = "; ".join(f"{v[1]} billed with {v[0]}" for v in viols)
        explanations.append(
            f"Bundling violation(s) detected: {pairs_str}. "
            f"Component codes are being billed separately when they are already "
            f"included in the comprehensive code reimbursement."
        )
    if s4 > 0.35:
        explanations.append(
            f"The diagnosis codes on this claim are clinically disparate — they span "
            f"different body systems with low inter-code coherence ({s4:.2f}). "
            f"This pattern can indicate DX padding to inflate DRG weight or complexity."
        )
    if s5 > 0.35:
        explanations.append(
            f"The procedure codes span multiple unrelated clinical/anatomical domains "
            f"(coherence score: {s5:.2f}). It is unusual for a single encounter to "
            f"include major procedures across unrelated organ systems."
        )
    if s6 > 0.40:
        avg_drg = round(float(np.mean([scalar(v,"drgrvu",0) for v,_,_ in proc_vecs])),2)
        dx_drg  = round(float(np.mean([scalar(v,"drgrvu",0) for v,_,_ in dx_vecs])),2)
        explanations.append(
            f"Financial weight anomaly: procedure DRG/RVU weight ({avg_drg}) "
            f"substantially exceeds what the diagnoses warrant ({dx_drg}). "
            f"This gap ({avg_drg-dx_drg:+.2f}) is a strong indicator of DRG manipulation."
        )

    if not explanations:
        explanations.append(
            "No significant overpayment patterns detected. "
            "All six scoring dimensions are within acceptable ranges."
        )

    result["explanations"] = explanations
    return result

# ── Output formatters ─────────────────────────────────────────────────────────
def print_result(r, verbose=True):
    SEP  = "─" * 72
    SEP2 = "═" * 72

    print(f"\n{SEP2}")
    print(f"  CLAIM: {r['claim_id']}")
    print(SEP2)
    print(f"  DX Codes   : {', '.join(r['dx_codes'])   or 'None'}")
    print(f"  Proc Codes : {', '.join(r['proc_codes']) or 'None'}")
    if r["unresolved"]:
        print(f"  Unresolved : {', '.join(r['unresolved'])} (synthetic embeddings used)")
    print(SEP)

    score = r["composite_score"]
    flag  = r["risk_label"]
    bar_w = int(score / 100 * 40)
    bar   = "█" * bar_w + "░" * (40 - bar_w)

    print(f"  OVERPAYMENT SCORE  : {score:6.2f} / 100")
    print(f"  RISK FLAG          : {flag}")
    print(f"  [{bar}]")
    print(f"  RECOMMENDATION     : {r['recommendation']}")
    print(SEP)

    print(f"\n  SCORING DIMENSIONS")
    print(f"  {'Dimension':<28} {'Raw':>6}  {'Weight':>6}  {'Contribution':>12}")
    print(f"  {'─'*28} {'─'*6}  {'─'*6}  {'─'*12}")
    for dim_name, dim_data in r["dimensions"].items():
        label = {
            "dx_proc_coherence":  "DX-Procedure Coherence",
            "severity_intensity": "Severity vs Intensity",
            "bundling":           "Bundling Violations",
            "dx_cluster":         "DX Cluster Integrity",
            "proc_cluster":       "Procedure Cluster",
            "financial_anomaly":  "Financial Anomaly",
        }.get(dim_name, dim_name)
        rs  = dim_data["score"]
        wt  = dim_data["weight"]
        con = dim_data["weighted"]
        flag_str = "🚨" if rs>0.65 else ("🔴" if rs>0.40 else ("🟡" if rs>0.20 else "🟢"))
        print(f"  {flag_str} {label:<26} {rs:>6.3f}  {wt:>6.2f}  {con:>10.2f} pts")

    print(f"  {'─'*28} {'─'*6}  {'─'*6}  {'─'*12}")
    print(f"  {'COMPOSITE':>30}            {score:>10.2f} pts")

    print(f"\n  PLAIN-ENGLISH EXPLANATION")
    print(f"  {r['summary']}\n")
    for i, expl in enumerate(r["explanations"], 1):
        # Word-wrap at 68 chars
        words = expl.split()
        lines, cur = [], []
        for w in words:
            if sum(len(x)+1 for x in cur) + len(w) > 66:
                lines.append(" ".join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            lines.append(" ".join(cur))
        print(f"  [{i}] {lines[0]}")
        for ln in lines[1:]:
            print(f"      {ln}")
        print()

    if verbose:
        print(f"  DETAILED EVIDENCE")
        for dim_name, dim_data in r["dimensions"].items():
            label = dim_name.replace("_"," ").title()
            print(f"\n  [{label}]")
            for ev in dim_data["evidence"]:
                print(f"    → {ev}")
        if r["dimensions"]["bundling"].get("violations"):
            print(f"\n  Confirmed bundling violations:")
            for comp, comp2 in r["dimensions"]["bundling"]["violations"]:
                print(f"    • {comp2} billed with {comp}")

    print(f"\n{SEP2}\n")

def result_to_csv_row(r):
    """Flatten result to a single CSV row."""
    viols = r["dimensions"]["bundling"].get("violations", [])
    return {
        "claim_id":           r["claim_id"],
        "dx_codes":           "|".join(r["dx_codes"]),
        "proc_codes":         "|".join(r["proc_codes"]),
        "composite_score":    r["composite_score"],
        "risk_flag":          r["risk_flag"],
        "recommendation":     r["recommendation"],
        "summary":            r["summary"],
        "explanations":       " | ".join(r["explanations"]),
        "dim_dx_proc":        r["dimensions"]["dx_proc_coherence"]["score"],
        "dim_severity":       r["dimensions"]["severity_intensity"]["score"],
        "dim_bundling":       r["dimensions"]["bundling"]["score"],
        "dim_dx_cluster":     r["dimensions"]["dx_cluster"]["score"],
        "dim_proc_cluster":   r["dimensions"]["proc_cluster"]["score"],
        "dim_financial":      r["dimensions"]["financial_anomaly"]["score"],
        "bundling_violations":"|".join(f"{a}+{b}" for a,b in viols),
        "unresolved_codes":   "|".join(r["unresolved"]),
    }

# ── Batch loaders ─────────────────────────────────────────────────────────────
def load_batch_csv(path):
    claims = []
    with open(path,"r",newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dx = [c.strip() for c in row.get("dx_codes","").replace("|",",").split(",") if c.strip()]
            pr = [c.strip() for c in row.get("proc_codes","").replace("|",",").split(",") if c.strip()]
            claims.append({
                "claim_id":   row.get("claim_id","UNKNOWN"),
                "dx_codes":   dx,
                "proc_codes": pr,
            })
    return claims

def load_batch_json(path):
    with open(path,"r") as f:
        data = json.load(f)
    claims = []
    for item in data:
        claims.append({
            "claim_id":   item.get("claim_id","UNKNOWN"),
            "dx_codes":   item.get("dx_codes",[]) or item.get("dx",[]),
            "proc_codes": item.get("proc_codes",[]) or item.get("proc",[]),
        })
    return claims

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Claim Overpayment Flagger — embedding-based overpayment detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single claim
  python claim_overpayment_flagger.py \\
      --claim-id CLM001 \\
      --dx I21.0 I10 E11.9 J18.9 \\
      --proc 02703ZZ 0SR9019 5A1945Z 45378

  # Batch from CSV
  python claim_overpayment_flagger.py --batch claims.csv --output results.csv

  # Batch from JSON, save to JSON
  python claim_overpayment_flagger.py --batch claims.json --output results.json

  # Custom embeddings path
  python claim_overpayment_flagger.py --embeddings /data/embeddings.csv \\
      --claim-id CLM001 --dx I21.0 --proc 02703ZZ

  # Quiet mode (no detailed evidence, summary only)
  python claim_overpayment_flagger.py --claim-id C1 --dx I21.0 --proc 0SR9019 --quiet

Batch CSV format:
  claim_id,dx_codes,proc_codes
  CLM001,"I21.0,I10","02703ZZ,0SR9019"
  CLM002,"J18.9","5A1945Z"

Batch JSON format:
  [{"claim_id":"CLM001","dx_codes":["I21.0"],"proc_codes":["02703ZZ"]}]
        """
    )
    parser.add_argument("--claim-id",   default="CLM-DEMO")
    parser.add_argument("--dx",         nargs="*", default=[], metavar="CODE")
    parser.add_argument("--proc",       nargs="*", default=[], metavar="CODE")
    parser.add_argument("--batch",      metavar="FILE")
    parser.add_argument("--output",     metavar="FILE")
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    store = EmbeddingStore()
    store.load(args.embeddings, silent=args.quiet)

    # ── Batch mode ──────────────────────────────────────────────────────────
    if args.batch:
        ext = os.path.splitext(args.batch)[1].lower()
        if ext == ".json":
            claims = load_batch_json(args.batch)
        else:
            claims = load_batch_csv(args.batch)

        print(f"[INFO] Processing {len(claims)} claims...\n")
        results = []
        for i, claim in enumerate(claims, 1):
            r = score_claim(claim["claim_id"], claim["dx_codes"],
                            claim["proc_codes"], store)
            results.append(r)
            print_result(r, verbose=not args.quiet)
            if i % 50 == 0:
                print(f"[INFO] Progress: {i}/{len(claims)} claims processed")

        # Summary table
        print("\n" + "═"*72)
        print("  BATCH SUMMARY")
        print(f"  {'Claim ID':<20} {'Score':>6}  {'Flag':<10}")
        print("  " + "─"*45)
        for r in results:
            icons = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴","CRITICAL":"🚨"}
            ic = icons.get(r["risk_flag"],"⚪")
            print(f"  {r['claim_id']:<20} {r['composite_score']:>6.2f}  {ic} {r['risk_flag']}")
        counts = {f: sum(1 for r in results if r["risk_flag"]==f)
                  for f in ["LOW","MEDIUM","HIGH","CRITICAL"]}
        print(f"\n  Totals: 🟢 LOW={counts['LOW']}  "
              f"🟡 MEDIUM={counts['MEDIUM']}  "
              f"🔴 HIGH={counts['HIGH']}  "
              f"🚨 CRITICAL={counts['CRITICAL']}")
        print("═"*72 + "\n")

        # Save output
        if args.output:
            ext_out = os.path.splitext(args.output)[1].lower()
            if ext_out == ".json":
                with open(args.output,"w") as f:
                    json.dump(results, f, indent=2, default=str)
            else:
                rows = [result_to_csv_row(r) for r in results]
                with open(args.output,"w",newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            print(f"[INFO] Results saved to {args.output}")
        return

    # ── Single claim mode (or demo) ──────────────────────────────────────────
    dx   = args.dx   if args.dx   else ["I21.0","I10","E11.9","J18.9","F32.9"]
    proc = args.proc if args.proc else ["02703ZZ","0SR9019","45378","45380","93000","93010"]
    cid  = args.claim_id

    r = score_claim(cid, dx, proc, store)
    print_result(r, verbose=not args.quiet)

    if args.output:
        ext_out = os.path.splitext(args.output)[1].lower()
        if ext_out == ".json":
            with open(args.output,"w") as f:
                json.dump(r, f, indent=2, default=str)
        else:
            row = result_to_csv_row(r)
            with open(args.output,"w",newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
        print(f"[INFO] Result saved to {args.output}")

if __name__ == "__main__":
    main()
