#!/usr/bin/env python3
"""
12-Step Dimensional Payment Integrity Auditor
==============================================
Analyzes inpatient claims using all 100 embedding dimensions
across 10 semantic axes, following the 12-step audit protocol.

Input modes:
  A) Direct CLI  — --claim-id, --dx codes, --proc codes
  B) Batch CSV   — claim_id, dx_codes, proc_codes columns
  C) Batch JSON  — list of claim objects

Embedding file: Medical_Code_FWA_Embeddings_100k.csv
"""

import numpy as np
import csv, json, os, sys, time, argparse, textwrap
from itertools import combinations

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_EMB = os.environ.get(
    "EMBEDDINGS_CSV",
    "/mnt/user-data/outputs/Medical_Code_FWA_Embeddings_100k.csv"
)
DIM = 100

# ── Axis definitions ──────────────────────────────────────────────────────────
AXES = {
    "clinical":   {"range": (0,  10), "label": "Clinical Domain (0–9)",
                   "desc": "Cardio·Neuro·Ortho·GI·Pulm·Metabolic·Renal·MH·Infect·Prev"},
    "severity":   {"range": (10, 20), "label": "Severity / CC-MCC (10–19)",
                   "desc": "Complication tier, MCC/CC weighting, acuity"},
    "intensity":  {"range": (20, 30), "label": "Service Intensity (20–29)",
                   "desc": "E&M level, procedural complexity"},
    "anatomical": {"range": (30, 40), "label": "Anatomical Site (30–39)",
                   "desc": "Head·Chest·Spine·Abdomen·Extremity·Skin"},
    "episode":    {"range": (40, 50), "label": "Episode Type (40–49)",
                   "desc": "Acute·Chronic·Post-op·Preventive"},
    "billing":    {"range": (50, 60), "label": "Billing Channel (50–59)",
                   "desc": "Inpatient·DME·Outpatient·Professional"},
    "bundling":   {"range": (60, 70), "label": "Bundling Cohesion (60–69)",
                   "desc": "Component vs comprehensive service boundary"},
    "dxproc":     {"range": (70, 80), "label": "DX-Proc Link (70–79)",
                   "desc": "Medical necessity DX→procedure compatibility"},
    "fwa":        {"range": (80, 90), "label": "FWA Risk Signals (80–89)",
                   "desc": "Upcoding·Unbundling·Phantom·MCC-stacking risk"},
    "financial":  {"range": (90, 100),"label": "DRG / RVU Proxy (90–99)",
                   "desc": "Financial weight, reimbursement alignment"},
}

DOMAIN_NAMES = {
    0:"Cardiovascular", 1:"Neurological",  2:"Orthopedic/MSK",
    3:"Gastrointestinal",4:"Pulmonary",    5:"Metabolic/Endocrine",
    6:"Renal/GU",       7:"Mental Health", 8:"Infectious",
    9:"Preventive/Administrative"
}

ANAT_NAMES = {
    0:"Head/CNS", 1:"Chest/Cardiac", 2:"Spine/Back",
    3:"Abdomen/GI",4:"Extremities",  5:"Skin/Integument",
    6:"Pelvis/GU", 7:"Systemic",     8:"Multi-site", 9:"Unspecified"
}

NCCI = {
    "93000":["93005","93010"], "70553":["70551","70552"],
    "74178":["74176","74177"], "45378":["45380","45384","45381","45385"],
    "43235":["43239","43247"], "80053":["80048"],
    "99291":["99292"],         "5A1945Z":["5A1935Z"],
    "02703ZZ":["02703Z3"],     "0SRC069":["0SRC06Z"],
}

# ── Embedding store ───────────────────────────────────────────────────────────
class Store:
    def __init__(self):
        self.idx = {}; self.mat = None
        self.descs = []; self.types = []

    def load(self, path, silent=False):
        if not os.path.exists(path):
            if not silent:
                print(f"[WARN] {path} not found — synthetic embeddings will be used.")
            return
        if not silent:
            print(f"[INFO] Loading embeddings ...", flush=True)
        t0 = time.time()
        codes_, types_, descs_, vecs_ = [], [], [], []
        with open(path, "r", newline="", buffering=1<<24) as f:
            rdr = csv.reader(f); hdr = next(rdr)
            try: ds = hdr.index("D001")
            except: ds = 7
            for row in rdr:
                if len(row) < ds + DIM: continue
                codes_.append(row[0].strip().upper())
                types_.append(row[1]); descs_.append(row[2])
                vecs_.append(row[ds:ds+DIM])
        self.mat   = np.array(vecs_, dtype=np.float32)
        norms = np.linalg.norm(self.mat, axis=1, keepdims=True)
        norms[norms == 0] = 1; self.mat /= norms
        self.idx   = {c: i for i, c in enumerate(codes_)}
        self.descs = descs_; self.types = types_
        if not silent:
            print(f"[INFO] {len(codes_):,} codes loaded in {time.time()-t0:.1f}s\n")

    def get(self, code):
        c = code.strip().upper()
        for k in [c, c.replace(".",""), c[:7], c[:6], c[:5], c[:4], c[:3]]:
            if k in self.idx:
                i = self.idx[k]
                return self.mat[i], self.descs[i], self.types[i], True
        rng = np.random.default_rng(abs(hash(code)) % (2**31))
        v = rng.standard_normal(DIM).astype(np.float32) * 0.28
        ch = code[0].upper() if code else "R"
        dm = {"A":8,"B":8,"C":None,"E":5,"F":7,"G":1,"I":0,
              "J":4,"K":3,"M":2,"N":6,"Z":9}
        d  = dm.get(ch)
        if d is not None: v[d] = 0.58
        n = np.linalg.norm(v); v = v/n if n > 0 else v
        return v, f"Synthetic:{code}", "UNKNOWN", False

# ── Vector math ───────────────────────────────────────────────────────────────
def cos(a, b):
    na,nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na>0 and nb>0 else 0.0

def seg(v, ax):
    s,e = AXES[ax]["range"]; return v[s:e]

def sc(v, ax, i=0):
    s,_ = AXES[ax]["range"]; return float(v[s+i])

def dominant(v, ax, names):
    sl = seg(v, ax)
    i  = int(np.argmax(sl))
    return names.get(i, f"Dim-{i}"), float(sl[i])

def mean_sim(vecs_a, vecs_b, ax=None):
    sims = []
    for va,_,_ in vecs_a:
        for vb,_,_ in vecs_b:
            a = seg(va,ax) if ax else va
            b = seg(vb,ax) if ax else vb
            sims.append(cos(a,b))
    return float(np.mean(sims)) if sims else 0.0

def pairwise(vecs, ax=None):
    """Return dict (code_a,code_b)->similarity for all unique pairs."""
    result = {}
    for (va,_,ca),(vb,_,cb) in combinations(vecs,2):
        a = seg(va,ax) if ax else va
        b = seg(vb,ax) if ax else vb
        result[(ca,cb)] = cos(a,b)
    return result

def sim_tag(s):
    if s>=.85: return "VERY HIGH"
    if s>=.65: return "HIGH"
    if s>=.40: return "MODERATE"
    if s>=.20: return "LOW"
    if s>=.00: return "MINIMAL"
    return "NEGATIVE"

def wrap(text, width=72, indent="    "):
    return "\n".join(
        textwrap.fill(line, width=width, initial_indent=indent,
                      subsequent_indent=indent)
        if line.strip() else ""
        for line in text.split("\n")
    )

# ══════════════════════════════════════════════════════════════════════════════
#  12-STEP ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class StepResult:
    def __init__(self, step_no, title, ax_range, severity, risk, findings,
                 implicated=None, dimension_detail=None):
        self.step_no   = step_no
        self.title     = title
        self.ax_range  = ax_range
        self.severity  = severity        # OK / LOW / MEDIUM / HIGH / CRITICAL
        self.risk      = risk            # 0.0 – 1.0
        self.findings  = findings        # list of strings
        self.implicated= implicated or []
        self.dim_detail= dimension_detail or {}  # axis -> {mean, interpretation, ...}


# ── Step 1: Clinical Domain Consistency (Dims 0–9) ────────────────────────────
def step1(dx_vecs, proc_vecs):
    findings = []
    dim_detail = {}

    dx_domains   = [(dominant(v,"clinical",DOMAIN_NAMES), c) for v,_,c in dx_vecs]
    proc_domains = [(dominant(v,"clinical",DOMAIN_NAMES), c) for v,_,c in proc_vecs]

    dx_dom_map   = {c: dom for (dom,_),c in dx_domains}
    proc_dom_map = {c: dom for (dom,_),c in proc_domains}

    # Domain distribution
    dx_dom_counts   = {}
    for (dom,_),c in dx_domains:
        dx_dom_counts[dom] = dx_dom_counts.get(dom,0)+1
    proc_dom_counts = {}
    for (dom,_),c in proc_domains:
        proc_dom_counts[dom] = proc_dom_counts.get(dom,0)+1

    findings.append(f"DX domain distribution:   {dict(sorted(dx_dom_counts.items(), key=lambda x:-x[1]))}")
    findings.append(f"Proc domain distribution: {dict(sorted(proc_dom_counts.items(), key=lambda x:-x[1]))}")

    # Cross-domain pairs (DX domain ≠ Proc domain)
    mismatches = []
    for (dx_dom,dx_str),dx_c in dx_domains:
        for (pr_dom,pr_str),pr_c in proc_domains:
            if dx_dom != pr_dom and dx_str>0.25 and pr_str>0.25:
                cl_sim = cos(seg(next(v for v,_,c in dx_vecs if c==dx_c),"clinical"),
                             seg(next(v for v,_,c in proc_vecs if c==pr_c),"clinical"))
                if cl_sim < 0.18:
                    mismatches.append((dx_c, dx_dom, pr_c, pr_dom, cl_sim))

    # Average cross-domain similarity
    avg_cl = mean_sim(dx_vecs, proc_vecs, "clinical")
    findings.append(f"Avg clinical-domain similarity (DX↔Proc): {avg_cl:.3f} [{sim_tag(avg_cl)}]")

    implicated = []
    risk = 0.08
    sev  = "OK"

    if mismatches:
        unique_mis = list({(d,p) for d,_,p,_,_ in mismatches})
        n = len(unique_mis)
        risk = min(1.0, 0.22 * n)
        sev  = "HIGH" if n>=4 else "MEDIUM" if n>=2 else "LOW"
        findings.append(f"\n  CROSS-DOMAIN MISMATCHES ({n} unique DX-Proc pairs):")
        for dx_c, dx_d, pr_c, pr_d, cl_sim in sorted(mismatches, key=lambda x:x[4])[:6]:
            findings.append(
                f"    • DX {dx_c} [{dx_d}] ↔ Proc {pr_c} [{pr_d}] | "
                f"clinical_sim={cl_sim:.3f}"
            )
            findings.append(
                f"      → Procedure domain ({pr_d}) is inconsistent with "
                f"diagnosis domain ({dx_d}). Verify medical necessity linkage."
            )
            implicated += [dx_c, pr_c]

    # Same-system clustering in DX
    max_cluster = max(dx_dom_counts.values()) if dx_dom_counts else 0
    dom_cluster = max(dx_dom_counts, key=dx_dom_counts.get) if dx_dom_counts else ""
    if max_cluster >= 3 and len(dx_dom_counts) <= 2:
        risk = max(risk, 0.55)
        sev  = "HIGH" if sev in ("OK","LOW","MEDIUM") else sev
        findings.append(
            f"\n  SAME-SYSTEM CLUSTERING: {max_cluster} DX codes in [{dom_cluster}] "
            f"with only {len(dx_dom_counts)} total domain(s) present."
        )
        findings.append(
            f"    → Dense clustering within one domain may indicate multiple "
            f"conditions from the same pathway being billed as separate MCCs/CCs."
        )

    if sev == "OK":
        findings.append("  ✓ Clinical domains are broadly consistent across DX and procedure codes")

    dim_detail["clinical"] = {
        "axis_mean_dx_proc": round(avg_cl, 4),
        "unique_dx_domains": len(dx_dom_counts),
        "unique_proc_domains": len(proc_dom_counts),
        "cross_domain_pairs": len(mismatches),
        "interpretation": (
            f"DX codes span {len(dx_dom_counts)} domain(s); "
            f"procedures span {len(proc_dom_counts)} domain(s). "
            f"Cross-domain pairs: {len(mismatches)}."
        )
    }

    return StepResult(1,"Clinical Domain Consistency","Dims 0–9",
                      sev, risk, findings, list(set(implicated)), dim_detail)


# ── Step 2: Severity Validation (Dims 10–19) ──────────────────────────────────
def step2(dx_vecs, mcc_count=None, cc_count=None):
    findings = []
    dim_detail = {}
    implicated = []

    avg_sev  = float(np.mean([sc(v,"severity",0) for v,_,_ in dx_vecs])) if dx_vecs else 0
    avg_cc   = float(np.mean([sc(v,"severity",1) for v,_,_ in dx_vecs])) if dx_vecs else 0
    sev_vals = [(sc(v,"severity",0), sc(v,"severity",1), c) for v,_,c in dx_vecs]

    findings.append(f"Avg severity score across DX codes: {avg_sev:.3f}")
    findings.append(f"Avg CC/MCC vector score:            {avg_cc:.3f}")

    # Intra-DX severity pairwise similarity
    sev_sims = {}
    for (va,_,ca),(vb,_,cb) in combinations(dx_vecs,2):
        sev_sims[(ca,cb)] = cos(seg(va,"severity"), seg(vb,"severity"))

    avg_sev_sim = float(np.mean(list(sev_sims.values()))) if sev_sims else 0
    findings.append(f"Avg intra-DX severity similarity:   {avg_sev_sim:.3f} [{sim_tag(avg_sev_sim)}]")

    # MCC stacking — high intra-similarity among high-severity codes
    high_sev_dx = [(v,d,c) for v,d,c in dx_vecs if sc(v,"severity",0) > 0.45]
    mcc_stack_risk = False
    if len(high_sev_dx) >= 2:
        mcc_sims = [cos(seg(va,"severity"),seg(vb,"severity"))
                    for (va,_,_),(vb,_,_) in combinations(high_sev_dx,2)]
        avg_mcc_sim = float(np.mean(mcc_sims)) if mcc_sims else 0
        findings.append(f"\n  High-severity DX codes ({len(high_sev_dx)}): "
                        f"{', '.join(c for _,_,c in high_sev_dx)}")
        findings.append(f"  Intra-MCC severity similarity: {avg_mcc_sim:.3f}")
        if avg_mcc_sim > 0.78:
            mcc_stack_risk = True
            findings.append(
                f"  ⚑ MCC STACKING SIGNAL: High-severity codes are very similar "
                f"to each other in severity space ({avg_mcc_sim:.3f})."
            )
            findings.append(
                f"    → These may represent the same disease process coded at "
                f"multiple levels (e.g., sepsis + septicemia + SIRS)."
            )
            implicated += [c for _,_,c in high_sev_dx]

    # Declared vs embedded MCC count discrepancy
    if mcc_count is not None:
        emb_high_sev = sum(1 for v,_,_ in dx_vecs if sc(v,"severity",0) > 0.45)
        findings.append(f"\n  Declared MCC count: {mcc_count} | "
                        f"Embedding-supported high-severity codes: {emb_high_sev}")
        if mcc_count > emb_high_sev + 1:
            findings.append(
                f"  ⚑ SEVERITY INFLATION: Declared MCC count ({mcc_count}) "
                f"exceeds embedding-supported count ({emb_high_sev}) by "
                f"{mcc_count - emb_high_sev}."
            )
            findings.append(
                f"    → Request documentation for each declared MCC to confirm "
                f"it affected care during this admission."
            )

    # Low avg severity but high declared complexity
    total_cx = (mcc_count or 0) + (cc_count or 0)
    risk = 0.08; sev = "OK"
    if avg_sev < 0.15 and total_cx >= 3:
        risk = 0.72; sev = "HIGH"
        findings.append(
            f"\n  ⚑ LOW EMBEDDING SEVERITY + HIGH DECLARED COMPLEXITY: "
            f"avg_sev={avg_sev:.3f} but {total_cx} MCCs/CCs declared."
        )
        findings.append(
            f"    → Severity vectors do not support MCC/CC tier in the "
            f"embedding space. High risk of severity inflation."
        )
    elif mcc_stack_risk:
        risk = max(risk, 0.60); sev = "HIGH"
    elif avg_sev < 0.20 and total_cx >= 2:
        risk = max(risk, 0.45); sev = "MEDIUM"

    if sev == "OK":
        findings.append("  ✓ Severity vectors are consistent with declared MCC/CC complexity")

    dim_detail["severity"] = {
        "avg_severity_score": round(avg_sev, 4),
        "avg_cc_mcc_score": round(avg_cc, 4),
        "intra_dx_severity_similarity": round(avg_sev_sim, 4),
        "high_severity_count": len(high_sev_dx),
        "mcc_stack_detected": mcc_stack_risk,
        "interpretation": (
            f"Avg severity={avg_sev:.3f}; {len(high_sev_dx)} high-sev DX codes; "
            f"intra-MCC sim={avg_sev_sim:.3f}."
        )
    }
    return StepResult(2,"Severity Validation","Dims 10–19",
                      sev, risk, findings, list(set(implicated)), dim_detail)


# ── Step 3: Service Intensity Alignment (Dims 20–29) ──────────────────────────
def step3(dx_vecs, proc_vecs, los=None):
    findings = []
    dim_detail = {}
    implicated = []

    avg_dx_sev  = float(np.mean([sc(v,"severity",0) for v,_,_ in dx_vecs])) if dx_vecs else 0
    avg_proc_cx = float(np.mean([sc(v,"intensity",1) for v,_,_ in proc_vecs])) if proc_vecs else 0
    avg_proc_int= float(np.mean([sc(v,"intensity",0) for v,_,_ in proc_vecs])) if proc_vecs else 0
    gap         = avg_proc_cx - avg_dx_sev

    findings.append(f"Avg DX severity (10–19):          {avg_dx_sev:.3f}")
    findings.append(f"Avg procedure E&M intensity (20): {avg_proc_int:.3f}")
    findings.append(f"Avg procedure complexity (21):    {avg_proc_cx:.3f}")
    findings.append(f"Intensity–Severity gap:            {gap:+.3f}")

    high_cx_procs = [(v,d,c) for v,d,c in proc_vecs if sc(v,"intensity",1) > 0.60]
    if high_cx_procs:
        findings.append(f"\n  High-complexity procedures (complexity>0.60):")
        for _,d,c in high_cx_procs:
            findings.append(f"    • {c}: {d[:55]}")
        implicated += [c for _,_,c in high_cx_procs]

    risk = 0.08; sev = "OK"

    if gap > 0.40:
        risk = 0.80; sev = "HIGH"
        findings.append(
            f"\n  ⚑ LARGE INTENSITY–SEVERITY GAP (+{gap:.3f}): "
            f"Procedures are far more complex than diagnoses are severe."
        )
        findings.append(
            f"    → Classic upcoding pattern: high-intensity procedures billed "
            f"for low-severity clinical conditions."
        )
    elif gap > 0.20:
        risk = 0.50; sev = "MEDIUM"
        findings.append(
            f"\n  ⚑ MODERATE GAP (+{gap:.3f}): Procedure complexity moderately "
            f"exceeds diagnosis severity — documentation review needed."
        )
    elif gap < -0.30:
        risk = max(risk, 0.35); sev = "LOW"
        findings.append(
            f"\n  NOTE: Diagnosis severity exceeds procedure intensity (gap={gap:.3f}). "
            f"Possible under-coding of procedures or overcoding of diagnoses."
        )

    # LOS vs intensity check
    if los is not None:
        if avg_proc_cx > 0.55 and los <= 1:
            risk = max(risk, 0.70); sev = "HIGH"
            findings.append(
                f"\n  ⚑ HIGH-INTENSITY PROCEDURES + {los}-DAY LOS: "
                f"Major procedures (complexity={avg_proc_cx:.3f}) with ≤1-day stay "
                f"is clinically atypical."
            )
            findings.append(
                f"    → Consider whether inpatient status is appropriate or "
                f"whether outpatient classification was applicable."
            )

    if sev == "OK":
        findings.append("  ✓ Procedure intensity is appropriately aligned with diagnosis severity")

    dim_detail["intensity"] = {
        "avg_dx_severity": round(avg_dx_sev, 4),
        "avg_proc_intensity": round(avg_proc_int, 4),
        "avg_proc_complexity": round(avg_proc_cx, 4),
        "gap": round(gap, 4),
        "interpretation": f"Intensity–severity gap={gap:+.3f}; "
                          f"{len(high_cx_procs)} high-complexity procedures detected."
    }
    return StepResult(3,"Service Intensity Alignment","Dims 20–29",
                      sev, risk, findings, list(set(implicated)), dim_detail)


# ── Step 4: Anatomical Coherence (Dims 30–39) ────────────────────────────────
def step4(dx_vecs, proc_vecs):
    findings = []
    dim_detail = {}
    implicated = []

    avg_anat = mean_sim(dx_vecs, proc_vecs, "anatomical")
    findings.append(f"Avg DX–Proc anatomical-site similarity: {avg_anat:.3f} [{sim_tag(avg_anat)}]")

    # Per-pair anatomical mismatches
    mismatches = []
    for vd,_,cd in dx_vecs:
        for vp,_,cp in proc_vecs:
            s = cos(seg(vd,"anatomical"), seg(vp,"anatomical"))
            dx_site,_  = dominant(vd,"anatomical",ANAT_NAMES)
            pr_site,pr_ = dominant(vp,"anatomical",ANAT_NAMES)
            if s < 0.10 and dx_site != pr_site:
                mismatches.append((cd, dx_site, cp, pr_site, s))

    risk = 0.08; sev = "OK"

    if mismatches:
        unique_pairs = list({(d,p) for d,_,p,_,_ in mismatches})
        n = len(unique_pairs)
        risk = min(0.90, 0.18*n); sev = "HIGH" if n>=4 else "MEDIUM" if n>=2 else "LOW"
        findings.append(f"\n  ANATOMICAL MISMATCHES ({n} DX–Proc pairs):")
        for cd, ds, cp, ps, s in sorted(mismatches, key=lambda x:x[4])[:5]:
            findings.append(
                f"    • DX {cd} [{ds}] ↔ Proc {cp} [{ps}] | "
                f"anatomical_sim={s:.3f}"
            )
            findings.append(
                f"      → Procedure operates on [{ps}] but principal DX "
                f"is [{ds}]. Confirm anatomical justification in operative notes."
            )
            implicated += [cd, cp]

    if avg_anat < 0.10 and not mismatches:
        sev = "MEDIUM"; risk = max(risk, 0.40)
        findings.append(
            f"\n  NOTE: Low overall anatomical similarity ({avg_anat:.3f}) even "
            f"without specific outlier pairs — diffuse anatomical inconsistency."
        )

    if sev == "OK":
        findings.append("  ✓ Procedure and diagnosis anatomical sites are broadly consistent")

    dim_detail["anatomical"] = {
        "avg_dx_proc_anat_sim": round(avg_anat, 4),
        "mismatch_pairs": len(mismatches),
        "interpretation": (
            f"Avg anatomical similarity={avg_anat:.3f}; "
            f"{len(mismatches)} site-mismatch pairs detected."
        )
    }
    return StepResult(4,"Anatomical Coherence","Dims 30–39",
                      sev, risk, findings, list(set(implicated)), dim_detail)


# ── Step 5: Episode Type Consistency (Dims 40–49) ────────────────────────────
def step5(dx_vecs, proc_vecs):
    findings = []
    dim_detail = {}
    implicated = []

    avg_ep = mean_sim(dx_vecs, proc_vecs, "episode")

    dx_acute  = float(np.mean([sc(v,"episode",0) for v,_,_ in dx_vecs])) if dx_vecs else 0
    dx_chron  = float(np.mean([sc(v,"episode",1) for v,_,_ in dx_vecs])) if dx_vecs else 0
    pr_acute  = float(np.mean([sc(v,"episode",0) for v,_,_ in proc_vecs])) if proc_vecs else 0
    pr_chron  = float(np.mean([sc(v,"episode",1) for v,_,_ in proc_vecs])) if proc_vecs else 0
    pr_prev   = float(np.mean([sc(v,"episode",3) for v,_,_ in proc_vecs])) if proc_vecs else 0

    findings.append(f"DX episode profile  — acute: {dx_acute:.3f} | chronic: {dx_chron:.3f}")
    findings.append(f"Proc episode profile — acute: {pr_acute:.3f} | chronic: {pr_chron:.3f} | preventive: {pr_prev:.3f}")
    findings.append(f"DX–Proc episode similarity: {avg_ep:.3f} [{sim_tag(avg_ep)}]")

    risk = 0.08; sev = "OK"

    # Preventive procedures in acute inpatient context
    prev_procs = [(v,d,c) for v,d,c in proc_vecs if sc(v,"episode",3) > 0.55]
    if prev_procs and dx_acute > 0.40:
        risk = max(risk, 0.55); sev = "MEDIUM"
        findings.append(
            f"\n  ⚑ PREVENTIVE PROCEDURES IN ACUTE CONTEXT: "
            f"{len(prev_procs)} procedure(s) have strong preventive episode signal "
            f"while diagnoses are acute (acute={dx_acute:.3f})."
        )
        findings.append(
            f"    → Preventive/screening procedures ({', '.join(c for _,_,c in prev_procs)}) "
            f"may not be appropriate for this inpatient admission."
        )
        implicated += [c for _,_,c in prev_procs]

    # Chronic-only diagnoses with acute high-complexity procedures
    if dx_chron > 0.70 and dx_acute < 0.20 and pr_acute > 0.60:
        risk = max(risk, 0.60); sev = "HIGH"
        findings.append(
            f"\n  ⚑ CHRONIC DX + ACUTE PROCEDURE MISMATCH: "
            f"Diagnoses are predominantly chronic (chronic={dx_chron:.3f}) "
            f"but procedures signal acute intervention (acute={pr_acute:.3f})."
        )
        findings.append(
            f"    → Confirm whether an acute exacerbation or complication triggered "
            f"this admission. Without it, inpatient status may be unwarranted."
        )

    if sev == "OK":
        findings.append("  ✓ Episode type is consistent between diagnoses and procedures")

    dim_detail["episode"] = {
        "dx_acute": round(dx_acute,4), "dx_chronic": round(dx_chron,4),
        "proc_acute": round(pr_acute,4), "proc_preventive": round(pr_prev,4),
        "avg_ep_sim": round(avg_ep,4),
        "interpretation": (
            f"DX acute={dx_acute:.3f}/chronic={dx_chron:.3f}; "
            f"Proc acute={pr_acute:.3f}/prev={pr_prev:.3f}; "
            f"episode_sim={avg_ep:.3f}."
        )
    }
    return StepResult(5,"Episode Type Consistency","Dims 40–49",
                      sev, risk, findings, list(set(implicated)), dim_detail)


# ── Step 6: Billing Channel Integrity (Dims 50–59) ───────────────────────────
def step6(proc_vecs):
    findings = []
    dim_detail = {}
    implicated = []

    avg_inp  = float(np.mean([sc(v,"billing",0) for v,_,_ in proc_vecs])) if proc_vecs else 0
    avg_dme  = float(np.mean([sc(v,"billing",1) for v,_,_ in proc_vecs])) if proc_vecs else 0
    avg_outp = float(np.mean([sc(v,"billing",2) for v,_,_ in proc_vecs])) if proc_vecs else 0

    findings.append(f"Avg inpatient billing signal:    {avg_inp:.3f}")
    findings.append(f"Avg outpatient billing signal:   {avg_outp:.3f}")
    findings.append(f"Avg DME billing signal:          {avg_dme:.3f}")

    risk = 0.08; sev = "OK"

    # Outpatient-only codes in inpatient claim
    outp_codes = [(v,d,c) for v,d,c in proc_vecs if sc(v,"billing",2) > 0.65 and sc(v,"billing",0) < 0.30]
    if outp_codes:
        risk = max(risk, 0.60); sev = "HIGH"
        findings.append(f"\n  ⚑ OUTPATIENT-PATTERN CODES IN INPATIENT CLAIM ({len(outp_codes)}):")
        for _,d,c in outp_codes:
            findings.append(f"    • {c}: outp_signal={sc(next(v for v,_,cc in proc_vecs if cc==c),'billing',2):.3f}")
            findings.append(f"      → This code has a strong outpatient/professional billing profile.")
            findings.append(f"        Verify whether inpatient setting is appropriate for this service.")
        implicated += [c for _,_,c in outp_codes]

    # DME codes billed on inpatient claim
    dme_codes = [(v,d,c) for v,d,c in proc_vecs if sc(v,"billing",1) > 0.55]
    if dme_codes:
        risk = max(risk, 0.65); sev = "HIGH" if sev in ("OK","LOW","MEDIUM") else sev
        findings.append(f"\n  ⚑ DME CODES IN INPATIENT CLAIM ({len(dme_codes)}):")
        for _,d,c in dme_codes:
            findings.append(f"    • {c}: dme_signal={sc(next(v for v,_,cc in proc_vecs if cc==c),'billing',1):.3f}")
            findings.append(f"      → DME is typically bundled under inpatient DRG and should not be separately billed.")
        implicated += [c for _,_,c in dme_codes]

    if sev == "OK":
        findings.append("  ✓ Billing channel profiles are consistent with inpatient context")

    dim_detail["billing"] = {
        "avg_inpatient": round(avg_inp,4),
        "avg_outpatient": round(avg_outp,4),
        "avg_dme": round(avg_dme,4),
        "outp_code_count": len(outp_codes),
        "dme_code_count": len(dme_codes),
        "interpretation": (
            f"Inpatient={avg_inp:.3f}, Outpatient={avg_outp:.3f}, DME={avg_dme:.3f}. "
            f"{len(outp_codes)} outpatient-profile and {len(dme_codes)} DME codes detected."
        )
    }
    return StepResult(6,"Billing Channel Integrity","Dims 50–59",
                      sev, risk, findings, list(set(implicated)), dim_detail)


# ── Step 7: Bundling and Redundancy (Dims 60–69) ─────────────────────────────
def step7(proc_vecs):
    findings = []
    dim_detail = {}
    implicated = []

    codes = [c for _,_,c in proc_vecs]
    avg_bund = float(np.mean([sc(v,"bundling",0) for v,_,_ in proc_vecs])) if proc_vecs else 0
    findings.append(f"Avg bundling cohesion signal across procedures: {avg_bund:.3f}")

    # NCCI explicit violations
    ncci_viols = []
    for comp, subs in NCCI.items():
        if comp in codes:
            for sub in subs:
                if sub in codes:
                    va = next((v for v,_,c in proc_vecs if c==comp), None)
                    vb = next((v for v,_,c in proc_vecs if c==sub), None)
                    s  = cos(va,vb) if va is not None and vb is not None else 0
                    ncci_viols.append((comp, sub, s))

    # Embedding-based high-similarity pairs
    pp_sims = pairwise(proc_vecs, "bundling")
    near_dups = [(a,b,s) for (a,b),s in pp_sims.items() if s >= 0.88]
    high_comp  = [(a,b,s) for (a,b),s in pp_sims.items() if 0.72<=s<0.88]

    risk = 0.05; sev = "OK"

    if ncci_viols:
        n = len(ncci_viols); risk = min(0.90, 0.28*n); sev = "HIGH"
        findings.append(f"\n  ⚑ NCCI BUNDLING VIOLATIONS ({n}):")
        for comp, sub, s in ncci_viols:
            findings.append(
                f"    • {sub} billed WITH {comp} | bundling_sim={s:.3f}"
            )
            findings.append(
                f"      → {sub} is a component service included in {comp}. "
                f"Separate billing violates NCCI and results in overpayment."
            )
            implicated += [comp, sub]

    if near_dups:
        risk = max(risk, 0.70); sev = "HIGH" if sev in ("OK","LOW","MEDIUM") else sev
        findings.append(f"\n  ⚑ NEAR-DUPLICATE PROCEDURE PAIRS ({len(near_dups)}):")
        for a,b,s in near_dups:
            findings.append(
                f"    • {a} ↔ {b}: bundling_axis_sim={s:.3f} — "
                f"nearly identical bundling profiles suggest same service."
            )
            implicated += [a,b]

    if high_comp:
        risk = max(risk, 0.40); sev = "MEDIUM" if sev=="OK" else sev
        findings.append(f"\n  HIGH BUNDLING SIMILARITY PAIRS ({len(high_comp)}) — review needed:")
        for a,b,s in high_comp[:4]:
            findings.append(f"    • {a} ↔ {b}: bundling_sim={s:.3f}")

    if sev == "OK":
        findings.append("  ✓ No bundling violations or duplicate procedures detected")

    dim_detail["bundling"] = {
        "avg_bundling_signal": round(avg_bund,4),
        "ncci_violations": len(ncci_viols),
        "near_duplicate_pairs": len(near_dups),
        "high_similarity_pairs": len(high_comp),
        "interpretation": (
            f"NCCI violations={len(ncci_viols)}; near-dups={len(near_dups)}; "
            f"high-sim={len(high_comp)}; avg_bundling={avg_bund:.3f}."
        )
    }
    return StepResult(7,"Bundling and Redundancy","Dims 60–69",
                      sev, risk, findings, list(set(implicated)), dim_detail)


# ── Step 8: DX–Procedure Link (Dims 70–79) ───────────────────────────────────
def step8(dx_vecs, proc_vecs):
    findings = []
    dim_detail = {}
    implicated = []

    # Per procedure: best-matching DX on dxproc axis
    avg_link = mean_sim(dx_vecs, proc_vecs, "dxproc")
    findings.append(f"Avg DX–Proc link similarity (dims 70–79): {avg_link:.3f} [{sim_tag(avg_link)}]")

    unsupported = []
    weak        = []
    for vp,_,cp in proc_vecs:
        link_sims = [cos(seg(vd,"dxproc"), seg(vp,"dxproc")) for vd,_,_ in dx_vecs]
        best = max(link_sims); avg = float(np.mean(link_sims))
        if best < 0.08 and avg < 0.10:
            unsupported.append((cp, best, avg))
        elif best < 0.20 or avg < 0.15:
            weak.append((cp, best, avg))

    # Per DX: does any procedure link back?
    orphan_dx = []
    for vd,_,cd in dx_vecs:
        link_sims = [cos(seg(vd,"dxproc"), seg(vp,"dxproc")) for vp,_,_ in proc_vecs]
        best = max(link_sims) if link_sims else 0
        if best < 0.08:
            orphan_dx.append((cd, best))

    risk = 0.08; sev = "OK"

    if unsupported:
        risk = min(0.90, 0.25*len(unsupported)); sev = "HIGH"
        findings.append(f"\n  ⚑ MEDICALLY UNSUPPORTED PROCEDURES ({len(unsupported)}):")
        for cp,best,avg in unsupported:
            findings.append(
                f"    • {cp}: best_dx_link={best:.3f}, avg_link={avg:.3f} — "
                f"NO diagnosis in embedding space supports this procedure."
            )
            findings.append(
                f"      → Request operative/clinical notes confirming medical "
                f"necessity. High phantom billing risk."
            )
            implicated.append(cp)

    if weak:
        risk = max(risk, 0.45); sev = "HIGH" if sev=="OK" else sev
        findings.append(f"\n  WEAKLY SUPPORTED PROCEDURES ({len(weak)}):")
        for cp,best,avg in weak:
            findings.append(
                f"    • {cp}: best_dx_link={best:.3f}, avg_link={avg:.3f} — "
                f"limited clinical justification from listed diagnoses."
            )
            implicated.append(cp)

    if orphan_dx:
        risk = max(risk, 0.40)
        findings.append(f"\n  ORPHAN DIAGNOSES ({len(orphan_dx)}) — no procedure links back:")
        for cd, best in orphan_dx:
            findings.append(
                f"    • {cd}: best_proc_link={best:.3f} — "
                f"diagnosis may not have been actively treated in this encounter."
            )
            findings.append(
                f"      → Diagnoses not treated may not qualify for CC/MCC credit "
                f"under UHDDS guidelines."
            )
            implicated.append(cd)

    if sev == "OK":
        findings.append("  ✓ Procedures are adequately supported by listed diagnoses")

    dim_detail["dxproc"] = {
        "avg_link_sim": round(avg_link,4),
        "unsupported_procs": len(unsupported),
        "weakly_supported_procs": len(weak),
        "orphan_dx_count": len(orphan_dx),
        "interpretation": (
            f"avg_link={avg_link:.3f}; unsupported={len(unsupported)}; "
            f"weak={len(weak)}; orphan_dx={len(orphan_dx)}."
        )
    }
    return StepResult(8,"DX–Procedure Link (Medical Necessity)","Dims 70–79",
                      sev, risk, findings, list(set(implicated)), dim_detail)


# ── Step 9: FWA Risk Pattern Detection (Dims 80–89) ──────────────────────────
def step9(dx_vecs, proc_vecs):
    findings = []
    dim_detail = {}
    implicated = []

    # Per-code FWA axis values
    # D080=upcoding, D081=unbundling, D082=over-ordering, D083=phantom billing
    def fwa_profile(vecs):
        return {
            "upcoding":    float(np.mean([sc(v,"fwa",0) for v,_,_ in vecs])) if vecs else 0,
            "unbundling":  float(np.mean([sc(v,"fwa",1) for v,_,_ in vecs])) if vecs else 0,
            "overorder":   float(np.mean([sc(v,"fwa",2) for v,_,_ in vecs])) if vecs else 0,
            "phantom":     float(np.mean([sc(v,"fwa",3) for v,_,_ in vecs])) if vecs else 0,
        }

    dx_fwa   = fwa_profile(dx_vecs)
    proc_fwa = fwa_profile(proc_vecs)
    all_fwa  = fwa_profile(dx_vecs + proc_vecs)

    findings.append(f"Portfolio FWA signal averages (dims 80–89):")
    findings.append(f"  Upcoding risk (80):   DX={dx_fwa['upcoding']:.3f}  Proc={proc_fwa['upcoding']:.3f}")
    findings.append(f"  Unbundling risk (81): DX={dx_fwa['unbundling']:.3f}  Proc={proc_fwa['unbundling']:.3f}")
    findings.append(f"  Over-ordering (82):   DX={dx_fwa['overorder']:.3f}  Proc={proc_fwa['overorder']:.3f}")
    findings.append(f"  Phantom billing (83): DX={dx_fwa['phantom']:.3f}  Proc={proc_fwa['phantom']:.3f}")

    risk = 0.08; sev = "OK"
    patterns = []

    # Upcoding signal
    if proc_fwa["upcoding"] > 0.45:
        risk = max(risk, 0.65); sev = "HIGH"
        high_up = [(v,d,c) for v,d,c in proc_vecs if sc(v,"fwa",0) > 0.50]
        patterns.append("Upcoding")
        findings.append(f"\n  ⚑ UPCODING SIGNAL: {len(high_up)} procedures have elevated upcoding risk.")
        for _,d,c in high_up:
            findings.append(f"    • {c}: upcoding_signal={sc(next(v for v,_,cc in proc_vecs if cc==c),'fwa',0):.3f}")
            implicated.append(c)
        findings.append(
            f"    → Verify service intensity documentation supports the billed "
            f"procedure/E&M level. Compare with peer providers."
        )

    # Unbundling signal
    if proc_fwa["unbundling"] > 0.40:
        risk = max(risk, 0.60); sev = "HIGH" if sev in ("OK","LOW","MEDIUM") else sev
        patterns.append("Unbundling")
        findings.append(
            f"\n  ⚑ UNBUNDLING SIGNAL: avg proc unbundling={proc_fwa['unbundling']:.3f}. "
            f"Multiple procedures likely represent components of a single service."
        )

    # Phantom billing
    if proc_fwa["phantom"] > 0.45:
        risk = max(risk, 0.70); sev = "HIGH"
        patterns.append("Phantom billing")
        phantom_procs = [(v,d,c) for v,d,c in proc_vecs if sc(v,"fwa",3) > 0.50]
        findings.append(f"\n  ⚑ PHANTOM BILLING SIGNAL: {len(phantom_procs)} procedures have elevated phantom risk.")
        for _,d,c in phantom_procs:
            findings.append(f"    • {c}: phantom_signal={sc(next(v for v,_,cc in proc_vecs if cc==c),'fwa',3):.3f}")
            implicated.append(c)
        findings.append(
            f"    → Request clinical documentation confirming these services were rendered."
        )

    # MCC stacking via FWA on DX side
    if dx_fwa["upcoding"] > 0.38:
        risk = max(risk, 0.55)
        patterns.append("MCC stacking (DX-side)")
        findings.append(
            f"\n  ⚑ MCC STACKING RISK (DX): DX codes have elevated upcoding "
            f"signal ({dx_fwa['upcoding']:.3f}) — severity may be inflated."
        )
        high_up_dx = [(v,d,c) for v,d,c in dx_vecs if sc(v,"fwa",0) > 0.40]
        for _,d,c in high_up_dx:
            implicated.append(c)

    if sev == "OK":
        findings.append("  ✓ No significant FWA risk signals detected across embedding axes 80–89")
    else:
        findings.append(f"\n  Active FWA patterns: {', '.join(patterns)}")

    dim_detail["fwa"] = {
        "avg_upcoding": round(all_fwa["upcoding"],4),
        "avg_unbundling": round(all_fwa["unbundling"],4),
        "avg_over_ordering": round(all_fwa["overorder"],4),
        "avg_phantom": round(all_fwa["phantom"],4),
        "patterns_detected": patterns,
        "interpretation": (
            f"upcoding={all_fwa['upcoding']:.3f}, unbundling={all_fwa['unbundling']:.3f}, "
            f"phantom={all_fwa['phantom']:.3f}. Patterns: {', '.join(patterns) or 'none'}."
        )
    }
    return StepResult(9,"FWA Risk Pattern Detection","Dims 80–89",
                      sev, risk, findings, list(set(implicated)), dim_detail)


# ── Step 10: Financial Alignment (Dims 90–99) ─────────────────────────────────
def step10(dx_vecs, proc_vecs):
    findings = []
    dim_detail = {}
    implicated = []

    avg_dx_fin   = float(np.mean([sc(v,"financial",0) for v,_,_ in dx_vecs]))  if dx_vecs  else 0
    avg_proc_fin = float(np.mean([sc(v,"financial",0) for v,_,_ in proc_vecs])) if proc_vecs else 0
    max_proc_fin = float(max(sc(v,"financial",0) for v,_,_ in proc_vecs))       if proc_vecs else 0
    fin_gap      = avg_proc_fin - avg_dx_fin
    fin_sim      = mean_sim(dx_vecs, proc_vecs, "financial")

    findings.append(f"Avg DX DRG/RVU weight (90–99):   {avg_dx_fin:.3f}")
    findings.append(f"Avg Proc DRG/RVU weight (90–99):  {avg_proc_fin:.3f}")
    findings.append(f"Max Proc DRG/RVU weight:           {max_proc_fin:.3f}")
    findings.append(f"Financial gap (proc–dx):           {fin_gap:+.3f}")
    findings.append(f"DX–Proc financial similarity:      {fin_sim:.3f} [{sim_tag(fin_sim)}]")

    high_fin = [(v,d,c) for v,d,c in proc_vecs if sc(v,"financial",0) > 0.72]
    if high_fin:
        findings.append(f"\n  High-financial-weight procedures (>0.72):")
        for _,d,c in high_fin:
            findings.append(f"    • {c}: fin_weight={sc(next(v for v,_,cc in proc_vecs if cc==c),'financial',0):.3f}")
        implicated += [c for _,_,c in high_fin]

    risk = 0.08; sev = "OK"

    if fin_gap > 0.40:
        risk = 0.82; sev = "HIGH"
        findings.append(
            f"\n  ⚑ LARGE FINANCIAL GAP (+{fin_gap:.3f}): Procedures carry much higher "
            f"DRG/RVU weight than diagnoses justify."
        )
        findings.append(
            f"    → Strong indicator of DRG manipulation: high-cost procedures "
            f"billed against low-acuity diagnoses."
        )
    elif fin_gap > 0.22:
        risk = 0.50; sev = "MEDIUM"
        findings.append(
            f"\n  ⚑ MODERATE FINANCIAL GAP (+{fin_gap:.3f}): Procedure weight moderately "
            f"exceeds diagnosis financial tier."
        )
    elif fin_sim < 0.15 and max_proc_fin > 0.65:
        risk = max(risk, 0.45); sev = "MEDIUM"
        findings.append(
            f"\n  ⚑ WEAK FINANCIAL ALIGNMENT (sim={fin_sim:.3f}) with high-cost "
            f"procedures: financial profiles do not align in embedding space."
        )

    if sev == "OK":
        findings.append("  ✓ Procedure financial weight is consistent with diagnosis acuity")

    dim_detail["financial"] = {
        "avg_dx_fin": round(avg_dx_fin,4),
        "avg_proc_fin": round(avg_proc_fin,4),
        "max_proc_fin": round(max_proc_fin,4),
        "fin_gap": round(fin_gap,4),
        "fin_sim": round(fin_sim,4),
        "interpretation": (
            f"fin_gap={fin_gap:+.3f}; max_proc={max_proc_fin:.3f}; "
            f"fin_sim={fin_sim:.3f}."
        )
    }
    return StepResult(10,"Financial Alignment","Dims 90–99",
                      sev, risk, findings, list(set(implicated)), dim_detail)


# ── Step 11: Cross-Dimensional Synthesis ──────────────────────────────────────
def step11(steps):
    findings = []
    contradictions = []

    # Map step results by step number
    s = {r.step_no: r for r in steps}
    # Helper to get risk of a step
    def risk(n): return s[n].risk if n in s else 0.0

    # Contradiction 1: High severity (10–19) + Low intensity (20–29)
    if risk(2) > 0.40 and risk(3) < 0.20:
        contradictions.append({
            "pattern": "High Severity + Low Service Intensity",
            "dims": "Dims 10–19 vs 20–29",
            "detail": (
                f"Severity risk={risk(2):.2f} (high) but intensity risk={risk(3):.2f} (low). "
                f"Clinical conditions declared as high-severity (MCC level) are paired "
                f"with low-complexity procedures — this is atypical for the documented acuity."
            ),
            "implication": "Possible severity inflation (MCCs declared without matching clinical intervention)."
        })

    # Contradiction 2: Strong DRG weight (90–99) + Weak DX-Proc link (70–79)
    if risk(10) > 0.40 and risk(8) > 0.45:
        contradictions.append({
            "pattern": "High Financial Weight + Weak Medical Necessity",
            "dims": "Dims 90–99 vs 70–79",
            "detail": (
                f"Financial risk={risk(10):.2f} (high-cost procedures) but "
                f"DX-Proc link risk={risk(8):.2f} (weak medical necessity). "
                f"Expensive procedures lack clinical support from listed diagnoses."
            ),
            "implication": "Classic overpayment scenario: high-cost services without documented medical necessity."
        })

    # Contradiction 3: High MCC count + Low domain diversity (0–9)
    if risk(2) > 0.50 and risk(1) < 0.25:
        contradictions.append({
            "pattern": "MCC Stacking + Low Clinical Diversity",
            "dims": "Dims 10–19 vs 0–9",
            "detail": (
                f"Severity risk={risk(2):.2f} (possible MCC inflation) but "
                f"clinical domain risk={risk(1):.2f} (low cross-domain mismatch = narrow focus). "
                f"Multiple MCCs from the same clinical domain suggest same-pathway coding."
            ),
            "implication": "ICD-10 combination codes may apply — separate MCCs could be collapsed."
        })

    # Contradiction 4: FWA risk (80–89) + OK bundling (60–69)
    if risk(9) > 0.50 and risk(7) < 0.25:
        contradictions.append({
            "pattern": "FWA Risk Signal Without Explicit Bundling Violation",
            "dims": "Dims 80–89 vs 60–69",
            "detail": (
                f"FWA risk={risk(9):.2f} but bundling risk={risk(7):.2f}. "
                f"FWA signals are elevated but no explicit NCCI violation found — "
                f"risk may be from subtle upcoding or coding pattern anomaly."
            ),
            "implication": "Investigate E&M level selection and documentation quality."
        })

    # Contradiction 5: Anatomical mismatch + Strong clinical coherence
    if risk(4) > 0.40 and risk(1) < 0.20:
        contradictions.append({
            "pattern": "Anatomical Mismatch + Broad Clinical Coherence",
            "dims": "Dims 30–39 vs 0–9",
            "detail": (
                f"Anatomical risk={risk(4):.2f} but clinical domain risk={risk(1):.2f}. "
                f"Procedures and diagnoses are in the same clinical domain but different "
                f"anatomical sites — possible specificity error in procedure coding."
            ),
            "implication": "Review laterality, specific anatomical site, and procedure code selection."
        })

    if contradictions:
        findings.append(f"CROSS-DIMENSIONAL CONTRADICTIONS DETECTED: {len(contradictions)}\n")
        for i, c in enumerate(contradictions, 1):
            findings.append(f"  [{i}] {c['pattern']} ({c['dims']})")
            findings.append(f"      Analysis: {c['detail']}")
            findings.append(f"      Implication: {c['implication']}")
            findings.append("")
    else:
        findings.append("  ✓ No major cross-dimensional contradictions detected")

    cross_risk = min(1.0, len(contradictions) * 0.22)
    sev = ("HIGH"   if cross_risk >= 0.55 else
           "MEDIUM" if cross_risk >= 0.30 else
           "LOW"    if cross_risk > 0     else "OK")

    return StepResult(11,"Cross-Dimensional Synthesis","All dims",
                      sev, cross_risk, findings, [], {"contradictions": contradictions})


# ── Step 12: Structured Audit Output ─────────────────────────────────────────
def step12(claim_id, dx_codes, proc_codes, all_steps, los=None, drg=None,
           mcc_count=None, cc_count=None):

    weights = [0.08, 0.15, 0.12, 0.08, 0.06, 0.07, 0.14, 0.12, 0.10, 0.08]
    # steps 1–10 indexed 0–9
    step_risks = [s.risk for s in all_steps[:10]]
    composite  = round(sum(r*w for r,w in zip(step_risks, weights)) * 100, 1)
    composite  = min(100.0, composite)

    if composite >= 68:   risk_level = "CRITICAL"; icon = "🚨"
    elif composite >= 48: risk_level = "HIGH";     icon = "🔴"
    elif composite >= 28: risk_level = "MEDIUM";   icon = "🟡"
    else:                 risk_level = "LOW";      icon = "🟢"

    priority = "HIGH" if composite >= 48 else "MEDIUM" if composite >= 28 else "LOW"

    # Top 3 drivers by weighted contribution
    driver_scores = [(s.risk*w, s) for s,w in zip(all_steps[:10], weights)]
    driver_scores.sort(key=lambda x:-x[0])
    primary_drivers = []
    for wt, step in driver_scores[:3]:
        if step.risk > 0.15:
            primary_drivers.append(
                f"{step.title} ({step.ax_range}) — severity={step.severity}, "
                f"risk={step.risk:.2f}, weighted_contribution={wt*100:.1f}pts"
            )

    # Overpayment type
    overpayment_types = _classify(all_steps)

    # Target codes
    all_impl = []
    for s in all_steps:
        all_impl.extend(s.implicated)
    target_codes = list(dict.fromkeys(all_impl))

    # Recommendation
    rec = _recommendation(priority, overpayment_types, target_codes)

    return {
        "claim_id":         claim_id,
        "dx_codes":         dx_codes,
        "proc_codes":       proc_codes,
        "los":              los,
        "drg":              drg,
        "mcc_count":        mcc_count,
        "cc_count":         cc_count,
        "overall_score":    composite,
        "risk_level":       risk_level,
        "risk_icon":        icon,
        "audit_priority":   priority,
        "primary_drivers":  primary_drivers,
        "overpayment_types":overpayment_types,
        "target_codes":     target_codes,
        "recommendation":   rec,
        "steps":            all_steps,
        "step_risks":       step_risks,
        "step_weights":     weights,
    }


def _classify(steps):
    s = {r.step_no: r for r in steps}
    types = []
    if s.get(7) and s[7].risk > 0.45:
        types.append("NCCI Unbundling / Procedure Component Separation")
    if s.get(2) and s[2].risk > 0.45:
        types.append("MCC/CC Severity Inflation")
    if s.get(8) and s[8].risk > 0.45:
        types.append("Medical Necessity Gap (Procedures Without Adequate DX Support)")
    if s.get(10) and s[10].risk > 0.40:
        types.append("DRG Financial Weight Manipulation")
    if s.get(9) and s[9].risk > 0.45:
        fwa_dim = s[9].dim_detail.get("fwa", {})
        patterns = fwa_dim.get("patterns_detected", [])
        if "Phantom billing" in patterns:
            types.append("Phantom Billing — Services Without Clinical Basis")
        if "Upcoding" in patterns:
            types.append("Upcoding — Inflated Service Intensity/Severity")
    if s.get(1) and s[1].risk > 0.45:
        types.append("Diagnosis Padding / Cross-Domain Inconsistency")
    return types or ["No Dominant Overpayment Pattern (Low Risk)"]


def _recommendation(priority, types, targets):
    if priority == "HIGH":
        return (
            "ROUTE TO CLINICAL AUDIT QUEUE IMMEDIATELY. "
            "Request: H&P, operative reports, discharge summary, nursing notes, "
            "and attending physician attestation for each flagged code. "
            f"Priority review codes: {', '.join(targets[:8]) if targets else 'see target list'}."
        )
    elif priority == "MEDIUM":
        return (
            "DOCUMENTATION REVIEW RECOMMENDED. "
            "Focus on CC/MCC sequencing, procedure bundling compliance, and "
            "medical necessity attestation for flagged procedures."
        )
    return "ROUTINE MONITORING. No immediate action required."


# ── Report printer ────────────────────────────────────────────────────────────
SEV_ICON = {"CRITICAL":"🚨","HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢","OK":"✅"}

def print_report(report, verbose=True):
    W = 78
    SEP  = "─"*W; SEP2 = "═"*W

    print(f"\n{SEP2}")
    print(f"  12-STEP DIMENSIONAL PAYMENT INTEGRITY AUDIT — {report['claim_id']}")
    print(SEP2)
    dx_str   = ", ".join(report["dx_codes"])
    proc_str = ", ".join(report["proc_codes"])
    print(f"  DX   [{len(report['dx_codes'])}]: {dx_str}")
    print(f"  PROC [{len(report['proc_codes'])}]: {proc_str}")
    meta = []
    if report["los"]       is not None: meta.append(f"LOS={report['los']}d")
    if report["drg"]       is not None: meta.append(f"DRG={report['drg']}")
    if report["mcc_count"] is not None: meta.append(f"MCC={report['mcc_count']}")
    if report["cc_count"]  is not None: meta.append(f"CC={report['cc_count']}")
    if meta: print(f"  {' | '.join(meta)}")
    print(SEP)

    score = report["overall_score"]
    bw    = int(score/100*48)
    bar   = "█"*bw + "░"*(48-bw)
    print(f"\n  ┌── OVERALL RISK SCORE ────────────────────────────────────────────┐")
    print(f"  │  Score  : {score:6.1f} / 100                                          │")
    print(f"  │  Level  : {report['risk_icon']} {report['risk_level']:<12}                              │")
    print(f"  │  Priority: {report['audit_priority']:<10}                                        │")
    print(f"  │  [{bar}]  │")
    print(f"  └───────────────────────────────────────────────────────────────────┘")

    print(f"\n  1. PRIMARY RISK DRIVERS (TOP 3)")
    for i,d in enumerate(report["primary_drivers"],1):
        for j,ln in enumerate(textwrap.wrap(d, 70)):
            print(f"     {'['+str(i)+']' if j==0 else '   '} {ln}")
    if not report["primary_drivers"]:
        print("     No significant risk drivers detected")

    print(f"\n  2. MOST LIKELY OVERPAYMENT TYPE(S)")
    for t in report["overpayment_types"]:
        print(f"     • {t}")

    print(f"\n  3. TARGET CODES FOR AUDIT")
    if report["target_codes"]:
        print(f"     {', '.join(report['target_codes'][:14])}")
        print(f"     ({len(report['target_codes'])} codes flagged across all steps)")
    else:
        print("     No specific codes flagged")

    print(f"\n  4. AUDIT RECOMMENDATION  [{report['audit_priority']} PRIORITY]")
    for ln in textwrap.wrap(report["recommendation"], 72):
        print(f"     {ln}")

    if verbose:
        print(f"\n  5. STEP-BY-STEP DIMENSIONAL ANALYSIS")
        print(SEP)
        wt_list = report["step_weights"]
        for step in report["steps"]:
            if step.step_no > 11: continue
            icon = SEV_ICON.get(step.severity,"⚪")
            wt   = wt_list[step.step_no-1] if step.step_no<=len(wt_list) else 0
            cont = step.risk*wt*100
            print(f"\n  {icon} STEP {step.step_no}: {step.title}  [{step.ax_range}]")
            print(f"     Severity: {step.severity}  |  Raw risk: {step.risk:.3f}  |  "
                  f"Weight: {wt:.2f}  |  Contribution: {cont:.1f}pts")
            if step.implicated:
                print(f"     Implicated codes: {', '.join(step.implicated)}")
            print()
            for line in step.findings:
                print(f"     {line}")

            # Dim detail box
            if step.dim_detail:
                for ax_name, detail in step.dim_detail.items():
                    if not isinstance(detail, dict): continue
                    interp = detail.get("interpretation","")
                    if interp:
                        print(f"\n     [Dimension summary — {ax_name}]")
                        for ln in textwrap.wrap(interp, 68):
                            print(f"       {ln}")

        # Step 11 separately
        s11 = next((s for s in report["steps"] if s.step_no==11), None)
        if s11:
            icon = SEV_ICON.get(s11.severity,"⚪")
            print(f"\n  {icon} STEP 11: {s11.title}")
            print(f"     Severity: {s11.severity}  |  Cross-risk: {s11.risk:.3f}")
            for line in s11.findings:
                print(f"     {line}")

        print(f"\n  6. DIMENSIONAL RISK SCORECARD")
        print(f"  {'Step':<44} {'Risk':>6}  {'Weight':>7}  {'Contrib':>8}")
        print(f"  {'─'*44} {'─'*6}  {'─'*7}  {'─'*8}")
        total = 0
        labels = [
            "1. Clinical Domain (0–9)",
            "2. Severity / CC-MCC (10–19)",
            "3. Service Intensity (20–29)",
            "4. Anatomical Coherence (30–39)",
            "5. Episode Type (40–49)",
            "6. Billing Channel (50–59)",
            "7. Bundling & Redundancy (60–69)",
            "8. DX–Proc Link (70–79)",
            "9. FWA Risk Signals (80–89)",
            "10. Financial Alignment (90–99)",
        ]
        for label, risk, wt in zip(labels, report["step_risks"], wt_list):
            cont = risk*wt*100; total += cont
            ico  = SEV_ICON.get(
                "CRITICAL" if risk>0.65 else "HIGH" if risk>0.40 else
                "MEDIUM" if risk>0.20 else "OK","✅"
            )
            print(f"  {ico} {label:<42} {risk:>6.3f}  {wt:>7.2f}  {cont:>7.1f}pts")
        print(f"  {'─'*44} {'─'*6}  {'─'*7}  {'─'*8}")
        print(f"  {'COMPOSITE':>45}            {total:>7.1f}pts")

    print(f"\n{SEP2}\n")


# ── Full claim audit ───────────────────────────────────────────────────────────
def audit(claim_id, dx_codes, proc_codes, los, drg,
          mcc_count, cc_count, store):

    dx_vecs   = [(v,d,c) for c in dx_codes
                 for v,d,t,m in [store.get(c)]]
    proc_vecs = [(v,d,c) for c in proc_codes
                 for v,d,t,m in [store.get(c)]]

    steps = [
        step1(dx_vecs, proc_vecs),
        step2(dx_vecs, mcc_count, cc_count),
        step3(dx_vecs, proc_vecs, los),
        step4(dx_vecs, proc_vecs),
        step5(dx_vecs, proc_vecs),
        step6(proc_vecs),
        step7(proc_vecs),
        step8(dx_vecs, proc_vecs),
        step9(dx_vecs, proc_vecs),
        step10(dx_vecs, proc_vecs),
        step11(steps if False else []),  # placeholder, filled below
    ]
    # Step 11 needs steps 1–10 done
    steps[10] = step11(steps[:10])

    return step12(claim_id, dx_codes, proc_codes, steps,
                  los, drg, mcc_count, cc_count)


# ── Batch utilities ───────────────────────────────────────────────────────────
def load_batch(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as f: data = json.load(f)
        return [{
            "claim_id":  d.get("claim_id","?"),
            "dx_codes":  d.get("dx_codes", d.get("dx",[])),
            "proc_codes":d.get("proc_codes",d.get("proc",[])),
            "los":       d.get("los"), "drg": d.get("drg"),
            "mcc_count": d.get("mcc_count",d.get("mcc")),
            "cc_count":  d.get("cc_count", d.get("cc")),
        } for d in data]
    claims = []
    with open(path,"r",newline="") as f:
        for row in csv.DictReader(f):
            dx  = [c.strip() for c in row.get("dx_codes","").replace("|",",").split(",") if c.strip()]
            prc = [c.strip() for c in row.get("proc_codes","").replace("|",",").split(",") if c.strip()]
            claims.append({
                "claim_id":  row.get("claim_id","?"),
                "dx_codes":  dx, "proc_codes": prc,
                "los":       int(row["los"]) if row.get("los") else None,
                "drg":       row.get("drg"),
                "mcc_count": int(row["mcc_count"]) if row.get("mcc_count") else None,
                "cc_count":  int(row["cc_count"])  if row.get("cc_count")  else None,
            })
    return claims


def to_csv_row(r):
    return {
        "claim_id":        r["claim_id"],
        "dx_codes":        "|".join(r["dx_codes"]),
        "proc_codes":      "|".join(r["proc_codes"]),
        "overall_score":   r["overall_score"],
        "risk_level":      r["risk_level"],
        "audit_priority":  r["audit_priority"],
        "overpayment_types":"|".join(r["overpayment_types"]),
        "driver_1": r["primary_drivers"][0] if len(r["primary_drivers"])>0 else "",
        "driver_2": r["primary_drivers"][1] if len(r["primary_drivers"])>1 else "",
        "driver_3": r["primary_drivers"][2] if len(r["primary_drivers"])>2 else "",
        "target_codes":    "|".join(r["target_codes"]),
        "recommendation":  r["recommendation"],
        **{f"step{s.step_no}_risk": s.risk for s in r["steps"] if s.step_no<=10},
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="12-Step Dimensional Payment Integrity Auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dim_auditor.py \\
    --claim-id CLM001 \\
    --dx I21.0 I10 E11.9 J18.9 N18.3 R65.21 I50.9 E66.01 F32.9 M54.5 \\
    --proc 02703ZZ 5A1945Z 0SR9019 45378 45380 93000 93010 27447 \\
    --los 2 --drg 207 --mcc 3 --cc 2

  python dim_auditor.py --batch claims.csv --output results.csv
  python dim_auditor.py --batch claims.json --output results.json --quiet
        """
    )
    p.add_argument("--claim-id",   default="DEMO")
    p.add_argument("--dx",         nargs="*", default=[])
    p.add_argument("--proc",       nargs="*", default=[])
    p.add_argument("--los",        type=int,  default=None)
    p.add_argument("--drg",        default=None)
    p.add_argument("--mcc",        type=int,  default=None, dest="mcc_count")
    p.add_argument("--cc",         type=int,  default=None, dest="cc_count")
    p.add_argument("--batch",      metavar="FILE")
    p.add_argument("--output",     metavar="FILE")
    p.add_argument("--embeddings", default=DEFAULT_EMB)
    p.add_argument("--quiet",      action="store_true")
    args = p.parse_args()

    store = Store()
    store.load(args.embeddings, silent=args.quiet)

    if args.batch:
        claims = load_batch(args.batch)
        print(f"[INFO] Auditing {len(claims)} claims ...\n")
        results = []
        for c in claims:
            r = audit(**c, store=store)
            results.append(r)
            print_report(r, verbose=not args.quiet)

        print("\n"+"═"*78)
        print("  BATCH SUMMARY")
        print(f"  {'Claim ID':<22} {'Score':>6}  {'Level':<10} {'Priority':<8}  Overpayment Type")
        print("  "+"─"*74)
        for r in results:
            ot = r["overpayment_types"][0][:36] if r["overpayment_types"] else "—"
            print(f"  {r['claim_id']:<22} {r['overall_score']:>6.1f}  "
                  f"{r['risk_level']:<10} {r['risk_icon']} {r['audit_priority']:<6}  {ot}")
        cnts = {l:sum(1 for r in results if r["risk_level"]==l)
                for l in ["LOW","MEDIUM","HIGH","CRITICAL"]}
        print(f"\n  Totals: 🟢LOW={cnts['LOW']} 🟡MEDIUM={cnts['MEDIUM']} "
              f"🔴HIGH={cnts['HIGH']} 🚨CRITICAL={cnts['CRITICAL']}")
        print("═"*78+"\n")

        if args.output:
            ext = os.path.splitext(args.output)[1].lower()
            if ext==".json":
                with open(args.output,"w") as f:
                    json.dump([{k:v for k,v in r.items() if k!="steps"}
                               for r in results],f,indent=2,default=str)
            else:
                rows = [to_csv_row(r) for r in results]
                with open(args.output,"w",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=rows[0].keys())
                    w.writeheader(); w.writerows(rows)
            print(f"[INFO] Saved → {args.output}")
        return

    # Single claim
    dx   = args.dx   or ["I21.0","I10","E11.9","J18.9","N18.3","R65.21","I50.9","E66.01","F32.9","M54.5"]
    proc = args.proc or ["02703ZZ","5A1945Z","0SR9019","45378","45380","93000","93010","27447"]
    r = audit(args.claim_id, dx, proc, args.los, args.drg,
              args.mcc_count, args.cc_count, store)
    print_report(r, verbose=not args.quiet)

    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        if ext==".json":
            with open(args.output,"w") as f:
                json.dump({k:v for k,v in r.items() if k!="steps"},f,indent=2,default=str)
        else:
            row = to_csv_row(r)
            with open(args.output,"w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=row.keys()); w.writeheader(); w.writerow(row)
        print(f"[INFO] Saved → {args.output}")

if __name__ == "__main__":
    main()
