#!/usr/bin/env python3
"""
Medical Code Similarity API — Backend for Dashboard
====================================================
Run with:  python similarity_api.py
Serves on: http://localhost:8765

Endpoints:
  GET  /search?q=TERM&limit=10   — search codes by code or description
  POST /compare                  — compare two codes with axis weights
  GET  /axes                     — return axis metadata
"""

import json, math, csv, os, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__),
                               "Medical_Embeddings_256_2026.csv")
PORT = 8765
DIM  = 256

# ── Axis definitions ──────────────────────────────────────────────────────────
AXES = [
    {"id":"clinical",  "name":"Clinical Domain",   "start":0,   "end":26,  "color":"#c84b2f",
     "desc":"Classifies the primary clinical specialty: Cardio, Neuro, Ortho, GI, Pulm, Metabolic, Renal, MH, Infect, Preventive",
     "fwa_relevance":"Cross-domain procedure/DX mismatch detection"},
    {"id":"severity",  "name":"Severity / CC-MCC", "start":26,  "end":52,  "color":"#b03020",
     "desc":"MCC activates 14 dims, CC activates 7, normal 2 — separation survives L2 norm",
     "fwa_relevance":"MCC stacking detection; severity inflation"},
    {"id":"intensity", "name":"Service Intensity",  "start":52,  "end":78,  "color":"#1a4b8c",
     "desc":"Procedure complexity, E&M level, AMA Work RVU signal",
     "fwa_relevance":"E&M upcoding; high-intensity proc on low-acuity DX"},
    {"id":"anatomical","name":"Anatomical Site",    "start":78,  "end":104, "color":"#1d6a52",
     "desc":"Head, Chest, Spine, Abdomen, Extremity, Skin, Pelvis, Systemic — 3 sub-dims per site",
     "fwa_relevance":"Wrong-site procedure detection; laterality errors"},
    {"id":"episode",   "name":"Episode Type",       "start":104, "end":130, "color":"#6a4a9a",
     "desc":"Acute, Chronic, Post-op, Preventive — episode timing and care context",
     "fwa_relevance":"Preventive codes in acute inpatient claims"},
    {"id":"billing",   "name":"Billing Channel",    "start":130, "end":156, "color":"#2a6080",
     "desc":"Inpatient, DME, Outpatient/Professional, ASC, Telehealth",
     "fwa_relevance":"Outpatient-only codes billed inpatient; site-of-service fraud"},
    {"id":"bundling",  "name":"Bundling Cohesion",  "start":156, "end":182, "color":"#805020",
     "desc":"Component vs comprehensive signal per NCCI edit table; add-on code flags",
     "fwa_relevance":"Unbundling detection; NCCI compliance violations"},
    {"id":"dxproc",    "name":"DX-Proc Link",       "start":182, "end":208, "color":"#c84b2f",
     "desc":"Body-system-specific medical necessity matrix — STEMI+PCI=0.98, UTI+PCI=0.23",
     "fwa_relevance":"Medical necessity gaps; phantom procedures; DX padding"},
    {"id":"fwa",       "name":"FWA Risk Signals",   "start":208, "end":234, "color":"#8b1a1a",
     "desc":"D209=upcoding, D210=unbundling, D211=over-ordering, D212=phantom billing",
     "fwa_relevance":"Direct fraud risk scoring; SIU referral prioritisation"},
    {"id":"financial", "name":"DRG / RVU Proxy",    "start":234, "end":256, "color":"#1a3a6a",
     "desc":"CMS MS-DRG relative weight, AMA Work RVU, financial tier signal",
     "fwa_relevance":"DRG upcoding; high-cost proc on low-acuity DX"},
]

# ── Load embeddings ───────────────────────────────────────────────────────────
print(f"Loading embeddings from {EMBEDDINGS_PATH} ...", flush=True)
store = {}          # code -> dict
code_list = []      # ordered list of (code, type, desc) for search
vectors = {}        # code -> np.array shape (256,)

with open(EMBEDDINGS_PATH, 'r', buffering=1<<24) as f:
    reader = csv.reader(f)
    hdr = next(reader)
    ds = hdr.index("D001")
    for row in reader:
        code = row[0].strip()
        t    = row[1].strip()
        desc = row[2].strip()
        cat  = row[3].strip()
        dw   = float(row[4]) if row[4] else 0
        vw   = float(row[5]) if row[5] else 0
        fc   = float(row[6]) if row[6] else 0
        vec  = np.array(row[ds:ds+DIM], dtype=np.float32)
        store[code] = {"code":code,"type":t,"desc":desc,"cat":cat,
                       "dollar_weight":dw,"volume_weight":vw,"fwa_composite":fc}
        vectors[code] = vec
        code_list.append((code.lower(), t, desc.lower(), code, t, desc))

print(f"Loaded {len(store):,} codes.", flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def axis_cos(va, vb, s, e):
    a = va[s:e]; b = vb[s:e]
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def weighted_similarity(va, vb, weights):
    """Compute weighted cosine across all 10 axes."""
    total_weight = sum(weights.values()) or 1.0
    score = 0.0
    for ax in AXES:
        w = weights.get(ax["id"], 1.0) / total_weight
        score += w * axis_cos(va, vb, ax["start"], ax["end"])
    return score

def axis_scores(va, vb):
    return {ax["id"]: round(axis_cos(va, vb, ax["start"], ax["end"]), 6) for ax in AXES}

def find_code(code_str):
    c = code_str.strip().upper()
    for k in [c, c.replace('.',''), c[:7], c[:6], c[:5], c[:4], c[:3]]:
        if k in store: return k
    return None

def search_codes(query, limit=12):
    q = query.strip().lower()
    results = []
    # Exact code match first
    for code_low, t, desc_low, code, ctype, desc in code_list:
        if code_low == q or code_low.startswith(q):
            results.append({"code":code,"type":ctype,"desc":desc,
                            "score":1.0 if code_low==q else 0.95})
    # Description keyword match
    words = q.split()
    for code_low, t, desc_low, code, ctype, desc in code_list:
        if any(word in desc_low for word in words):
            if not any(r["code"]==code for r in results):
                match_score = sum(1 for w in words if w in desc_low)/len(words)
                results.append({"code":code,"type":ctype,"desc":desc,"score":match_score})
    results.sort(key=lambda r: -r["score"])
    return results[:limit]

def explain(va, vb, axis_sims, weights):
    """Generate plain-English explanation of each axis contribution."""
    dom_names = ["Cardiovascular","Neurological","Orthopedic","GI","Pulmonary",
                 "Metabolic","Renal","Mental Health","Infectious","Preventive"]
    anat_names = ["Head/CNS","Chest","Spine","Abdomen","Extremity","Skin","Pelvis","Systemic"]

    explanations = {}

    # Clinical domain
    s1 = [max(float(va[i*2]), float(va[i*2+1])) for i in range(10)]
    s2 = [max(float(vb[i*2]), float(vb[i*2+1])) for i in range(10)]
    d1 = dom_names[int(np.argmax(s1))]; d2 = dom_names[int(np.argmax(s2))]
    sim_c = axis_sims["clinical"]
    if sim_c > 0.75:
        explanations["clinical"] = f"Both codes in the same clinical domain ({d1}). High domain coherence."
    elif sim_c > 0.40:
        explanations["clinical"] = f"Related domains: {d1} vs {d2}. Partial clinical overlap."
    else:
        explanations["clinical"] = f"Domain mismatch: {d1} vs {d2}. Clinically unrelated specialties."

    # Severity
    sim_s = axis_sims["severity"]
    sev1 = float(va[26]); sev2 = float(vb[26])
    act1 = sum(1 for x in va[26:52] if abs(float(x)) > 0.05)
    act2 = sum(1 for x in vb[26:52] if abs(float(x)) > 0.05)
    tier1 = "MCC" if act1 >= 10 else ("CC" if act1 >= 5 else "Normal")
    tier2 = "MCC" if act2 >= 10 else ("CC" if act2 >= 5 else "Normal")
    if sim_s > 0.85:
        explanations["severity"] = f"Similar severity tiers ({tier1} ≈ {tier2}). Watch for MCC stacking if both are in same clinical domain."
    else:
        explanations["severity"] = f"Different severity tiers: Code A={tier1}, Code B={tier2}. Severity mismatch — review medical necessity."

    # DX-Proc link
    sim_d = axis_sims["dxproc"]
    if sim_d > 0.75:
        explanations["dxproc"] = f"Strong medical necessity link ({sim_d:.3f}). Procedure is well-supported by diagnosis."
    elif sim_d > 0.40:
        explanations["dxproc"] = f"Moderate medical necessity link ({sim_d:.3f}). Review clinical documentation."
    else:
        explanations["dxproc"] = f"Weak medical necessity link ({sim_d:.3f}). High risk of medical necessity denial — DX does not support this procedure."

    # Anatomical
    sim_a = axis_sims["anatomical"]
    an1 = anat_names[int(np.argmax([max(float(va[78+i*3]),float(va[79+i*3])) for i in range(8)]))]
    an2 = anat_names[int(np.argmax([max(float(vb[78+i*3]),float(vb[79+i*3])) for i in range(8)]))]
    if sim_a > 0.75:
        explanations["anatomical"] = f"Same anatomical region ({an1}). Site-of-service consistent."
    elif sim_a < 0.10:
        explanations["anatomical"] = f"Anatomical mismatch: {an1} vs {an2}. Possible wrong-site attribution."
    else:
        explanations["anatomical"] = f"Different anatomical regions: {an1} vs {an2}. Review site documentation."

    # Bundling
    sim_b = axis_sims["bundling"]
    b1 = float(vb[156]); b2 = float(va[156])
    if sim_b > 0.75:
        explanations["bundling"] = "Both codes have similar bundling profiles. Check NCCI edits — possible unbundling."
    elif b1 > 0.60 or b2 > 0.60:
        explanations["bundling"] = "One code is a component (high bundling signal). NCCI violation risk if both billed together."
    else:
        explanations["bundling"] = "Both codes appear to be comprehensive services. No bundling conflict detected."

    # FWA
    fwa_up_a = float(va[208]); fwa_up_b = float(vb[208])
    fwa_unb_a = float(va[209]); fwa_unb_b = float(vb[209])
    flags = []
    if max(fwa_up_a, fwa_up_b) > 0.18: flags.append("upcoding risk")
    if max(fwa_unb_a, fwa_unb_b) > 0.18: flags.append("unbundling risk")
    if float(va[211]) > 0.25 or float(vb[211]) > 0.25: flags.append("phantom billing signal")
    explanations["fwa"] = f"FWA signals: {', '.join(flags) if flags else 'No elevated FWA signals detected'}."

    # Financial
    fin_a = float(va[234]); fin_b = float(vb[234])
    fin_gap = abs(fin_a - fin_b)
    if fin_gap > 0.15:
        explanations["financial"] = f"Financial weight gap: {fin_gap:.3f}. High-DRG procedure paired with low-weight diagnosis — financial anomaly."
    else:
        explanations["financial"] = f"Financial weights are aligned. No financial gap anomaly detected."

    # Billing
    inp_a = float(va[130]); inp_b = float(vb[130])
    outp_a = float(va[132]); outp_b = float(vb[132])
    if (inp_a > 0.12 and outp_b > 0.12) or (outp_a > 0.12 and inp_b > 0.12):
        explanations["billing"] = "Mixed billing channels detected. One code is inpatient-oriented, the other outpatient — verify claim type."
    else:
        explanations["billing"] = "Billing channels are consistent between codes."

    # Episode
    ac_a = float(va[104]); ac_b = float(vb[104])
    pr_a = float(va[107]); pr_b = float(vb[107])
    if (ac_a > 0.12 and pr_b > 0.12) or (pr_a > 0.12 and ac_b > 0.12):
        explanations["episode"] = "Episode type conflict: one code is acute, the other preventive. Check for preventive codes on acute claims."
    else:
        explanations["episode"] = "Episode types are consistent."

    # Intensity
    int_a = float(va[52]); int_b = float(vb[52])
    int_gap = abs(int_a - int_b)
    if int_gap > 0.10:
        explanations["intensity"] = f"Intensity gap: high-complexity procedure ({max(int_a,int_b):.3f}) paired with low-complexity service ({min(int_a,int_b):.3f}). Verify documentation."
    else:
        explanations["intensity"] = "Service intensity is consistent between codes."

    return explanations

def risk_level(weighted_sim, axis_sims):
    dxproc = axis_sims["dxproc"]
    anat   = axis_sims["anatomical"]
    clin   = axis_sims["clinical"]
    fwa_composite_a = 0  # would need individual vectors

    if dxproc < 0.25 and clin < 0.30:
        return "CRITICAL", "Cross-domain mismatch with no medical necessity support"
    if dxproc < 0.40:
        return "HIGH", "Weak medical necessity link between diagnosis and procedure"
    if anat < 0.10:
        return "HIGH", "Anatomical site mismatch"
    if weighted_sim > 0.80:
        return "LOW", "Codes are clinically coherent"
    if weighted_sim > 0.60:
        return "LOW", "Generally coherent — routine review"
    return "MEDIUM", "Partial clinical alignment — documentation review recommended"


# ── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass  # suppress default logging

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json")
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/axes":
            self.send_json(AXES)

        elif parsed.path == "/search":
            q = qs.get("q",[""])[0]
            limit = int(qs.get("limit",[12])[0])
            self.send_json(search_codes(q, limit))

        elif parsed.path == "/code":
            code_str = qs.get("code",[""])[0]
            key = find_code(code_str)
            if key:
                info = dict(store[key])
                v = vectors[key]
                info["axis_profile"] = {}
                for ax in AXES:
                    seg = v[ax["start"]:ax["end"]]
                    info["axis_profile"][ax["id"]] = {
                        "max": round(float(np.max(seg)),6),
                        "mean": round(float(np.mean(seg)),6),
                        "active_dims": int(np.sum(np.abs(seg) > 0.05)),
                    }
                self.send_json(info)
            else:
                self.send_json({"error": f"Code '{code_str}' not found"}, 404)

        elif parsed.path == "/health":
            self.send_json({"status":"ok","codes":len(store),"dim":DIM})
        else:
            self.send_json({"error":"Not found"},404)

    def do_POST(self):
        if self.path == "/compare":
            length = int(self.headers.get("Content-Length",0))
            body   = json.loads(self.rfile.read(length))
            a = find_code(body.get("code_a",""))
            b = find_code(body.get("code_b",""))
            weights = body.get("weights", {ax["id"]:1.0 for ax in AXES})
            if not a:
                self.send_json({"error":f"Code A '{body.get('code_a','')}' not found"},404); return
            if not b:
                self.send_json({"error":f"Code B '{body.get('code_b','')}' not found"},404); return

            va = vectors[a]; vb = vectors[b]
            ax_sims = axis_scores(va, vb)
            wsim    = weighted_similarity(va, vb, weights)
            expls   = explain(va, vb, ax_sims, weights)
            risk, reason = risk_level(wsim, ax_sims)

            self.send_json({
                "code_a": store[a],
                "code_b": store[b],
                "weighted_similarity": round(wsim, 6),
                "full_cosine": round(float(np.dot(va,vb)), 6),
                "axis_similarities": ax_sims,
                "explanations": expls,
                "risk_level": risk,
                "risk_reason": reason,
                "weights_used": weights,
            })
        else:
            self.send_json({"error":"Not found"},404)

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Similarity API running on http://localhost:{PORT}")
    print(f"Endpoints: GET /search?q=TERM  |  POST /compare  |  GET /axes  |  GET /health")
    server.serve_forever()
