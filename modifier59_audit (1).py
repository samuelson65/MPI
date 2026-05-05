"""
Modifier 59 Audit System — Embedding-Driven Payment Integrity
=============================================================
Reads CPT embeddings from a CSV file.
CSV format:
    - Column "Code"       : CPT code identifier
    - Columns D001–D100   : 100-dimensional embedding vector

No diagnosis codes, anatomical sites, or provider specialty used.
All relationships inferred purely from cosine similarity.
"""

import csv
import json
import math
import sys
import os
from itertools import combinations

# ─── Configuration ────────────────────────────────────────────────────────────

EMBEDDING_CSV = "cpt_embeddings.csv"   # <-- change path if needed
CODE_COLUMN   = "Code"
DIM_PREFIX    = "D"
N_DIMS        = 100

# Similarity thresholds
THRESHOLD_HIGH = 0.85   # > this → Likely Invalid
THRESHOLD_MOD  = 0.65   # > this → Questionable  (else Valid)

# ─── CPT Descriptions (extend as needed) ──────────────────────────────────────

CPT_DESCRIPTIONS = {
    "99213": "Office Visit, Est. – Low Complexity",
    "99214": "Office Visit, Est. – Mod. Complexity",
    "20610": "Arthrocentesis, Major Joint",
    "20600": "Arthrocentesis, Small Joint",
    "97110": "Therapeutic Exercise",
    "97140": "Manual Therapy",
    "93000": "ECG w/ Interpretation",
    "93010": "ECG Interpretation Only",
    "11042": "Debridement, Subcutaneous Tissue",
    "11043": "Debridement, Muscle/Fascia",
}

# ─── Synthetic Claims ─────────────────────────────────────────────────────────

CLAIMS = [
    {
        "claim_id":       "C1",
        "scenario":       "Clearly Bundled Case",
        "cpt_codes":      ["93000", "93010"],
        "modifier_59_on": "93010",
    },
    {
        "claim_id":       "C2",
        "scenario":       "Clearly Distinct Case",
        "cpt_codes":      ["11042", "93000", "97110"],
        "modifier_59_on": "11042",
    },
    {
        "claim_id":       "C3",
        "scenario":       "Borderline Case",
        "cpt_codes":      ["97110", "97140"],
        "modifier_59_on": "97140",
    },
    {
        "claim_id":       "C4",
        "scenario":       "Adversarial Case (high similarity, attempted justification)",
        "cpt_codes":      ["20610", "20600"],
        "modifier_59_on": "20600",
    },
    {
        "claim_id":       "C5",
        "scenario":       "Noise Case (unrelated CPT added)",
        "cpt_codes":      ["99214", "20610", "93000"],
        "modifier_59_on": "20610",
    },
]

# ─── Embedding Loader ─────────────────────────────────────────────────────────

def load_embeddings(csv_path):
    """
    Load CPT embeddings from CSV.

    Expected columns: Code, D001, D002, ..., D100
    Returns: dict { cpt_code (str) -> embedding (list[float]) }
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] Embedding file not found: {csv_path}")
        sys.exit(1)

    embeddings = {}
    dim_cols = [f"{DIM_PREFIX}{str(i).zfill(3)}" for i in range(1, N_DIMS + 1)]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Validate Code column
        if CODE_COLUMN not in headers:
            print(f"[ERROR] Column '{CODE_COLUMN}' not found in CSV.")
            print(f"        Found columns: {headers}")
            sys.exit(1)

        # Find which dimension columns are actually present
        available_dims = [c for c in dim_cols if c in headers]
        if not available_dims:
            print(f"[ERROR] No dimension columns (D001–D{N_DIMS:03d}) found in CSV.")
            sys.exit(1)

        if len(available_dims) < N_DIMS:
            print(f"[WARNING] Expected {N_DIMS} dims, found {len(available_dims)}. Proceeding.")

        for row in reader:
            code = row[CODE_COLUMN].strip()
            try:
                vec = [float(row[d]) for d in available_dims]
            except (ValueError, KeyError) as e:
                print(f"[WARNING] Skipping code '{code}': {e}")
                continue
            embeddings[code] = vec

    print(f"[INFO] Loaded {len(embeddings)} CPT embeddings "
          f"({len(available_dims)} dims) from '{csv_path}'")
    return embeddings


# ─── Math Utilities ───────────────────────────────────────────────────────────

def l2_norm(vec):
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(a, b):
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot   = sum(x * y for x, y in zip(a, b))
    denom = l2_norm(a) * l2_norm(b)
    if denom == 0:
        return 0.0
    return dot / denom


# ─── Claim Analysis Helpers ───────────────────────────────────────────────────

def get_pairwise_similarities(cpt_codes, embeddings):
    """Return all pairwise cosine similarities for CPT codes in a claim."""
    pairs = {}
    for a, b in combinations(cpt_codes, 2):
        key = f"{a}_vs_{b}"
        if a in embeddings and b in embeddings:
            pairs[key] = round(cosine_similarity(embeddings[a], embeddings[b]), 4)
        else:
            missing = [c for c in (a, b) if c not in embeddings]
            print(f"[WARNING] Embedding missing for: {missing} — pair {key} skipped.")
    return pairs


def classify_tier(val):
    """Map similarity value to tier label."""
    if val > THRESHOLD_HIGH:
        return "HIGH"
    if val >= THRESHOLD_MOD:
        return "MODERATE"
    return "LOW"


def determine_status(similarities):
    """Determine Modifier 59 status from max pairwise similarity."""
    values = [v for v in similarities.values() if v is not None]
    if not values:
        return "Unknown"
    max_sim = max(values)
    if max_sim > THRESHOLD_HIGH:
        return "Likely Invalid"
    if max_sim >= THRESHOLD_MOD:
        return "Questionable"
    return "Valid"


def compute_risk_score(similarities):
    """
    Risk score 0-1.
    max_sim drives 70%, avg_sim drives 30%.
    """
    values = [v for v in similarities.values() if v is not None]
    if not values:
        return 0.0
    max_sim = max(values)
    avg_sim = sum(values) / len(values)
    return round(min(0.70 * max_sim + 0.30 * avg_sim, 1.0), 4)


def determine_confidence(similarities):
    """Confidence is high when similarity is far from thresholds."""
    values = [v for v in similarities.values() if v is not None]
    if not values:
        return "Low"
    max_sim = max(values)
    if max_sim > THRESHOLD_HIGH or max_sim < 0.55:
        return "High"
    return "Medium"


def get_max_pair(similarities):
    """Return the pair key and value with the highest similarity."""
    return max(similarities.items(), key=lambda x: x[1])


# ─── Explainability Builders ──────────────────────────────────────────────────

def build_key_drivers(cpt_codes, similarities, modifier_59_on):
    drivers = []
    sorted_pairs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    for pair_key, val in sorted_pairs[:3]:
        a, b = pair_key.split("_vs_")
        tier = classify_tier(val)
        if tier == "HIGH":
            drivers.append(
                f"Very high similarity between {a} and {b} ({val:.4f}) — "
                f"exceeds {THRESHOLD_HIGH} bundling threshold; strong procedural overlap"
            )
        elif tier == "MODERATE":
            drivers.append(
                f"Moderate similarity between {a} and {b} ({val:.4f}) — "
                f"related procedures; possible time-based or staged bundling conflict"
            )
        else:
            drivers.append(
                f"Low similarity between {a} and {b} ({val:.4f}) — "
                f"distinct procedural families; embedding supports independence"
            )

    desc  = CPT_DESCRIPTIONS.get(modifier_59_on, "")
    label = f"({desc})" if desc else ""
    drivers.append(
        f"Modifier 59 applied to {modifier_59_on} {label} — "
        f"evaluated against all co-billed codes on this claim"
    )
    return drivers[:3]


def build_reasoning(similarities, status):
    values  = [v for v in similarities.values() if v is not None]
    max_sim = max(values) if values else 0.0
    avg_sim = sum(values) / len(values) if values else 0.0
    mp_key, mp_val = get_max_pair(similarities)
    a, b = mp_key.split("_vs_")
    pair_label = f"{a} ↔ {b}"
    tier = classify_tier(max_sim)

    if tier == "HIGH":
        interp = (
            f"The dominant pair ({pair_label}) has a cosine similarity of {mp_val:.4f}, "
            f"computed directly from the 100-dimensional CPT embeddings. This places it firmly "
            f"in the >{THRESHOLD_HIGH} 'likely duplicate or component' zone. The embedding vectors "
            f"are nearly collinear, indicating near-identical procedural semantics. "
            f"The residual distance ({1 - mp_val:.4f}) represents noise rather than a clinically "
            f"meaningful distinction."
        )
        bundling = (
            "HIGH likelihood of component bundling or duplication. One code likely subsumes "
            "the other under NCCI logic. Billing both separately constitutes unbundling unless "
            "extraordinary circumstances (separate sites, separate providers, separate encounters) "
            "are explicitly documented."
        )
    elif tier == "MODERATE":
        interp = (
            f"The dominant pair ({pair_label}) has a cosine similarity of {mp_val:.4f} "
            f"from the 100-dimensional embeddings, falling in the borderline "
            f"{THRESHOLD_MOD}–{THRESHOLD_HIGH} zone. The vectors share substantial directional "
            f"variance but are not collinear, encoding partial but not complete procedural overlap. "
            f"Ambiguity is inherent at this similarity level."
        )
        bundling = (
            "MODERATE bundling likelihood. Codes are related but not confirmed as strict components. "
            "Whether they represent independent services depends on documentation of distinct timed "
            "units, separate clinical goals, or anatomically distinct sites — none detectable "
            "from embeddings alone."
        )
    else:
        interp = (
            f"All pairwise similarities are below {THRESHOLD_MOD} "
            f"(max: {max_sim:.4f}, avg: {avg_sim:.4f}), computed from 100-dimensional embeddings. "
            f"The codes occupy geometrically distant regions of procedure space, encoding "
            f"fundamentally different clinical activities with no shared technique or indication cluster."
        )
        bundling = (
            "LOW bundling likelihood. No embedding evidence of shared procedural logic, component "
            "relationships, or technique overlap. These codes are most consistent with independent "
            "services across separate clinical domains."
        )

    uncertainty = "LOW"  if tier in ("HIGH", "LOW") else "HIGH"
    decisive    = "decisive" if tier in ("HIGH", "LOW") else "borderline"

    limitations = (
        "IMPORTANT: This analysis is based solely on CPT embedding cosine similarity computed "
        "from the provided 100-dimensional vectors. No diagnosis codes, anatomical site labels, "
        "provider specialty, or documentation context are available. Embedding similarity is a "
        "proxy for procedural relatedness — it cannot confirm or deny: (a) different anatomical "
        "sites, (b) separate clinical encounters, (c) distinct medical necessity, or "
        f"(d) payer-specific bundling policies. Uncertainty: {uncertainty} (similarity is {decisive})."
    )

    return {
        "embedding_interpretation": interp,
        "bundling_likelihood":      bundling,
        "limitations":              limitations,
    }


def build_axis_contributions(similarities, status):
    values  = [v for v in similarities.values() if v is not None]
    max_sim = max(values) if values else 0.0
    avg_sim = sum(values) / len(values) if values else 0.0

    # 60-69: Bundling Cohesion
    if max_sim > THRESHOLD_HIGH:
        a60 = (
            f"VERY HIGH ({max_sim:.4f}). Near-maximum bundling cohesion from embedding geometry. "
            f"The dominant pair's 100-dim vectors are nearly collinear, strongly implying a "
            f"comprehensive/component relationship. Primary driver of elevated risk score."
        )
    elif max_sim >= THRESHOLD_MOD:
        a60 = (
            f"MODERATE ({max_sim:.4f}). Meaningful cohesion without crossing the definitive "
            f"threshold. Shared procedural logic is present but a strict parent-child "
            f"relationship is not confirmed by the embedding geometry."
        )
    else:
        a60 = (
            f"VERY LOW (max: {max_sim:.4f}, avg: {avg_sim:.4f}). Near-zero bundling cohesion "
            f"across all pairs in 100-dim space. No component or comprehensive relationship "
            f"implied by the embedding vectors."
        )

    # 70-79: Procedure Relationship
    if avg_sim > 0.75:
        a70 = (
            f"HIGH (avg similarity: {avg_sim:.4f}). All codes cluster tightly in the same "
            f"procedural region of the 100-dim embedding space — consistent with same-technique "
            f"or same-system billing."
        )
    elif avg_sim >= 0.50:
        a70 = (
            f"MIXED (avg similarity: {avg_sim:.4f}). Some codes share procedural context; "
            f"others are more distant in embedding space. The claim spans more than one "
            f"procedural family, partially supporting independent service delivery."
        )
    else:
        a70 = (
            f"LOW (avg similarity: {avg_sim:.4f}). Codes occupy maximally separated procedural "
            f"clusters in 100-dim space — consistent with multi-system or multi-technique "
            f"service delivery. Embedding strongly supports procedural independence."
        )

    # 80-89: FWA Risk Signal
    if status == "Likely Invalid":
        a80 = (
            f"HIGH. High-similarity pair billed with Modifier 59 is a textbook unbundling pattern. "
            f"Embedding similarity ({max_sim:.4f}) independently corroborates the risk without "
            f"external labels. Automated denial or pre-payment review is supported."
        )
    elif status == "Questionable":
        a80 = (
            f"MODERATE. Same-cluster billing with Modifier 59 elevates FWA risk above baseline. "
            f"Pattern warrants post-payment review or documentation request. Automated denial "
            f"not supported without additional evidence."
        )
    else:
        a80 = (
            f"LOW. No procedural overlap signal detected from embeddings. Multi-system same-day "
            f"billing carries low inherent FWA risk from a bundling standpoint. Medical necessity "
            f"risk (if any) is outside the scope of embedding-based analysis."
        )

    return {"60-69": a60, "70-79": a70, "80-89": a80}


def build_counterfactual(similarities, status):
    mp_key, mp_val = get_max_pair(similarities)
    a, b = mp_key.split("_vs_")

    if status == "Likely Invalid":
        return (
            f"Modifier 59 could be reconsidered ONLY if: "
            f"(1) Medical records explicitly document that {a} and {b} were performed on "
            f"anatomically distinct sites (e.g., different body regions or laterality), "
            f"(2) Separate encounters or distinct timed blocks are documented with individual "
            f"procedure notes, "
            f"(3) Different performing providers (separate NPIs) executed each service, or "
            f"(4) A payer-specific policy explicitly permits unbundled billing for this pair. "
            f"None of these conditions are detectable from embedding similarity alone. "
            f"Current evidence (similarity: {mp_val:.4f}) supports denial or human review escalation."
        )
    elif status == "Questionable":
        return (
            f"Modifier 59 could be CONFIRMED VALID if: "
            f"(1) Notes document distinct timed units with separate clinical goals for {a} and {b}, "
            f"(2) Services were applied to anatomically distinct body regions within the same encounter, "
            f"(3) Payer policy explicitly permits same-session concurrent billing with Modifier 59. "
            f"Modifier 59 would be INVALIDATED if a single continuous time block was split across "
            f"both codes. Similarity ({mp_val:.4f}) alone is insufficient to decide — "
            f"documentation review is required."
        )
    else:
        return (
            f"Modifier 59 is well-supported by embedding evidence (max similarity: {mp_val:.4f}). "
            f"What would INVALIDATE it: if any code pair shared similarity >{THRESHOLD_HIGH}, "
            f"indicating one subsumed the other — which is not the case here. "
            f"Secondary concern: verify that any E&M codes carry Modifier 25 (not 59) when "
            f"co-billed with same-day procedures, as this distinction is outside the scope "
            f"of embedding-based analysis."
        )


# ─── Audit Engine ─────────────────────────────────────────────────────────────

def audit_claim(claim, embeddings):
    """Run full Modifier 59 audit on a single claim."""
    cpt_codes   = claim["cpt_codes"]
    modifier_on = claim["modifier_59_on"]

    missing = [c for c in cpt_codes if c not in embeddings]
    if missing:
        print(f"[WARNING] Claim {claim['claim_id']}: no embedding for {missing} — skipping.")
        return None

    similarities   = get_pairwise_similarities(cpt_codes, embeddings)
    status         = determine_status(similarities)
    risk_score     = compute_risk_score(similarities)
    confidence     = determine_confidence(similarities)
    key_drivers    = build_key_drivers(cpt_codes, similarities, modifier_on)
    reasoning      = build_reasoning(similarities, status)
    axes           = build_axis_contributions(similarities, status)
    counterfactual = build_counterfactual(similarities, status)

    return {
        "claim_id":           claim["claim_id"],
        "scenario":           claim["scenario"],
        "cpt_codes":          cpt_codes,
        "modifier_59_on":     modifier_on,
        "similarities":       similarities,
        "modifier_59_status": status,
        "risk_score":         risk_score,
        "confidence":         confidence,
        "key_drivers":        key_drivers,
        "reasoning":          reasoning,
        "axis_contributions": axes,
        "counterfactual":     counterfactual,
    }


# ─── Display ──────────────────────────────────────────────────────────────────

STATUS_SYMBOLS = {
    "Valid":          "✓  VALID",
    "Questionable":   "⚠  QUESTIONABLE",
    "Likely Invalid": "✕  LIKELY INVALID",
}


def wrap(text, width=64, indent="    "):
    words  = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > width:
            lines.append(indent + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(indent + " ".join(line))
    return "\n".join(lines)


def risk_bar(score, width=10):
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def sim_bar(val, width=10):
    filled = round(val * width)
    return "▓" * filled + "·" * (width - filled)


def sep(char="─", width=72):
    print(char * width)


def print_claim(r):
    sep("═")
    print(f"  {r['claim_id']}  |  {r['scenario']}")
    sep()

    print("\n  CPT CODES")
    for code in r["cpt_codes"]:
        tag  = " ← MOD-59" if code == r["modifier_59_on"] else ""
        desc = CPT_DESCRIPTIONS.get(code, "(no description)")
        print(f"    {code}{tag:<12}  {desc}")

    print("\n  PAIRWISE SIMILARITY  (cosine, 100-dim embeddings)")
    for pair, val in r["similarities"].items():
        a, b = pair.split("_vs_")
        tier = classify_tier(val)
        print(f"    {a} ↔ {b}  {sim_bar(val)}  {val:.4f}  [{tier}]")

    print("\n  DECISION")
    print(f"    Status     :  {STATUS_SYMBOLS.get(r['modifier_59_status'], r['modifier_59_status'])}")
    print(f"    Risk Score :  [{risk_bar(r['risk_score'])}]  {r['risk_score']:.4f}")
    print(f"    Confidence :  {r['confidence']}")

    print("\n  KEY DRIVERS")
    for i, d in enumerate(r["key_drivers"], 1):
        lines = wrap(d, width=63, indent="       ").split("\n")
        print(f"    {i}. {lines[0].strip()}")
        for line in lines[1:]:
            print(line)

    print("\n  REASONING")
    for section, content in r["reasoning"].items():
        label = section.replace("_", " ").upper()
        print(f"\n    [{label}]")
        print(wrap(content, width=63, indent="      "))

    print("\n  AXIS CONTRIBUTIONS")
    axis_labels = {
        "60-69": "60-69  Bundling Cohesion",
        "70-79": "70-79  Procedure Relationship",
        "80-89": "80-89  FWA Risk Signal",
    }
    for k, label in axis_labels.items():
        print(f"\n    [{label}]")
        print(wrap(r["axis_contributions"][k], width=63, indent="      "))

    print("\n  COUNTERFACTUAL")
    print(wrap(r["counterfactual"], width=63, indent="    "))
    print()


def print_summary(results):
    sep("═")
    print("  AUDIT SUMMARY")
    sep()
    print(f"  {'ID':<6} {'SCENARIO':<40} {'STATUS':<16} {'RISK':>6}  {'CONF'}")
    sep()
    for r in results:
        short = r["modifier_59_status"].replace("Likely ", "")
        print(f"  {r['claim_id']:<6} {r['scenario'][:39]:<40} {short:<16} "
              f"{r['risk_score']:>6.4f}  {r['confidence']}")
    sep()
    valid = sum(1 for r in results if r["modifier_59_status"] == "Valid")
    quest = sum(1 for r in results if r["modifier_59_status"] == "Questionable")
    inval = sum(1 for r in results if r["modifier_59_status"] == "Likely Invalid")
    avg   = sum(r["risk_score"] for r in results) / len(results)
    print(f"\n  Valid: {valid}  |  Questionable: {quest}  |  Likely Invalid: {inval}")
    print(f"  Average Risk Score: {avg:.4f}")
    sep("═")
    print("  CONSTRAINT: Analysis based solely on 100-dim CPT embedding cosine similarity.")
    print("  No dx codes · No anatomical sites · No provider specialty · No documentation.")
    sep("═")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Optional: override CSV path via command line
    # Usage: python modifier59_audit.py path/to/cpt_embeddings.csv
    csv_path = sys.argv[1] if len(sys.argv) > 1 else EMBEDDING_CSV

    print()
    print("  MODIFIER 59 AUDIT ENGINE — EMBEDDING-DRIVEN PAYMENT INTEGRITY")
    print(f"  Embedding file  : {csv_path}")
    print(f"  Dimensions      : {N_DIMS}")
    print(f"  Thresholds      : >{THRESHOLD_HIGH} BUNDLE  |  "
          f"{THRESHOLD_MOD}–{THRESHOLD_HIGH} BORDERLINE  |  <{THRESHOLD_MOD} DISTINCT")
    print()

    embeddings = load_embeddings(csv_path)

    results = []
    for claim in CLAIMS:
        result = audit_claim(claim, embeddings)
        if result:
            results.append(result)
            print_claim(result)

    if not results:
        print("[ERROR] No claims could be audited. Check that CPT codes exist in the CSV.")
        sys.exit(1)

    print_summary(results)

    out_json = "modifier59_audit_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON results written to: {out_json}\n")


if __name__ == "__main__":
    main()
