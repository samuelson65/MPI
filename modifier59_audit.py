"""
Modifier 59 Audit System — Embedding-Driven Payment Integrity
=============================================================
Pure CPT embedding cosine similarity approach.
No diagnosis codes, no anatomical sites, no provider specialty.
"""

import json
from itertools import combinations

# ─── CPT Universe ─────────────────────────────────────────────────────────────

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

# Pairwise cosine similarity matrix (synthetic, embedding-derived)
SIMILARITY_MATRIX = {
    "99213": {"99213":1.00,"99214":0.94,"20610":0.52,"20600":0.50,"97110":0.48,"97140":0.46,"93000":0.55,"93010":0.53,"11042":0.41,"11043":0.39},
    "99214": {"99213":0.94,"99214":1.00,"20610":0.54,"20600":0.51,"97110":0.49,"97140":0.47,"93000":0.57,"93010":0.55,"11042":0.43,"11043":0.41},
    "20610": {"99213":0.52,"99214":0.54,"20610":1.00,"20600":0.91,"97110":0.61,"97140":0.59,"93000":0.38,"93010":0.36,"11042":0.44,"11043":0.46},
    "20600": {"99213":0.50,"99214":0.51,"20610":0.91,"20600":1.00,"97110":0.59,"97140":0.57,"93000":0.37,"93010":0.35,"11042":0.42,"11043":0.44},
    "97110": {"99213":0.48,"99214":0.49,"20610":0.61,"20600":0.59,"97110":1.00,"97140":0.79,"93000":0.31,"93010":0.29,"11042":0.35,"11043":0.33},
    "97140": {"99213":0.46,"99214":0.47,"20610":0.59,"20600":0.57,"97110":0.79,"97140":1.00,"93000":0.30,"93010":0.28,"11042":0.33,"11043":0.31},
    "93000": {"99213":0.55,"99214":0.57,"20610":0.38,"20600":0.37,"97110":0.31,"97140":0.30,"93000":1.00,"93010":0.88,"11042":0.29,"11043":0.27},
    "93010": {"99213":0.53,"99214":0.55,"20610":0.36,"20600":0.35,"97110":0.29,"97140":0.28,"93000":0.88,"93010":1.00,"11042":0.27,"11043":0.25},
    "11042": {"99213":0.41,"99214":0.43,"20610":0.44,"20600":0.42,"97110":0.35,"97140":0.33,"93000":0.29,"93010":0.27,"11042":1.00,"11043":0.87},
    "11043": {"99213":0.39,"99214":0.41,"20610":0.46,"20600":0.44,"97110":0.33,"97140":0.31,"93000":0.27,"93010":0.25,"11042":0.87,"11043":1.00},
}

# ─── Synthetic Claims ──────────────────────────────────────────────────────────

CLAIMS = [
    {
        "claim_id": "C1",
        "scenario": "Clearly Bundled Case",
        "cpt_codes": ["93000", "93010"],
        "modifier_59_on": "93010",
    },
    {
        "claim_id": "C2",
        "scenario": "Clearly Distinct Case",
        "cpt_codes": ["11042", "93000", "97110"],
        "modifier_59_on": "11042",
    },
    {
        "claim_id": "C3",
        "scenario": "Borderline Case",
        "cpt_codes": ["97110", "97140"],
        "modifier_59_on": "97140",
    },
    {
        "claim_id": "C4",
        "scenario": "Adversarial Case (high similarity, attempted justification)",
        "cpt_codes": ["20610", "20600"],
        "modifier_59_on": "20600",
    },
    {
        "claim_id": "C5",
        "scenario": "Noise Case (unrelated CPT added)",
        "cpt_codes": ["99214", "20610", "93000"],
        "modifier_59_on": "20610",
    },
]

# ─── Core Logic ───────────────────────────────────────────────────────────────

def get_similarity(cpt_a, cpt_b):
    """Return cosine similarity between two CPT codes."""
    return SIMILARITY_MATRIX.get(cpt_a, {}).get(cpt_b)


def get_pairwise_similarities(cpt_codes):
    """Return all pairwise similarities for a list of CPT codes."""
    pairs = {}
    for a, b in combinations(cpt_codes, 2):
        key = f"{a}_vs_{b}"
        pairs[key] = get_similarity(a, b)
    return pairs


def classify_similarity(val):
    """Map a similarity score to a tier label."""
    if val is None:
        return "UNKNOWN"
    if val > 0.85:
        return "HIGH"
    if val >= 0.65:
        return "MODERATE"
    return "LOW"


def compute_risk_score(similarities):
    """
    Derive a 0-1 risk score from pairwise similarities.
    Max pairwise similarity drives the score; secondary pairs contribute.
    """
    values = [v for v in similarities.values() if v is not None]
    if not values:
        return 0.0
    max_sim = max(values)
    avg_sim = sum(values) / len(values)
    # Weighted: max similarity is dominant signal
    raw = 0.70 * max_sim + 0.30 * avg_sim
    return round(min(raw, 1.0), 2)


def determine_modifier_59_status(similarities):
    """
    Apply embedding-based heuristics to determine Modifier 59 validity.

    > 0.85 max similarity  → Likely Invalid
    0.65–0.85              → Questionable
    < 0.65                 → Valid
    """
    values = [v for v in similarities.values() if v is not None]
    if not values:
        return "Unknown"
    max_sim = max(values)
    if max_sim > 0.85:
        return "Likely Invalid"
    if max_sim >= 0.65:
        return "Questionable"
    return "Valid"


def determine_confidence(similarities):
    """Confidence is high when similarity is far from the thresholds."""
    values = [v for v in similarities.values() if v is not None]
    if not values:
        return "Low"
    max_sim = max(values)
    # Far from thresholds → high confidence
    if max_sim > 0.85 or max_sim < 0.55:
        return "High"
    if 0.70 <= max_sim <= 0.82 or 0.55 <= max_sim < 0.65:
        return "Medium"
    return "Medium"


def build_key_drivers(cpt_codes, similarities, modifier_59_on):
    """Generate top-3 explainable drivers from similarity patterns."""
    drivers = []
    sorted_pairs = sorted(similarities.items(), key=lambda x: x[1] or 0, reverse=True)

    for pair_key, val in sorted_pairs[:3]:
        a, b = pair_key.split("_vs_")
        tier = classify_similarity(val)
        if tier == "HIGH":
            drivers.append(
                f"Very high similarity between {a} and {b} ({val:.2f}) — "
                f"exceeds 0.85 bundling threshold; strong procedural overlap signal"
            )
        elif tier == "MODERATE":
            drivers.append(
                f"Moderate similarity between {a} and {b} ({val:.2f}) — "
                f"related procedures; possible time-based or staged bundling conflict"
            )
        else:
            drivers.append(
                f"Low similarity between {a} and {b} ({val:.2f}) — "
                f"distinct procedural families; embedding supports independence"
            )

    if not drivers:
        drivers.append("Insufficient similarity data to generate drivers.")

    # Add Modifier 59 target note
    drivers.append(
        f"Modifier 59 applied to {modifier_59_on} "
        f"({CPT_DESCRIPTIONS.get(modifier_59_on, 'Unknown')}) — "
        f"evaluated against all co-billed codes on this claim"
    )

    return drivers[:3]


def build_reasoning(cpt_codes, similarities, status):
    """Build structured reasoning block from embedding patterns."""
    values = [v for v in similarities.values() if v is not None]
    max_sim = max(values) if values else 0
    avg_sim = sum(values) / len(values) if values else 0
    max_pair = max(similarities.items(), key=lambda x: x[1] or 0)

    # Embedding interpretation
    tier = classify_similarity(max_sim)
    if tier == "HIGH":
        interp = (
            f"The dominant pair ({max_pair[0].replace('_vs_', ' ↔ ')}) yields a cosine similarity "
            f"of {max_pair[1]:.2f}, placing it firmly in the >0.85 'likely duplicate or component' zone. "
            f"The embedding model encodes these codes as near-identical in procedural semantics — "
            f"consistent with a parent-child or component billing relationship. "
            f"The residual difference ({1 - max_pair[1]:.2f}) represents noise rather than clinically meaningful distinction."
        )
    elif tier == "MODERATE":
        interp = (
            f"The dominant pair ({max_pair[0].replace('_vs_', ' ↔ ')}) yields a cosine similarity "
            f"of {max_pair[1]:.2f}, falling in the borderline 0.65–0.85 zone. "
            f"The embedding encodes meaningful shared variance — both codes likely occupy the same "
            f"procedural cluster — but maintains enough directional separation to avoid a definitive "
            f"bundling classification. Ambiguity is inherent at this similarity level."
        )
    else:
        interp = (
            f"All pairwise similarities are below 0.65 (max: {max_sim:.2f}, avg: {avg_sim:.2f}). "
            f"The embedding model places these codes in geometrically distant regions of procedure space, "
            f"indicating they encode fundamentally different clinical activities with no shared technique, "
            f"instrument, or indication cluster."
        )

    # Bundling likelihood
    if tier == "HIGH":
        bundling = (
            "HIGH likelihood of component bundling or duplication. One code likely subsumes the other "
            "under standard NCCI logic. Billing both separately constitutes unbundling unless extraordinary "
            "circumstances (separate sites, separate providers, separate encounters) are documented."
        )
    elif tier == "MODERATE":
        bundling = (
            "MODERATE bundling likelihood. Codes are related but not confirmed as components. "
            "Whether these represent independent services depends on documentation of distinct timed units, "
            "separate clinical goals, or anatomically distinct sites — none of which are detectable "
            "from embeddings alone."
        )
    else:
        bundling = (
            "LOW bundling likelihood. No embedding evidence of shared procedural logic, component "
            "relationships, or technique overlap. These codes are most consistent with independent "
            "services across separate clinical domains."
        )

    # Limitations
    limitations = (
        "IMPORTANT: This analysis is based solely on CPT embedding similarity. "
        "No diagnosis codes, anatomical site labels, provider specialty, or documentation context "
        "are available. Embedding similarity is a proxy for procedural relatedness — it cannot "
        "confirm or deny: (a) different anatomical sites, (b) separate clinical encounters, "
        "(c) distinct medical necessity for each service, or (d) payer-specific bundling policies. "
        f"Uncertainty level: {'LOW' if tier in ('HIGH', 'LOW') else 'HIGH'} "
        f"(similarity {'is decisive' if tier in ('HIGH', 'LOW') else 'is borderline'})."
    )

    return {
        "embedding_interpretation": interp,
        "bundling_likelihood": bundling,
        "limitations": limitations,
    }


def build_axis_contributions(similarities, status):
    """Map embedding signals to the three audit framework axes."""
    values = [v for v in similarities.values() if v is not None]
    max_sim = max(values) if values else 0
    avg_sim = sum(values) / len(values) if values else 0

    # Axis 60-69: Bundling Cohesion
    if max_sim > 0.85:
        axis_60 = (
            f"VERY HIGH ({max_sim:.2f}). Near-maximum bundling cohesion detected. "
            f"The dominant pair's embedding proximity strongly implies a comprehensive/component "
            f"relationship. This axis is the primary driver of the high risk score."
        )
    elif max_sim >= 0.65:
        axis_60 = (
            f"MODERATE ({max_sim:.2f}). Meaningful cohesion without crossing the definitive threshold. "
            f"Shared procedural logic is present but a strict parent-child relationship is not confirmed."
        )
    else:
        axis_60 = (
            f"VERY LOW (max: {max_sim:.2f}, avg: {avg_sim:.2f}). Near-zero bundling cohesion across all pairs. "
            f"No component or comprehensive relationship is implied by the embedding geometry."
        )

    # Axis 70-79: Procedure Relationship
    if avg_sim > 0.75:
        axis_70 = (
            f"HIGH (avg similarity: {avg_sim:.2f}). All codes cluster tightly in the same procedural family. "
            f"High intra-claim procedural homogeneity — consistent with same-technique or same-system billing."
        )
    elif avg_sim >= 0.50:
        axis_70 = (
            f"MIXED (avg similarity: {avg_sim:.2f}). Some codes share procedural context; "
            f"others are more distant. The claim spans more than one procedural family, "
            f"which partially supports independent service delivery."
        )
    else:
        axis_70 = (
            f"LOW (avg similarity: {avg_sim:.2f}). Codes occupy maximally separated procedural clusters. "
            f"The claim reflects multi-system or multi-technique service delivery — "
            f"embedding strongly supports procedural independence."
        )

    # Axis 80-89: FWA Risk Signal
    if status == "Likely Invalid":
        axis_80 = (
            f"HIGH. The combination of high similarity and Modifier 59 usage is a documented FWA pattern "
            f"(unbundling). The embedding independently corroborates the risk without external labels. "
            f"Automated denial or pre-payment review is supported."
        )
    elif status == "Questionable":
        axis_80 = (
            f"MODERATE. Same-cluster billing with Modifier 59 elevates FWA risk above baseline. "
            f"This pattern warrants post-payment review or documentation request. "
            f"Automated denial is not supported without additional evidence."
        )
    else:
        axis_80 = (
            f"LOW. No procedural overlap signal detected. Multi-system same-day billing carries "
            f"low inherent FWA risk from a bundling standpoint. Medical necessity risk "
            f"(if any) is outside the scope of embedding-based analysis."
        )

    return {"60-69": axis_60, "70-79": axis_70, "80-89": axis_80}


def build_counterfactual(cpt_codes, similarities, status):
    """Generate counterfactual evidence that would change the decision."""
    values = [v for v in similarities.values() if v is not None]
    max_sim = max(values) if values else 0
    max_pair = max(similarities.items(), key=lambda x: x[1] or 0)
    a, b = max_pair[0].split("_vs_")

    if status == "Likely Invalid":
        return (
            f"Modifier 59 on the flagged code could be reconsidered ONLY if: "
            f"(1) Medical records explicitly document that {a} and {b} were performed "
            f"on anatomically distinct sites (e.g., different body regions, different laterality), "
            f"(2) Separate encounters or distinct timed blocks are documented with individual procedure notes, "
            f"(3) Different performing providers (separate NPIs) executed each service, or "
            f"(4) A payer-specific policy explicitly permits unbundled billing for this pair. "
            f"None of these conditions are detectable from embedding similarity alone. "
            f"Current evidence (similarity: {max_sim:.2f}) supports denial or human review escalation."
        )
    elif status == "Questionable":
        return (
            f"Modifier 59 could be CONFIRMED VALID if: "
            f"(1) Therapy/procedure notes document distinct timed units with separate clinical goals "
            f"for {a} and {b}, "
            f"(2) Services were applied to anatomically distinct body regions within the same encounter, "
            f"(3) Payer policy explicitly permits same-session concurrent billing for this pair with Modifier 59. "
            f"Modifier 59 would be INVALIDATED if a single continuous time block was split across both codes. "
            f"Recommendation: Request documentation before rendering a final determination. "
            f"Similarity ({max_sim:.2f}) alone is insufficient to decide."
        )
    else:
        return (
            f"Modifier 59 is well-supported by embedding evidence (max similarity: {max_sim:.2f}). "
            f"What would INVALIDATE it: if any code pair were found to share similarity >0.85, "
            f"indicating one subsumed the other — which is not the case here. "
            f"Secondary concern: verify that E&M codes carry Modifier 25 (not 59) when co-billed "
            f"with same-day procedures, as this distinction is outside the scope of embedding analysis."
        )


# ─── Audit Engine ─────────────────────────────────────────────────────────────

def audit_claim(claim):
    """Run full Modifier 59 audit on a single claim. Returns structured result."""
    cpt_codes     = claim["cpt_codes"]
    modifier_on   = claim["modifier_59_on"]

    similarities  = get_pairwise_similarities(cpt_codes)
    status        = determine_modifier_59_status(similarities)
    risk_score    = compute_risk_score(similarities)
    confidence    = determine_confidence(similarities)
    key_drivers   = build_key_drivers(cpt_codes, similarities, modifier_on)
    reasoning     = build_reasoning(cpt_codes, similarities, status)
    axes          = build_axis_contributions(similarities, status)
    counterfactual = build_counterfactual(cpt_codes, similarities, status)

    return {
        "claim_id":            claim["claim_id"],
        "scenario":            claim["scenario"],
        "cpt_codes":           cpt_codes,
        "modifier_59_on":      modifier_on,
        "similarities":        {k: round(v, 2) for k, v in similarities.items() if v is not None},
        "modifier_59_status":  status,
        "risk_score":          risk_score,
        "confidence":          confidence,
        "key_drivers":         key_drivers,
        "reasoning":           reasoning,
        "axis_contributions":  axes,
        "counterfactual":      counterfactual,
    }


# ─── Display ──────────────────────────────────────────────────────────────────

STATUS_SYMBOLS = {
    "Valid":          "✓  VALID",
    "Questionable":   "⚠  QUESTIONABLE",
    "Likely Invalid": "✕  LIKELY INVALID",
}

RISK_BARS = {
    (0.00, 0.30): "░░░░░░░░░░",
    (0.30, 0.50): "███░░░░░░░",
    (0.50, 0.70): "█████░░░░░",
    (0.70, 0.85): "███████░░░",
    (0.85, 1.01): "██████████",
}

def risk_bar(score):
    for (lo, hi), bar in RISK_BARS.items():
        if lo <= score < hi:
            filled = round(score * 10)
            return "█" * filled + "░" * (10 - filled)
    return "██████████"


def print_separator(char="─", width=72):
    print(char * width)


def print_claim(result):
    w = 72
    print_separator("═", w)
    print(f"  {result['claim_id']}  |  {result['scenario']}")
    print_separator("─", w)

    # CPT codes
    print("\n  CPT CODES")
    for code in result["cpt_codes"]:
        tag = " ← MOD-59" if code == result["modifier_59_on"] else ""
        desc = CPT_DESCRIPTIONS.get(code, "")
        print(f"    {code}{tag:12}  {desc}")

    # Similarities
    print("\n  PAIRWISE SIMILARITY MATRIX")
    for pair, val in result["similarities"].items():
        a, b = pair.split("_vs_")
        tier = classify_similarity(val)
        tier_tag = f"[{tier:8}]"
        bar = "▓" * round(val * 10) + "·" * (10 - round(val * 10))
        print(f"    {a} ↔ {b}  {bar}  {val:.2f}  {tier_tag}")

    # Decision
    print("\n  DECISION")
    status_str = STATUS_SYMBOLS.get(result["modifier_59_status"], result["modifier_59_status"])
    rbar = risk_bar(result["risk_score"])
    print(f"    Status     :  {status_str}")
    print(f"    Risk Score :  [{rbar}]  {result['risk_score']:.2f}")
    print(f"    Confidence :  {result['confidence']}")

    # Key Drivers
    print("\n  KEY DRIVERS")
    for i, d in enumerate(result["key_drivers"], 1):
        # word-wrap at 65 chars
        words = d.split()
        lines, line = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 63:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        print(f"    {i}. {lines[0]}")
        for l in lines[1:]:
            print(f"       {l}")

    # Reasoning
    print("\n  REASONING")
    for section, content in result["reasoning"].items():
        label = section.replace("_", " ").upper()
        print(f"\n    [{label}]")
        words = content.split()
        lines, line = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 63:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for l in lines:
            print(f"      {l}")

    # Axis Contributions
    print("\n  AXIS CONTRIBUTIONS")
    axis_labels = {
        "60-69": "60–69  Bundling Cohesion",
        "70-79": "70–79  Procedure Relationship",
        "80-89": "80–89  FWA Risk Signal",
    }
    for axis_key, axis_label in axis_labels.items():
        content = result["axis_contributions"].get(axis_key, "")
        print(f"\n    [{axis_label}]")
        words = content.split()
        lines, line = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 63:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for l in lines:
            print(f"      {l}")

    # Counterfactual
    print("\n  COUNTERFACTUAL")
    words = result["counterfactual"].split()
    lines, line = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > 63:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    for l in lines:
        print(f"    {l}")

    print()


def print_summary(results):
    print_separator("═", 72)
    print("  AUDIT SUMMARY")
    print_separator("─", 72)
    print(f"  {'ID':<6} {'SCENARIO':<38} {'STATUS':<18} {'RISK':>6}  {'CONF'}")
    print_separator("─", 72)
    for r in results:
        status_short = r["modifier_59_status"].replace("Likely ", "")
        print(f"  {r['claim_id']:<6} {r['scenario'][:37]:<38} {status_short:<18} {r['risk_score']:>6.2f}  {r['confidence']}")
    print_separator("─", 72)

    valid = sum(1 for r in results if r["modifier_59_status"] == "Valid")
    quest = sum(1 for r in results if r["modifier_59_status"] == "Questionable")
    inval = sum(1 for r in results if r["modifier_59_status"] == "Likely Invalid")
    avg   = sum(r["risk_score"] for r in results) / len(results)

    print(f"\n  Valid: {valid}  |  Questionable: {quest}  |  Likely Invalid: {inval}")
    print(f"  Average Risk Score: {avg:.2f}")
    print_separator("═", 72)
    print("  CONSTRAINT REMINDER: Analysis based solely on CPT embedding similarity.")
    print("  No dx codes · No anatomical sites · No provider specialty · No documentation.")
    print_separator("═", 72)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("  MODIFIER 59 AUDIT ENGINE — EMBEDDING-DRIVEN PAYMENT INTEGRITY")
    print("  Similarity thresholds:  >0.85 BUNDLE  |  0.65-0.85 BORDERLINE  |  <0.65 DISTINCT")
    print()

    results = []
    for claim in CLAIMS:
        result = audit_claim(claim)
        results.append(result)
        print_claim(result)

    print_summary(results)

    # Also write JSON output
    with open("modifier59_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  JSON results written to: modifier59_audit_results.json\n")


if __name__ == "__main__":
    main()
