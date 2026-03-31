"""
Clinical Coherence Engine — Production Grade
=============================================
Scope:
  - Code Types  : ICD-10 DX, CPT, HCPCS, CCS categories
  - Rules       : DX-Procedure domain matching, Explicit valid/invalid pairs,
                  Global period violations
  - Output      : Claim-level risk label + explainable audit reason

Run:
    python coherence_engine.py
"""

import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class RiskLabel(str, Enum):
    LOW      = "LOW RISK"
    MEDIUM   = "MEDIUM RISK"
    HIGH     = "HIGH RISK"
    CRITICAL = "CRITICAL RISK"


class CoherenceLabel(str, Enum):
    COHERENT   = "COHERENT"
    INCOHERENT = "INCOHERENT"
    WARNING    = "WARNING"


@dataclass
class ProcedureResult:
    proc_code:        str
    code_type:        str           # CPT or HCPCS
    units:            int
    coherence_label:  CoherenceLabel
    coherence_score:  float         # 0.0 to 1.0
    rule_triggered:   str           # which layer caught this
    audit_reason:     str           # plain English explanation
    matched_dx:       List[str]     # DX codes that justify this procedure
    flag:             str           # short flag for the report


@dataclass
class ClaimResult:
    claim_id:               str
    member_id:              str
    claim_date:             str
    dx_codes:               List[str]
    dx_domains:             List[str]
    total_procedures:       int
    coherent_count:         int
    incoherent_count:       int
    warning_count:          int
    claim_coherence_score:  float
    risk_label:             RiskLabel
    audit_summary:          str           # one paragraph plain English summary
    procedure_results:      List[ProcedureResult]
    global_period_flags:    List[str]     # any global period violations found


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────

ICD10_DESC = {
    "E11.9":  "Type 2 diabetes w/o complications",
    "I10":    "Essential hypertension",
    "J18.9":  "Pneumonia, unspecified",
    "M54.5":  "Low back pain",
    "F32.1":  "Major depressive disorder",
    "N18.3":  "Chronic kidney disease stage 3",
    "I25.10": "Coronary artery disease",
    "Z79.4":  "Long-term insulin use",
    "E78.5":  "Hyperlipidemia",
    "G89.29": "Chronic pain",
    "M17.11": "Primary osteoarthritis, right knee",
    "Z87.39": "Personal history of musculoskeletal disorder",
    "J44.1":  "COPD with acute exacerbation",
    "F41.1":  "Generalized anxiety disorder",
    "K21.0":  "GERD with esophagitis",
}

CPT_DESC = {
    "99213": "Office visit, est patient, low complexity",
    "99214": "Office visit, est patient, mod complexity",
    "99215": "Office visit, est patient, high complexity",
    "99232": "Subsequent hospital care",
    "93000": "ECG with interpretation",
    "85025": "Complete blood count (CBC)",
    "80053": "Comprehensive metabolic panel",
    "71046": "Chest X-ray, 2 views",
    "27447": "Total knee arthroplasty",
    "27310": "Arthrotomy, knee",
    "29881": "Knee arthroscopy w/ meniscectomy",
    "90837": "Psychotherapy, 60 min",
    "90834": "Psychotherapy, 45 min",
    "43239": "Upper GI endoscopy with biopsy",
    "94640": "Nebulizer treatment",
}

HCPCS_DESC = {
    "A4253": "Blood glucose test strips",
    "E0601": "CPAP device",
    "J1040": "Methylprednisolone injection",
    "G0439": "Annual wellness visit",
    "G2212": "Prolonged office visit add-on",
    "J7644": "Ipratropium bromide, inhalation solution",
}

# CCS: DX code → (ccs_id, clinical domain)
CCS_MAP = {
    "E11.9":  ("049", "endocrinology"),
    "I10":    ("098", "cardiology"),
    "J18.9":  ("122", "pulmonology"),
    "M54.5":  ("204", "orthopedics"),
    "F32.1":  ("657", "psychiatry"),
    "N18.3":  ("158", "nephrology"),
    "I25.10": ("101", "cardiology"),
    "Z79.4":  ("660", "endocrinology"),
    "E78.5":  ("053", "endocrinology"),
    "G89.29": ("203", "orthopedics"),
    "M17.11": ("203", "orthopedics"),
    "Z87.39": ("212", "orthopedics"),
    "J44.1":  ("127", "pulmonology"),
    "F41.1":  ("651", "psychiatry"),
    "K21.0":  ("138", "gastroenterology"),
}

# Layer 1: Procedure → allowed clinical domains
PROC_DOMAIN_MAP = {
    # Office visits — valid in all domains
    "99213": ["all"],
    "99214": ["all"],
    "99215": ["all"],
    "99232": ["pulmonology", "cardiology", "nephrology", "endocrinology", "gastroenterology"],
    "G0439": ["all"],
    "G2212": ["all"],
    # Cardiac
    "93000": ["cardiology", "nephrology"],
    # Labs — valid across all domains
    "85025": ["all"],
    "80053": ["all"],
    # Radiology
    "71046": ["pulmonology", "cardiology"],
    # Orthopedic surgery
    "27447": ["orthopedics"],
    "27310": ["orthopedics"],
    "29881": ["orthopedics"],
    # Psychiatry
    "90837": ["psychiatry"],
    "90834": ["psychiatry"],
    # GI
    "43239": ["gastroenterology"],
    # Pulmonology
    "94640": ["pulmonology"],
    "E0601": ["pulmonology"],
    "J7644": ["pulmonology"],
    # Endocrinology / diabetes
    "A4253": ["endocrinology"],
    # Injection — broad use
    "J1040": ["orthopedics", "pulmonology", "cardiology", "nephrology"],
}

# Layer 2a: Explicit VALID DX-procedure pairs (clinical whitelist)
# Procedure → list of DX codes that clinically justify it
EXPLICIT_VALID_PAIRS = {
    "27447": ["M17.11", "M54.5", "Z87.39", "G89.29"],
    "29881": ["M17.11", "M54.5", "Z87.39", "G89.29"],
    "27310": ["M17.11", "M54.5"],
    "93000": ["I25.10", "I10", "N18.3"],
    "E0601": ["J18.9", "J44.1"],
    "90837": ["F32.1", "F41.1"],
    "90834": ["F32.1", "F41.1"],
    "A4253": ["E11.9", "Z79.4"],
    "94640": ["J18.9", "J44.1"],
    "J7644": ["J18.9", "J44.1"],
    "43239": ["K21.0"],
    "71046": ["J18.9", "J44.1", "I25.10"],
}

# Layer 2b: Explicit INVALID DX-procedure pairs (known billing error patterns)
# Procedure → DX codes that make this claim suspicious
EXPLICIT_INVALID_PAIRS = {
    # Knee surgery under purely psychiatric diagnosis
    "27447": {
        "invalid_dx":  ["F32.1", "F41.1"],
        "reason":      "Knee arthroplasty billed with psychiatric-only diagnosis — no orthopedic indication",
    },
    "29881": {
        "invalid_dx":  ["F32.1", "F41.1"],
        "reason":      "Knee arthroscopy billed with psychiatric-only diagnosis",
    },
    # Psychotherapy under purely orthopedic or cardiac diagnosis
    "90837": {
        "invalid_dx":  ["M17.11", "M54.5", "I25.10", "I10", "E11.9"],
        "reason":      "Psychotherapy billed with no psychiatric diagnosis on claim",
    },
    "90834": {
        "invalid_dx":  ["M17.11", "M54.5", "I25.10"],
        "reason":      "Psychotherapy billed with no psychiatric diagnosis on claim",
    },
    # CPAP under non-respiratory diagnosis
    "E0601": {
        "invalid_dx":  ["E11.9", "I10", "F32.1", "M17.11"],
        "reason":      "CPAP device billed with no respiratory diagnosis on claim",
    },
    # GI endoscopy under non-GI diagnosis
    "43239": {
        "invalid_dx":  ["I10", "F32.1", "M17.11", "E11.9"],
        "reason":      "Upper GI endoscopy billed with no GI diagnosis",
    },
}

# Layer 3: Global period rules
# Procedure → global period in days + procedures covered in that period
GLOBAL_PERIOD_RULES = {
    "27447": {
        "period_days": 90,
        "covered_codes": ["99213", "99214", "99215", "99232", "29881", "27310"],
        "description": "Total knee arthroplasty has a 90-day global period",
    },
    "29881": {
        "period_days": 90,
        "covered_codes": ["99213", "99214", "99215", "99232"],
        "description": "Knee arthroscopy has a 90-day global period",
    },
    "43239": {
        "period_days": 10,
        "covered_codes": ["99213", "99214"],
        "description": "Upper GI endoscopy has a 10-day global period",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# COHERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalCoherenceEngine:
    """
    Production-grade coherence engine.

    Layers:
      1. CCS domain matching  — DX domains vs procedure allowed domains
      2a. Explicit valid pairs — whitelist of known valid DX-procedure combos
      2b. Explicit invalid pairs — blacklist of known billing error patterns
      3. Global period         — detects procedures billed inside surgery windows
    """

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _dx_domains(self, dx_codes: List[str]) -> set:
        return {CCS_MAP[dx][1] for dx in dx_codes if dx in CCS_MAP}

    def _risk_label(self, score: float) -> RiskLabel:
        if score >= 0.75:   return RiskLabel.LOW
        if score >= 0.50:   return RiskLabel.MEDIUM
        if score >= 0.25:   return RiskLabel.HIGH
        return RiskLabel.CRITICAL

    def _proc_desc(self, code: str) -> str:
        return CPT_DESC.get(code) or HCPCS_DESC.get(code) or code

    def _dx_desc(self, code: str) -> str:
        return ICD10_DESC.get(code, code)

    # ── Layer 1: Domain Matching ──────────────────────────────────────────────

    def _check_domain(self, proc_code: str, dx_codes: List[str]) -> Optional[ProcedureResult]:
        allowed  = PROC_DOMAIN_MAP.get(proc_code, ["all"])
        domains  = self._dx_domains(dx_codes)

        if "all" in allowed:
            return ProcedureResult(
                proc_code       = proc_code,
                code_type       = "CPT" if proc_code in CPT_DESC else "HCPCS",
                units           = 1,
                coherence_label = CoherenceLabel.COHERENT,
                coherence_score = 1.0,
                rule_triggered  = "Layer 1 — Domain match (universal)",
                audit_reason    = f"{self._proc_desc(proc_code)} is clinically appropriate across all domains.",
                matched_dx      = dx_codes,
                flag            = "",
            )

        overlap = domains.intersection(set(allowed))
        if overlap:
            matched = [dx for dx in dx_codes if CCS_MAP.get(dx, (None,None))[1] in overlap]
            return ProcedureResult(
                proc_code       = proc_code,
                code_type       = "CPT" if proc_code in CPT_DESC else "HCPCS",
                units           = 1,
                coherence_label = CoherenceLabel.COHERENT,
                coherence_score = round(len(overlap) / len(allowed), 2),
                rule_triggered  = "Layer 1 — Domain match",
                audit_reason    = (
                    f"{self._proc_desc(proc_code)} is clinically appropriate. "
                    f"Procedure requires {set(allowed)} domain(s); claim has {overlap} "
                    f"via DX: {', '.join(matched)}."
                ),
                matched_dx      = matched,
                flag            = "",
            )
        return None  # no domain match — continue to next layer

    # ── Layer 2a: Explicit Valid Pairs ────────────────────────────────────────

    def _check_valid_pairs(self, proc_code: str, dx_codes: List[str]) -> Optional[ProcedureResult]:
        valid_dx = EXPLICIT_VALID_PAIRS.get(proc_code, [])
        matched  = [dx for dx in dx_codes if dx in valid_dx]
        if matched:
            return ProcedureResult(
                proc_code       = proc_code,
                code_type       = "CPT" if proc_code in CPT_DESC else "HCPCS",
                units           = 1,
                coherence_label = CoherenceLabel.COHERENT,
                coherence_score = 1.0,
                rule_triggered  = "Layer 2a — Explicit valid pair",
                audit_reason    = (
                    f"{self._proc_desc(proc_code)} is clinically justified by "
                    f"{', '.join([f'{dx} ({self._dx_desc(dx)})' for dx in matched])}."
                ),
                matched_dx      = matched,
                flag            = "",
            )
        return None

    # ── Layer 2b: Explicit Invalid Pairs ─────────────────────────────────────

    def _check_invalid_pairs(self, proc_code: str, dx_codes: List[str]) -> Optional[ProcedureResult]:
        rule = EXPLICIT_INVALID_PAIRS.get(proc_code)
        if not rule:
            return None

        invalid_dx  = rule["invalid_dx"]
        # Only fire if ALL dx codes on the claim are in the invalid set
        # (a valid DX elsewhere on the claim may still justify the procedure)
        non_invalid = [dx for dx in dx_codes if dx not in invalid_dx]
        if not non_invalid:
            return ProcedureResult(
                proc_code       = proc_code,
                code_type       = "CPT" if proc_code in CPT_DESC else "HCPCS",
                units           = 1,
                coherence_label = CoherenceLabel.INCOHERENT,
                coherence_score = 0.0,
                rule_triggered  = "Layer 2b — Explicit invalid pair",
                audit_reason    = (
                    f"BILLING ERROR: {rule['reason']} "
                    f"DX on claim: {', '.join([f'{dx} ({self._dx_desc(dx)})' for dx in dx_codes])}."
                ),
                matched_dx      = [],
                flag            = "⚠️ Known billing error pattern",
            )
        return None

    # ── Layer 3: Global Period Violation ─────────────────────────────────────

    def _check_global_period(
        self,
        proc_code: str,
        claim_date: datetime,
        member_history: List[dict],
    ) -> List[str]:
        """
        Check if proc_code falls inside the global period of a prior surgery.
        member_history: list of prior claims [{claim_date, proc_codes: [str]}]
        Returns list of violation strings (empty if none).
        """
        violations = []
        for prior in member_history:
            prior_date  = prior["claim_date"]
            prior_procs = prior["proc_codes"]
            for surgery_code, rule in GLOBAL_PERIOD_RULES.items():
                if surgery_code not in prior_procs:
                    continue
                days_since = (claim_date - prior_date).days
                if 0 < days_since <= rule["period_days"]:
                    if proc_code in rule["covered_codes"]:
                        violations.append(
                            f"GLOBAL PERIOD VIOLATION: {self._proc_desc(proc_code)} ({proc_code}) "
                            f"billed {days_since} days after {self._proc_desc(surgery_code)} ({surgery_code}) "
                            f"on {prior_date.strftime('%Y-%m-%d')}. "
                            f"{rule['description']} — follow-up visits are already included in the surgical fee."
                        )
        return violations

    # ── Main: Check One Procedure ─────────────────────────────────────────────

    def check_procedure(
        self,
        proc_code: str,
        dx_codes: List[str],
        units: int = 1,
        claim_date: Optional[datetime] = None,
        member_history: Optional[List[dict]] = None,
    ) -> ProcedureResult:
        """
        Run all layers for one procedure against the claim's DX codes.
        Priority: 2b (invalid) → 2a (valid) → 1 (domain) → fallback (incoherent)
        """
        # Layer 2b first — known bad patterns take priority
        result = self._check_invalid_pairs(proc_code, dx_codes)
        if result:
            result.units = units
            return result

        # Layer 2a — known good pairs
        result = self._check_valid_pairs(proc_code, dx_codes)
        if result:
            result.units = units
            return result

        # Layer 1 — domain matching
        result = self._check_domain(proc_code, dx_codes)
        if result:
            result.units = units
            return result

        # Fallback — no layer justified this procedure
        domains  = self._dx_domains(dx_codes)
        allowed  = PROC_DOMAIN_MAP.get(proc_code, ["unknown"])
        return ProcedureResult(
            proc_code       = proc_code,
            code_type       = "CPT" if proc_code in CPT_DESC else "HCPCS",
            units           = units,
            coherence_label = CoherenceLabel.INCOHERENT,
            coherence_score = 0.0,
            rule_triggered  = "Fallback — no rule matched",
            audit_reason    = (
                f"NO CLINICAL JUSTIFICATION: {self._proc_desc(proc_code)} requires "
                f"{set(allowed)} domain(s), but claim DX codes only cover "
                f"{domains if domains else 'unknown'} domain(s). "
                f"DX on claim: {', '.join([f'{dx} ({self._dx_desc(dx)})' for dx in dx_codes])}."
            ),
            matched_dx      = [],
            flag            = "⚠️ No clinical justification found",
        )

    # ── Main: Check Full Claim ────────────────────────────────────────────────

    def check_claim(
        self,
        claim: dict,
        member_history: Optional[List[dict]] = None,
    ) -> ClaimResult:
        """
        Run all coherence layers on a full claim.

        claim = {
            claim_id, member_id, claim_date,
            dx_codes: [str],
            proc_codes: [{code, type, units}]
        }
        member_history = [
            {claim_date: datetime, proc_codes: [str]}
        ]
        """
        dx_codes    = claim["dx_codes"]
        claim_date  = claim["claim_date"]
        if isinstance(claim_date, str):
            claim_date = datetime.strptime(claim_date, "%Y-%m-%d")

        # ── Per-procedure checks ──────────────────────────────────
        proc_results = []
        for item in claim["proc_codes"]:
            result = self.check_procedure(
                proc_code      = item["code"],
                dx_codes       = dx_codes,
                units          = item.get("units", 1),
                claim_date     = claim_date,
                member_history = member_history,
            )
            proc_results.append(result)

        # ── Global period check across all procedures ─────────────
        gp_flags = []
        if member_history:
            for item in claim["proc_codes"]:
                gp_flags.extend(
                    self._check_global_period(item["code"], claim_date, member_history)
                )

        # ── Claim-level scoring ───────────────────────────────────
        scores          = [r.coherence_score for r in proc_results]
        claim_score     = round(sum(scores) / len(scores), 3) if scores else 0.0
        coherent_count  = sum(1 for r in proc_results if r.coherence_label == CoherenceLabel.COHERENT)
        incoherent_count= sum(1 for r in proc_results if r.coherence_label == CoherenceLabel.INCOHERENT)
        warning_count   = sum(1 for r in proc_results if r.coherence_label == CoherenceLabel.WARNING)
        risk_label      = self._risk_label(claim_score)

        # ── Adjust risk upward if global period violations found ──
        if gp_flags:
            if risk_label == RiskLabel.LOW:    risk_label = RiskLabel.MEDIUM
            elif risk_label == RiskLabel.MEDIUM: risk_label = RiskLabel.HIGH

        # ── Audit summary (plain English paragraph) ───────────────
        dx_domain_list  = list(self._dx_domains(dx_codes))
        flagged_procs   = [r for r in proc_results if r.coherence_label == CoherenceLabel.INCOHERENT]

        if not flagged_procs and not gp_flags:
            audit_summary = (
                f"Claim {claim['claim_id']} for member {claim['member_id']} on "
                f"{claim_date.strftime('%Y-%m-%d')} passed all coherence checks. "
                f"All {len(proc_results)} procedure(s) are clinically justified by the "
                f"diagnosis codes on the claim ({', '.join(dx_codes)}). "
                f"Risk level: {risk_label.value}."
            )
        else:
            issues = []
            for r in flagged_procs:
                issues.append(r.audit_reason)
            for gp in gp_flags:
                issues.append(gp)

            audit_summary = (
                f"Claim {claim['claim_id']} for member {claim['member_id']} on "
                f"{claim_date.strftime('%Y-%m-%d')} has {len(issues)} coherence issue(s). "
                f"Clinical domains on claim: {dx_domain_list}. "
                f"Issues found: " + " | ".join(issues) +
                f" Overall coherence score: {claim_score:.2f}. "
                f"Risk level: {risk_label.value}."
            )

        return ClaimResult(
            claim_id              = claim["claim_id"],
            member_id             = claim["member_id"],
            claim_date            = claim_date.strftime("%Y-%m-%d"),
            dx_codes              = dx_codes,
            dx_domains            = dx_domain_list,
            total_procedures      = len(proc_results),
            coherent_count        = coherent_count,
            incoherent_count      = incoherent_count,
            warning_count         = warning_count,
            claim_coherence_score = claim_score,
            risk_label            = risk_label,
            audit_summary         = audit_summary,
            procedure_results     = proc_results,
            global_period_flags   = gp_flags,
        )


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_claim_report(result: ClaimResult):
    W = 70
    print("\n" + "=" * W)
    print(f"  COHERENCE AUDIT REPORT")
    print(f"  Claim: {result.claim_id}  |  Member: {result.member_id}  |  Date: {result.claim_date}")
    print("=" * W)

    print(f"\n  DX Codes on Claim:")
    for dx in result.dx_codes:
        domain = CCS_MAP.get(dx, (None, "unknown"))[1]
        print(f"    • {dx} — {ICD10_DESC.get(dx, dx)}  [{domain}]")

    print(f"\n  Clinical Domains Detected : {result.dx_domains}")
    print(f"  Claim Coherence Score     : {result.claim_coherence_score:.3f}")
    print(f"  Risk Label                : {result.risk_label.value}")

    print(f"\n  {'─'*W}")
    print(f"  PROCEDURE BREAKDOWN")
    print(f"  {'─'*W}")
    print(f"  {'Code':<8} {'Type':<6} {'Units':<6} {'Result':<12} {'Layer':<30} Matched DX")
    print(f"  {'─'*W}")

    for r in result.procedure_results:
        icon  = "✅" if r.coherence_label == CoherenceLabel.COHERENT else (
                "⚠️ " if r.coherence_label == CoherenceLabel.WARNING else "❌")
        layer = r.rule_triggered[:28]
        matched = ", ".join(r.matched_dx) if r.matched_dx else "—"
        print(f"  {r.proc_code:<8} {r.code_type:<6} {r.units:<6} {icon} {r.coherence_label.value:<10} {layer:<30} {matched}")
        if r.flag:
            print(f"           {r.flag}")

    if result.global_period_flags:
        print(f"\n  {'─'*W}")
        print(f"  GLOBAL PERIOD VIOLATIONS")
        print(f"  {'─'*W}")
        for gp in result.global_period_flags:
            print(f"  🚫 {gp}")

    print(f"\n  {'─'*W}")
    print(f"  AUDIT SUMMARY")
    print(f"  {'─'*W}")
    # Word-wrap the summary
    words, line = result.audit_summary.split(), ""
    for word in words:
        if len(line) + len(word) + 1 > 66:
            print(f"  {line}")
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        print(f"  {line}")

    print(f"\n  Coherent: {result.coherent_count}  |  "
          f"Incoherent: {result.incoherent_count}  |  "
          f"Global Period Flags: {len(result.global_period_flags)}")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    engine = ClinicalCoherenceEngine()

    # ── Case 1: Mixed claim — some coherent, some not ─────────────
    claim_1 = {
        "claim_id":  "CLM998877",
        "member_id": "MBR00042",
        "claim_date": "2024-03-15",
        "dx_codes":  ["I25.10", "E11.9", "E78.5", "I10"],
        "proc_codes": [
            {"code": "93000", "type": "CPT",   "units": 1},
            {"code": "80053", "type": "CPT",   "units": 1},
            {"code": "27447", "type": "CPT",   "units": 1},   # ❌ no ortho DX
            {"code": "A4253", "type": "HCPCS", "units": 3},
            {"code": "90837", "type": "CPT",   "units": 2},   # ❌ no psych DX
        ],
    }

    # ── Case 2: Global period violation ───────────────────────────
    claim_2 = {
        "claim_id":  "CLM112233",
        "member_id": "MBR00099",
        "claim_date": "2024-05-10",
        "dx_codes":  ["M17.11", "G89.29"],
        "proc_codes": [
            {"code": "99214", "type": "CPT", "units": 1},   # ❌ in global period
            {"code": "27310", "type": "CPT", "units": 1},   # ❌ in global period
        ],
    }

    # Member had knee surgery 30 days ago
    member_history = [
        {
            "claim_date": datetime(2024, 4, 10),
            "proc_codes": ["27447"],
        }
    ]

    # ── Case 3: Clean claim — all coherent ────────────────────────
    claim_3 = {
        "claim_id":  "CLM445566",
        "member_id": "MBR00077",
        "claim_date": "2024-06-01",
        "dx_codes":  ["F32.1", "F41.1"],
        "proc_codes": [
            {"code": "90837", "type": "CPT", "units": 1},
            {"code": "99214", "type": "CPT", "units": 1},
        ],
    }

    print_claim_report(engine.check_claim(claim_1))
    print_claim_report(engine.check_claim(claim_2, member_history=member_history))
    print_claim_report(engine.check_claim(claim_3))


if __name__ == "__main__":
    run_demo()
