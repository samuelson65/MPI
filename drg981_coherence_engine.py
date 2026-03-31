"""
DRG 981 Family Coherence Engine
================================
Covers : DRG 981 (w/ MCC), DRG 982 (w/ CC), DRG 983 (w/o CC/MCC)
         Extensive O.R. Procedures Unrelated to Principal Diagnosis

Flag Types:
  F1 — O.R. proc unrelated to principal DX
  F2 — O.R. proc unrelated to ANY DX on claim
  F3 — CCS category mismatch with DRG assignment
  F4 — CPT not matching expected inpatient O.R. setting

Output:
  - Per-claim audit report (detailed)
  - Batch summary table (all claims)

Run:
    python drg981_coherence_engine.py
"""

import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class DRG(str, Enum):
    DRG_981 = "981"   # w/ MCC  — Most Complex, highest weight ~4.2
    DRG_982 = "982"   # w/ CC   — Moderate,     weight ~2.6
    DRG_983 = "983"   # w/o CC/MCC — Lowest,    weight ~1.8

class RiskLabel(str, Enum):
    LOW      = "LOW RISK"
    MEDIUM   = "MEDIUM RISK"
    HIGH     = "HIGH RISK"
    CRITICAL = "CRITICAL RISK"


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE: DRG WEIGHTS & EXPECTED REIMBURSEMENT
# ─────────────────────────────────────────────────────────────────────────────

DRG_CONFIG = {
    DRG.DRG_981: {
        "desc":           "Extensive O.R. Proc Unrelated to PDX w/ MCC",
        "weight":         4.1964,
        "geometric_los":  9.4,
        "requires_mcc":   True,
        "requires_cc":    False,
        "base_rate":      6000,   # approximate Medicare base rate USD
    },
    DRG.DRG_982: {
        "desc":           "Extensive O.R. Proc Unrelated to PDX w/ CC",
        "weight":         2.6342,
        "geometric_los":  6.2,
        "requires_mcc":   False,
        "requires_cc":    True,
        "base_rate":      6000,
    },
    DRG.DRG_983: {
        "desc":           "Extensive O.R. Proc Unrelated to PDX w/o CC/MCC",
        "weight":         1.8156,
        "geometric_los":  4.1,
        "requires_mcc":   False,
        "requires_cc":    False,
        "base_rate":      6000,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE: ICD-10 DX CODES
# ─────────────────────────────────────────────────────────────────────────────

ICD10_DESC = {
    # Respiratory
    "J18.9":  ("Pneumonia, unspecified",                    "pulmonology"),
    "J44.1":  ("COPD with acute exacerbation",              "pulmonology"),
    "J96.00": ("Acute respiratory failure, unspecified",    "pulmonology"),
    # Cardiac
    "I21.9":  ("Acute MI, unspecified",                     "cardiology"),
    "I25.10": ("Coronary artery disease",                   "cardiology"),
    "I50.9":  ("Heart failure, unspecified",                "cardiology"),
    "I10":    ("Essential hypertension",                    "cardiology"),
    # Orthopedic
    "M17.11": ("Primary osteoarthritis, right knee",        "orthopedics"),
    "M54.5":  ("Low back pain",                             "orthopedics"),
    "S72.001A":("Femoral neck fracture, unspecified",       "orthopedics"),
    # GI
    "K92.1":  ("Melena",                                    "gastroenterology"),
    "K57.30": ("Diverticulosis, large intestine",           "gastroenterology"),
    "K56.60": ("Unspecified intestinal obstruction",        "gastroenterology"),
    # Neurological
    "G35":    ("Multiple sclerosis",                        "neurology"),
    "I63.9":  ("Cerebral infarction, unspecified",          "neurology"),
    # Endocrine
    "E11.9":  ("Type 2 diabetes w/o complications",         "endocrinology"),
    "E11.649":("Type 2 diabetes w/ hypoglycemia",           "endocrinology"),
    # Psychiatric
    "F32.1":  ("Major depressive disorder",                 "psychiatry"),
    "F20.9":  ("Schizophrenia, unspecified",                "psychiatry"),
    # Sepsis / MCC
    "A41.9":  ("Sepsis, unspecified organism",              "infectious_disease"),
    "R65.21": ("Severe sepsis with septic shock",           "infectious_disease"),
    # Renal
    "N17.9":  ("Acute kidney failure, unspecified",         "nephrology"),
    "N18.3":  ("Chronic kidney disease, stage 3",           "nephrology"),
}

# MCC codes (Major Complication/Comorbidity) — drives DRG 981
MCC_CODES = {
    "A41.9", "R65.21", "J96.00", "I21.9", "I50.9",
    "N17.9", "G35", "I63.9", "E11.649",
}

# CC codes (Complication/Comorbidity) — drives DRG 982
CC_CODES = {
    "J44.1", "I25.10", "I10", "N18.3", "E11.9",
    "K56.60", "K92.1", "F32.1",
}


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE: CCS MAPPING
# ─────────────────────────────────────────────────────────────────────────────

CCS_MAP = {
    "J18.9":   ("122", "Pneumonia"),
    "J44.1":   ("127", "COPD"),
    "J96.00":  ("131", "Respiratory failure"),
    "I21.9":   ("100", "Acute MI"),
    "I25.10":  ("101", "Coronary artery disease"),
    "I50.9":   ("108", "Heart failure"),
    "I10":     ("098", "Essential hypertension"),
    "M17.11":  ("203", "Osteoarthritis"),
    "M54.5":   ("204", "Back problems"),
    "S72.001A":("226", "Fracture of neck of femur"),
    "K92.1":   ("149", "Biliary tract disease"),
    "K57.30":  ("145", "Intestinal obstruction"),
    "K56.60":  ("145", "Intestinal obstruction"),
    "G35":     ("079", "Multiple sclerosis"),
    "I63.9":   ("109", "Stroke"),
    "E11.9":   ("049", "Diabetes mellitus"),
    "E11.649": ("049", "Diabetes mellitus"),
    "F32.1":   ("657", "Mood disorders"),
    "F20.9":   ("659", "Schizophrenia"),
    "A41.9":   ("002", "Septicemia"),
    "R65.21":  ("002", "Septicemia"),
    "N17.9":   ("157", "Acute renal failure"),
    "N18.3":   ("158", "Chronic kidney disease"),
}

# CCS → broad MDC (Major Diagnostic Category) domain for DRG alignment
CCS_TO_MDC = {
    "122": "MDC04_respiratory",
    "127": "MDC04_respiratory",
    "131": "MDC04_respiratory",
    "100": "MDC05_circulatory",
    "101": "MDC05_circulatory",
    "108": "MDC05_circulatory",
    "098": "MDC05_circulatory",
    "203": "MDC08_musculoskeletal",
    "204": "MDC08_musculoskeletal",
    "226": "MDC08_musculoskeletal",
    "149": "MDC06_digestive",
    "145": "MDC06_digestive",
    "079": "MDC01_nervous",
    "109": "MDC01_nervous",
    "049": "MDC10_endocrine",
    "657": "MDC19_mental",
    "659": "MDC19_mental",
    "002": "MDC18_infectious",
    "157": "MDC11_renal",
    "158": "MDC11_renal",
}


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE: O.R. PROCEDURE CODES (CPT / ICD-10-PCS proxies)
# ─────────────────────────────────────────────────────────────────────────────

# CPT O.R. procedures with their clinical domain and OR-setting validity
OR_PROC_MAP = {
    # ── Orthopedic ────────────────────────────────────────────────
    "27447": {
        "desc":    "Total knee arthroplasty",
        "domain":  "MDC08_musculoskeletal",
        "or_setting": True,
        "valid_pdx_ccs": ["203", "204", "226"],
        "valid_dx": ["M17.11", "M54.5", "S72.001A"],
    },
    "27130": {
        "desc":    "Total hip arthroplasty",
        "domain":  "MDC08_musculoskeletal",
        "or_setting": True,
        "valid_pdx_ccs": ["203", "226"],
        "valid_dx": ["M17.11", "S72.001A"],
    },
    "22612": {
        "desc":    "Lumbar spinal fusion",
        "domain":  "MDC08_musculoskeletal",
        "or_setting": True,
        "valid_pdx_ccs": ["204", "203"],
        "valid_dx": ["M54.5"],
    },
    # ── Cardiac ───────────────────────────────────────────────────
    "33533": {
        "desc":    "CABG, arterial, single",
        "domain":  "MDC05_circulatory",
        "or_setting": True,
        "valid_pdx_ccs": ["100", "101"],
        "valid_dx": ["I21.9", "I25.10"],
    },
    "92928": {
        "desc":    "Percutaneous coronary intervention (PCI)",
        "domain":  "MDC05_circulatory",
        "or_setting": True,
        "valid_pdx_ccs": ["100", "101"],
        "valid_dx": ["I21.9", "I25.10"],
    },
    # ── GI / Abdominal ────────────────────────────────────────────
    "44140": {
        "desc":    "Colectomy, partial",
        "domain":  "MDC06_digestive",
        "or_setting": True,
        "valid_pdx_ccs": ["145", "149"],
        "valid_dx": ["K57.30", "K56.60"],
    },
    "43239": {
        "desc":    "Upper GI endoscopy w/ biopsy",
        "domain":  "MDC06_digestive",
        "or_setting": False,  # ← endoscopy is NOT a typical inpatient O.R. proc
        "valid_pdx_ccs": ["149"],
        "valid_dx": ["K92.1"],
    },
    # ── Respiratory ───────────────────────────────────────────────
    "32480": {
        "desc":    "Lobectomy, lung",
        "domain":  "MDC04_respiratory",
        "or_setting": True,
        "valid_pdx_ccs": ["122", "127", "131"],
        "valid_dx": ["J18.9", "J44.1"],
    },
    "31622": {
        "desc":    "Bronchoscopy, diagnostic",
        "domain":  "MDC04_respiratory",
        "or_setting": False,  # ← typically not a major O.R. setting procedure
        "valid_pdx_ccs": ["122", "127", "131"],
        "valid_dx": ["J18.9", "J44.1", "J96.00"],
    },
    # ── Neuro ─────────────────────────────────────────────────────
    "61510": {
        "desc":    "Craniotomy for brain tumor",
        "domain":  "MDC01_nervous",
        "or_setting": True,
        "valid_pdx_ccs": ["079", "109"],
        "valid_dx": ["G35", "I63.9"],
    },
    # ── Renal ─────────────────────────────────────────────────────
    "50340": {
        "desc":    "Kidney transplant",
        "domain":  "MDC11_renal",
        "or_setting": True,
        "valid_pdx_ccs": ["157", "158"],
        "valid_dx": ["N17.9", "N18.3"],
    },
    # ── Non-OR CPTs sometimes billed as OR (F4 flag targets) ─────
    "99213": {
        "desc":    "Office visit, low complexity",
        "domain":  "all",
        "or_setting": False,
        "valid_pdx_ccs": [],
        "valid_dx": [],
    },
    "93000": {
        "desc":    "ECG with interpretation",
        "domain":  "MDC05_circulatory",
        "or_setting": False,
        "valid_pdx_ccs": [],
        "valid_dx": [],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcFlag:
    proc_code:   str
    flag_type:   str          # F1 / F2 / F3 / F4
    severity:    str          # HIGH / MEDIUM
    audit_reason: str


@dataclass
class DRG981Result:
    claim_id:            str
    member_id:           str
    claim_date:          str
    assigned_drg:        DRG
    drg_desc:            str
    drg_weight:          float
    estimated_payment:   float

    principal_dx:        str
    principal_dx_desc:   str
    principal_ccs:       str
    principal_ccs_desc:  str
    principal_mdc:       str

    secondary_dx:        List[str]
    has_mcc:             bool
    has_cc:              bool
    drg_cc_mcc_valid:    bool      # does DRG tier match actual CC/MCC on claim?

    or_procedures:       List[str]

    # Per-flag results
    f1_flags:            List[ProcFlag]   # O.R. proc unrelated to PDX
    f2_flags:            List[ProcFlag]   # O.R. proc unrelated to ANY DX
    f3_flags:            List[ProcFlag]   # CCS mismatch with DRG
    f4_flags:            List[ProcFlag]   # CPT not an O.R. setting proc

    coherence_score:     float
    risk_label:          RiskLabel
    audit_summary:       str
    recommended_drg:     Optional[str]    # suggested correct DRG if mismatch found


# ─────────────────────────────────────────────────────────────────────────────
# DRG 981 COHERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DRG981CoherenceEngine:
    """
    Coherence engine scoped to DRG 981/982/983 family.

    Flag Types:
      F1 — O.R. proc clinically unrelated to principal DX
      F2 — O.R. proc unrelated to ANY DX on the claim
      F3 — CCS category of principal DX mismatches DRG MDC expectation
      F4 — CPT billed is not a true inpatient O.R. setting procedure
    """

    def _get_dx_info(self, dx: str):
        desc, domain = ICD10_DESC.get(dx, (dx, "unknown"))
        ccs_id, ccs_desc = CCS_MAP.get(dx, ("???", "Unknown"))
        mdc = CCS_TO_MDC.get(ccs_id, "MDC_unknown")
        return desc, domain, ccs_id, ccs_desc, mdc

    def _estimated_payment(self, drg: DRG) -> float:
        cfg = DRG_CONFIG[drg]
        return round(cfg["weight"] * cfg["base_rate"], 2)

    def _validate_drg_tier(
        self, drg: DRG, secondary_dx: List[str]
    ) -> tuple:
        """Check if DRG tier (981/982/983) matches CC/MCC codes on claim."""
        has_mcc = any(dx in MCC_CODES for dx in secondary_dx)
        has_cc  = any(dx in CC_CODES  for dx in secondary_dx)

        if drg == DRG.DRG_981 and not has_mcc:
            return has_mcc, has_cc, False, "DRG 981 requires MCC but no MCC found in secondary DX"
        if drg == DRG.DRG_982 and not has_cc:
            return has_mcc, has_cc, False, "DRG 982 requires CC but no CC/MCC found in secondary DX"
        return has_mcc, has_cc, True, ""

    def _suggest_drg(self, principal_mdc: str, has_mcc: bool, has_cc: bool) -> str:
        """Suggest a more appropriate DRG based on principal DX domain."""
        suggestions = {
            "MDC05_circulatory":    ("216", "217", "218"),  # cardiac valve / other cardiac
            "MDC08_musculoskeletal":("470", "471", "472"),  # joint replacement
            "MDC04_respiratory":    ("177", "178", "179"),  # respiratory infections
            "MDC06_digestive":      ("329", "330", "331"),  # major small/large bowel
            "MDC01_nervous":        ("025", "026", "027"),  # craniotomy
            "MDC11_renal":          ("652", "653", "654"),  # kidney transplant
            "MDC10_endocrine":      ("637", "638", "639"),  # diabetes
            "MDC19_mental":         ("876", "877", "885"),  # mental health O.R.
        }
        tiers = suggestions.get(principal_mdc, ("???", "???", "???"))
        if has_mcc: return f"DRG {tiers[0]} (MCC tier for {principal_mdc})"
        if has_cc:  return f"DRG {tiers[1]} (CC tier for {principal_mdc})"
        return f"DRG {tiers[2]} (base tier for {principal_mdc})"

    # ── Flag F1: O.R. proc unrelated to PRINCIPAL DX ─────────────

    def _flag_f1(self, proc_code: str, pdx: str, pdx_ccs: str) -> Optional[ProcFlag]:
        proc = OR_PROC_MAP.get(proc_code)
        if not proc:
            return None
        valid_ccs = proc["valid_pdx_ccs"]
        valid_dx  = proc["valid_dx"]
        if pdx in valid_dx or pdx_ccs in valid_ccs:
            return None  # coherent with PDX
        pdx_desc, _, _, _, _ = self._get_dx_info(pdx)
        return ProcFlag(
            proc_code    = proc_code,
            flag_type    = "F1",
            severity     = "HIGH",
            audit_reason = (
                f"F1 — O.R. PROC UNRELATED TO PRINCIPAL DX: "
                f"{proc['desc']} ({proc_code}) requires CCS {set(valid_ccs)} or "
                f"DX {set(valid_dx)}, but principal DX is {pdx} "
                f"({pdx_desc}, CCS {pdx_ccs}). "
                f"This is the core criterion for DRG 981 assignment."
            ),
        )

    # ── Flag F2: O.R. proc unrelated to ANY DX on claim ──────────

    def _flag_f2(
        self, proc_code: str, all_dx: List[str]
    ) -> Optional[ProcFlag]:
        proc = OR_PROC_MAP.get(proc_code)
        if not proc:
            return None
        valid_dx  = set(proc["valid_dx"])
        valid_ccs = set(proc["valid_pdx_ccs"])
        # Check all DX codes on claim
        for dx in all_dx:
            ccs_id, _ = CCS_MAP.get(dx, ("???", ""))
            if dx in valid_dx or ccs_id in valid_ccs:
                return None  # at least one DX justifies it
        return ProcFlag(
            proc_code    = proc_code,
            flag_type    = "F2",
            severity     = "CRITICAL",
            audit_reason = (
                f"F2 — O.R. PROC UNRELATED TO ANY DX: "
                f"{proc['desc']} ({proc_code}) cannot be clinically justified "
                f"by any of the {len(all_dx)} DX codes on this claim "
                f"({', '.join(all_dx)}). "
                f"No valid clinical indication found — strong overpayment signal."
            ),
        )

    # ── Flag F3: CCS mismatch with DRG MDC ───────────────────────

    def _flag_f3(
        self, proc_code: str, pdx_ccs: str, pdx_mdc: str
    ) -> Optional[ProcFlag]:
        proc = OR_PROC_MAP.get(proc_code)
        if not proc:
            return None
        proc_mdc = proc["domain"]
        if proc_mdc == "all" or proc_mdc == pdx_mdc:
            return None
        return ProcFlag(
            proc_code    = proc_code,
            flag_type    = "F3",
            severity     = "HIGH",
            audit_reason = (
                f"F3 — CCS/MDC MISMATCH: "
                f"{proc['desc']} ({proc_code}) belongs to {proc_mdc}, "
                f"but principal DX CCS {pdx_ccs} maps to {pdx_mdc}. "
                f"MDC mismatch confirms O.R. procedure is in a different "
                f"clinical category than the admitting diagnosis — "
                f"DRG 981 assignment may be appropriate but requires "
                f"clinical documentation review."
            ),
        )

    # ── Flag F4: CPT not an inpatient O.R. setting procedure ──────

    def _flag_f4(self, proc_code: str) -> Optional[ProcFlag]:
        proc = OR_PROC_MAP.get(proc_code)
        if not proc:
            # Unknown CPT billed as O.R. proc — flag it
            return ProcFlag(
                proc_code    = proc_code,
                flag_type    = "F4",
                severity     = "MEDIUM",
                audit_reason = (
                    f"F4 — UNKNOWN O.R. CPT: {proc_code} is not in the "
                    f"recognized inpatient O.R. procedure reference. "
                    f"Verify this CPT triggers DRG 981 assignment per "
                    f"CMS grouper logic."
                ),
            )
        if not proc["or_setting"]:
            return ProcFlag(
                proc_code    = proc_code,
                flag_type    = "F4",
                severity     = "HIGH",
                audit_reason = (
                    f"F4 — NOT AN O.R. SETTING PROCEDURE: "
                    f"{proc['desc']} ({proc_code}) is typically performed "
                    f"outside the O.R. (clinic/endoscopy suite/bedside). "
                    f"Billing this as an inpatient O.R. procedure to trigger "
                    f"DRG 981 is a known upcoding pattern."
                ),
            )
        return None

    # ── Main: Analyze One Claim ───────────────────────────────────

    def analyze(self, claim: dict) -> DRG981Result:
        """
        claim = {
            claim_id, member_id, claim_date,
            assigned_drg: "981" | "982" | "983",
            principal_dx: str,
            secondary_dx: [str],
            or_procedures: [str]   # CPT codes
        }
        """
        claim_id   = claim["claim_id"]
        member_id  = claim["member_id"]
        claim_date = claim["claim_date"]
        drg        = DRG(claim["assigned_drg"])
        pdx        = claim["principal_dx"]
        sec_dx     = claim.get("secondary_dx", [])
        all_dx     = [pdx] + sec_dx
        or_procs   = claim["or_procedures"]

        # Principal DX info
        pdx_desc, pdx_domain, pdx_ccs, pdx_ccs_desc, pdx_mdc = self._get_dx_info(pdx)

        # DRG tier validation
        has_mcc, has_cc, drg_valid, drg_tier_reason = self._validate_drg_tier(drg, sec_dx)

        # Run all 4 flag types across all O.R. procedures
        f1, f2, f3, f4 = [], [], [], []
        for proc in or_procs:
            r1 = self._flag_f1(proc, pdx, pdx_ccs)
            r2 = self._flag_f2(proc, all_dx)
            r3 = self._flag_f3(proc, pdx_ccs, pdx_mdc)
            r4 = self._flag_f4(proc)
            if r1: f1.append(r1)
            if r2: f2.append(r2)
            if r3: f3.append(r3)
            if r4: f4.append(r4)

        # DRG tier mismatch as F3 flag
        if not drg_valid:
            f3.append(ProcFlag(
                proc_code    = "DRG_TIER",
                flag_type    = "F3",
                severity     = "HIGH",
                audit_reason = f"F3 — DRG TIER MISMATCH: {drg_tier_reason}.",
            ))

        # Scoring
        total_checks  = len(or_procs) * 4 + 1  # 4 flags per proc + tier check
        total_flags   = len(f1) + len(f2) + len(f3) + len(f4) + (0 if drg_valid else 1)
        # F2 (critical) counts double
        weighted_flags = len(f1) + len(f2) * 2 + len(f3) + len(f4) + (0 if drg_valid else 1)
        score = max(0.0, round(1.0 - (weighted_flags / max(total_checks, 1)), 3))

        if score >= 0.75:   risk = RiskLabel.LOW
        elif score >= 0.50: risk = RiskLabel.MEDIUM
        elif score >= 0.25: risk = RiskLabel.HIGH
        else:               risk = RiskLabel.CRITICAL

        # Escalate if F2 (no DX justifies the proc at all)
        if f2:
            risk = RiskLabel.CRITICAL

        # Recommended DRG
        rec_drg = self._suggest_drg(pdx_mdc, has_mcc, has_cc) if (f1 or f2) else None

        # Audit summary
        all_flags = f1 + f2 + f3 + f4
        if not all_flags:
            summary = (
                f"Claim {claim_id} (Member {member_id}, {claim_date}) assigned "
                f"DRG {drg.value} passed all coherence checks. "
                f"Principal DX {pdx} ({pdx_desc}, CCS {pdx_ccs}) is consistent "
                f"with the O.R. procedures billed: {', '.join(or_procs)}. "
                f"DRG tier {'valid — MCC confirmed' if has_mcc else 'valid — CC confirmed' if has_cc else 'valid'}. "
                f"Risk: {risk.value}."
            )
        else:
            flag_texts = " | ".join([f.audit_reason for f in all_flags])
            summary = (
                f"Claim {claim_id} (Member {member_id}, {claim_date}) assigned "
                f"DRG {drg.value} has {len(all_flags)} coherence issue(s). "
                f"Principal DX: {pdx} ({pdx_desc}, CCS {pdx_ccs}, MDC: {pdx_mdc}). "
                f"O.R. Procedures billed: {', '.join(or_procs)}. "
                f"Issues: {flag_texts}. "
                + (f"Suggested correct DRG: {rec_drg}. " if rec_drg else "")
                + f"Coherence score: {score:.3f}. Risk: {risk.value}."
            )

        return DRG981Result(
            claim_id            = claim_id,
            member_id           = member_id,
            claim_date          = claim_date,
            assigned_drg        = drg,
            drg_desc            = DRG_CONFIG[drg]["desc"],
            drg_weight          = DRG_CONFIG[drg]["weight"],
            estimated_payment   = self._estimated_payment(drg),
            principal_dx        = pdx,
            principal_dx_desc   = pdx_desc,
            principal_ccs       = pdx_ccs,
            principal_ccs_desc  = pdx_ccs_desc,
            principal_mdc       = pdx_mdc,
            secondary_dx        = sec_dx,
            has_mcc             = has_mcc,
            has_cc              = has_cc,
            drg_cc_mcc_valid    = drg_valid,
            or_procedures       = or_procs,
            f1_flags            = f1,
            f2_flags            = f2,
            f3_flags            = f3,
            f4_flags            = f4,
            coherence_score     = score,
            risk_label          = risk,
            audit_summary       = summary,
            recommended_drg     = rec_drg,
        )


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT: AUDIT REPORT (per claim)
# ─────────────────────────────────────────────────────────────────────────────

def print_audit_report(r: DRG981Result):
    W = 72
    RISK_ICON = {
        RiskLabel.LOW: "🟢", RiskLabel.MEDIUM: "🟡",
        RiskLabel.HIGH: "🔴", RiskLabel.CRITICAL: "🚨",
    }
    print(f"\n{'='*W}")
    print(f"  DRG 981 FAMILY — COHERENCE AUDIT REPORT")
    print(f"  Claim: {r.claim_id}  |  Member: {r.member_id}  |  Date: {r.claim_date}")
    print(f"{'='*W}")

    print(f"\n  Assigned DRG  : {r.assigned_drg.value} — {r.drg_desc}")
    print(f"  DRG Weight    : {r.drg_weight}   Est. Payment: ${r.estimated_payment:,.2f}")
    cc_status = "✅ MCC confirmed" if r.has_mcc else ("✅ CC confirmed" if r.has_cc else "❌ No CC/MCC found")
    print(f"  CC/MCC Status : {cc_status}  |  DRG Tier Valid: {'✅' if r.drg_cc_mcc_valid else '❌'}")

    print(f"\n  ── Principal Diagnosis ──────────────────────────────────────")
    print(f"  Code : {r.principal_dx} — {r.principal_dx_desc}")
    print(f"  CCS  : {r.principal_ccs} — {r.principal_ccs_desc}")
    print(f"  MDC  : {r.principal_mdc}")

    if r.secondary_dx:
        print(f"\n  ── Secondary DX (CC/MCC) ────────────────────────────────────")
        for dx in r.secondary_dx:
            desc, _, ccs_id, ccs_desc, _ = (
                ICD10_DESC.get(dx, (dx, ""))[0],
                "", *CCS_MAP.get(dx, ("???","Unknown")), ""
            )
            mcc_tag = " [MCC]" if dx in MCC_CODES else (" [CC]" if dx in CC_CODES else "")
            print(f"  • {dx} — {ICD10_DESC.get(dx,('?',''))[0]}{mcc_tag}")

    print(f"\n  ── O.R. Procedures Billed ───────────────────────────────────")
    for proc in r.or_procedures:
        info     = OR_PROC_MAP.get(proc, {})
        desc     = info.get("desc", "Unknown procedure")
        or_flag  = "✅ O.R." if info.get("or_setting") else "⚠️  Non-O.R."
        print(f"  • {proc} — {desc}  [{or_flag}]")

    def print_flags(flags, label):
        if not flags:
            return
        print(f"\n  ── {label} ──────────────────────────────────────────────────")
        for f in flags:
            sev_icon = "🚨" if f.severity == "CRITICAL" else "🔴"
            print(f"  {sev_icon} [{f.flag_type}] {f.proc_code}")
            # word wrap reason
            words, line = f.audit_reason.split(), ""
            for w in words:
                if len(line) + len(w) + 1 > 64:
                    print(f"       {line}")
                    line = w
                else:
                    line = f"{line} {w}".strip()
            if line: print(f"       {line}")

    print_flags(r.f1_flags, "F1: O.R. Proc Unrelated to Principal DX")
    print_flags(r.f2_flags, "F2: O.R. Proc Unrelated to ANY DX")
    print_flags(r.f3_flags, "F3: CCS / MDC Mismatch")
    print_flags(r.f4_flags, "F4: CPT Not an O.R. Setting Procedure")

    if r.recommended_drg:
        print(f"\n  💡 Recommended DRG : {r.recommended_drg}")

    print(f"\n  ── Audit Summary ────────────────────────────────────────────")
    words, line = r.audit_summary.split(), ""
    for w in words:
        if len(line) + len(w) + 1 > 66:
            print(f"  {line}")
            line = w
        else:
            line = f"{line} {w}".strip()
    if line: print(f"  {line}")

    total_flags = len(r.f1_flags)+len(r.f2_flags)+len(r.f3_flags)+len(r.f4_flags)
    print(f"\n  Coherence Score: {r.coherence_score:.3f}  |  "
          f"Total Flags: {total_flags}  |  "
          f"{RISK_ICON[r.risk_label]} {r.risk_label.value}")
    print(f"{'='*W}")


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT: BATCH SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_batch_summary(results: List[DRG981Result]):
    rows = []
    for r in results:
        rows.append({
            "claim_id":     r.claim_id,
            "member_id":    r.member_id,
            "drg":          r.assigned_drg.value,
            "pdx":          r.principal_dx,
            "ccs":          r.principal_ccs,
            "mdc":          r.principal_mdc.split("_")[0],
            "or_procs":     ", ".join(r.or_procedures),
            "F1":           len(r.f1_flags),
            "F2":           len(r.f2_flags),
            "F3":           len(r.f3_flags),
            "F4":           len(r.f4_flags),
            "score":        r.coherence_score,
            "risk":         r.risk_label.value,
            "est_payment":  f"${r.estimated_payment:,.0f}",
            "rec_drg":      r.recommended_drg or "—",
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*100}")
    print(f"  DRG 981 FAMILY — BATCH SUMMARY  ({len(results)} claims)")
    print(f"{'='*100}")
    print(df.to_string(index=False))

    print(f"\n  ── Risk Distribution ─────────────────────────────")
    print(df["risk"].value_counts().to_string())
    print(f"\n  ── Flag Counts ───────────────────────────────────")
    print(f"  F1 (PDX mismatch)  : {df['F1'].sum()}")
    print(f"  F2 (No DX match)   : {df['F2'].sum()}")
    print(f"  F3 (CCS/MDC/Tier)  : {df['F3'].sum()}")
    print(f"  F4 (Non-O.R. CPT)  : {df['F4'].sum()}")
    print(f"{'='*100}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    engine = DRG981CoherenceEngine()

    test_claims = [
        # Case 1: CRITICAL — knee surgery billed under pneumonia PDX, no ortho DX anywhere
        {
            "claim_id":     "CLM100001",
            "member_id":    "MBR00042",
            "claim_date":   "2024-03-15",
            "assigned_drg": "981",
            "principal_dx": "J18.9",          # Pneumonia
            "secondary_dx": ["A41.9", "I10"], # Sepsis (MCC) + HTN
            "or_procedures": ["27447"],        # ❌ Knee replacement — no ortho DX
        },
        # Case 2: HIGH — cardiac PDX but ortho O.R. proc + non-OR CPT billed
        {
            "claim_id":     "CLM100002",
            "member_id":    "MBR00055",
            "claim_date":   "2024-04-02",
            "assigned_drg": "982",
            "principal_dx": "I21.9",           # Acute MI
            "secondary_dx": ["I10", "E11.9"],  # HTN (CC) + Diabetes
            "or_procedures": ["92928", "27130", "43239"],
            # 92928 PCI ✅, 27130 hip replacement ❌, 43239 endoscopy ❌ non-OR
        },
        # Case 3: HIGH — DRG tier mismatch (983 assigned but no MCC/CC at all)
        {
            "claim_id":     "CLM100003",
            "member_id":    "MBR00071",
            "claim_date":   "2024-05-20",
            "assigned_drg": "982",             # Claims CC tier
            "principal_dx": "M17.11",          # Knee OA
            "secondary_dx": ["M54.5"],         # Back pain — NOT a CC
            "or_procedures": ["27447"],        # ✅ TKA matches PDX
        },
        # Case 4: MEDIUM — GI proc under respiratory PDX, but secondary GI DX present
        {
            "claim_id":     "CLM100004",
            "member_id":    "MBR00088",
            "claim_date":   "2024-06-10",
            "assigned_drg": "983",
            "principal_dx": "J44.1",           # COPD
            "secondary_dx": ["K57.30"],        # Diverticulosis
            "or_procedures": ["32480", "44140"],
            # 32480 lobectomy ✅ matches COPD, 44140 colectomy — justified by K57.30 secondary
        },
        # Case 5: LOW RISK — clean claim, proc matches PDX perfectly
        {
            "claim_id":     "CLM100005",
            "member_id":    "MBR00099",
            "claim_date":   "2024-07-01",
            "assigned_drg": "981",
            "principal_dx": "I25.10",          # CAD
            "secondary_dx": ["N17.9", "I10"],  # AKF (MCC) + HTN
            "or_procedures": ["33533"],        # ✅ CABG — matches CAD perfectly
        },
    ]

    results = [engine.analyze(c) for c in test_claims]

    # Per-claim audit reports
    for r in results:
        print_audit_report(r)

    # Batch summary table
    print_batch_summary(results)


if __name__ == "__main__":
    from typing import List
    run_demo()
