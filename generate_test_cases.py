#!/usr/bin/env python3
"""
Synthetic Inpatient Claim Test Case Generator
==============================================
Generates 20 structured test cases for validating the 12-step
dimensional payment integrity audit system.

Outputs:
  - test_cases.json       : Full structured test cases
  - test_cases_batch.csv  : Batch-ready input for dim_auditor.py
  - test_cases_report.txt : Human-readable test case documentation

Run:
  python generate_test_cases.py
  python generate_test_cases.py --run-audit   # auto-run through dim_auditor
"""

import json, csv, os, sys, argparse, textwrap

# ── Test Case Catalog ──────────────────────────────────────────────────────────
# Each test case follows the full schema:
# id, name, category, scenario, targeted_axes, dx_codes, proc_codes,
# los, drg, mcc_count, cc_count,
# expected_flag, expected_risk, expected_root_cause,
# explanation_guidance, notes

TEST_CASES = [

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 1: VALID CONTROL CASES (3 cases)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-001",
        "name": "AMI with PCI — Textbook Inpatient",
        "category": "Valid Control",
        "scenario": (
            "62-year-old male admitted for anterior STEMI. Treated with emergent "
            "percutaneous coronary intervention (PCI). Comorbid hypertension and "
            "type 2 diabetes without complications. 3-day LOS. "
            "This is the canonical high-acuity cardiac inpatient case."
        ),
        "targeted_axes": ["0–9 (clinical coherence)", "10–19 (severity justified)", 
                           "70–79 (DX-Proc link strong)", "90–99 (DRG weight appropriate)"],
        "input": {
            "claim_id":  "TC-001",
            "dx_codes":  ["I21.01", "I10", "E11.9", "Z79.01"],
            "proc_codes": ["02703ZZ", "B2041ZZ"],
            "los": 3, "drg": "246", "mcc_count": 1, "cc_count": 2
        },
        "expected_flag": False,
        "expected_risk": "LOW",
        "expected_root_cause": "None — valid, coherent acute cardiac claim",
        "explanation_guidance": (
            "Dims 0–9: All codes in Cardiovascular domain. "
            "Dims 10–19: I21.01 (STEMI) is a legitimate MCC; I10 and E11.9 are valid CCs. "
            "Dims 70–79: PCI (02703ZZ) is directly medically necessary for STEMI. "
            "Dims 90–99: DRG 246 financial weight is appropriate for this acuity. "
            "Expected: ALL steps pass, composite score < 25."
        ),
        "notes": "Baseline for calibration. Any system flagging this is over-sensitive."
    },

    {
        "id": "TC-002",
        "name": "Hip Fracture Repair — Legitimate Orthopedic Inpatient",
        "category": "Valid Control",
        "scenario": (
            "78-year-old female admitted after fall with right hip fracture. "
            "Underwent open reduction internal fixation (ORIF) next day. "
            "Comorbid osteoporosis and essential hypertension. "
            "5-day LOS. Routine post-surgical recovery."
        ),
        "targeted_axes": ["0–9 (ortho domain)", "30–39 (anatomical alignment)", 
                           "40–49 (acute episode)", "70–79 (DX-Proc justified)"],
        "input": {
            "claim_id":  "TC-002",
            "dx_codes":  ["S72.001A", "M81.0", "I10", "W19.XXXA"],
            "proc_codes": ["0QS604Z", "0QR60JZ"],
            "los": 5, "drg": "481", "mcc_count": 0, "cc_count": 2
        },
        "expected_flag": False,
        "expected_risk": "LOW",
        "expected_root_cause": "None — valid orthopedic trauma case",
        "explanation_guidance": (
            "Dims 0–9: DX and procedures aligned in Orthopedic/MSK domain. "
            "Dims 30–39: Both DX (lower extremity) and procedure (lower bone repair) site-matched. "
            "Dims 40–49: Acute trauma episode type consistent across DX and proc. "
            "Dims 70–79: ORIF directly supports hip fracture diagnosis. "
            "Expected: Score < 20."
        ),
        "notes": "Good test for anatomical coherence step. Should pass Step 4 cleanly."
    },

    {
        "id": "TC-003",
        "name": "Sepsis with Mechanical Ventilation — High Acuity Valid",
        "category": "Valid Control",
        "scenario": (
            "55-year-old admitted with septic shock secondary to pneumonia. "
            "Required mechanical ventilation > 96 hours, vasopressor support. "
            "ICU-level care. 8-day LOS. Legitimately the highest DRG weight tier."
        ),
        "targeted_axes": ["10–19 (MCC fully justified)", "20–29 (high intensity appropriate)", 
                           "90–99 (high DRG weight warranted)"],
        "input": {
            "claim_id":  "TC-003",
            "dx_codes":  ["A41.9", "R65.21", "J18.9", "J96.01"],
            "proc_codes": ["5A1945Z", "0BH17EZ", "3E033XZ"],
            "los": 8, "drg": "870", "mcc_count": 3, "cc_count": 0
        },
        "expected_flag": False,
        "expected_risk": "LOW",
        "expected_root_cause": "None — legitimate high-complexity ICU sepsis case",
        "explanation_guidance": (
            "Dims 10–19: A41.9 (sepsis), R65.21 (septic shock), J96.01 (acute resp failure) "
            "are all legitimate MCCs from separate pathways. "
            "Dims 20–29: Mechanical ventilation >96h (5A1945Z) is high-intensity and appropriate. "
            "Dims 90–99: DRG 870 (sepsis with MV) carries the highest financial weight appropriately. "
            "Step 11 should show NO contradictions. All axes aligned."
        ),
        "notes": "Stress test for false-positive suppression on high-acuity legitimate claims."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 2: SEVERITY INFLATION (2 cases)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-004",
        "name": "MCC Stacking — Same Pathway Cardiovascular Codes",
        "category": "Severity Inflation",
        "scenario": (
            "Patient admitted for atrial fibrillation management. "
            "Coder lists I48.91 (AF unspecified), I50.9 (HF unspecified), "
            "I11.0 (hypertensive HD with HF) — all from the same cardiovascular "
            "failure pathway, each declared as separate MCCs. "
            "2-day LOS with only rhythm monitoring and medication adjustment."
        ),
        "targeted_axes": ["10–19 (MCC stacking)", "0–9 (same-domain clustering)", 
                           "20–29 (low intensity vs high severity claim)"],
        "input": {
            "claim_id":  "TC-004",
            "dx_codes":  ["I48.91", "I50.9", "I11.0", "I10", "E78.5"],
            "proc_codes": ["02HK3MZ", "99232"],
            "los": 2, "drg": "291", "mcc_count": 3, "cc_count": 1
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "MCC stacking — multiple cardiac failure MCCs from same disease pathway",
        "explanation_guidance": (
            "Dims 0–9: 3 of 5 DX codes in Cardiovascular domain — same-system clustering alert. "
            "Dims 10–19: Intra-MCC severity similarity will be VERY HIGH (>0.85) "
            "because I48.91, I50.9, I11.0 are all cardiac failure manifestations. "
            "Step 2 should flag: 'MCC codes are very similar to each other in severity space'. "
            "Dims 20–29: Only cardiac monitoring + subsequent hospital care — low intensity "
            "for 3 declared MCCs. Step 3 intensity gap will be negative. "
            "Step 11: Contradiction — High Severity (10–19) + Low Intensity (20–29). "
            "ICD-10 coding note: I11.0 already includes the concept of hypertensive HF; "
            "coding I50.9 separately may violate combination code rules."
        ),
        "notes": "Tests Step 2 (MCC stacking) and Step 11 cross-dimensional contradiction."
    },

    {
        "id": "TC-005",
        "name": "Weak MCC Elevation — Malnutrition CC Used as MCC",
        "category": "Severity Inflation",
        "scenario": (
            "72-year-old admitted for elective knee replacement. "
            "Coder adds E43 (unspecified severe malnutrition) as a secondary DX "
            "to elevate DRG, despite BMI being 24 and no nutritional workup in records. "
            "E43 qualifies as MCC under MS-DRG, artificially inflating DRG weight. "
            "Patient discharged day 3 in good condition."
        ),
        "targeted_axes": ["10–19 (false MCC)", "70–79 (DX not treated)", 
                           "90–99 (DRG weight inflated)", "80–89 (upcoding signal)"],
        "input": {
            "claim_id":  "TC-005",
            "dx_codes":  ["M17.11", "E43", "I10", "E11.9", "Z96.641"],
            "proc_codes": ["0SRD069", "0SRC069", "99223"],
            "los": 3, "drg": "469", "mcc_count": 1, "cc_count": 2
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "False MCC — malnutrition diagnosis not supported by clinical evidence or treatment",
        "explanation_guidance": (
            "Dims 10–19: E43 (severe malnutrition) has high severity vector but low DX-Proc link "
            "to knee replacement procedures — no nutritional intervention in procedure list. "
            "Dims 70–79: E43 will appear as 'orphan DX' — no procedure supports it. "
            "Step 8 should flag E43 as unlinked to any procedure. "
            "Dims 80–89: E43 has elevated upcoding FWA signal when paired with elective surgery. "
            "Dims 90–99: DRG 469 (MCC) vs DRG 470 (no MCC) — ~$8,000 DRG gap. "
            "Financial gap step will flag elevated proc weight vs low DX diversity. "
            "Auditor should verify: nutritional assessment, albumin levels, dietitian note."
        ),
        "notes": "Classic CMS target — 'diagnosis present on admission' vs 'coded to inflate DRG'."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 3: DX-PROCEDURE MISMATCH (2 cases)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-006",
        "name": "Wrong-Site Procedure — Cardiac Procedure for GI Diagnosis",
        "category": "DX-Procedure Mismatch",
        "scenario": (
            "Patient admitted primarily for acute pancreatitis with dehydration. "
            "Procedure list includes coronary angiography (93458) and PCI (02703ZZ) "
            "which belong to a completely different clinical encounter or were "
            "incorrectly attributed to this claim. "
            "Classic medical record attribution error or deliberate cross-claim fraud."
        ),
        "targeted_axes": ["0–9 (cross-domain: GI vs Cardio)", "70–79 (zero DX-Proc link)", 
                           "30–39 (abdomen DX vs chest proc)", "80–89 (phantom billing risk)"],
        "input": {
            "claim_id":  "TC-006",
            "dx_codes":  ["K85.9", "E86.0", "K86.1", "E11.9"],
            "proc_codes": ["02703ZZ", "93458", "B2041ZZ"],
            "los": 4, "drg": "301", "mcc_count": 0, "cc_count": 2
        },
        "expected_flag": True,
        "expected_risk": "CRITICAL",
        "expected_root_cause": "DX-Procedure mismatch — cardiac interventions have no clinical basis in GI/pancreatic admission",
        "explanation_guidance": (
            "Dims 0–9: DX codes in GI/Metabolic domain; procedures in Cardiovascular domain. "
            "Step 1 will flag CROSS-DOMAIN MISMATCHES for all DX-Proc pairs. "
            "Dims 30–39: DX site = Abdomen; procedure site = Chest/Cardiac. "
            "Step 4 will flag every DX-Proc pair as anatomical mismatch. "
            "Dims 70–79: DX-Proc link similarity ≈ 0 for all pairs. "
            "Step 8 will flag 02703ZZ and 93458 as 'medically unsupported procedures'. "
            "Dims 80–89: Phantom billing signal elevated for cardiac procedures on GI admission. "
            "Expected: Steps 1, 4, 8 all CRITICAL. Composite score > 75."
        ),
        "notes": "Most severe mismatch pattern. Good test for Steps 1, 4, 8 cascade firing."
    },

    {
        "id": "TC-007",
        "name": "Psychiatric DX with Orthopedic Procedures",
        "category": "DX-Procedure Mismatch",
        "scenario": (
            "Inpatient claim where principal DX is major depressive disorder (F32.2) "
            "with secondary anxiety (F41.1). Procedures include bilateral knee "
            "arthroscopy and physical therapy. Either the DX/proc belong to different "
            "admissions, or the psych DX was added to inflate the claim while "
            "the actual admission was orthopedic."
        ),
        "targeted_axes": ["0–9 (MH vs Ortho domain)", "70–79 (DX-Proc link weak)", 
                           "40–49 (chronic psych vs acute ortho)"],
        "input": {
            "claim_id":  "TC-007",
            "dx_codes":  ["F32.2", "F41.1", "F33.1", "Z87.39"],
            "proc_codes": ["0S9D0ZZ", "0S9C0ZZ", "97110"],
            "los": 2, "drg": "882", "mcc_count": 1, "cc_count": 1
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "DX-Procedure domain mismatch — psychiatric diagnoses do not support orthopedic procedures",
        "explanation_guidance": (
            "Dims 0–9: All DX codes in Mental Health domain; procedures in Orthopedic domain. "
            "Step 1: Cross-domain alert — MH diagnoses with ortho procedures. "
            "Dims 70–79: Mental health DX vector has near-zero DX-Proc link to knee procedures. "
            "Step 8: All procedures will appear 'weakly/unsupported'. "
            "Dims 40–49: Chronic episode type (psych) vs acute/post-op (ortho) inconsistency. "
            "Step 5: Episode conflict. "
            "This case tests whether the system catches domain-level mismatch "
            "when individual code similarity might be low but not obviously zero."
        ),
        "notes": "Tests system sensitivity to mental health + physical procedure cross-domain fraud."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 4: PROCEDURE REDUNDANCY / UNBUNDLING (2 cases)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-008",
        "name": "ECG + Interpretation Unbundling",
        "category": "Procedure Redundancy / Unbundling",
        "scenario": (
            "Cardiac monitoring admission for AF. Biller separately codes 93000 "
            "(ECG complete), 93005 (ECG tracing only), AND 93010 (ECG interpretation only). "
            "93000 already includes both components. Separate billing of 93005 and 93010 "
            "alongside 93000 constitutes an NCCI violation and double-billing of the "
            "interpretation and tracing components."
        ),
        "targeted_axes": ["60–69 (bundling cohesion)", "80–89 (unbundling FWA signal)"],
        "input": {
            "claim_id":  "TC-008",
            "dx_codes":  ["I48.91", "I10", "R00.1"],
            "proc_codes": ["93000", "93005", "93010", "99232"],
            "los": 1, "drg": "309", "mcc_count": 0, "cc_count": 1
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "NCCI unbundling — ECG component codes (93005, 93010) billed with comprehensive ECG (93000)",
        "explanation_guidance": (
            "Dims 60–69: 93005 and 93010 have high bundling cohesion signal as component codes. "
            "Bundling similarity between 93005/93010 and 93000 will be very high (>0.85). "
            "Step 7 will flag: NCCI violations for 93005+93000 and 93010+93000. "
            "Dims 80–89: Both component codes have high unbundling FWA signal (D081). "
            "Step 9 will flag unbundling pattern. "
            "Corrective action: Remove 93005 and 93010; retain only 93000. "
            "Dollar impact: 2 erroneous line items."
        ),
        "notes": "Classic NCCI violation. Simplest unbundling test case for Step 7."
    },

    {
        "id": "TC-009",
        "name": "Colonoscopy with Biopsy — Unbundled Diagnostic + Therapeutic",
        "category": "Procedure Redundancy / Unbundling",
        "scenario": (
            "GI admission for colorectal cancer screening with polyp. "
            "Biller codes 45378 (diagnostic colonoscopy) AND 45380 (with biopsy) "
            "on same claim, same date. Once a biopsy is taken, 45378 is subsumed "
            "into 45380; the diagnostic component cannot be billed separately. "
            "Additionally adds 45381 (submucosal injection) as separate line."
        ),
        "targeted_axes": ["60–69 (bundling)", "80–89 (unbundling risk)", "70–79 (DX-Proc link ok)"],
        "input": {
            "claim_id":  "TC-009",
            "dx_codes":  ["K63.5", "Z12.11", "D12.6"],
            "proc_codes": ["45378", "45380", "45381", "G0105"],
            "los": 1, "drg": "392", "mcc_count": 0, "cc_count": 0
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "NCCI unbundling — diagnostic colonoscopy (45378) subsumed by therapeutic (45380, 45381)",
        "explanation_guidance": (
            "Dims 60–69: 45378 has very high bundling cohesion signal as a component of 45380. "
            "Step 7: NCCI violation alert — 45378+45380 is a known bundling pair. "
            "Bundling axis similarity between 45378 and 45380 approaches 1.0. "
            "Dims 80–89: All three colonoscopy codes have high unbundling risk. "
            "Note: DX-Proc link (70–79) should be OK — colonoscopy IS supported by K63.5/Z12.11. "
            "This tests the system's ability to flag procedure-procedure issues "
            "while not falsely flagging the DX-Proc dimension."
        ),
        "notes": "Important: Tests bundling WITHOUT DX-Proc mismatch. Steps 7 fires; Step 8 should NOT."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 5: CROSS-DOMAIN INCONSISTENCY (1 case)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-010",
        "name": "Multi-Specialty Claim — Procedure Domain Fragmentation",
        "category": "Cross-Domain Inconsistency",
        "scenario": (
            "Patient admitted for COPD exacerbation. Claim includes procedures from "
            "4 different clinical domains: bronchoscopy (pulm), knee arthroscopy (ortho), "
            "EEG (neuro), and colonoscopy (GI). No plausible clinical pathway connects "
            "these. Likely result of cross-claim attribution errors or coordinated billing "
            "of unrelated procedures to a single high-acuity admission."
        ),
        "targeted_axes": ["0–9 (cross-domain fragmentation)", "30–39 (multi-site anatomical)", 
                           "70–79 (DX-Proc mismatch)", "80–89 (phantom billing)"],
        "input": {
            "claim_id":  "TC-010",
            "dx_codes":  ["J44.1", "J96.01", "I10"],
            "proc_codes": ["0BTJ4ZZ", "0S9D0ZZ", "4A000Z4", "0DTJ4ZZ"],
            "los": 3, "drg": "190", "mcc_count": 1, "cc_count": 1
        },
        "expected_flag": True,
        "expected_risk": "CRITICAL",
        "expected_root_cause": "Cross-domain fragmentation — procedures from 4 unrelated clinical domains on a single pulmonary admission",
        "explanation_guidance": (
            "Dims 0–9: DX codes in Pulmonary domain; procedures span Pulmonary, Ortho, Neuro, GI. "
            "Step 1: Multiple cross-domain mismatches. Procedure domain count = 4 vs DX domain count = 1. "
            "Dims 30–39: DX site = Chest; procedures include Lower Extremity (knee), Head (EEG), Abdomen (colon). "
            "Step 4: Severe anatomical mismatch across all non-pulmonary procedures. "
            "Dims 70–79: Only bronchoscopy has any DX-Proc link; knee/EEG/colon are unsupported. "
            "Step 8: 3 of 4 procedures flagged as medically unsupported. "
            "Expected: Steps 1, 4, 8 all fire CRITICAL."
        ),
        "notes": "Extreme multi-domain fragmentation. Good stress test for Steps 1, 4, 8 in combination."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 6: ANATOMICAL MISMATCH (1 case)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-011",
        "name": "Left vs Right — Laterality Coding Error at Anatomical Level",
        "category": "Anatomical Mismatch",
        "scenario": (
            "Patient admitted for left total hip arthroplasty (M16.12 - left hip OA). "
            "Procedure codes include right hip replacement (0SR9019) instead of left (0SRB019). "
            "Bilateral replacement codes are NOT present. This is a specificity/laterality error "
            "where the procedure code does not match the documented diagnosis side. "
            "May also indicate wrong-side surgery or coding error."
        ),
        "targeted_axes": ["30–39 (anatomical laterality mismatch)", "70–79 (DX-Proc specificity)"],
        "input": {
            "claim_id":  "TC-011",
            "dx_codes":  ["M16.12", "I10", "E11.9", "Z96.641"],
            "proc_codes": ["0SR9019", "99221", "99231"],
            "los": 3, "drg": "470", "mcc_count": 0, "cc_count": 2
        },
        "expected_flag": True,
        "expected_risk": "MEDIUM",
        "expected_root_cause": "Anatomical laterality mismatch — left hip OA diagnosis paired with right hip replacement procedure",
        "explanation_guidance": (
            "Dims 30–39: M16.12 (LEFT hip OA) vs 0SR9019 (RIGHT hip replacement). "
            "Anatomical similarity will be reduced due to laterality difference in embedding. "
            "Step 4 should detect site-mismatch. "
            "Dims 70–79: DX-Proc link will be moderate (hip OA → hip replacement is valid) "
            "but reduced due to laterality inconsistency. "
            "This tests system sensitivity to sub-domain specificity, not just broad mismatch. "
            "Important note: The system may rate this MEDIUM (not CRITICAL) because the "
            "overall clinical relationship (hip OA → hip replacement) is valid — "
            "only the laterality is wrong. Auditor should verify operative report for correct side."
        ),
        "notes": "Tests sub-domain anatomical specificity. Expect MEDIUM risk, not CRITICAL."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 7: EPISODE TYPE CONFLICT (1 case)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-012",
        "name": "Preventive Codes in Acute Inpatient Context",
        "category": "Episode Type Conflict",
        "scenario": (
            "Patient admitted emergently for acute MI. During the 2-day stay, "
            "biller appends Z12.31 (screening mammography encounter) and G0101 "
            "(cervical cancer screening) to the claim — these are preventive/wellness "
            "codes that would only apply in an outpatient preventive visit. "
            "Adding them to an acute inpatient claim is inappropriate and inflates "
            "the secondary diagnosis count."
        ),
        "targeted_axes": ["40–49 (episode conflict: acute vs preventive)", 
                           "50–59 (billing channel mismatch)", "70–79 (DX not treated in this episode)"],
        "input": {
            "claim_id":  "TC-012",
            "dx_codes":  ["I21.01", "I10", "Z12.31", "Z01.411", "Z00.00"],
            "proc_codes": ["02703ZZ", "B2041ZZ", "99232"],
            "los": 2, "drg": "247", "mcc_count": 1, "cc_count": 1
        },
        "expected_flag": True,
        "expected_risk": "MEDIUM",
        "expected_root_cause": "Episode type conflict — preventive/screening codes appended to acute MI inpatient claim",
        "explanation_guidance": (
            "Dims 40–49: I21.01 and cardiac proc have acute episode signal. "
            "Z12.31, Z01.411, Z00.00 have strong preventive episode signal. "
            "Step 5: Preventive procedures in acute context alert. "
            "Dims 50–59: Preventive codes have outpatient billing channel profile; "
            "Step 6 may flag billing channel inconsistency. "
            "Dims 70–79: Z-codes are orphan DX — no procedure on this claim supports them. "
            "Step 8: Z12.31, Z01.411, Z00.00 flagged as orphan diagnoses. "
            "Clinical note: Preventive codes may be technically present on admission "
            "but should NOT appear on an acute inpatient claim per CMS billing rules."
        ),
        "notes": "Tests Step 5 and Step 6 specifically. Important rule: POA ≠ appropriate for inpatient billing."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 8: SERVICE INTENSITY MISMATCH (2 cases)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-013",
        "name": "Major Surgery for Minor Diagnosis",
        "category": "Service Intensity Mismatch",
        "scenario": (
            "Patient admitted with simple UTI (N39.0) and back pain (M54.5). "
            "Procedure list includes CABG (coronary artery bypass graft), "
            "mechanical ventilation >96h, and bilateral hip replacement. "
            "These procedures have no clinical basis in the listed diagnoses. "
            "Classic phantom billing pattern — high-intensity procedures on low-severity diagnoses."
        ),
        "targeted_axes": ["20–29 (intensity >> severity)", "10–19 (low severity vs high proc weight)", 
                           "90–99 (extreme financial gap)", "80–89 (phantom risk)"],
        "input": {
            "claim_id":  "TC-013",
            "dx_codes":  ["N39.0", "M54.5", "R51.9"],
            "proc_codes": ["02100Z8", "5A1945Z", "0SR9019", "0SRB019"],
            "los": 1, "drg": "981", "mcc_count": 0, "cc_count": 0
        },
        "expected_flag": True,
        "expected_risk": "CRITICAL",
        "expected_root_cause": "Extreme service intensity mismatch — major cardiac/orthopedic procedures for UTI/back pain diagnoses",
        "explanation_guidance": (
            "Dims 10–19: DX severity vectors very low (UTI, back pain = low acuity). "
            "Dims 20–29: Procedure complexity vectors very high (CABG, bilateral THA, MV). "
            "Gap = massive. Step 3: CRITICAL intensity–severity mismatch. "
            "Dims 90–99: Procedure DRG weight ≈ 0.9+ vs DX DRG weight ≈ 0.1. "
            "Step 10: Financial gap will be extreme (>0.70). "
            "Dims 80–89: Phantom billing signal on high-cost procedures without clinical basis. "
            "Step 9: Phantom billing pattern detected. "
            "Step 11: Multiple contradictions across all dimensions. "
            "Expected composite score > 85."
        ),
        "notes": "Maximum stress test. Every step should fire. Used to validate score ceiling."
    },

    {
        "id": "TC-014",
        "name": "E/M Level Inflation — Subsequent Care Upcoding",
        "category": "Service Intensity Mismatch",
        "scenario": (
            "Routine 2-day admission for stable hypertension management. "
            "Biller codes 99233 (high complexity subsequent hospital care) "
            "when documentation supports only 99231 (low complexity). "
            "Additionally bills 99291 (critical care) which has no ICU note "
            "or critical care documentation in the record."
        ),
        "targeted_axes": ["20–29 (E/M upcoding)", "10–19 (low severity vs high E/M)", 
                           "80–89 (upcoding FWA signal)"],
        "input": {
            "claim_id":  "TC-014",
            "dx_codes":  ["I10", "E78.5", "Z87.39"],
            "proc_codes": ["99233", "99291", "99292"],
            "los": 2, "drg": "812", "mcc_count": 0, "cc_count": 1
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "E/M upcoding — high-complexity and critical care E/M for stable hypertension admission",
        "explanation_guidance": (
            "Dims 10–19: DX codes (I10, E78.5) have low severity vectors — chronic stable conditions. "
            "Dims 20–29: 99233 and 99291 have high E/M intensity signals. "
            "Gap = significant. Step 3: Moderate intensity–severity mismatch. "
            "Dims 80–89: 99233 and 99291 have elevated upcoding FWA signal. "
            "Step 9: Upcoding pattern detected. "
            "Note: 99291 + 99292 — critical care codes are also near-duplicate (Step 7). "
            "99292 is an add-on code to 99291 and has high bundling signal. "
            "Step 7: Near-duplicate pair 99291/99292 flagged. "
            "Auditor should request: physician notes, nursing time documentation, ICU transfer record."
        ),
        "notes": "Tests E/M upcoding via intensity axis. Also catches add-on code bundling."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 9: FINANCIAL MISALIGNMENT (1 case)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-015",
        "name": "DRG Maximization via High-Weight Procedures on Low-Acuity Admission",
        "category": "Financial Misalignment",
        "scenario": (
            "Patient admitted with mild cellulitis and dehydration. "
            "Biller adds cardiac catheterization and coronary angiography to the claim — "
            "procedures with extremely high DRG/RVU weights. These procedures either "
            "did not occur or occurred in a separate outpatient encounter being "
            "improperly bundled into this inpatient claim to maximize reimbursement."
        ),
        "targeted_axes": ["90–99 (DRG weight inflation)", "70–79 (DX-Proc link weak)", 
                           "0–9 (domain mismatch)", "80–89 (phantom billing)"],
        "input": {
            "claim_id":  "TC-015",
            "dx_codes":  ["L03.115", "E86.0", "I10"],
            "proc_codes": ["93458", "93459", "B2041ZZ", "3E033XZ"],
            "los": 2, "drg": "247", "mcc_count": 0, "cc_count": 1
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "Financial misalignment — high-DRG cardiac procedures attributed to low-acuity skin/dehydration admission",
        "explanation_guidance": (
            "Dims 90–99: Cardiac cath procedures (93458, 93459) have very high DRG/RVU weight vectors. "
            "DX codes (cellulitis, dehydration) have very low DRG weight vectors. "
            "Financial gap will be large (>0.45). Step 10: HIGH risk. "
            "Dims 0–9: DX in Skin/Infectious domain; procedures in Cardiovascular domain. "
            "Step 1: Cross-domain mismatch. "
            "Dims 70–79: No DX supports cardiac catheterization. Step 8: Unsupported procedures. "
            "Dims 80–89: Phantom billing signal on cardiac procedures. "
            "Step 9: Phantom billing detected. "
            "Also: 93458 and 93459 are very similar procedures — "
            "Step 7 may flag near-duplicate pair."
        ),
        "notes": "Pure financial exploitation pattern. Steps 1, 8, 9, 10 all fire."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 10: ADVERSARIAL CASES (3 cases)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-016",
        "name": "Adversarial — High Similarity But Clinically Invalid Combination",
        "category": "Adversarial",
        "scenario": (
            "Sophisticated coder selects DX and procedure codes that all share the same "
            "clinical domain (cardiovascular) and similar embedding signatures, but the "
            "specific combination is clinically implausible: "
            "Codes for heart transplant aftercare (Z94.1) as principal DX "
            "with initial encounter codes for first-time MI procedures. "
            "Embedding similarity will be HIGH (all cardiac) but the clinical logic fails "
            "because post-transplant patients cannot have native coronary procedures."
        ),
        "targeted_axes": ["40–49 (acute vs post-op conflict)", "70–79 (clinically invalid link)", 
                           "10–19 (severity mismatch in same domain)"],
        "input": {
            "claim_id":  "TC-016",
            "dx_codes":  ["Z94.1", "I25.10", "I10", "I50.22"],
            "proc_codes": ["02703ZZ", "02100Z8", "B2041ZZ"],
            "los": 4, "drg": "246", "mcc_count": 1, "cc_count": 2
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "Adversarial high-similarity invalid combination — post-transplant patient cannot have native coronary interventions",
        "explanation_guidance": (
            "SYSTEM WEAKNESS: Embedding similarity will be HIGH across all axes "
            "(all codes in cardiovascular domain). Naive cosine similarity would NOT flag this. "
            "Dims 40–49: Z94.1 is a chronic/post-op episode code; PCI (02703ZZ) is an acute "
            "procedure on native coronary arteries. Episode conflict. "
            "Dims 70–79: DX-Proc link may appear moderate because all codes are cardiac — "
            "this is where the adversarial design tries to fool the system. "
            "The system must use episode type (40–49) and temporal flags to catch this. "
            "Step 5 should flag: Post-op chronic DX (Z94.1) with acute native-vessel procedure. "
            "Step 11 should catch: High-similarity codes but episode-type contradiction. "
            "Clinical note: Post-transplant patients undergo transplant vessel monitoring, "
            "not native coronary PCI — these are anatomically incompatible."
        ),
        "notes": "Designed to fool similarity-based systems. Tests Step 5 and Step 11 cross-synthesis."
    },

    {
        "id": "TC-017",
        "name": "Adversarial — MCC Added in Same Embedding Cluster as Principal DX",
        "category": "Adversarial",
        "scenario": (
            "Clever coder adds a CC/MCC that is semantically close to the principal DX "
            "in embedding space (high cosine similarity) but represents a distinct "
            "condition with its own DRG impact. "
            "Example: Acute MI (I21.0) as principal, plus I21.4 (NSTEMI) coded as secondary. "
            "Both are AMI variants — highly similar in embedding space — but coding both "
            "is incorrect per ICD-10 guidelines when they represent the same MI event."
        ),
        "targeted_axes": ["10–19 (intra-DX similarity)", "0–9 (same domain clustering)", 
                           "70–79 (DX-Proc coherent but DX-DX redundant)"],
        "input": {
            "claim_id":  "TC-017",
            "dx_codes":  ["I21.01", "I21.4", "I21.9", "I10", "E11.9"],
            "proc_codes": ["02703ZZ", "B2041ZZ", "99232"],
            "los": 3, "drg": "246", "mcc_count": 2, "cc_count": 1
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "Adversarial MCC duplication — multiple AMI specificity variants coded for what may be a single MI event",
        "explanation_guidance": (
            "SYSTEM WEAKNESS: DX-Proc link (Step 8) will appear VALID because "
            "MI diagnoses DO support PCI procedures — the adversarial element is in "
            "the DX-DX dimension, not the DX-Proc dimension. "
            "Dims 10–19: I21.01, I21.4, I21.9 are very similar in severity space — "
            "near-duplicate DX alert. Intra-DX severity similarity > 0.90. "
            "Step 2: MCC stacking detected (same pathway, different specificity). "
            "Dims 0–9: All 3 AMI codes in Cardiovascular domain — same-system clustering. "
            "Step 1: Dense clustering in cardiac domain. "
            "ICD-10 Rule: I21.01 (STEMI LAD) and I21.4 (NSTEMI) cannot coexist for "
            "the same MI event — they represent mutually exclusive types. "
            "The system should catch this via DX-DX redundancy analysis (Step 2)."
        ),
        "notes": "Adversarial case that appears clinically coherent but violates ICD-10 sequencing rules."
    },

    {
        "id": "TC-018",
        "name": "Adversarial — Legitimate-Looking Multi-System but DRG-Maximizing",
        "category": "Adversarial",
        "scenario": (
            "Claim appears multi-system and plausible: AMI + CKD + DM + Sepsis. "
            "Each diagnosis from a different domain, procedures appear to match. "
            "However, the sepsis code (A41.9) and respiratory failure (J96.01) "
            "were resolved BEFORE admission per the clinical notes, yet are coded "
            "as if present during the entire stay to maximize DRG complexity. "
            "The MCC count of 4 is technically supported by the codes but not "
            "by the clinical timeline."
        ),
        "targeted_axes": ["10–19 (MCC count review)", "40–49 (episode timing conflict)", 
                           "70–79 (DX-Proc partial link)", "90–99 (financial weight with weak support)"],
        "input": {
            "claim_id":  "TC-018",
            "dx_codes":  ["I21.01", "N18.4", "E11.65", "A41.9", "J96.01", "R65.21"],
            "proc_codes": ["02703ZZ", "5A1D00Z", "B2041ZZ", "3E033XZ"],
            "los": 5, "drg": "870", "mcc_count": 4, "cc_count": 1
        },
        "expected_flag": True,
        "expected_risk": "MEDIUM",
        "expected_root_cause": "Adversarial multi-system DRG maximization — MCC codes may not represent conditions present and treated throughout stay",
        "explanation_guidance": (
            "SYSTEM CHALLENGE: Each DX-Proc pair has plausible links "
            "(MI→PCI, CKD→dialysis, sepsis→antibiotics). The system should NOT "
            "score this as CRITICAL based on embedding similarity alone. "
            "Dims 10–19: 4 MCCs from different domains — diversity seems legitimate. "
            "However, intra-MCC similarity is moderate (mixed domains). "
            "Step 2 should NOT fire MCC stacking (diverse domains). Risk = MEDIUM. "
            "Dims 40–49: A41.9 and J96.01 are acute episode codes. "
            "If they were resolved pre-admission, the temporal flag should catch this. "
            "The system correctly scores MEDIUM, not CRITICAL — leaving investigation "
            "to the auditor who must review the timeline in the clinical record. "
            "This tests the system's calibration — it should not over-flag plausible-looking claims."
        ),
        "notes": "MEDIUM risk expected. Tests calibration: system must not over-flag multi-system legitimate claims."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 11: NOISE INJECTION (2 cases)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-019",
        "name": "Noise Injection — Irrelevant DX Padding to Inflate CC Count",
        "category": "Noise Injection",
        "scenario": (
            "Routine knee replacement admission. Primary DX and procedures are valid. "
            "However, coder adds 8 secondary diagnoses — many unrelated to orthopedic "
            "care and several are history codes (Z87.x) or chronic conditions "
            "not treated in this admission. "
            "Goal: push CC count from 1 to 5, elevating DRG reimbursement tier."
        ),
        "targeted_axes": ["0–9 (multi-domain noise)", "10–19 (CC inflation)", 
                           "70–79 (orphan DX — not treated)"],
        "input": {
            "claim_id":  "TC-019",
            "dx_codes":  ["M17.11", "I10", "E11.9", "F32.9", "N18.2", 
                           "Z87.891", "Z87.39", "H54.7", "G89.29", "J45.20"],
            "proc_codes": ["0SRD069", "99221", "99231"],
            "los": 3, "drg": "470", "mcc_count": 0, "cc_count": 5
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "Diagnosis padding — multiple secondary diagnoses unrelated to orthopedic admission used to inflate CC count",
        "explanation_guidance": (
            "Dims 0–9: DX codes span 7+ clinical domains — fragmented multi-domain distribution. "
            "Step 1: Low clinical diversity on procedure side vs high diversity on DX side. "
            "Dims 10–19: CC count of 5 with only knee replacement procedures. "
            "Step 2: Declared CC=5 exceeds embedding-supported complexity. "
            "Dims 70–79: F32.9, N18.2, H54.7, G89.29, J45.20 are orphan DX — "
            "no procedures address these conditions. "
            "Step 8: Multiple orphan diagnoses flagged. "
            "Z-codes (Z87.891, Z87.39) are history codes and should never "
            "count toward CC/MCC per CMS guidelines. "
            "Expected: Steps 1, 2, 8 all fire. Composite HIGH."
        ),
        "notes": "Tests noise robustness — the valid ortho core should not suppress the inflated CC detection."
    },

    {
        "id": "TC-020",
        "name": "Noise Injection — Redundant Procedure Variations",
        "category": "Noise Injection",
        "scenario": (
            "Cardiac admission with valid PCI. Biller adds multiple near-identical "
            "variants of the same procedure: 02703ZZ (1-site PCI), 02713ZZ (2-site PCI), "
            "02723ZZ (3-site PCI), and 02733ZZ (4-site PCI) — all in addition to the "
            "original PCI code. Only one vessel intervention was documented. "
            "This artificially inflates procedure count and potentially DRG weight."
        ),
        "targeted_axes": ["60–69 (procedure near-duplicates)", "80–89 (unbundling/phantom)", 
                           "20–29 (inflated procedure intensity)"],
        "input": {
            "claim_id":  "TC-020",
            "dx_codes":  ["I21.01", "I10", "E11.9"],
            "proc_codes": ["02703ZZ", "02713ZZ", "02723ZZ", "02733ZZ", "B2041ZZ"],
            "los": 3, "drg": "246", "mcc_count": 1, "cc_count": 1
        },
        "expected_flag": True,
        "expected_risk": "HIGH",
        "expected_root_cause": "Redundant procedure variants — multiple site-count variants of PCI coded when only one vessel was treated",
        "explanation_guidance": (
            "Dims 60–69: 02703ZZ, 02713ZZ, 02723ZZ, 02733ZZ are all PCI codes varying only "
            "in vessel count. Bundling axis similarity will be very high among all four. "
            "Step 7: Multiple near-duplicate procedure pairs. "
            "Dims 80–89: Unbundling/phantom signal elevated for the extra site variants. "
            "Step 9: Unbundling pattern. "
            "Dims 20–29: 4 PCI codes artificially inflates procedure intensity signal. "
            "Step 3: Procedure intensity appears inflated. "
            "Clinical note: ICD-10-PCS requires coder to count distinct vessel sites — "
            "if only one vessel was stented, only 02703ZZ should be coded. "
            "Auditor should review cardiac cath report for vessel count."
        ),
        "notes": "Tests procedure variant noise injection. High similarity among proc codes should trigger Step 7."
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CATEGORY 12: RARE BUT VALID EDGE CASES (2 cases)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "TC-021",
        "name": "Complex OB Case — Valid Multi-System Delivery Claim",
        "category": "Valid Edge Case",
        "scenario": (
            "32-year-old admitted for vaginal delivery complicated by preeclampsia, "
            "gestational diabetes, and postpartum hemorrhage requiring blood transfusion. "
            "Multiple clinical domains legitimately present: OB, metabolic, cardiovascular. "
            "This edge case should NOT be flagged despite apparent multi-domain diversity."
        ),
        "targeted_axes": ["0–9 (multi-domain valid)", "40–49 (acute delivery episode)", 
                           "70–79 (DX-Proc coherent)", "10–19 (severity justified)"],
        "input": {
            "claim_id":  "TC-021",
            "dx_codes":  ["O14.05", "O24.419", "O72.1", "Z37.0", "O09.523"],
            "proc_codes": ["10E0XZZ", "30233N1", "0UJ07ZZ"],
            "los": 3, "drg": "765", "mcc_count": 2, "cc_count": 1
        },
        "expected_flag": False,
        "expected_risk": "LOW",
        "expected_root_cause": "None — valid complex OB delivery with justified complications",
        "explanation_guidance": (
            "Dims 0–9: All codes in OB domain (O-prefix DX, OB procedure codes). "
            "Step 1: Clinical domain consistent — should NOT fire. "
            "Dims 10–19: O14.05 (preeclampsia MCC) and O72.1 (PPH MCC) are from "
            "different pathways within OB — legitimate multi-MCC. "
            "Step 2: Intra-MCC similarity will be moderate (different OB complications) — "
            "should NOT fire MCC stacking. "
            "Dims 70–79: Delivery procedure (10E0XZZ) directly supports all OB diagnoses. "
            "Blood transfusion (30233N1) directly supports O72.1 (PPH). "
            "Step 8: All DX well-linked to procedures. "
            "System challenge: Multi-domain diversity + multi-MCC must NOT trigger "
            "false positives when all codes are clinically valid within one episode."
        ),
        "notes": "Critical for false-positive calibration. OB claims appear multi-system but are legitimate."
    },

    {
        "id": "TC-022",
        "name": "Oncology Inpatient — Valid High-Cost Chemotherapy Admission",
        "category": "Valid Edge Case",
        "scenario": (
            "55-year-old admitted for inpatient chemotherapy for acute myeloid leukemia. "
            "Multi-drug regimen with supportive care. High cost, long LOS (7 days), "
            "multiple procedure codes for different drug infusions. "
            "This should NOT be flagged despite high financial weight and multi-procedure claim."
        ),
        "targeted_axes": ["90–99 (high DRG weight but justified)", "20–29 (high intensity justified)", 
                           "70–79 (DX-Proc link strong)", "60–69 (infusion codes — not unbundled)"],
        "input": {
            "claim_id":  "TC-022",
            "dx_codes":  ["C92.00", "D61.9", "D64.9", "Z85.6"],
            "proc_codes": ["30233N1", "96413", "96415", "96415", "3E033XZ"],
            "los": 7, "drg": "834", "mcc_count": 2, "cc_count": 1
        },
        "expected_flag": False,
        "expected_risk": "LOW",
        "expected_root_cause": "None — valid high-acuity oncology chemotherapy admission",
        "explanation_guidance": (
            "Dims 0–9: C92.00 (AML), D61.9 (aplastic anemia), D64.9 (anemia) — "
            "all in hematology/oncology domain. Procedures: chemo infusion, transfusion. "
            "Step 1: Consistent domain. "
            "Dims 20–29: Chemotherapy infusion (96413/96415) is appropriately high-intensity for AML. "
            "Step 3: Intensity gap justified by MCC-level DX. "
            "Dims 60–69: 96415 is an add-on code to 96413 — Step 7 may flag as component code. "
            "IMPORTANT: 96415 is a LEGITIMATE add-on code — the system should recognize "
            "that add-on codes billed with their parent are NOT an NCCI violation. "
            "Dims 90–99: High DRG weight (DRG 834) is appropriate for AML chemo. "
            "Step 10: Financial weight matches clinical acuity — should NOT fire. "
            "System challenge: Must distinguish 96415 as legitimate add-on vs unbundling."
        ),
        "notes": "Tests add-on code handling in Step 7. 96415 with 96413 is valid, not a violation."
    },
]

# ── Output functions ──────────────────────────────────────────────────────────

def write_json(cases, path):
    with open(path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"[OK] JSON written → {path}")

def write_batch_csv(cases, path):
    """Write batch-ready CSV for dim_auditor.py"""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_id","dx_codes","proc_codes","los","drg","mcc_count","cc_count"])
        for tc in cases:
            inp = tc["input"]
            writer.writerow([
                inp["claim_id"],
                ",".join(inp["dx_codes"]),
                ",".join(inp["proc_codes"]),
                inp.get("los",""),
                inp.get("drg",""),
                inp.get("mcc_count",""),
                inp.get("cc_count",""),
            ])
    print(f"[OK] Batch CSV written → {path}")

def write_report(cases, path):
    lines = []
    W = 76

    lines.append("=" * W)
    lines.append("  SYNTHETIC INPATIENT CLAIM TEST CASES")
    lines.append("  Payment Integrity Audit System Validation")
    lines.append(f"  Total test cases: {len(cases)}")
    lines.append("=" * W)

    # Summary table
    lines.append(f"\n  {'ID':<8} {'Name':<45} {'Category':<26} {'Expected Risk'}")
    lines.append("  " + "─"*74)
    for tc in cases:
        lines.append(
            f"  {tc['id']:<8} {tc['name'][:44]:<45} "
            f"{tc['category']:<26} {tc['expected_risk']}"
        )

    # Category counts
    from collections import Counter
    cats = Counter(tc["category"] for tc in cases)
    lines.append(f"\n  CATEGORY DISTRIBUTION:")
    for cat, n in sorted(cats.items()):
        lines.append(f"    {cat:<32}: {n} case(s)")

    lines.append(f"\n  AXIS COVERAGE SUMMARY:")
    all_axes = []
    for tc in cases:
        all_axes.extend(tc["targeted_axes"])
    for ax_range in ["0–9","10–19","20–29","30–39","40–49","50–59","60–69","70–79","80–89","90–99"]:
        count = sum(1 for a in all_axes if ax_range in a)
        lines.append(f"    Dims {ax_range}: covered by {count} test cases")

    lines.append("\n" + "=" * W)

    # Full detail per case
    for tc in cases:
        lines.append(f"\n{'─'*W}")
        lines.append(f"  {tc['id']} | {tc['name']}")
        lines.append(f"  Category: {tc['category']}")
        lines.append(f"  Expected Risk: {tc['expected_risk']}  |  "
                     f"Should Flag: {'YES' if tc['expected_flag'] else 'NO'}")
        lines.append(f"  Targeted Axes: {', '.join(tc['targeted_axes'])}")
        lines.append(f"\n  SCENARIO:")
        for ln in textwrap.wrap(tc["scenario"], 72):
            lines.append(f"    {ln}")
        lines.append(f"\n  INPUT DATA:")
        inp = tc["input"]
        lines.append(f"    Claim ID  : {inp['claim_id']}")
        lines.append(f"    DX Codes  : {', '.join(inp['dx_codes'])}")
        lines.append(f"    Proc Codes: {', '.join(inp['proc_codes'])}")
        meta = []
        if inp.get("los"):       meta.append(f"LOS={inp['los']}d")
        if inp.get("drg"):       meta.append(f"DRG={inp['drg']}")
        if inp.get("mcc_count"): meta.append(f"MCC={inp['mcc_count']}")
        if inp.get("cc_count"):  meta.append(f"CC={inp['cc_count']}")
        if meta: lines.append(f"    {' | '.join(meta)}")
        lines.append(f"\n  EXPECTED ROOT CAUSE:")
        lines.append(f"    {tc['expected_root_cause']}")
        lines.append(f"\n  EXPLANATION GUIDANCE:")
        for ln in textwrap.wrap(tc["explanation_guidance"], 72):
            lines.append(f"    {ln}")
        if tc.get("notes"):
            lines.append(f"\n  NOTES: {tc['notes']}")

    lines.append(f"\n{'='*W}\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Report written → {path}")


def print_summary(cases):
    W = 76
    print("\n" + "="*W)
    print("  SYNTHETIC TEST CASE GENERATOR — SUMMARY")
    print("="*W)
    print(f"  Generated {len(cases)} test cases\n")

    risk_icons = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴","CRITICAL":"🚨"}
    flag_icons = {True:"⚠️ YES", False:"✓  NO "}

    print(f"  {'ID':<8} {'Expected Risk':<12} {'Flag':<8} {'Category':<28} Name")
    print("  " + "─"*72)
    for tc in cases:
        icon = risk_icons.get(tc["expected_risk"],"⚪")
        fi   = flag_icons.get(tc["expected_flag"],"?")
        print(f"  {tc['id']:<8} {icon} {tc['expected_risk']:<10} {fi}  "
              f"{tc['category']:<28} {tc['name'][:28]}")

    print("="*W + "\n")


def main():
    p = argparse.ArgumentParser(
        description="Synthetic Inpatient Test Case Generator for Payment Integrity Audit System"
    )
    p.add_argument("--run-audit", action="store_true",
                   help="Auto-run generated cases through dim_auditor.py")
    p.add_argument("--embeddings",
                   default="/mnt/user-data/outputs/Medical_Code_FWA_Embeddings_100k.csv")
    p.add_argument("--out-dir", default="/mnt/user-data/outputs")
    args = p.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    json_path   = os.path.join(out, "test_cases.json")
    csv_path    = os.path.join(out, "test_cases_batch.csv")
    report_path = os.path.join(out, "test_cases_report.txt")

    write_json(TEST_CASES, json_path)
    write_batch_csv(TEST_CASES, csv_path)
    write_report(TEST_CASES, report_path)
    print_summary(TEST_CASES)

    print(f"  Files generated:")
    print(f"    {json_path}")
    print(f"    {csv_path}")
    print(f"    {report_path}")

    if args.run_audit:
        print(f"\n  Running batch audit through dim_auditor.py ...\n")
        audit_out = os.path.join(out, "test_cases_audit_results.csv")
        cmd = (f"python {out}/dim_auditor.py "
               f"--embeddings {args.embeddings} "
               f"--batch {csv_path} "
               f"--output {audit_out} --quiet")
        os.system(cmd)
        print(f"\n  Audit results → {audit_out}")

if __name__ == "__main__":
    main()
