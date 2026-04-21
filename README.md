import pandas as pd
import numpy as np

# =========================================================
# LOAD EMBEDDINGS
# =========================================================
class CPTEmbeddingValidator:
    def __init__(self, file_path="all.csv"):
        self.df = pd.read_csv(file_path)

        # Ensure correct format
        if "code" not in self.df.columns:
            raise ValueError("CSV must contain 'code' column")

        # Set index
        self.df.set_index("code", inplace=True)

        # Convert to dict for fast lookup
        self.embeddings = {
            code: row.values.astype(float)
            for code, row in self.df.iterrows()
        }

        # Define dimension groups
        self.groups = {
            "clinical": range(0, 10),
            "severity": range(10, 20),
            "intensity": range(20, 30),
            "anatomy": range(30, 40),
            "episode": range(40, 50),
            "billing": range(50, 60),
            "bundling": range(60, 70),
            "dx_proc_link": range(70, 80),
            "risk": range(80, 90),
            "financial": range(90, 100)
        }

    # =========================================================
    # COSINE SIMILARITY
    # =========================================================
    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    # =========================================================
    # WEIGHTED SIMILARITY
    # =========================================================
    def weighted_similarity(self, code1, code2, weights):
        if code1 not in self.embeddings or code2 not in self.embeddings:
            raise ValueError(f"Missing code: {code1} or {code2}")

        v1 = self.embeddings[code1]
        v2 = self.embeddings[code2]

        score = 0.0
        total_weight = 0.0
        group_scores = {}

        for group, idxs in self.groups.items():
            sub_v1 = v1[list(idxs)]
            sub_v2 = v2[list(idxs)]

            sim = self.cosine_similarity(sub_v1, sub_v2)
            w = weights.get(group, 1.0)

            group_scores[group] = sim
            score += sim * w
            total_weight += w

        final_score = score / total_weight if total_weight else 0

        return final_score, group_scores

    # =========================================================
    # MODIFIER 59 RISK EVALUATION
    # =========================================================
    def evaluate_mod59(self, code1, code2):
        weights = {
            "bundling": 3.0,
            "anatomy": 2.5,
            "risk": 2.0,
            "clinical": 1.5,
            "intensity": 1.5,
            "dx_proc_link": 1.5,
            "financial": 1.0,
            "billing": 1.0,
            "episode": 1.0,
            "severity": 1.0
        }

        score, group_scores = self.weighted_similarity(code1, code2, weights)

        # Risk classification
        if score > 0.75:
            risk = "🚨 HIGH RISK (Likely Unbundling)"
        elif score > 0.5:
            risk = "⚖️ MODERATE (Needs Review)"
        else:
            risk = "✅ LOW RISK (Likely Valid)"

        return score, risk, group_scores


# =========================================================
# TEST SUITE
# =========================================================
def run_tests(validator):
    test_cases = [

        # HIGH RISK
        ("45378", "45380", "Same family endoscopy"),
        ("17000", "17003", "Add-on misuse"),
        ("47562", "47563", "Same organ procedures"),
        ("99213", "99215", "E/M upcoding"),
        ("27130", "27132", "Procedure vs variant"),
        ("64483", "64484", "Injection add-on"),

        # LOW RISK
        ("29881", "43239", "Different systems"),
        ("11042", "71020", "Surgery vs imaging"),
        ("93000", "11042", "Diagnostic vs therapeutic"),
        ("93000", "45378", "Random unrelated"),

        # MODERATE
        ("17110", "17111", "Lesion count variation"),
        ("71250", "71020", "CT vs X-ray"),
        ("20550", "20610", "Nearby anatomy"),

        # VALID SAME SYSTEM DIFFERENT SITE
        ("29881", "29827", "Knee vs shoulder"),
    ]

    print("\n================ MODIFIER 59 VALIDATION ================\n")

    for c1, c2, desc in test_cases:
        try:
            score, risk, group_scores = validator.evaluate_mod59(c1, c2)

            print(f"🧪 Test: {desc}")
            print(f"CPT Pair: {c1} vs {c2}")
            print(f"Score: {round(score, 3)}")
            print(f"Risk: {risk}")

            print("🔍 Dimension Breakdown:")
            for k, v in group_scores.items():
                print(f"   {k:15}: {round(v, 3)}")

            print("-" * 60)

        except Exception as e:
            print(f"Error with {c1}-{c2}: {e}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    validator = CPTEmbeddingValidator("all.csv")
    run_tests(validator)
