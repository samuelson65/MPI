import numpy as np
import pandas as pd


class Mod59EmbeddingValidator:
    def __init__(self, path="all.csv"):
        self.df = pd.read_csv(path)
        self.embeddings = self.load_embeddings()

        self.groups = {
            "clinical_domain": range(0, 10),
            "severity": range(10, 20),
            "service_intensity": range(20, 30),
            "anatomical_site": range(30, 40),
            "episode_type": range(40, 50),
            "billing_channel": range(50, 60),
            "bundling": range(60, 70),
            "dx_proc_link": range(70, 80),
            "fwa_risk": range(80, 90),
            "drg_rvu": range(90, 100),
        }

    # ---------------------------
    # Load embeddings
    # ---------------------------
    def load_embeddings(self):
        embeddings = {}
        feature_cols = [f"D{str(i).zfill(3)}" for i in range(1, 101)]

        for _, row in self.df.iterrows():
            code = str(row["code"]).strip()
            vec = row[feature_cols].values.astype(float)
            embeddings[code] = vec

        return embeddings

    # ---------------------------
    # Weighted cosine
    # ---------------------------
    def weighted_cosine(self, v1, v2, weight_vector):
        v1_w = v1 * weight_vector
        v2_w = v2 * weight_vector

        dot = np.dot(v1_w, v2_w)
        norm1 = np.linalg.norm(v1_w)
        norm2 = np.linalg.norm(v2_w)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    # ---------------------------
    # Build weight vector
    # ---------------------------
    def build_weights(self, group_weights):
        W = np.ones(100)

        for group, weight in group_weights.items():
            for idx in self.groups[group]:
                W[idx] = weight

        return W

    # ---------------------------
    # Core similarity functions
    # ---------------------------
    def similarity(self, code1, code2, weights):
        if code1 not in self.embeddings or code2 not in self.embeddings:
            raise ValueError(f"{code1} or {code2} not found")

        W = self.build_weights(weights)
        v1 = self.embeddings[code1]
        v2 = self.embeddings[code2]

        return self.weighted_cosine(v1, v2, W)

    # ---------------------------
    # Mod 59 scoring
    # ---------------------------
    def mod59_score(self, cpt1, cpt2):
        bundling_weights = {"bundling": 3.0, "fwa_risk": 2.5}
        anatomy_weights = {"anatomical_site": 3.0}
        clinical_weights = {"clinical_domain": 2.0, "dx_proc_link": 2.5}

        bundling_sim = self.similarity(cpt1, cpt2, bundling_weights)
        anatomy_sim = self.similarity(cpt1, cpt2, anatomy_weights)
        clinical_sim = self.similarity(cpt1, cpt2, clinical_weights)

        # Risk score
        risk_score = (
            0.5 * bundling_sim +
            0.3 * anatomy_sim -
            0.2 * clinical_sim
        )

        return {
            "bundling_sim": round(bundling_sim, 4),
            "anatomy_sim": round(anatomy_sim, 4),
            "clinical_sim": round(clinical_sim, 4),
            "risk_score": round(risk_score, 4)
        }


# =========================================================
# 🔥 TEST CASES
# =========================================================
if __name__ == "__main__":

    validator = Mod59EmbeddingValidator("all.csv")

    # -----------------------------------------
    # Replace with CPTs present in your file
    # -----------------------------------------

    test_cases = [
        {
            "name": "🚨 Likely Unbundling (Same procedure family)",
            "cpt1": "45378",  # Colonoscopy diagnostic
            "cpt2": "45380",  # Colonoscopy with biopsy
            "expected": "HIGH RISK"
        },
        {
            "name": "🚨 Likely Same Anatomy Conflict",
            "cpt1": "17000",  # Skin lesion removal
            "cpt2": "17003",  # Additional lesions
            "expected": "HIGH RISK"
        },
        {
            "name": "✅ Likely Valid Mod 59 (Different anatomy)",
            "cpt1": "29881",  # Knee arthroscopy
            "cpt2": "11042",  # Wound debridement
            "expected": "LOW RISK"
        },
        {
            "name": "⚖️ Edge Case",
            "cpt1": "93000",  # ECG
            "cpt2": "71020",  # Chest X-ray
            "expected": "MODERATE"
        }
    ]

    print("\n================ MOD 59 VALIDATION RESULTS ================\n")

    for test in test_cases:
        c1 = test["cpt1"]
        c2 = test["cpt2"]

        try:
            result = validator.mod59_score(c1, c2)

            print(f"Test: {test['name']}")
            print(f"CPT Pair: {c1} vs {c2}")
            print(f"Expected: {test['expected']}")
            print("Output:", result)

            # Basic interpretation
            if result["risk_score"] > 0.6:
                print("➡️ Flag: HIGH RISK (Possible Mod 59 misuse)")
            elif result["risk_score"] > 0.3:
                print("➡️ Flag: MODERATE RISK")
            else:
                print("➡️ Flag: LOW RISK (Likely valid)")

            print("-" * 60)

        except Exception as e:
            print(f"Error with {c1}, {c2}: {e}")
