import numpy as np
import pandas as pd


class WeightedCodeSimilarity:
    def __init__(self, csv_path="all.csv"):
        self.embeddings = self.load_embeddings(csv_path)
        self.dim_groups = self.define_dimension_groups()

    # ---------------------------
    # Load Embeddings (Strict Schema)
    # ---------------------------
    def load_embeddings(self, path):
        df = pd.read_csv(path)

        # Expect: code + D001 ... D100
        expected_cols = ["code"] + [f"D{str(i).zfill(3)}" for i in range(1, 101)]

        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        embeddings = {}

        for _, row in df.iterrows():
            code = self.normalize_code(row["code"])
            vec = row[expected_cols[1:]].values.astype(float)

            if len(vec) != 100:
                raise ValueError(f"Invalid vector length for {code}")

            embeddings[code] = vec

        return embeddings

    # ---------------------------
    # Normalize Codes
    # ---------------------------
    def normalize_code(self, code):
        return str(code).replace('.', '').upper()

    # ---------------------------
    # Dimension Groups
    # ---------------------------
    def define_dimension_groups(self):
        return {
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
    # Build Weight Vector
    # ---------------------------
    def build_weight_vector(self, group_weights):
        W = np.ones(100)

        for group, weight in group_weights.items():
            if group not in self.dim_groups:
                raise ValueError(f"Unknown group: {group}")

            for idx in self.dim_groups[group]:
                W[idx] = weight

        return W

    # ---------------------------
    # Weighted Cosine Similarity
    # ---------------------------
    def weighted_cosine(self, v1, v2, W):
        v1_w = v1 * W
        v2_w = v2 * W

        dot = np.dot(v1_w, v2_w)
        norm1 = np.linalg.norm(v1_w)
        norm2 = np.linalg.norm(v2_w)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    # ---------------------------
    # Public API
    # ---------------------------
    def get_similarity(self, code1, code2, group_weights=None):
        code1 = self.normalize_code(code1)
        code2 = self.normalize_code(code2)

        if code1 not in self.embeddings:
            raise ValueError(f"{code1} not found")
        if code2 not in self.embeddings:
            raise ValueError(f"{code2} not found")

        v1 = self.embeddings[code1]
        v2 = self.embeddings[code2]

        W = self.build_weight_vector(group_weights) if group_weights else np.ones(100)

        return self.weighted_cosine(v1, v2, W)

    # ---------------------------
    # Top-K Similar
    # ---------------------------
    def top_k_similar(self, code, k=5, group_weights=None):
        code = self.normalize_code(code)

        if code not in self.embeddings:
            raise ValueError(f"{code} not found")

        target_vec = self.embeddings[code]
        W = self.build_weight_vector(group_weights) if group_weights else np.ones(100)

        scores = []

        for other_code, vec in self.embeddings.items():
            if other_code == code:
                continue

            sim = self.weighted_cosine(target_vec, vec, W)
            scores.append((other_code, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# =========================================================
# 🔥 TEST CASES
# =========================================================
if __name__ == "__main__":

    sim = WeightedCodeSimilarity("all.csv")

    code_a = "N17.9"
    code_b = "I10"
    code_c = "E11.9"

    # ---------------------------
    # Base Similarity
    # ---------------------------
    print("\n=== BASE SIMILARITY ===")
    print(f"{code_a} vs {code_b}:",
          round(sim.get_similarity(code_a, code_b), 4))

    # ---------------------------
    # Clinical Similarity
    # ---------------------------
    clinical_weights = {
        "clinical_domain": 2.5,
        "anatomical_site": 2.0,
        "episode_type": 1.5
    }

    print("\n=== CLINICAL SIMILARITY ===")
    print(f"{code_a} vs {code_b}:",
          round(sim.get_similarity(code_a, code_b, clinical_weights), 4))

    # ---------------------------
    # FWA Similarity
    # ---------------------------
    fwa_weights = {
        "fwa_risk": 3.0,
        "severity": 2.5,
        "drg_rvu": 2.0,
        "bundling": 2.0
    }

    print("\n=== FWA SIMILARITY ===")
    print(f"{code_a} vs {code_c}:",
          round(sim.get_similarity(code_a, code_c, fwa_weights), 4))

    # ---------------------------
    # Medical Necessity
    # ---------------------------
    med_weights = {
        "dx_proc_link": 3.0,
        "clinical_domain": 2.0,
        "anatomical_site": 2.0
    }

    print("\n=== MEDICAL NECESSITY SIMILARITY ===")
    print(f"{code_b} vs {code_c}:",
          round(sim.get_similarity(code_b, code_c, med_weights), 4))

    # ---------------------------
    # Top-K Similar (FWA Focus)
    # ---------------------------
    print("\n=== TOP 5 SIMILAR (FWA FOCUS) ===")
    for c, s in sim.top_k_similar(code_a, k=5, group_weights=fwa_weights):
        print(c, round(s, 4))
