import numpy as np
import pandas as pd
import ast

class CodeSimilarity:
    def __init__(self, csv_path):
        self.embeddings = self.load_embeddings(csv_path)

    def load_embeddings(self, path):
        df = pd.read_csv(path)

        embeddings = {}

        # Detect format
        if 'embedding' in df.columns:
            # Format B
            for _, row in df.iterrows():
                code = row['code']
                vec = np.array(ast.literal_eval(row['embedding']))
                embeddings[code] = vec
        else:
            # Format A
            feature_cols = [col for col in df.columns if col != 'code']
            for _, row in df.iterrows():
                code = row['code']
                vec = row[feature_cols].values.astype(float)
                embeddings[code] = vec

        return embeddings

    def cosine_similarity(self, v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def get_similarity(self, code1, code2):
        if code1 not in self.embeddings:
            raise ValueError(f"{code1} not found")
        if code2 not in self.embeddings:
            raise ValueError(f"{code2} not found")

        v1 = self.embeddings[code1]
        v2 = self.embeddings[code2]

        return self.cosine_similarity(v1, v2)

    def top_k_similar(self, code, k=5):
        if code not in self.embeddings:
            raise ValueError(f"{code} not found")

        target_vec = self.embeddings[code]
        scores = []

        for other_code, vec in self.embeddings.items():
            if other_code == code:
                continue

            sim = self.cosine_similarity(target_vec, vec)
            scores.append((other_code, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# -------- Usage --------
if __name__ == "__main__":
    sim = CodeSimilarity("embeddings.csv")

    code1 = "N179"
    code2 = "I10"

    print("Similarity:", sim.get_similarity(code1, code2))

    print("\nTop 5 similar to N179:")
    for c, s in sim.top_k_similar("N179", 5):
        print(c, round(s, 4))
