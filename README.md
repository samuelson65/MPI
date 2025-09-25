import pandas as pd
import itertools

def compute_metrics(setA, setB):
    """Compute symmetric and asymmetric metrics between two sets."""
    A, B = set(setA), set(setB)
    inter = A & B
    union = A | B
    a_len, b_len = len(A), len(B)
    inter_len, union_len = len(inter), len(union)

    jaccard = inter_len / union_len if union_len else 0.0
    overlap = inter_len / min(a_len, b_len) if min(a_len, b_len) > 0 else 0.0

    contain_A_in_B = inter_len / a_len if a_len > 0 else 0.0
    contain_B_in_A = inter_len / b_len if b_len > 0 else 0.0

    unique_count_A = len(A - B)
    unique_frac_A = unique_count_A / a_len if a_len > 0 else 0.0

    unique_count_B = len(B - A)
    unique_frac_B = unique_count_B / b_len if b_len > 0 else 0.0

    return {
        "jaccard": jaccard,
        "overlap_coeff": overlap,
        "contain_A_in_B": contain_A_in_B,
        "contain_B_in_A": contain_B_in_A,
        "unique_count_A": unique_count_A,
        "unique_frac_A": unique_frac_A,
        "unique_count_B": unique_count_B,
        "unique_frac_B": unique_frac_B
    }

def recommend_action(metrics, thresh_jaccard=0.8, thresh_contain=0.99, thresh_unique=0.05):
    """Provide recommendation based on thresholds."""
    if metrics["contain_A_in_B"] >= thresh_contain and metrics["unique_count_A"] == 0:
        return "A redundant (subset of B)"
    if metrics["contain_B_in_A"] >= thresh_contain and metrics["unique_count_B"] == 0:
        return "B redundant (subset of A)"
    if metrics["jaccard"] >= thresh_jaccard and metrics["unique_frac_A"] <= thresh_unique:
        return "Merge (A mostly covered by B)"
    if metrics["jaccard"] >= thresh_jaccard and metrics["unique_frac_B"] <= thresh_unique:
        return "Merge (B mostly covered by A)"
    return "Keep separate"

def query_similarity(df, thresh_jaccard=0.8, thresh_contain=0.99, thresh_unique=0.05):
    """
    Input: df with columns [query_name, claim_id]
    Output: pairwise similarity dataframe with metrics + recommendation
    """
    # group claims per query
    query_claims = df.groupby("query_name")["claim_id"].apply(set).to_dict()

    results = []
    for (q1, claims1), (q2, claims2) in itertools.combinations(query_claims.items(), 2):
        metrics = compute_metrics(claims1, claims2)
        rec = recommend_action(metrics, thresh_jaccard, thresh_contain, thresh_unique)
        results.append({
            "Query_A": q1,
            "Query_B": q2,
            **metrics,
            "Recommendation": rec
        })

    return pd.DataFrame(results)


# -------------------------------
# 🔹 Example usage
if __name__ == "__main__":
    # Example toy data
    data = [
        ("A", "C1"), ("A", "C2"), ("A", "C3"),
        ("B", "C1"), ("B", "C2"), ("B", "C3"), ("B", "C4"),
        ("C", "C5"), ("C", "C6"),
        ("D", "C2"), ("D", "C3")
    ]
    df = pd.DataFrame(data, columns=["query_name", "claim_id"])

    results_df = query_similarity(df)

    print("\nPairwise Query Similarity:")
    print(results_df)
