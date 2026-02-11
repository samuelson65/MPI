import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering

def analyze_audit_highlights(corpus, claim_amounts=None):
    # 1. Setup DataFrame
    df = pd.DataFrame({'comment': corpus})
    # If no amounts provided, we'll just count occurrences
    df['amount'] = claim_amounts if claim_amounts else [1] * len(df)
    
    # 2. Semantic Embedding (CPU-friendly)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['comment'].astype(str).tolist(), convert_to_tensor=True)

    # 3. Agglomerative Clustering (Unsupervised)
    # distance_threshold: 1.2 is a good middle ground for PI audits
    cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.2)
    df['cluster_id'] = cluster_model.fit_predict(embeddings.cpu())

    # 4. Highlight Extraction (The 'Central' Sentence)
    highlights = {}
    for cid in df['cluster_id'].unique():
        idx = df[df['cluster_id'] == cid].index
        if len(idx) > 0:
            c_embeds = embeddings[idx]
            # Find the sentence that best represents the whole group
            sim_matrix = util.cos_sim(c_embeds, c_embeds)
            central_idx = sim_matrix.sum(dim=1).argmax().item()
            actual_text = df.loc[idx[central_idx], 'comment']
            
            # Formatting the highlight to be punchy
            highlights[cid] = (actual_text[:75] + '...') if len(actual_text) > 75 else actual_text

    df['key_highlight'] = df['cluster_id'].map(highlights)

    # 5. Summary Aggregation for Storytelling
    summary = df.groupby('key_highlight').agg(
        frequency=('comment', 'count'),
        total_impact=('amount', 'sum')
    ).sort_values(by='total_impact', ascending=False).reset_index()

    return df, summary

# --- Input for your Hackathon ---
# Example: Adding claim amounts to show 'Total Impact'
corpus = [
    "the patient was readmitted to the same hospital and the provider is responsible for such things",
    "provider liability indicated as patient returned to same facility within 48 hours",
    "incorrect DRG applied to the claim, documentation does not support MCC",
    "DRG 291 was billed but the clinical records fail to justify the secondary diagnosis",
    "documentation gap found: no evidence of sepsis in the lab results for this discharge",
    "the provider should be held liable for the readmission to the same hospital",
]
# Mock data representing the financial value of each claim
amounts = [5400, 4200, 12000, 11500, 8900, 5100]

full_results, summary_table = analyze_audit_highlights(corpus, amounts)

print("--- CLUSTERED DATA ---")
print(full_results[['key_highlight', 'comment']])
print("\n--- STORYTELLING SUMMARY (HACKATHON VIEW) ---")
print(summary_table)
