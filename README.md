import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering

def cluster_with_highlights(corpus):
    # 1. Clean data and remove duplicates to avoid bias
    unique_corpus = list(set([str(c).strip() for c in corpus if c]))
    df = pd.DataFrame({'comment': unique_corpus})
    
    # 2. Semantic Embeddings (Understand the meaning, not just words)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(unique_corpus, convert_to_tensor=True)

    # 3. Dynamic Clustering
    # Adjust distance_threshold (e.g., 1.0 to 1.5) to make clusters tighter or broader
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.3)
    df['cluster_id'] = clustering_model.fit_predict(embeddings.cpu())

    # 4. Extract the "Highlight" for each cluster
    # We find the 'Centroid' sentence—the one most similar to all others in its group
    highlights = {}
    for cluster_id in df['cluster_id'].unique():
        cluster_indices = df[df['cluster_id'] == cluster_id].index
        
        if len(cluster_indices) == 1:
            # Only one comment? That comment is the highlight
            highlights[cluster_id] = df.loc[cluster_indices[0], 'comment']
        else:
            # Find the sentence with the highest average similarity to others in the cluster
            cluster_embeds = embeddings[cluster_indices]
            cos_scores = util.cos_sim(cluster_embeds, cluster_embeds)
            
            # Sum the scores for each sentence (excluding itself)
            central_idx_in_cluster = cos_scores.sum(dim=1).argmax().item()
            real_index = cluster_indices[central_idx_in_cluster]
            
            # Clean up the highlight (take the first 60 chars or the first sentence)
            full_text = df.loc[real_index, 'comment']
            highlight = full_text.split('.')[0][:80] # Get the primary point
            highlights[cluster_id] = f"Highlight: {highlight}..."

    # 5. Map back to DataFrame
    df['cluster_heading'] = df['cluster_id'].map(highlights)
    
    # Final output merged back to original list order if needed
    original_df = pd.DataFrame({'comment': corpus})
    return original_df.merge(df, on='comment', how='left').fillna("N/A")

# --- Example with Audit Data ---
corpus = [
    "the patient was readmitted to the same hospital and the provider is responsible for such things",
    "provider liability indicated as patient returned to same facility within 48 hours",
    "incorrect DRG applied to the claim, documentation does not support MCC",
    "DRG 291 was billed but the clinical records fail to justify the secondary diagnosis",
    "documentation gap found: no evidence of sepsis in the lab results for this discharge"
]

final_df = cluster_with_highlights(corpus)
print(final_df[['cluster_heading', 'comment']])
