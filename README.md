import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

def classify_comments_advanced(comments):
    # --- 1. Data Cleaning & Edge Case: Empty List ---
    if not comments or len(comments) < 2:
        return pd.DataFrame({'comment': comments, 'cluster_id': [0]*len(comments), 'heading': ['Insufficient Data']})

    # Remove duplicates and empty strings to avoid skewing clusters
    clean_comments = list(dict.fromkeys([str(c).strip() for c in comments if len(str(c).strip()) > 0]))
    
    # --- 2. Generate Embeddings ---
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(clean_comments)

    # --- 3. Find Optimal Number of Clusters (Silhouette Method) ---
    # We test ranges from 2 up to 10 (or the max possible)
    max_k = min(len(clean_comments) - 1, 10)
    best_k = 2
    if max_k > 2:
        sil_scores = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(embeddings)
            sil_scores.append(silhouette_score(embeddings, km.labels_))
        best_k = range(2, max_k + 1)[np.argmax(sil_scores)]
    
    # --- 4. Final Clustering ---
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    df = pd.DataFrame({'comment': clean_comments, 'cluster_id': clusters})

    # --- 5. Generate Intelligent Headings (using TF-IDF keywords) ---
    # This finds words that are frequent in one cluster but rare in others
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
    headings = {}
    
    for i in range(best_k):
        cluster_texts = df[df['cluster_id'] == i]['comment']
        # Fallback if cluster is too small for TF-IDF
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            keywords = vectorizer.get_feature_names_out()
            headings[i] = "Topic: " + ", ".join(keywords[:3])
        except:
            headings[i] = f"Cluster {i} (General)"

    df['cluster_heading'] = df['cluster_id'].map(headings)
    
    # Merge back to original list to preserve original order and duplicates if necessary
    original_df = pd.DataFrame({'comment': comments})
    final_df = original_df.merge(df, on='comment', how='left').fillna("Empty/Filtered")
    
    return final_df

# --- Test Drive ---
input_data = [
    "The app keeps crashing on my iPhone", "Crashing constantly", 
    "I love the new UI design", "The interface looks very modern",
    "How do I reset my password?", "Password recovery is not working",
    " ", # Edge case: empty string
    "The app keeps crashing on my iPhone" # Edge case: duplicate
]

df_output = classify_comments_advanced(input_data)
print(df_output)
