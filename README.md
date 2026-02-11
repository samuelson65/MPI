import pandas as pd
from sentence_transformers import SentenceTransformer, util

def classify_pi_simple(corpus):
    # 1. Clean the input list
    clean_corpus = [str(c) if c else "" for c in corpus]

    # 2. Define your actual Audit Categories (The "Headings")
    # Change these strings to match exactly what you want to see in your report
    target_categories = [
        "Clinical Validation / MCC-CC Issues",
        "30-Day Related Readmission",
        "Coding & DRG Sequencing Error",
        "Medical Necessity / Documentation",
        "Unrelated Readmission"
    ]

    # 3. Load a lightweight model (Runs fast on any laptop CPU)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Create "Embeddings" for both comments and headings
    comment_embeddings = model.encode(clean_corpus, convert_to_tensor=True)
    category_embeddings = model.encode(target_categories, convert_to_tensor=True)

    # 5. Find the best match for each comment
    results = []
    # cosine_scores compares every comment against every category
    cosine_scores = util.cos_sim(comment_embeddings, category_embeddings)

    for i, comment in enumerate(clean_corpus):
        if not comment.strip():
            results.append({"comment": comment, "cluster_heading": "Empty/No Data", "confidence": 0})
            continue
            
        # Find which category had the highest similarity score
        best_category_idx = cosine_scores[i].argmax().item()
        
        results.append({
            "comment": comment,
            "cluster_heading": target_categories[best_category_idx],
            "confidence": round(float(cosine_scores[i][best_category_idx]), 3)
        })

    return pd.DataFrame(results)

# --- Usage ---
corpus = [
    "Readmission within 30 days for same DRG 291",
    "Clinical validation failed; MCC not supported by labs",
    "Principal diagnosis sequencing is incorrect",
    "Patient returned with sepsis following discharge",
    " " # Empty case
]

df = classify_pi_simple(corpus)
print(df)
