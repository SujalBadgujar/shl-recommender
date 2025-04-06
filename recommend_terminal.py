from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import torch

# 1. Load model & saved data
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = torch.tensor(np.load("corpus_embeddings.npy"))
with open("corpus_data.json") as f:
    shl_data = json.load(f)

# Optional: lowercased corpus for keyword boosting
corpus_texts = [(
    item["title"] + " " + item["description"]
).lower() for item in shl_data]

# 2. Recommender with keyword boosting
def recommend(query, top_k=10):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, corpus_embeddings)[0]

    # Boost score if keyword appears in the original text
    keywords = query.lower().split()  # simple keyword split
    for i, text in enumerate(corpus_texts):
        if any(keyword in text for keyword in keywords):
            scores[i] += 0.2  # boost amount (you can tune this)

    top_results = torch.topk(scores, k=top_k)

    print(f"\nTop {top_k} SHL Test Recommendations:\n")
    print("{:<50} {:<10} {:<10} {:<10} {:<10}".format(
        "Title", "Remote", "Adaptive", "Duration", "Type"))
    print("=" * 90)

    for score, idx in zip(top_results.values, top_results.indices):
        item = shl_data[idx]
        print("{:<50} {:<10} {:<10} {:<10} {:<10}".format(
            item["title"][:47] + "..." if len(item["title"]) > 50 else item["title"],
            item["remote_testing"],
            item["adaptive_rt"],
            item["duration"],
            item["test_type"]
        ))

# 3. Run in terminal
if __name__ == "__main__":
    user_query = input("🔍 Enter job description or query: ")
    recommend(user_query)
