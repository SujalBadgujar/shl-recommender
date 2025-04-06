# model.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = torch.tensor(np.load("corpus_embeddings.npy"))
with open("corpus_data.json") as f:
    shl_data = json.load(f)

corpus_texts = [
    (item["title"] + " " + item["description"]).lower()
    for item in shl_data
]

def recommend(query, top_k=10):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, corpus_embeddings)[0]

    keywords = query.lower().split()
    for i, text in enumerate(corpus_texts):
        if any(keyword in text for keyword in keywords):
            scores[i] += 0.2

    top_results = torch.topk(scores, k=top_k)
    return [shl_data[i] for i in top_results.indices]
