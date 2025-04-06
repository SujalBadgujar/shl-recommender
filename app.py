from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import json

app = Flask(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = torch.tensor(np.load("corpus_embeddings.npy"))
with open("corpus_data.json") as f:
    shl_data = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        query = request.form["query"]
        query_emb = model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_emb, corpus_embeddings)[0]
        top_results = torch.topk(scores, k=10)
        recommendations = [shl_data[i] for i in top_results.indices]

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
