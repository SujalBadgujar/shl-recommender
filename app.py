import os
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import torch
import numpy as np
import json

app = Flask(__name__)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Gemini API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Missing Gemini API key. Please set GEMINI_API_KEY environment variable.")

# Configure Gemini model
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Load corpus and embeddings
corpus_embeddings = torch.tensor(np.load("corpus_embeddings.npy"))
with open("data_scraped.json") as f:
    shl_data = json.load(f)

# Prepare lowercased corpus texts for keyword matching
corpus_texts = [
    (item.get("title", "") + " " + item.get("description", "")).lower()
    for item in shl_data
]

def extract_keywords_with_gemini(text):
    """Use Gemini to extract important keywords/skills from the query"""
    prompt = (
        f"Extract the main skills, job roles, or keywords from the following text "
        f"for matching with assessments. Respond with only a comma-separated list of keywords:\n\n{text}"
    )
    try:
        response = gemini_model.generate_content(prompt)
        keywords = response.text.strip().lower().replace("\n", "").split(",")
        return [kw.strip() for kw in keywords if kw.strip()]
    except Exception as e:
        print("Gemini error:", e)
        return []

def recommend(query, top_k=10):
    """Recommend SHL tests based on semantic + Gemini keyword match"""
    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, corpus_embeddings)[0]

    # Use Gemini to extract keywords from the query
    keywords = extract_keywords_with_gemini(query)
    print("Gemini Keywords:", keywords)

    # Boost scores based on keyword overlap with corpus
    for i, text in enumerate(corpus_texts):
        if any(keyword in text for keyword in keywords):
            scores[i] += 0.2  # Boost if keyword is present

    top_results = torch.topk(scores, k=top_k)
    return [shl_data[i] for i in top_results.indices]

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        query = request.form["query"]
        recommendations = recommend(query)
    return render_template("index.html", recommendations=recommendations)

@app.route("/api", methods=["POST"])
def api():
    data = request.get_json()
    query = data.get("query")
    results = recommend(query)
    return jsonify({"results": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
