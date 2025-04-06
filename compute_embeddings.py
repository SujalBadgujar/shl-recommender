from sentence_transformers import SentenceTransformer
import numpy as np
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('shl_product_catalog_with_duration.json') as f:
    shl_data = json.load(f)

titles = [item['title'] for item in shl_data]
title_embeddings = model.encode(titles, convert_to_numpy=True)

np.save('title_embeddings.npy', title_embeddings)
with open('titles.json', 'w') as f:
    json.dump(shl_data, f)

print("✅ Embeddings saved.")
