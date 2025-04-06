import json

with open("data_scraped.json", "r") as f_in, open("data.jsonl", "w") as f_out:
    data = json.load(f_in)
    for entry in data:
        f_out.write(json.dumps(entry) + "\n")
