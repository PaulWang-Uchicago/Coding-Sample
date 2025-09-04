import json
from math import ceil

# Load your master URL list
with open("urls.json") as f:
    urls = json.load(f)

num_batches = 8
batch_size  = ceil(len(urls) / num_batches)

for i in range(num_batches):
    start = i * batch_size
    end   = start + batch_size
    chunk = urls[start:end]
    with open(f"batch{i+1}.json", "w") as out:
        json.dump(chunk, out, indent=2)
    print(f"Wrote batch{i+1}.json with {len(chunk)} URLs")