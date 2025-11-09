import os
import requests
import time
import numpy

from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
# BASE_URL = "https://api.openai.com"

def fetchembeddings(docs) -> float:
    # Just a dummy token for local testing
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjbGllbnQucHkiLCJpYXQiOjE3NjI3MjQ4MjMsImV4cCI6MTc5NDI2MDgyM30.-p_fzBBeUSIqYfPjkmnaoGsdamppyMp6OtXPNVapDt0"
    if BASE_URL == "https://api.openai.com":
        token = os.getenv("OPENAI_API_KEY", "")
    url = f"{BASE_URL}/v1/embeddings"
    payload = {
        "model": "MiniLM-L6-v2",
        "dimension": 384,
        "input": docs,
    }
    if BASE_URL == "https://api.openai.com":
        payload["model"] = "text-embedding-3-small"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers)
    data = resp.json()
    return [
            d["embedding"] for d in data["data"]
        ]

def doctext(doc):
    return f"**Subject**:\r\n{doc["subject"]}\r\n**Body**:\r\n{doc["body"]}"

def fetchbatch(sql, offset: int, limit: int = 128):
    if limit <= 0:
        raise ValueError("'limit' must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit > 128:
        raise ValueError("'limit' must be at most 128")
    if offset >= len(sql):
        return []
    batch = sql[offset:offset+limit]
    return batch

def batchsize():
    limit = 128 if BASE_URL.endswith("127.0.0.1:8000") or BASE_URL == "https://api.openai.com" else 10
    return limit

def buildFromScratch():
    import yaml
    print(f"Loading {str(Path(__file__).with_suffix(".yaml"))}")
    with open(str(Path(__file__).with_suffix(".yaml")), "r", encoding="utf-8") as f:
        db = yaml.safe_load(f)[:1000]
    offset = 0
    b = 0
    while True:
        batch = fetchbatch(db, offset, limit=batchsize())
        if not batch or len(batch) == 0:
            break
        docs = [doctext(t) for t in batch]
        print(f"Processing batch[{b}] of {len(docs)}...")
        offset += len(docs)
        b += 1
        embeddings = fetchembeddings(docs)
        assert len(embeddings) == len(docs)
        for i in range(len(batch)):
            batch[i]["embedding"] = embeddings[i]
        time.sleep(0.3) # small sleep to respect rate limits
    return db

if __name__ == "__main__":
    # Build a pretend vector db from scratch
    db = buildFromScratch()
    vecs = []
    for r in db:
        v = numpy.array(r["embedding"], dtype=numpy.float32)
        norm = numpy.linalg.norm(v) # L2 normalization. Don't really need this if the server already does it.
        if norm > 0:
            v = v / norm
        vecs.append(v)
    vecs = numpy.vstack(vecs) # shape: (N, D)
    # Start prompting
    while True:
        try:
            queryString = input(f"\x1b[38;2;{66};{244};{66}m{"Model"}\x1b[0m>").strip()
            if queryString == "cls":
                os.system('cls' if os.name == 'nt' else 'clear')
                print("\x1b[H\x1b[2J\x1b[3J", end="")
                continue
            if queryString:
                print(f"\x1b[38;2;{255};{255};{255}m")
                try:
                    q = fetchembeddings(queryString)
                    q = numpy.array(q, dtype=numpy.float32) # shape: (1, D)
                    q = numpy.squeeze(q) # shape: (D,)
                    norm = numpy.linalg.norm(q) # L2 normalization. Don't really need this if the server already does it.
                    if norm > 0:
                        q = q / norm
                    sims = vecs @ q
                    topk = 5
                    topk = numpy.argsort(sims)[-topk:][::-1]
                    for rank, idx in enumerate(topk, 1):
                        print(f"cosine: {sims[idx]:.3f}")
                        print(f"id: {db[idx]["id"]}")
                        print(f"account: {db[idx]["account"]}")
                        print(f"channel: {db[idx]["channel"]}")
                        print(f"priority: {db[idx]["priority"]}")
                        print(f"subject: {db[idx]["subject"]}")
                        print(f"body: {db[idx]["body"]}")
                        print()
                finally:
                    print("\x1b[0m")
        except KeyboardInterrupt:
            exit()
