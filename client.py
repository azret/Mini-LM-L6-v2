import os, json, requests, time, jwt

BASE_URL = "http://127.0.0.1:8000"
# BASE_URL = "https://minilm-l6-v2-fsdhaggedqfrddhg.eastus-01.azurewebsites.net"
# BASE_URL = "https://api.openai.com"

JWT_SECRET = os.getenv("APP_JWT_SECRET", "{CHANGEME}")
JWT_ALG = os.getenv("APP_JWT_ALG", "HS256")
JWT_ISS = os.getenv("APP_JWT_ISS", "")
JWT_LEEWAY_SECONDS = int(os.getenv("APP_JWT_LEEWAY", "30"))

def issue_token(
    subject: str,
    expires_in_seconds: int = 365 * 24 * 60 * 60, # 1 year
    extra_claims: dict | None = None,
) -> str:
    r"""
    Issue a long-lived JWT (default: 1 year).
    """
    now = int(time.time())
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_in_seconds,
    }
    if JWT_ISS:
        payload["iss"] = JWT_ISS
    if extra_claims:
        payload.update(extra_claims)
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return token

from concurrent.futures import ThreadPoolExecutor, as_completed

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "FastAPI is handling this request on Azure right now.",
    "Latency tests should include both warm and cold runs.",
    "Embeddings let us compare text by semantic similarity.",
    "This is a short sentence.",
    "This is a slightly longer sentence that should still fit comfortably under the max token limit.",
    "🚀 Unicode emojis should not break tokenization.",
    "Caffè con panna is delicious.",
    "こんにちは、これは埋め込みのテストです。",
    "    The model should ignore leading and trailing whitespace.    ",
    "HTTP 200 is good; HTTP 404 is still useful for latency baselines.",
    "Azure App Service adds a bit of overhead compared to localhost.",
    "OpenAI’s endpoint is usually between 200ms and 400ms for small payloads.",
    "Please summarize the following paragraph in one sentence.",
    "The capital of France is Paris.",
    "1 2 3 4 5 6 7 8 9 10",
    "Special characters: !@#$%^&*()_+[]{}|;':\",./<>?",
    "Here is some code: `def hello(name): return f\"Hello, {name}\"`.",
    "Large language models can generate and embed text.",
    "Running multiple concurrent requests should hit the worker pool.",
    "This sentence is intentionally verbose so that we can check how the model handles inputs that approach the maximum text length parameter configured on the server.",
    "The weather today is cloudy with a chance of performance regressions.",
    "Please check CPU usage and memory consumption during this batch.",
    "Similar sentence to number 4: embeddings allow us to measure how close two pieces of text are.",
    "Totally unrelated topic: penguins live in the Southern Hemisphere.",
    "Another unrelated topic: GPU utilization on Azure can vary.",
    "lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Multiline\ntext\nshould\nstill\nwork.",
    "I wonder how fast this request will be when four workers are busy.",
    "End of test batch.",
]

def do_one_request(token: str, idx: int) -> float:
    t0 = time.time()
    url = f"{BASE_URL}/v1/embeddings"
    payload = {
        "model": "MiniLM-L6-v2",
        "dimension": 384,
        "input": sentences,
    }
    if BASE_URL == "https://api.openai.com":
        payload["model"] = "text-embedding-3-small"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers)
    data = resp.json()
    elapsed = time.time() - t0
    return (idx, resp.status_code, elapsed, len(data["data"]))

def loadtest():
    token = issue_token(subject="client.py")
    if BASE_URL == "https://api.openai.com":
        token = os.getenv("OPENAI_API_KEY", "")
    total_requests = 100 # how many requests in total
    concurrency = 1 # how many to run at the same time
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(do_one_request, token, i)
            for i in range(total_requests)
        ]
        for fut in as_completed(futures):
            idx, status, elapsed, len_ = fut.result()
            print(f"[{idx:03d}] status={status} latency={elapsed*1000:.2f}ms batch={len_}")

if __name__ == "__main__":
    loadtest()
