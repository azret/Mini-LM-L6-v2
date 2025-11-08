import json, requests

BASE_URL = "http://127.0.0.1:8000"

def main():
    url = f"{BASE_URL}/v1/embeddings"

    payload = {
        "model": "MiniLM-L6-v2",
        "input": "The quick brown fox"
    }

    resp = requests.post(url, json=payload)

    print("status:", resp.status_code)

    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    main()
