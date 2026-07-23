import requests
import json

print("=== Full Q&A Test ===")
r = requests.post(
    "http://localhost:8000/api/chat/stream",
    json={"query": "皮影戏是什么", "similarity_threshold": 0.0},
    stream=True
)

full = ""
for line in r.iter_lines():
    if line:
        chunk = line.decode("utf-8")
        if chunk.startswith("data: "):
            data = chunk[6:]
            if data.startswith("[DONE]"):
                meta = json.loads(data[6:])
                print(f"\n\n--- Result ---")
                print(f"RAG: {meta.get('has_rag')}")
                print(f"Channel: {meta.get('channel')}")
                print(f"Tokens: {meta.get('token_count')}")
                print(f"Sources: {meta.get('sources', [])[:2]}")
            else:
                full += data
                print(data, end="", flush=True)

if not full:
    print("No response!")
