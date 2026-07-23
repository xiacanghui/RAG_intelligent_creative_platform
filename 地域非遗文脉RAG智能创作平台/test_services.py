import requests

print("1. Testing RAG...")
rag_resp = requests.post("http://localhost:8001/search", json={"query":"test","n_results":5,"similarity_threshold":0.0}, timeout=10)
rag_data = rag_resp.json()
print(f"   RAG: has_results={rag_data.get('has_results')}")

print("2. Testing LLM (non-stream)...")
try:
    llm_resp = requests.post("http://localhost:8002/generate", json={"prompt":"hello","system_prompt":"test","max_tokens":10}, timeout=60)
    print(f"   LLM: {llm_resp.status_code}")
    print(f"   Response: {llm_resp.json()}")
except Exception as e:
    print(f"   LLM Error: {e}")

print("3. Testing LLM (stream)...")
try:
    llm_resp = requests.post("http://localhost:8002/generate/stream", json={"prompt":"hello","system_prompt":"test","max_tokens":10,"stream":True}, stream=True, timeout=60)
    print(f"   LLM Stream Status: {llm_resp.status_code}")
    for i, line in enumerate(llm_resp.iter_lines()):
        if line:
            print(f"   Line {i}: {line.decode('utf-8')[:100]}")
        if i > 5:
            break
except Exception as e:
    print(f"   LLM Stream Error: {e}")
