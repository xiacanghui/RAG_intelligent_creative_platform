"""
双推理引擎微服务 (端口 8002)
职责：本地 Ollama 推理 + DeepSeek 云端 API 推理，自动切换
"""
import os
import time
import json
import psutil
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Generator
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from common.config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    MEMORY_THRESHOLD_LOW, MEMORY_THRESHOLD_HIGH, LLM_SERVICE_PORT
)

# ============================================
# FastAPI 应用
# ============================================
app = FastAPI(title="LLM Inference Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================
# 推理通道状态管理
# ============================================
class InferenceState:
    def __init__(self):
        self.ollama_running = False
        self.ollama_model_loaded = False
        self.last_inference_time = 0
        self.local_model = OLLAMA_MODEL
        self.cloud_model = DEEPSEEK_MODEL
        self.cloud_enabled = bool(DEEPSEEK_API_KEY)
        self.total_local_tokens = 0
        self.total_cloud_tokens = 0

state = InferenceState()

# ============================================
# 数据模型
# ============================================
class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = "你是一个专业的非遗文化知识助手。"
    use_cloud: Optional[bool] = None  # None=自动判断
    stream: bool = False
    max_tokens: int = 2048

class GenerateResponse(BaseModel):
    response: str
    source: str  # "local" or "cloud"
    token_count: int
    latency_ms: int

# ============================================
# 内存监控
# ============================================
def get_memory_usage() -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    return {
        "percent": mem.percent,
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "total_gb": round(mem.total / (1024**3), 2)
    }

def should_use_cloud() -> bool:
    """根据内存阈值判断是否使用云端推理"""
    mem = get_memory_usage()
    if mem["percent"] >= MEMORY_THRESHOLD_HIGH:
        return True
    if state.cloud_enabled and mem["percent"] >= MEMORY_THRESHOLD_LOW:
        return True
    return False

# ============================================
# Ollama 本地推理
# ============================================
def check_ollama_health() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            state.ollama_running = True
            models = resp.json().get("models", [])
            state.ollama_model_loaded = any(
                m.get("name", "").startswith(state.local_model) for m in models
            )
            return True
    except:
        pass
    state.ollama_running = False
    state.ollama_model_loaded = False
    return False

def generate_local(prompt: str, system_prompt: str, max_tokens: int = 2048) -> Dict:
    """本地 Ollama 推理"""
    start = time.time()
    
    full_prompt = f"【系统指令】{system_prompt}\n\n{prompt}"
    
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": state.local_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            },
            timeout=120
        )
        result = resp.json()
        latency = int((time.time() - start) * 1000)
        token_count = result.get("eval_count", len(result.get("response", "")) // 2)
        state.total_local_tokens += token_count
        state.last_inference_time = time.time()
        
        return {
            "response": result.get("response", ""),
            "source": "local",
            "token_count": token_count,
            "latency_ms": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"本地推理失败: {str(e)}")

def generate_local_stream(prompt: str, system_prompt: str, max_tokens: int = 2048) -> Generator:
    """本地 Ollama 流式推理"""
    full_prompt = f"【系统指令】{system_prompt}\n\n{prompt}"
    
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": state.local_model,
                "prompt": full_prompt,
                "stream": True,
                "options": {"num_predict": max_tokens}
            },
            stream=True,
            timeout=120
        )
        
        full_text = ""
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                full_text += token
                yield token
        
        token_count = len(full_text) // 2
        state.total_local_tokens += token_count
        state.last_inference_time = time.time()
        
        meta = {"source": "local", "token_count": token_count}
        yield f"\n[DONE]{json.dumps(meta)}"
        
    except Exception as e:
        yield f"\n[ERROR]{str(e)}"

# ============================================
# DeepSeek 云端推理
# ============================================
def generate_cloud(prompt: str, system_prompt: str, max_tokens: int = 2048) -> Dict:
    """DeepSeek API 推理"""
    if not state.cloud_enabled:
        raise HTTPException(status_code=400, detail="DeepSeek API 未配置")
    
    start = time.time()
    
    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": state.cloud_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": max_tokens
            },
            timeout=60
        )
        
        if resp.status_code != 200:
            raise Exception(f"API 错误: {resp.status_code}")
        
        result = resp.json()
        latency = int((time.time() - start) * 1000)
        content = result["choices"][0]["message"]["content"]
        token_count = result.get("usage", {}).get("total_tokens", len(content) // 2)
        state.total_cloud_tokens += token_count
        
        return {
            "response": content,
            "source": "cloud",
            "token_count": token_count,
            "latency_ms": latency
        }
    except Exception as e:
        # 降级到本地推理
        print(f"[Cloud] 调用失败，降级本地: {e}")
        return generate_local(prompt, system_prompt, max_tokens)

def generate_cloud_stream(prompt: str, system_prompt: str, max_tokens: int = 2048) -> Generator:
    """DeepSeek API 流式推理"""
    if not state.cloud_enabled:
        yield json.dumps({"error": "DeepSeek API 未配置"})
        return
    
    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": state.cloud_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "stream": True
            },
            stream=True,
            timeout=60
        )
        
        full_text = ""
        for line in resp.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        # DeepSeek推理模型可能用reasoning_content
                        content = delta.get("content", "") or delta.get("reasoning_content", "")
                        if content:
                            full_text += content
                            yield content
                    except:
                        pass
        
        token_count = len(full_text) // 2
        state.total_cloud_tokens += token_count
        meta = {"source": "cloud", "token_count": token_count}
        yield f"\n[DONE]{json.dumps(meta)}"
        
    except Exception as e:
        print(f"[Cloud Stream] 降级本地: {e}")
        yield from generate_local_stream(prompt, system_prompt, max_tokens)

# ============================================
# API 路由
# ============================================
@app.get("/health")
async def health():
    check_ollama_health()
    mem = get_memory_usage()
    return {
        "status": "ok",
        "ollama_running": state.ollama_running,
        "model_loaded": state.ollama_model_loaded,
        "cloud_enabled": state.cloud_enabled,
        "memory": mem,
        "local_tokens": state.total_local_tokens,
        "cloud_tokens": state.total_cloud_tokens
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """统一推理接口（自动路由本地/云端）"""
    use_cloud = request.use_cloud
    if use_cloud is None:
        use_cloud = should_use_cloud()
    
    if use_cloud and state.cloud_enabled:
        return generate_cloud(request.prompt, request.system_prompt, request.max_tokens)
    elif state.ollama_model_loaded:
        return generate_local(request.prompt, request.system_prompt, request.max_tokens)
    else:
        raise HTTPException(status_code=503, detail="无可用推理通道")

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """统一流式推理接口"""
    use_cloud = request.use_cloud
    if use_cloud is None:
        use_cloud = should_use_cloud()
    
    def event_stream():
        if use_cloud and state.cloud_enabled:
            yield from generate_cloud_stream(request.prompt, request.system_prompt, request.max_tokens)
        elif state.ollama_model_loaded:
            yield from generate_local_stream(request.prompt, request.system_prompt, request.max_tokens)
        else:
            yield json.dumps({"error": "无可用推理通道"})
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/status")
async def status():
    """推理服务状态"""
    check_ollama_health()
    return {
        "local_model": state.local_model,
        "cloud_model": state.cloud_model,
        "cloud_enabled": state.cloud_enabled,
        "ollama_running": state.ollama_running,
        "model_loaded": state.ollama_model_loaded,
        "memory": get_memory_usage(),
        "recommended_channel": "cloud" if should_use_cloud() else "local"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LLM_SERVICE_PORT)
