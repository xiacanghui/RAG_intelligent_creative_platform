"""
业务网关微服务 (端口 8000)
职责：前端接口、会话管理、任务分发、资源监控
"""
import os
import json
import time
import traceback
import psutil
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from collections import deque
from threading import Lock

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from common.config import (
    GATEWAY_PORT, RAG_SERVICE_PORT, LLM_SERVICE_PORT,
    MEMORY_THRESHOLD_LOW, MEMORY_THRESHOLD_HIGH,
    KNOWLEDGE_BASE_DIR, LOG_DIR
)

# ============================================
# FastAPI 应用
# ============================================
app = FastAPI(title="非遗创作平台网关")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 微服务地址
RAG_SERVICE_URL = f"http://localhost:{RAG_SERVICE_PORT}"
LLM_SERVICE_URL = f"http://localhost:{LLM_SERVICE_PORT}"

# ============================================
# 并发控制
# ============================================
MAX_CONCURRENT = 3
_lock = Lock()
_active = 0

def _acquire():
    global _active
    with _lock:
        if _active >= MAX_CONCURRENT:
            return False
        _active += 1
        return True

def _release():
    global _active
    with _lock:
        _active = max(0, _active - 1)

# 请求日志
request_logs = deque(maxlen=100)

def log_request(endpoint, status, detail=""):
    request_logs.append({
        "time": time.strftime("%H:%M:%S"),
        "endpoint": endpoint,
        "status": status,
        "detail": detail
    })

# ============================================
# 数据模型
# ============================================
class ChatRequest(BaseModel):
    query: str
    model_source: str = "local"
    similarity_threshold: float = 0.1
    top_k: int = 5

class PatternRequest(BaseModel):
    category: str
    style_description: str

class ContentGenerateRequest(BaseModel):
    template_type: str
    topic: str
    extra_info: str = ""

# ============================================
# 内存监控
# ============================================
def get_memory():
    mem = psutil.virtual_memory()
    return {
        "percent": mem.percent,
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2)
    }

def get_recommended_channel():
    mem = get_memory()
    if mem["percent"] >= MEMORY_THRESHOLD_HIGH:
        return "cloud"
    return "local"

# ============================================
# 内容过滤（简化版）
# ============================================
class ContentFilter:
    BLOCKED = ["暴力", "色情", "违法"]
    def check(self, text):
        for word in self.BLOCKED:
            if word in text:
                return None
        return text

content_filter = ContentFilter()

# ============================================
# API 路由 - 健康检测
# ============================================
@app.get("/api/model/health")
async def model_health():
    """Ollama 健康检测（返回前端期望格式）"""
    try:
        resp = requests.get(f"{LLM_SERVICE_URL}/status", timeout=5)
        data = resp.json()
        return {
            "ollama_running": data.get("ollama_running", False),
            "model_available": data.get("model_loaded", False),
            "model_name": data.get("local_model", ""),
            "available_models": [data.get("local_model", "")],
            "message": "" if data.get("model_loaded") else "模型未加载"
        }
    except:
        return {
            "ollama_running": False,
            "model_available": False,
            "model_name": "",
            "available_models": [],
            "message": "推理服务未启动"
        }

@app.get("/api/model/list")
async def model_list():
    """可用模型列表"""
    try:
        resp = requests.get(f"{LLM_SERVICE_URL}/status", timeout=5)
        data = resp.json()
        return {"models": [data.get("local_model", "")], "current": data.get("local_model", "")}
    except:
        return {"models": [], "current": ""}

@app.post("/api/model/switch")
async def switch_model(source: str, model_name: str = None):
    """切换模型通道"""
    return {"message": f"已切换至{source}通道", "current_model": model_name or "auto"}

@app.get("/api/memory")
async def get_memory_api():
    """系统资源监控"""
    return get_memory()

# ============================================
# API 路由 - 智能问答
# ============================================
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """非遗智能问答（流式输出）"""
    if not _acquire():
        raise HTTPException(status_code=429, detail="请求过多，请稍后")
    
    try:
        log_request("/api/chat/stream", "started", request.query[:50])
        filtered = content_filter.check(request.query)
        if not filtered:
            raise HTTPException(status_code=400, detail="输入包含敏感词")
        
        # 1. 向量检索
        try:
            rag_resp = requests.post(
                f"{RAG_SERVICE_URL}/search",
                json={
                    "query": filtered,
                    "n_results": request.top_k,
                    "similarity_threshold": request.similarity_threshold
                },
                timeout=10
            )
            rag_data = rag_resp.json()
        except Exception as e:
            print(f"[Gateway] RAG请求失败: {e}")
            rag_data = {"sources": [], "context": "", "has_results": False}
        
        sources = rag_data.get("sources", [])
        context = rag_data.get("context", "")
        has_rag = rag_data.get("has_results", False)
        
        # 2. 构建 prompt
        if has_rag:
            system_prompt = "你是一个专业的非遗文化知识助手。请基于提供的非遗资料回答。"
            prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{filtered}\n\n请基于参考资料回答："
        else:
            system_prompt = "你是一个专业的非遗文化知识助手。本地知识库暂无匹配。"
            prompt = f"用户问题：{filtered}\n\n请回答："
        
        # 3. 推理
        use_cloud = get_recommended_channel() == "cloud"
        
        def event_stream():
            full_text = ""
            try:
                resp = requests.post(
                    f"{LLM_SERVICE_URL}/generate/stream",
                    json={
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "use_cloud": use_cloud,
                        "stream": True
                    },
                    stream=True,
                    timeout=120
                )
                
                for line in resp.iter_lines():
                    if line:
                        chunk = line.decode('utf-8')
                        if '[DONE]' in chunk:
                            break
                        full_text += chunk
                        yield f"data: {chunk}\n\n"
            except Exception as e:
                print(f"[Gateway] 推理错误: {e}")
                yield f"data: [ERROR]{str(e)}\n\n"
            finally:
                _release()
            
            meta = {
                "sources": sources,
                "has_rag": has_rag,
                "token_count": len(full_text) // 2,
                "channel": "cloud" if use_cloud else "local"
            }
            yield f"data: [DONE]{json.dumps(meta, ensure_ascii=False)}\n\n"
            log_request("/api/chat/stream", "completed", f"rag={has_rag}")
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    
    except HTTPException:
        _release()
        raise
    except Exception as e:
        _release()
        traceback.print_exc()
        log_request("/api/chat/stream", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# API 路由 - 纹样生成
# ============================================
@app.post("/api/pattern/generate")
async def generate_pattern(request: PatternRequest):
    """纹样生成（自动关联知识库）"""
    # 获取知识库素材
    kb_context = ""
    try:
        rag_resp = requests.post(
            f"{RAG_SERVICE_URL}/search",
            json={"query": f"{request.category}纹样工艺特点", "n_results": 3},
            timeout=10
        )
        rag_data = rag_resp.json()
        if rag_data.get("has_results"):
            kb_context = f"\n\n【知识库参考】\n{rag_data['context'][:500]}"
    except:
        pass
    
    prompt = f"传统{request.category}纹样设计，{request.style_description}，高清细节，8K分辨率{kb_context}\n\n请基于非遗史料生成绘图提示词："
    
    try:
        resp = requests.post(
            f"{LLM_SERVICE_URL}/generate",
            json={
                "prompt": prompt,
                "system_prompt": "你是一个非遗纹样设计专家，生成专业的AI绘图提示词。",
                "max_tokens": 1024
            },
            timeout=120
        )
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# API 路由 - 文创内容
# ============================================
@app.post("/api/content/generate")
async def generate_content(request: ContentGenerateRequest):
    """文创内容生成（RAG绑定）"""
    # 获取知识库素材
    kb_context = ""
    try:
        rag_resp = requests.post(
            f"{RAG_SERVICE_URL}/search",
            json={"query": request.topic, "n_results": 3},
            timeout=10
        )
        rag_data = rag_resp.json()
        if rag_data.get("has_results"):
            kb_context = f"\n\n【参考史料】\n{rag_data['context'][:600]}"
    except:
        pass
    
    templates = {
        "导览解说": f"请为{request.topic}撰写导览解说文案，要求详实准确、语言生动{kb_context}",
        "短视频脚本": f"请为{request.topic}创作非遗短视频脚本{kb_context}",
        "宣传文案": f"请为{request.topic}撰写文创宣传文案{kb_context}",
        "朋友圈文案": f"请为{request.topic}创作国风朋友圈文案{kb_context}"
    }
    
    prompt = templates.get(request.template_type, templates["导览解说"])
    
    try:
        resp = requests.post(
            f"{LLM_SERVICE_URL}/generate",
            json={
                "prompt": prompt,
                "system_prompt": "你是一个专业的非遗文创内容创作助手。",
                "max_tokens": 2048
            },
            timeout=120
        )
        result = resp.json()
        return {
            "content": result.get("response", ""),
            "template_type": request.template_type,
            "model_used": result.get("source", "unknown"),
            "token_count": result.get("token_count", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# API 路由 - 知识库管理
# ============================================
@app.get("/api/knowledge/status")
async def knowledge_status():
    """知识库状态"""
    try:
        resp = requests.get(f"{RAG_SERVICE_URL}/stats", timeout=5)
        return resp.json()
    except:
        return {"total_documents": 0, "categories": [], "documents": []}

@app.get("/api/knowledge/documents")
async def list_documents():
    """文档列表"""
    try:
        resp = requests.get(f"{RAG_SERVICE_URL}/documents", timeout=5)
        return resp.json()
    except:
        return {"documents": [], "total": 0}

@app.post("/api/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    """上传文档"""
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="仅支持 TXT/MD 格式")
    
    try:
        content = await file.read()
        files = {"file": (file.filename, content, "text/plain")}
        resp = requests.post(f"{RAG_SERVICE_URL}/upload", files=files, timeout=30)
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str):
    """删除文档"""
    try:
        resp = requests.delete(f"{RAG_SERVICE_URL}/documents/{doc_id}", timeout=5)
        return resp.json()
    except:
        raise HTTPException(status_code=404, detail="文档不存在")

# ============================================
# API 路由 - 日志
# ============================================
@app.get("/api/logs")
async def get_logs():
    return {"logs": list(request_logs), "active_requests": _active}

@app.get("/api/opensource/license")
async def get_license():
    return [
        {"name": "FastAPI", "license": "MIT License", "url": "https://github.com/tiangolo/fastapi"},
        {"name": "TailwindCSS", "license": "MIT License", "url": "https://github.com/tailwindlabs/tailwindcss"},
        {"name": "Ollama", "license": "MIT License", "url": "https://github.com/ollama/ollama"},
        {"name": "Qwen3", "license": "Apache License 2.0", "url": "https://github.com/QwenLM/Qwen3"},
        {"name": "DeepSeek", "license": "MIT License", "url": "https://github.com/deepseek-ai"}
    ]

# ============================================
# 静态文件服务
# ============================================
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>服务启动中...</h1>")

# ============================================
# 启动
# ============================================
@app.on_event("startup")
async def startup():
    LOG_DIR.mkdir(exist_ok=True)
    print(f"[Gateway] 网关服务启动，端口: {GATEWAY_PORT}")
    print(f"[Gateway] 向量服务: {RAG_SERVICE_URL}")
    print(f"[Gateway] 推理服务: {LLM_SERVICE_URL}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)
