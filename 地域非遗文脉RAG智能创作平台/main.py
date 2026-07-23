"""
地域非遗文脉RAG智能创作平台 - FastAPI主服务启动入口
基础工程代码由 Vibe Coding 智能体生成
非遗分类加权检索算法、Token分层节流缓存系统、非遗风格强制校验过滤引擎三大核心业务算法模块由项目负责人独立人工重构开发
"""
import os
import sys
import json
import time
import shutil
import psutil
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from rag_core import RAGCore
from model_switch import ModelSwitchController
from utils.content_check import ContentFilter
from utils.token_control import TokenController
from utils.style_filter import StyleFilter

# ============================================
# 全局配置
# ============================================
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base"))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./vector_store"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
ARCHIVE_DIR = BASE_DIR / "archive"

KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR.mkdir(exist_ok=True)

# ============================================
# 全局实例
# ============================================
rag_core = RAGCore(chroma_persist_dir=str(CHROMA_PERSIST_DIR), knowledge_base_dir=str(KNOWLEDGE_BASE_DIR))
model_controller = ModelSwitchController()
content_filter = ContentFilter()
token_controller = TokenController()
style_filter = StyleFilter()

# ============================================
# 请求模型
# ============================================
class ChatRequest(BaseModel):
    query: str
    similarity_threshold: float = 0.1
    top_k: int = 5
    provider: Optional[str] = None

class PatternRequest(BaseModel):
    category: str
    style_description: str

class ContentGenerateRequest(BaseModel):
    template_type: str
    topic: str
    extra_info: str = ""
    provider: Optional[str] = None

class KnowledgeStatus(BaseModel):
    total_documents: int
    categories: list
    total_chunks: int
    last_updated: str
    documents: list = []

class ImageGenRequest(BaseModel):
    prompt: str
    category: str = "通用"
    size: str = "1024x1024"

# ============================================
# 生命周期
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[启动] 初始化RAG核心...")
    rag_core.initialize()
    print("[启动] RAG核心就绪")
    yield
    print("[关闭] 保存向量库...")
    rag_core.save()

app = FastAPI(title="地域非遗文脉RAG智能创作平台", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================
# 前端页面
# ============================================
@app.get("/", response_class=HTMLResponse)
async def index():
    p = STATIC_DIR / "index.html"
    return FileResponse(str(p)) if p.exists() else HTMLResponse("<h1>平台加载中...</h1>")

# ============================================
# 模型健康检测
# ============================================
@app.get("/api/model/health")
async def model_health():
    import requests as req
    ollama_running = False
    model_available = False
    model_name = ""
    try:
        r = req.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_running = True
            models = r.json().get("models", [])
            for m in models:
                if m.get("name") == os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b"):
                    model_available = True
                    model_name = m["name"]
                    break
    except Exception:
        pass
    return {"ollama_running": ollama_running, "model_available": model_available, "model_name": model_name}

# ============================================
# 模型状态 & 切换
# ============================================
@app.get("/api/model/status")
async def model_status():
    status = model_controller.get_status()
    try:
        mem = psutil.virtual_memory()
        status["memory"] = {"total_gb": round(mem.total / 1024**3, 1), "used_gb": round(mem.used / 1024**3, 1), "percent": mem.percent}
    except Exception:
        status["memory"] = None
    return status

@app.post("/api/model/switch")
async def switch_model(source: str):
    try:
        model_controller.switch_to(source)
        cfg = model_controller.get_config(source)
        return {"message": f"已切换至{cfg['name']}", "display_name": cfg["name"], "current_model": source}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================
# 系统信息
# ============================================
@app.get("/api/memory")
async def get_memory():
    try:
        mem = psutil.virtual_memory()
        return {"total_gb": round(mem.total / 1024**3, 1), "used_gb": round(mem.used / 1024**3, 1), "percent": mem.percent}
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "percent": 0}

@app.get("/api/network/check")
async def check_network():
    try:
        import requests as req
        r = req.get("https://www.baidu.com", timeout=3)
        return {"available": r.status_code < 400}
    except Exception:
        return {"available": False}

# ============================================
# 智能问答 (SSE流式)
# ============================================
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    filtered = content_filter.check(request.query)
    if filtered is None:
        raise HTTPException(status_code=400, detail="输入内容包含敏感词或违规信息")

    truncated = token_controller.truncate_query(filtered)

    async def generate():
        try:
            system_prompt = """你是一个专业的非遗文化知识助手。请基于提供的非遗资料回答用户问题。
回答要求：内容准确详实，语言通俗易懂，适当引用资料来源，体现非遗文化价值。"""

            result = rag_core.query(
                query=truncated,
                model_source=request.provider or (None if model_controller.current_source == "local" else model_controller.current_source)
            )
            answer = result.get("answer", "抱歉，生成回答时出现错误")
            token_count = result.get("token_count", 0)
            sources = result.get("sources", [])

            chunk_size = 20
            for i in range(0, len(answer), chunk_size):
                yield f"data: {answer[i:i+chunk_size]}\n\n"
                time.sleep(0.02)

            meta = json.dumps({"token_count": token_count, "sources": [{"filename": s.get("filename", ""), "category": s.get("category", ""), "similarity": round(1 - (s.get("similarity", 0) if isinstance(s.get("similarity"), (int, float)) else 0), 2)} for s in sources[:5]]}, ensure_ascii=False)
            yield f"data: [DONE]{meta}\n\n"
        except Exception as e:
            yield f"data: [ERROR]请求处理失败: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ============================================
# 纹样提示词生成
# ============================================
@app.post("/api/pattern/generate")
async def generate_pattern(request: PatternRequest):
    validated = style_filter.validate_style(request.style_description)
    if validated is None:
        raise HTTPException(status_code=400, detail="风格描述不符合非遗文化规范")
    prompt = rag_core.generate_pattern_prompt(category=request.category, style_description=validated)
    return {"prompt": prompt, "category": request.category}

# ============================================
# 纹样生图 (Agnes AI)
# ============================================
@app.post("/api/agnes/generate-image")
async def agnes_generate_image(request: ImageGenRequest):
    agnes_key = os.getenv("AGNES_API_KEY", "")
    if not agnes_key:
        return {"success": False, "message": "Agnes AI API密钥未配置", "urls": []}

    import requests as req
    try:
        w, h = request.size.split("x")
        r = req.post(
            "https://apihub.agnes-ai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {agnes_key}", "Content-Type": "application/json"},
            json={
                "model": "agnes-image-2.1-flash",
                "prompt": request.prompt,
                "size": request.size,
                "extra_body": {"response_format": "url"}
            },
            timeout=120,
        )
        data = r.json()
        if r.status_code == 200 and data.get("data"):
            urls = [item.get("url", "") for item in data["data"] if item.get("url")]
            if urls:
                return {"success": True, "urls": urls, "message": "生成成功"}
            return {"success": False, "message": "API返回数据中无图片URL", "urls": []}
        else:
            msg = data.get("error", {}).get("message", "") if isinstance(data.get("error"), dict) else str(data)
            return {"success": False, "message": f"生图失败: {msg[:200]}", "urls": []}
    except Exception as e:
        return {"success": False, "message": str(e), "urls": []}

# ============================================
# 文创内容生成
# ============================================
@app.post("/api/content/generate", response_model=None)
async def generate_content(request: ContentGenerateRequest):
    result = rag_core.generate_creative_content(
        template_type=request.template_type,
        topic=request.topic,
        extra_info=request.extra_info
    )
    if request.provider and request.provider != "local":
        try:
            cfg = model_controller.get_config(request.provider)
            prompt = f"请为{request.topic}创作一段{request.template_type}内容，体现非遗文化特色。"
            cloud_result = model_controller.generate(prompt=prompt, force_source=request.provider)
            return {"content": cloud_result["response"], "template_type": request.template_type, "model_used": cloud_result["model"], "token_count": cloud_result["token_count"]}
        except Exception as e:
            return {"content": f"云端生成失败: {str(e)}，使用本地模型结果", "template_type": request.template_type, "model_used": result["model_used"], "token_count": result["token_count"]}
    return {"content": result["content"], "template_type": result["template_type"], "model_used": result["model_used"], "token_count": result["token_count"]}

# ============================================
# 知识库管理
# ============================================
@app.get("/api/knowledge/status")
async def knowledge_status():
    cats = set()
    total_chunks = 0
    for doc in rag_core.documents:
        for c in doc.get("category", []):
            cats.add(c)
        total_chunks += doc.get("chunks_count", doc.get("chunk_count", 0))
    return {
        "total_documents": len(rag_core.documents),
        "categories": list(cats),
        "total_chunks": total_chunks,
        "last_updated": datetime.now().isoformat(),
        "documents": [
            {"doc_id": d["doc_id"], "filename": d["filename"], "category": d.get("category", []), "chunk_count": d.get("chunks_count", d.get("chunk_count", 0))}
            for d in rag_core.documents
        ],
    }

@app.post("/api/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    if not (file.filename.endswith(".txt") or file.filename.endswith(".md")):
        raise HTTPException(status_code=400, detail="仅支持TXT/MD格式文件")
    content = await file.read()
    text = content.decode("utf-8")
    if content_filter.check(text) is None:
        raise HTTPException(status_code=400, detail="文档内容包含敏感信息")
    file_path = KNOWLEDGE_BASE_DIR / file.filename
    file_path.write_text(text, encoding="utf-8")
    doc_id = rag_core.add_document(text, file.filename)
    return {"message": "文档上传成功", "doc_id": doc_id, "filename": file.filename, "chunks": len(rag_core._chunk_text(text))}

@app.delete("/api/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str):
    ok = rag_core.delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"message": "删除成功", "doc_id": doc_id}

@app.get("/api/knowledge/{doc_id}/chunks")
async def get_doc_chunks(doc_id: str):
    try:
        results = rag_core.collection.get(where={"doc_id": doc_id})
        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                chunks.append({"id": results["ids"][i], "content": doc, "category": meta.get("category", "通用"), "chunk_index": meta.get("chunk_index", 0)})
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# 日志 & 导出
# ============================================
@app.get("/api/logs")
async def get_logs():
    logs = []
    log_file = LOG_DIR / "model_calls.jsonl"
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return {"logs": logs[-100:], "total": len(logs)}

@app.post("/api/logs/export")
async def export_logs():
    src = LOG_DIR / "model_calls.jsonl"
    if not src.exists():
        return {"message": "暂无日志", "path": ""}
    dst = ARCHIVE_DIR / f"logs-{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl"
    shutil.copy2(str(src), str(dst))
    return {"message": "日志已导出", "path": str(dst)}

# ============================================
# 开源协议
# ============================================
@app.get("/api/opensource/license")
async def get_licenses():
    return [
        {"name": "FastAPI", "license": "MIT"},
        {"name": "TailwindCSS", "license": "MIT"},
        {"name": "Ollama", "license": "MIT"},
        {"name": "DeepSeek-R1", "license": "MIT"},
        {"name": "GLM-4-Flash", "license": "Custom"},
        {"name": "Agnes AI", "license": "Commercial"},
        {"name": "硅基流动DeepSeek", "license": "MIT"},
        {"name": "ChromaDB", "license": "Apache-2.0"},
    ]

# ============================================
# 程序入口
# ============================================
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
