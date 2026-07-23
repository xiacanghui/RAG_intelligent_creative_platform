"""
向量知识库微服务 (端口 8001) - 简化版
"""
import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 全局配置
BASE_DIR = Path(__file__).parent.parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "knowledge_base")))
RAG_PORT = int(os.getenv("RAG_SERVICE_PORT", 8001))

print(f"[RAG] BASE_DIR: {BASE_DIR}")
print(f"[RAG] VECTOR_STORE_DIR: {VECTOR_STORE_DIR}")
print(f"[RAG] KNOWLEDGE_DIR: {KNOWLEDGE_DIR}")
print(f"[RAG] KNOWLEDGE_DIR exists: {KNOWLEDGE_DIR.exists()}")

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# TF-IDF 搜索引擎
# ============================================
class SimpleTFIDF:
    def __init__(self):
        self.documents = []
        self.doc_ids = []
        self.metadata = []
        self.doc_vectors = []
        self.idf = {}
        self.doc_count = 0

    def _tokenize(self, text):
        return [c for c in text if c.strip() and len(c) > 1]

    def _compute_tf(self, tokens):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for k in tf:
            tf[k] /= len(tokens)
        return tf

    def _compute_idf(self):
        self.idf = {}
        for vec in self.doc_vectors:
            for token in vec:
                self.idf[token] = self.idf.get(token, 0) + 1
        for token in self.idf:
            self.idf[token] = max(0.1, 1.0 - self.idf[token] / max(1, self.doc_count))

    def add(self, doc_id, text, meta=None):
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)
        self.documents.append(text)
        self.doc_ids.append(doc_id)
        self.metadata.append(meta or {})
        self.doc_vectors.append(tf)
        self.doc_count += 1
        self._compute_idf()

    def query(self, query_text, n_results=5):
        if self.doc_count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        query_tokens = self._tokenize(query_text)
        query_tf = self._compute_tf(query_tokens)
        scores = []
        for i, doc_tf in enumerate(self.doc_vectors):
            score = 0
            for token, q_weight in query_tf.items():
                if token in doc_tf:
                    score += q_weight * doc_tf[token] * self.idf.get(token, 1.0)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_n = scores[:n_results]
        return {
            "documents": [[self.documents[idx] for idx, _ in top_n]],
            "metadatas": [[self.metadata[idx] for idx, _ in top_n]],
            "distances": [[1.0 - score for _, score in top_n]]
        }

    def delete(self, doc_id):
        indices = [i for i, did in enumerate(self.doc_ids) if did.startswith(doc_id)]
        if not indices:
            return False
        for i in sorted(indices, reverse=True):
            self.documents.pop(i)
            self.doc_ids.pop(i)
            self.metadata.pop(i)
            self.doc_vectors.pop(i)
            self.doc_count -= 1
        self._compute_idf()
        return True

    def count(self):
        return self.doc_count

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"documents": self.documents, "doc_ids": self.doc_ids, "metadata": self.metadata}, f, ensure_ascii=False)

    def load(self, path):
        if not Path(path).exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = data.get("documents", [])
            self.doc_ids = data.get("doc_ids", [])
            self.metadata = data.get("metadata", [])
            self.doc_count = len(self.documents)
            self.doc_vectors = []
            for doc in self.documents:
                tokens = self._tokenize(doc)
                self.doc_vectors.append(self._compute_tf(tokens))
            self._compute_idf()
        except Exception as e:
            print(f"[RAG] 加载失败: {e}")

# ============================================
# 工具函数
# ============================================
def chunk_text(text, chunk_size=500):
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 <= chunk_size:
            current = current + "\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:chunk_size]]

def detect_category(text):
    keywords = {
        "皮影": ["皮影", "皮影戏", "驴皮", "雕刻"],
        "剪纸": ["剪纸", "剪刀", "窗花", "镂空"],
        "苏绣": ["苏绣", "刺绣", "丝线", "双面绣"],
        "湘绣": ["湘绣", "湖南", "刺绣"],
        "木雕": ["木雕", "雕刻", "浮雕"],
        "竹编": ["竹编", "竹子", "编织"],
        "年画": ["年画", "门神", "木版"]
    }
    found = []
    for cat, words in keywords.items():
        if any(w in text for w in words):
            found.append(cat)
    return found if found else ["通用"]

# ============================================
# FastAPI 应用
# ============================================
app = FastAPI(title="RAG Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = SimpleTFIDF()
documents_meta = []

# 初始化：加载已有数据 + 扫描知识库
def init():
    global documents_meta, engine
    data_file = VECTOR_STORE_DIR / "search_data.json"
    meta_file = VECTOR_STORE_DIR / "docs_meta.json"
    
    print(f"[RAG] init() 开始...")
    print(f"[RAG] data_file: {data_file}")
    print(f"[RAG] meta_file: {meta_file}")
    
    # 加载已有数据
    engine.load(str(data_file))
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                documents_meta = json.load(f)
            print(f"[RAG] 加载元数据: {len(documents_meta)} 文档")
        except Exception as e:
            print(f"[RAG] 元数据加载失败: {e}")
            documents_meta = []
    
    # 扫描知识库，加载新文件
    loaded = {d["filename"] for d in documents_meta}
    print(f"[RAG] 已加载文件: {loaded}")
    
    txt_files = list(KNOWLEDGE_DIR.glob("*.txt"))
    print(f"[RAG] 发现文件: {len(txt_files)}")
    
    for f in txt_files:
        print(f"[RAG] 检查: {f.name}, in loaded: {f.name in loaded}")
        if f.name not in loaded:
            try:
                content = f.read_text(encoding="utf-8")
                doc_id = str(uuid.uuid4())[:8]
                chunks = chunk_text(content)
                for i, chunk in enumerate(chunks):
                    engine.add(
                        doc_id=f"{doc_id}_{i}",
                        text=chunk,
                        meta={"doc_id": doc_id, "filename": f.name, "chunk_index": i, "category": ",".join(detect_category(chunk))}
                    )
                documents_meta.append({"doc_id": doc_id, "filename": f.name, "chunk_count": len(chunks), "category": detect_category(content)})
                print(f"[RAG] 加载新文件: {f.name}")
            except Exception as e:
                print(f"[RAG] 加载失败: {f.name}: {e}")
    
    # 保存
    engine.save(str(data_file))
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(documents_meta, f, ensure_ascii=False, indent=2)
    
    print(f"[RAG] 就绪: {len(documents_meta)} 文档, {engine.count()} 片段")

init()

# 如果元数据为空，强制重新加载
if len(documents_meta) == 0:
    print("[RAG] 元数据为空，强制加载知识库...")
    for f in KNOWLEDGE_DIR.glob("*.txt"):
        try:
            content = f.read_text(encoding="utf-8")
            doc_id = str(uuid.uuid4())[:8]
            chunks = chunk_text(content)
            for i, chunk in enumerate(chunks):
                engine.add(f"{doc_id}_{i}", chunk, {"doc_id": doc_id, "filename": f.name, "chunk_index": i, "category": ",".join(detect_category(chunk))})
            documents_meta.append({"doc_id": doc_id, "filename": f.name, "chunk_count": len(chunks), "category": detect_category(content)})
            print(f"[RAG] 加载: {f.name}")
        except Exception as e:
            print(f"[RAG] 失败: {e}")
    engine.save(str(VECTOR_STORE_DIR / "search_data.json"))
    with open(VECTOR_STORE_DIR / "docs_meta.json", "w", encoding="utf-8") as f:
        json.dump(documents_meta, f, ensure_ascii=False, indent=2)
    print(f"[RAG] 完成: {len(documents_meta)} 文档, {engine.count()} 片段")

# ============================================
# API
# ============================================
class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    similarity_threshold: float = 0.1

@app.get("/health")
async def health():
    return {"status": "ok", "documents": engine.count()}

@app.get("/stats")
async def stats():
    cats = set()
    for d in documents_meta:
        for c in d.get("category", []):
            cats.add(c)
    return {"total_documents": len(documents_meta), "total_chunks": engine.count(), "categories": list(cats)}

@app.get("/documents")
async def list_docs():
    return {"documents": documents_meta, "total": len(documents_meta)}

@app.post("/search")
async def search(req: SearchRequest):
    results = engine.query(req.query, n_results=req.n_results * 2)
    sources, context_docs, seen = [], [], set()
    
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            sim = 1 - (results["distances"][0][i] if results.get("distances") else 0)
            if sim < req.similarity_threshold:
                continue
            key = doc[:50].strip()
            if key in seen:
                continue
            seen.add(key)
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            sources.append({"content": doc, "filename": meta.get("filename", ""), "category": meta.get("category", ""), "similarity": round(sim, 3)})
            context_docs.append(doc)
            if len(sources) >= req.n_results:
                break
    
    merged = {}
    for s in sources:
        k = s["filename"]
        if k not in merged:
            merged[k] = {"filename": s["filename"], "category": s["category"], "similarity": s["similarity"], "count": 1}
        else:
            merged[k]["count"] += 1
    
    return {"sources": list(merged.values()), "context": "\n\n".join(context_docs), "has_results": len(sources) > 0}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(400, "仅支持 TXT/MD")
    
    for d in documents_meta:
        if d["filename"] == file.filename:
            raise HTTPException(400, f"'{file.filename}' 已存在")
    
    content = await file.read()
    text = content.decode('utf-8')
    (KNOWLEDGE_DIR / file.filename).write_text(text, encoding='utf-8')
    
    doc_id = str(uuid.uuid4())[:8]
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        engine.add(f"{doc_id}_{i}", text=chunk, meta={"doc_id": doc_id, "filename": file.filename, "chunk_index": i, "category": ",".join(detect_category(chunk))})
    
    documents_meta.append({"doc_id": doc_id, "filename": file.filename, "chunk_count": len(chunks), "category": detect_category(text)})
    engine.save(str(VECTOR_STORE_DIR / "search_data.json"))
    with open(VECTOR_STORE_DIR / "docs_meta.json", "w", encoding="utf-8") as f:
        json.dump(documents_meta, f, ensure_ascii=False, indent=2)
    
    return {"message": "成功", "doc_id": doc_id, "chunks": len(chunks)}

@app.delete("/documents/{doc_id}")
async def delete(doc_id: str):
    global documents_meta
    if not engine.delete(doc_id):
        raise HTTPException(404, "不存在")
    documents_meta = [d for d in documents_meta if d["doc_id"] != doc_id]
    engine.save(str(VECTOR_STORE_DIR / "search_data.json"))
    with open(VECTOR_STORE_DIR / "docs_meta.json", "w", encoding="utf-8") as f:
        json.dump(documents_meta, f, ensure_ascii=False, indent=2)
    return {"message": "删除成功"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=RAG_PORT)
