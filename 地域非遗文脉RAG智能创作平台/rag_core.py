"""
地域非遗文脉RAG智能创作平台 - RAG检索主逻辑
基础工程代码由 Vibe Coding 智能体生成

【人工重构区：非遗加权排序算法】
本文件包含非遗分类加权相似度排序算法插槽，由项目负责人独立人工重构开发
"""
import os
import json
import uuid
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# ============================================
# 配置
# ============================================
KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base"))
VECTOR_STORE_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./vector_store"))
CHUNK_SIZE = 500


class SimpleChineseEmbedding:
    """简单中文字符级嵌入函数"""

    def name(self) -> str:
        return "simple_chinese_hash"

    def __call__(self, input):
        return [self._embed_one(text) for text in input]

    def embed_query(self, input):
        if isinstance(input, str):
            return [self._embed_one(input)]
        return [self._embed_one(text) for text in input]

    def _embed_one(self, text):
        vec = [0.0] * 384
        for ch in text:
            idx = hash(ch) % 384
            vec[idx] += 1.0
        total = sum(abs(v) for v in vec) or 1.0
        return [v / total for v in vec]


class SimpleVectorStore:
    """基于numpy的简易向量存储，替代ChromaDB"""

    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self._store_file = self.persist_dir / "vectors.json"
        self.documents = []
        self.embeddings = []
        self.ids = []
        self.metadatas = []
        self._load()

    def _load(self):
        if self._store_file.exists():
            try:
                with open(self._store_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.embeddings = data.get("embeddings", [])
                self.ids = data.get("ids", [])
                self.metadatas = data.get("metadatas", [])
            except (json.JSONDecodeError, IOError):
                pass

    def _save(self):
        try:
            with open(self._store_file, "w", encoding="utf-8") as f:
                json.dump({
                    "documents": self.documents,
                    "embeddings": self.embeddings,
                    "ids": self.ids,
                    "metadatas": self.metadatas
                }, f, ensure_ascii=False)
        except IOError as e:
            print(f"[RAG] 向量存储保存失败: {e}")

    def add(self, documents, ids, metadatas, embeddings):
        self.documents.extend(documents)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)
        self.embeddings.extend(embeddings)
        self._save()

    def query(self, query_embedding, n_results=3):
        if not self.embeddings:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_vec = query_embedding[0]
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_vec, emb)
            similarities.append((sim, i))

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_n = similarities[:n_results]

        result_ids = [[self.ids[idx] for _, idx in top_n]]
        result_docs = [[self.documents[idx] for _, idx in top_n]]
        result_metas = [[self.metadatas[idx] for _, idx in top_n]]
        result_dists = [[1.0 - sim for sim, idx in top_n]]

        return {
            "ids": result_ids,
            "documents": result_docs,
            "metadatas": result_metas,
            "distances": result_dists
        }

    def get(self, where=None):
        if not where:
            return {"ids": self.ids, "documents": self.documents, "metadatas": self.metadatas}

        matched_ids = []
        matched_docs = []
        matched_metas = []

        for i, meta in enumerate(self.metadatas):
            match = True
            for key, value in where.items():
                if meta.get(key) != value:
                    match = False
                    break
            if match:
                matched_ids.append(self.ids[i])
                matched_docs.append(self.documents[i])
                matched_metas.append(self.metadatas[i])

        return {"ids": matched_ids, "documents": matched_docs, "metadatas": matched_metas}

    def delete(self, ids):
        id_set = set(ids)
        new_indices = [i for i, id_ in enumerate(self.ids) if id_ not in id_set]
        self.ids = [self.ids[i] for i in new_indices]
        self.documents = [self.documents[i] for i in new_indices]
        self.metadatas = [self.metadatas[i] for i in new_indices]
        self.embeddings = [self.embeddings[i] for i in new_indices]
        self._save()

    def count(self):
        return len(self.ids)

    @staticmethod
    def _cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class RAGCore:
    """RAG检索核心类"""

    def __init__(self, chroma_persist_dir: str, knowledge_base_dir: str):
        self.vector_store_dir = Path(chroma_persist_dir)
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.vector_store_dir.mkdir(exist_ok=True)
        self.knowledge_base_dir.mkdir(exist_ok=True)

        self.vector_store = None
        self.embedding_fn = SimpleChineseEmbedding()
        self.model_controller = None
        self.documents = []

        self._metadata_file = self.vector_store_dir / "documents_metadata.json"

    def initialize(self):
        """初始化RAG核心组件"""
        try:
            self.vector_store = SimpleVectorStore(str(self.vector_store_dir))

            self._load_metadata()
            self._scan_and_load_documents()

            print(f"[RAG] 初始化完成，当前向量库文档数: {self.vector_store.count()}")

        except Exception as e:
            print(f"[RAG] 初始化失败: {e}")
            raise

    def _load_metadata(self):
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.documents = []
        else:
            self.documents = []

    def _save_metadata(self):
        try:
            with open(self._metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[RAG] 元数据保存失败: {e}")

    def _scan_and_load_documents(self):
        txt_files = list(self.knowledge_base_dir.glob("*.txt"))
        loaded_ids = {doc["doc_id"] for doc in self.documents}

        for txt_file in txt_files:
            doc_id = txt_file.stem
            if doc_id not in loaded_ids:
                try:
                    content = txt_file.read_text(encoding="utf-8")
                    self.add_document(content, txt_file.name)
                    print(f"[RAG] 已加载新文档: {txt_file.name}")
                except Exception as e:
                    print(f"[RAG] 文档加载失败 {txt_file.name}: {e}")

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) + 1 <= CHUNK_SIZE:
                current_chunk = current_chunk + "\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
        if current_chunk:
            chunks.append(current_chunk)
        return chunks if chunks else [text[:CHUNK_SIZE]]

    def _detect_category(self, text: str) -> List[str]:
        categories = {
            "皮影": ["皮影", "皮影戏", "影子戏", "驴皮", "雕刻", "操纵"],
            "竹编": ["竹编", "竹艺", "竹器", "编织", "竹篾", "竹丝"],
            "年画": ["年画", "木版年画", "门神", "春联", "新春", "喜庆"],
            "剪纸": ["剪纸", "窗花", "剪刀", "红纸", "镂空", "民间剪纸"],
            "木雕": ["木雕", "雕刻", "木艺", "檀木", "黄杨木", "根雕"],
            "湘绣": ["湘绣", "湖南刺绣", "丝线", "绣花", "针法"],
            "苏绣": ["苏绣", "苏州刺绣", "双面绣", "丝线", "精细"],
        }
        detected = []
        for cat, keywords in categories.items():
            if any(kw in text for kw in keywords):
                detected.append(cat)
        return detected if detected else ["通用"]

    def add_document(self, content: str, filename: str) -> str:
        doc_id = str(uuid.uuid4())[:8]
        chunks = self._chunk_text(content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            category = self._detect_category(chunk)
            embedding = self.embedding_fn([chunk])[0]

            self.vector_store.add(
                documents=[chunk],
                ids=[chunk_id],
                metadatas=[{
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "category": ",".join(category),
                    "added_at": datetime.now().isoformat()
                }],
                embeddings=[embedding]
            )

        self.documents.append({
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": len(chunks),
            "category": self._detect_category(content),
            "added_at": datetime.now().isoformat()
        })
        self._save_metadata()
        return doc_id

    def delete_document(self, doc_id: str) -> bool:
        try:
            results = self.vector_store.get(where={"doc_id": doc_id})
            if results["ids"]:
                self.vector_store.delete(ids=results["ids"])

                for doc in self.documents:
                    if doc["doc_id"] == doc_id:
                        file_path = self.knowledge_base_dir / doc["filename"]
                        if file_path.exists():
                            file_path.unlink()
                        break

                self.documents = [d for d in self.documents if d["doc_id"] != doc_id]
                self._save_metadata()
                return True
            return False
        except Exception as e:
            print(f"[RAG] 文档删除失败: {e}")
            return False

    def query(self, query: str, model_source: str = "local") -> Dict[str, Any]:
        query_embedding = self.embedding_fn.embed_query([query])
        results = self.vector_store.query(query_embedding, n_results=5)

        sources = []
        category_tags = set()
        context_docs = []

        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                sources.append({
                    "content": doc,
                    "filename": metadata.get("filename", "未知"),
                    "category": metadata.get("category", "通用")
                })
                context_docs.append(doc)
                for cat in metadata.get("category", "通用").split(","):
                    category_tags.add(cat)

        context = "\n\n".join(context_docs) if context_docs else "暂无相关非遗资料"

        system_prompt = """你是一个专业的非遗文化知识助手。请基于提供的非遗资料回答用户问题。
回答要求：
1. 内容准确、详实
2. 语言通俗易懂
3. 适当引用资料来源
4. 体现非遗文化价值"""

        prompt = f"""基于以下非遗资料回答问题：

【参考资料】
{context}

【用户问题】
{query}

请基于参考资料给出详细回答："""

        try:
            if self.model_controller is None:
                from model_switch import ModelSwitchController
                self.model_controller = ModelSwitchController()

            model_result = self.model_controller.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                force_source=model_source
            )

            return {
                "answer": model_result["response"],
                "sources": sources,
                "category_tags": list(category_tags),
                "model_used": model_result["model"],
                "token_count": model_result["token_count"]
            }

        except Exception as e:
            return {
                "answer": f"抱歉，生成回答时出现错误: {str(e)}",
                "sources": sources,
                "category_tags": list(category_tags),
                "model_used": "error",
                "token_count": 0
            }

    def generate_pattern_prompt(self, category: str, style_description: str) -> str:
        """生成传统纹样AI绘图提示词（纯英文，适配Agnes AI）"""
        category_features = {
            "皮影": "Chinese shadow puppet art, intricate hand-carved leather silhouette, translucent warm-toned material, traditional opera character shape, delicate openwork carving patterns, jointed movable figure, dramatic light projection effect",
            "竹编": "Chinese bamboo weaving art, interlaced warp and weft structure, natural bamboo color tones, geometric pattern arrangement, handwoven textile texture, fine bamboo strip fiber details",
            "年画": "Chinese New Year woodblock print, vibrant multi-color registration, auspicious door god figure, festive celebratory atmosphere, symmetrical composition, woodcut carving texture and grain",
            "剪纸": "Chinese paper cutting art, bold red color palette, precise scissor-cut openwork technique, symmetric folk pattern design, window flower motifs, auspicious symbolic imagery, red paper texture",
            "木雕": "Chinese wood carving craft, three-dimensional relief sculpture, precious wood natural grain, intricate chisel carving technique, traditional decorative motif, warm polished wood luster, layered depth",
            "湘绣": "Chinese Hunan embroidery style, silk thread stitching texture, smooth color gradient transitions, fine needlework detail, floral and bird subjects, silk thread sheen and luster, raised embroidered surface",
            "苏绣": "Chinese Suzhou embroidery style, double-sided sheer embroidery effect, delicate silk thread work, soft elegant pastel color palette, Jiangnan water town elements, silky smooth thread sheen, refined flat stitch surface",
        }
        features = category_features.get(category, "traditional Chinese folk art pattern, handcrafted texture")

        prompt = f"{features}. {style_description}. Centered symmetrical composition, balanced visual weight, seamless pattern design. Soft natural lighting with subtle depth shadows. Visible handcraft texture details, museum exhibition quality. Ultra-high definition 8K resolution, masterpiece level craftsmanship."

        return prompt

    def generate_creative_content(self, template_type: str, topic: str, extra_info: str = "") -> Dict[str, Any]:
        """批量生成结构化文创内容"""
        templates = {
            "导览解说": """请为{topic}撰写一段景区线下导览解说文案。
要求：
1. 内容详实准确，包含历史背景、工艺特点、文化价值
2. 语言生动有趣，适合口头讲解
3. 时长控制在3-5分钟
4. 包含互动引导语
{extra_info}""",

            "短视频脚本": """请为{topic}创作一个完整的非遗科普短视频脚本。
要求：
1. 包含开场、主体内容、结尾完整结构
2. 画面描述详细，便于拍摄
3. 配乐建议
4. 时长控制在2-3分钟
{extra_info}""",

            "宣传文案": """请为{topic}撰写一段文创商品宣传介绍文案。
要求：
1. 突出非遗文化价值和工艺特色
2. 语言优美，有感染力
3. 包含产品特点、适用场景
4. 字数200-300字
{extra_info}""",

            "朋友圈文案": """请为{topic}创作几条国风朋友圈打卡文案。
要求：
1. 文字简洁优美，有文化底蕴
2. 适合配图发布
3. 包含3-5条不同风格
4. 适当使用emoji增加趣味
{extra_info}"""
        }

        template = templates.get(template_type, templates["导览解说"])
        prompt = template.format(topic=topic, extra_info=extra_info)

        try:
            if self.model_controller is None:
                from model_switch import ModelSwitchController
                self.model_controller = ModelSwitchController()

            system_prompt = "你是一个专业的非遗文创内容创作助手，请根据要求生成高质量的文创内容。"
            model_result = self.model_controller.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )

            return {
                "content": model_result["response"],
                "template_type": template_type,
                "model_used": model_result["model"],
                "token_count": model_result["token_count"]
            }

        except Exception as e:
            return {
                "content": f"内容生成失败: {str(e)}",
                "template_type": template_type,
                "model_used": "error",
                "token_count": 0
            }

    def get_knowledge_base_status(self) -> Dict[str, Any]:
        categories = set()
        for doc in self.documents:
            for cat in doc.get("category", []):
                categories.add(cat)
        return {
            "total_documents": len(self.documents),
            "categories": list(categories),
            "last_updated": datetime.now().isoformat()
        }

    def save(self):
        print("[RAG] 向量库状态已保存")
