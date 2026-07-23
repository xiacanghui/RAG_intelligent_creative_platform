"""
地域非遗文脉RAG智能创作平台 - RAG检索主逻辑
基础工程代码由 Vibe Coding 智能体生成

【人工重构区：非遗加权排序算法】
本文件包含非遗分类加权相似度排序算法插槽，由项目负责人独立人工重构开发
"""
import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# ============================================
# 配置
# ============================================
KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base"))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./vector_store"))
CHUNK_SIZE = 500  # 文本切片大小（字）


class SimpleChineseEmbedding:
    """简单中文字符级嵌入函数，无需下载模型"""

    def name(self) -> str:
        return "simple_chinese_hash"

    def __call__(self, input):
        return [self._embed_one(text) for text in input]

    def embed_query(self, input):
        return [self._embed_one(text) for text in input]

    def _embed_one(self, text):
        vec = [0.0] * 384
        for ch in text:
            idx = hash(ch) % 384
            vec[idx] += 1.0
        total = sum(abs(v) for v in vec) or 1.0
        return [v / total for v in vec]


class RAGCore:
    """RAG检索核心类"""

    def __init__(self, chroma_persist_dir: str, knowledge_base_dir: str):
        self.chroma_persist_dir = Path(chroma_persist_dir)
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.chroma_persist_dir.mkdir(exist_ok=True)
        self.knowledge_base_dir.mkdir(exist_ok=True)

        self.vector_store = None
        self.embeddings = None
        self.model_controller = None
        self.documents = []  # 文档元数据存储

        # 文档元数据文件
        self._metadata_file = self.chroma_persist_dir / "documents_metadata.json"

    def initialize(self):
        """初始化RAG核心组件"""
        try:
            import chromadb
            from chromadb.config import Settings

            # 初始化ChromaDB（不使用默认ONNX嵌入，避免下载79MB模型）
            self.vector_store = chromadb.PersistentClient(
                path=str(self.chroma_persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # 使用自定义嵌入函数（无需下载ONNX模型）
            self.embedding_fn = SimpleChineseEmbedding()
            self.collection = self.vector_store.get_or_create_collection(
                name="nonheritage_knowledge",
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_fn
            )

            # 加载文档元数据
            self._load_metadata()

            # 扫描知识库目录，加载新文档
            self._scan_and_load_documents()

            print(f"[RAG] 初始化完成，当前向量库文档数: {self.collection.count()}")

        except ImportError as e:
            print(f"[RAG] 依赖缺失: {e}")
            print("[RAG] 请运行: pip install chromadb")
            raise

    def _load_metadata(self):
        """加载文档元数据"""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.documents = []
        else:
            self.documents = []

    def _save_metadata(self):
        """保存文档元数据"""
        try:
            with open(self._metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[RAG] 元数据保存失败: {e}")

    def _scan_and_load_documents(self):
        """扫描知识库目录并加载新文档"""
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
        """文本智能切片"""
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
        """检测非遗工艺分类标签"""
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
        """添加文档到向量库"""
        doc_id = str(uuid.uuid4())[:8]
        chunks = self._chunk_text(content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            category = self._detect_category(chunk)

            self.collection.add(
                documents=[chunk],
                ids=[chunk_id],
                metadatas=[{
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "category": ",".join(category),
                    "added_at": datetime.now().isoformat()
                }]
            )

        # 保存文档元数据
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
        """从向量库删除文档"""
        try:
            # 查找并删除所有相关chunk
            results = self.collection.get(
                where={"doc_id": doc_id}
            )

            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])

                # 删除知识库文件
                for doc in self.documents:
                    if doc["doc_id"] == doc_id:
                        file_path = self.knowledge_base_dir / doc["filename"]
                        if file_path.exists():
                            file_path.unlink()
                        break

                # 更新元数据
                self.documents = [d for d in self.documents if d["doc_id"] != doc_id]
                self._save_metadata()
                return True

            return False
        except Exception as e:
            print(f"[RAG] 文档删除失败: {e}")
            return False

    def query(self, query: str, model_source: str = "local") -> Dict[str, Any]:
        """RAG查询接口"""
        # 1. 查询向量化
        results = self.collection.query(
            query_texts=[query],
            n_results=5
        )

        # 2. 提取检索结果
        sources = []
        category_tags = set()
        context_docs = []

        if results and results["documents"]:
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

        # 3. 构建RAG上下文
        context = "\n\n".join(context_docs) if context_docs else "暂无相关非遗资料"

        # 4. 调用大模型生成回答
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
            # 【人工重构区：非遗加权排序算法】
            # TODO: 由项目负责人实现非遗分类加权相似度排序算法
            # 当前为基础实现，直接使用向量相似度结果
            sorted_results = self._basic_similarity_sort(results)

            # 调用模型生成回答
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

    def _basic_similarity_sort(self, results: Dict) -> List[Dict]:
        """基础相似度排序（供人工重构区参考）

        【人工重构区：非遗加权排序算法】
        本函数为临时实现，由项目负责人重构为非遗分类加权相似度排序算法

        重构方向建议：
        1. 根据非遗工艺品类权重进行加权排序
        2. 考虑文档时效性（新增文档权重更高）
        3. 考虑用户历史查询偏好
        4. 引入非遗专业度评分机制
        """
        if not results or not results.get("documents"):
            return []

        sorted_items = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results.get("distances") else 0

            sorted_items.append({
                "content": doc,
                "metadata": metadata,
                "similarity": 1 - distance  # 转换为相似度
            })

        # 基础排序：按相似度降序
        sorted_items.sort(key=lambda x: x["similarity"], reverse=True)
        return sorted_items

    def generate_pattern_prompt(self, category: str, style_description: str) -> str:
        """生成传统纹样AI绘图提示词（纯英文，适配Agnes AI）"""
        category_features = {
            "皮影": "Chinese shadow puppet art, intricate hand-carved leather silhouette, translucent warm-toned material, traditional opera character造型, delicate openwork carving patterns, jointed movable figure, dramatic light projection effect",
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
        """获取知识库状态"""
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
        """保存向量库状态"""
        if self.vector_store:
            try:
                # ChromaDB PersistentClient 自动持久化
                print("[RAG] 向量库状态已保存")
            except Exception as e:
                print(f"[RAG] 保存失败: {e}")
