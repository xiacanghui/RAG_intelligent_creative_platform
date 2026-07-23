"""
公共配置模块 - 所有微服务共享
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 服务端口
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))

# Ollama 本地轻量化模型配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "deepseek-r1:1.5b")
OLLAMA_MODEL = LOCAL_MODEL_NAME

# 硅基流动 DeepSeek 云端 API 配置
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

# 智谱 GLM-4-Flash 云端 API 配置（免费）
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash")

# 知识库路径
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "knowledge_base")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))

# 内存阈值
MEMORY_THRESHOLD_LOW = int(os.getenv("MEMORY_THRESHOLD_LOW", 70))
MEMORY_THRESHOLD_HIGH = int(os.getenv("MEMORY_THRESHOLD_HIGH", 85))
