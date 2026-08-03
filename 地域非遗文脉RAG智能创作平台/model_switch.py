"""
地域非遗文脉RAG智能创作平台 - 双模型调度控制器
基础工程代码由 Vibe Coding 智能体生成
非遗分类加权检索算法、Token分层节流缓存系统、非遗风格强制校验过滤引擎三大核心业务算法模块由项目负责人独立人工重构开发

调度架构：
  本地DeepSeek蒸馏轻量化模型 (Ollama) → http://localhost:11434
  智谱GLM-4-Flash云端免费模型 → https://open.bigmodel.cn/api/paas/v4
"""
import os
import time
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ============================================
# 配置
# ============================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))

LOG_DIR.mkdir(exist_ok=True)

# ============================================
# 模型配置表
# ============================================
MODEL_CONFIGS = {
    "local": {
        "name": "本地 DeepSeek 蒸馏轻量化模型",
        "short": "DeepSeek-R1 1.5B",
        "source_type": "local",
        "tags": ["离线可用", "GPU加速", "轻量"],
    },
    "zhipu": {
        "name": "智谱 GLM-4-Flash 云端免费模型",
        "short": "GLM-4-Flash",
        "source_type": "cloud",
        "tags": ["依赖网络", "永久免费", "长文案"],
    },
}


class ModelSwitchController:
    """双模型调度控制器"""

    def __init__(self):
        self.current_source = "local"
        self.call_log = []
        self._log_file = LOG_DIR / "model_calls.jsonl"

    def get_config(self, source: str) -> Dict[str, Any]:
        return MODEL_CONFIGS.get(source, MODEL_CONFIGS["local"])

    def get_status(self) -> Dict[str, Any]:
        local_ok = self._check_ollama()
        config = self.get_config(self.current_source)
        return {
            "current_source": self.current_source,
            "source_type": config["source_type"],
            "local_available": local_ok,
            "total_calls": len(self.call_log),
        }

    def switch_to(self, source: str):
        if source not in MODEL_CONFIGS:
            raise ValueError(f"未知模型源: {source}")
        self.current_source = source
        self._log_event("switch", {"to": source})

    def _check_ollama(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                models = r.json().get("models", [])
                return any(m.get("name") == OLLAMA_MODEL for m in models)
        except requests.RequestException:
            pass
        return False

    # ============================================
    # 本地 Ollama 调用
    # ============================================
    def _call_ollama(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        start = time.time()
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_gpu": 33, "num_thread": 8, "num_ctx": 4096},
        }
        if system_prompt:
            payload["system"] = system_prompt

        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
        elapsed = time.time() - start

        if r.status_code != 200:
            raise Exception(f"Ollama API返回错误: {r.status_code}")

        result = r.json()
        tokens = result.get("eval_count", 0) + result.get("prompt_eval_count", 0)
        self._log_event("call", {
            "source": "local", "model": OLLAMA_MODEL,
            "token_count": tokens, "elapsed": round(elapsed, 2), "success": True,
        })
        return {"response": result.get("response", ""), "token_count": tokens, "model": OLLAMA_MODEL, "elapsed": elapsed}

    # ============================================
    # 智谱 GLM-4-Flash 调用
    # ============================================
    def _call_zhipu(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        if not ZHIPU_API_KEY:
            raise Exception("智谱API密钥未配置")

        start = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        r = requests.post(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            headers={"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"},
            json={"model": "glm-4-flash", "messages": messages, "temperature": 0.7},
            timeout=120,
        )
        elapsed = time.time() - start

        if r.status_code != 200:
            raise Exception(f"智谱API错误: {r.status_code} {r.text[:200]}")

        data = r.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        self._log_event("call", {
            "source": "zhipu", "model": "glm-4-flash",
            "token_count": tokens, "elapsed": round(elapsed, 2), "success": True,
        })
        return {"response": content, "token_count": tokens, "model": "glm-4-flash", "elapsed": elapsed}

    # ============================================
    # 统一生成接口
    # ============================================
    def generate(self, prompt: str, system_prompt: str = "", force_source: Optional[str] = None) -> Dict[str, Any]:
        source = force_source or self.current_source

        dispatch = {
            "local": self._call_ollama,
            "zhipu": self._call_zhipu,
        }

        if source in dispatch:
            try:
                return dispatch[source](prompt, system_prompt)
            except Exception as e:
                self._log_event("call", {"source": source, "success": False, "error": str(e)})
                raise

        raise ValueError(f"未知模型源: {source}")

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        entry = {"timestamp": datetime.now().isoformat(), "event_type": event_type, **data}
        self.call_log.append(entry)
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
