"""
地域非遗文脉RAG智能创作平台 - Token分层节流管控工具
基础工程代码由 Vibe Coding 智能体生成

【人工重构区：Token分层节流、重复问答本地缓存逻辑】
本文件包含Token分层节流缓存逻辑插槽，由项目负责人独立人工重构开发
"""
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ============================================
# 配置
# ============================================
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
CACHE_DIR = LOG_DIR / "cache"

# Token限制配置
MAX_QUERY_LENGTH = 800  # 单次输入汉字上限
MAX_RESPONSE_TOKENS = 2048  # 单次响应最大Token数
DAILY_TOKEN_LIMIT = 100000  # 每日Token使用上限
CACHE_EXPIRY_HOURS = 24  # 缓存有效期（小时）

CACHE_DIR.mkdir(parents=True, exist_ok=True)

class TokenController:
    """Token分层节流管控器"""

    def __init__(self):
        self.daily_usage = 0
        self.usage_file = LOG_DIR / "token_usage.json"
        self.cache_file = CACHE_DIR / "query_cache.json"
        self._load_usage()
        self._load_cache()

    def _load_usage(self):
        """加载今日Token使用量"""
        try:
            if self.usage_file.exists():
                with open(self.usage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 检查是否是今天的记录
                    if data.get("date") == datetime.now().strftime("%Y-%m-%d"):
                        self.daily_usage = data.get("total_tokens", 0)
                    else:
                        self.daily_usage = 0
                        self._save_usage()
        except (json.JSONDecodeError, IOError):
            self.daily_usage = 0

    def _save_usage(self):
        """保存Token使用量"""
        try:
            with open(self.usage_file, "w", encoding="utf-8") as f:
                json.dump({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "total_tokens": self.daily_usage
                }, f, ensure_ascii=False)
        except IOError:
            pass

    def _load_cache(self):
        """加载查询缓存"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except (json.JSONDecodeError, IOError):
            self.cache = {}

    def _save_cache(self):
        """保存查询缓存"""
        try:
            # 清理过期缓存
            now = datetime.now()
            expired_keys = []
            for key, entry in self.cache.items():
                cached_time = datetime.fromisoformat(entry.get("timestamp", now.isoformat()))
                if now - cached_time > timedelta(hours=CACHE_EXPIRY_HOURS):
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except IOError:
            pass

    def truncate_query(self, query: str) -> str:
        """输入文本截断

        【人工重构区：Token分层节流、重复问答本地缓存逻辑】
        本函数为临时实现，由项目负责人重构为更智能的分层节流策略

        重构方向建议：
        1. 实现分层截断策略（关键信息保留、次要信息截断）
        2. 智能识别并保留查询中的关键非遗术语
        3. 基于上下文长度动态调整截断策略
        """
        if len(query) <= MAX_QUERY_LENGTH:
            return query

        # 基础截断：保留前MAX_QUERY_LENGTH个字符
        truncated = query[:MAX_QUERY_LENGTH]

        # 尝试在句号、问号等标点处截断
        for punct in ["。", "？", "！", ".", "?", "!"]:
            last_pos = truncated.rfind(punct)
            if last_pos > MAX_QUERY_LENGTH * 0.8:  # 至少保留80%的内容
                truncated = truncated[:last_pos + 1]
                break

        return truncated

    def check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """检查缓存中是否有重复查询

        【人工重构区：Token分层节流、重复问答本地缓存逻辑】
        本函数为临时实现，由项目负责人重构为更智能的缓存匹配策略

        重构方向建议：
        1. 实现语义相似度缓存匹配（而非完全匹配）
        2. 支持模糊查询缓存命中
        3. 引入缓存优先级和淘汰机制
        4. 基于用户画像的个性化缓存策略
        """
        # 生成查询哈希
        query_hash = hashlib.md5(query.encode()).hexdigest()

        if query_hash in self.cache:
            entry = self.cache[query_hash]
            cached_time = datetime.fromisoformat(entry.get("timestamp", ""))

            # 检查缓存是否过期
            if datetime.now() - cached_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                return entry["response"]

        return None

    def update_cache(self, query: str, response: Dict[str, Any]):
        """更新查询缓存

        【人工重构区：Token分层节流、重复问答本地缓存逻辑】
        本函数为临时实现，由项目负责人重构为更智能的缓存更新策略

        重构方向建议：
        1. 基于响应质量评估决定是否缓存
        2. 实现缓存容量限制和LRU淘汰
        3. 支持缓存预热和手动刷新
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()

        self.cache[query_hash] = {
            "query": query[:100],  # 仅存储查询前100字符用于展示
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

        # 限制缓存大小
        if len(self.cache) > 1000:
            # 按时间排序，删除最旧的条目
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k].get("timestamp", "")
            )
            for key in sorted_keys[:100]:
                del self.cache[key]

        self._save_cache()

    def record_token_usage(self, token_count: int):
        """记录Token使用量"""
        self.daily_usage += token_count
        self._save_usage()

    def check_daily_limit(self) -> bool:
        """检查是否超过每日Token使用限制"""
        return self.daily_usage < DAILY_TOKEN_LIMIT

    def get_usage_stats(self) -> Dict[str, Any]:
        """获取Token使用统计"""
        return {
            "daily_usage": self.daily_usage,
            "daily_limit": DAILY_TOKEN_LIMIT,
            "remaining": max(0, DAILY_TOKEN_LIMIT - self.daily_usage),
            "cache_size": len(self.cache),
            "date": datetime.now().strftime("%Y-%m-%d")
        }

    def estimate_tokens(self, text: str) -> int:
        """估算文本Token数（简化版：1个汉字约等于1.5个Token）"""
        return int(len(text) * 1.5)
