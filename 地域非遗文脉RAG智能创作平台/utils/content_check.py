"""
地域非遗文脉RAG智能创作平台 - 通用内容合规过滤工具
本模块完整自动实现，无需人工修改
"""
import re
from typing import Optional, List, Dict
from datetime import datetime

# ============================================
# 敏感词库
# ============================================
SENSITIVE_KEYWORDS = [
    # 政治敏感词
    "颠覆国家政权", "分裂国家", "恐怖主义", "极端主义",
    "邪教", "非法集会", "扰乱社会秩序",

    # 违法内容
    "枪支弹药", "爆炸物制作", "制毒贩毒", "贩卖人口",
    "网络诈骗", "非法传销", "赌博网站",

    # 低俗内容
    "色情", "淫秽", "赌博", "暴力血腥",

    # 侵权内容
    "盗版", "破解软件", "非法下载",

    # 虚假信息
    "虚假广告", "夸大宣传", "假冒伪劣",

    # 商业违规
    "传销", "庞氏骗局", "非法集资",
]

# 危险模式（正则表达式）
DANGEROUS_PATTERNS = [
    r"如何制造(枪支|炸弹|毒品)",
    r"(赌博|博彩)网站(地址|链接)",
    r"(翻墙|VPN)下载",
    r"(色情|成人)网站",
]

# 内容长度限制
MAX_CONTENT_LENGTH = 10000  # 最大内容长度
MIN_CONTENT_LENGTH = 5      # 最小内容长度


class ContentFilter:
    """通用内容合规过滤器"""

    def __init__(self):
        self.filter_count = 0
        self.pass_count = 0
        self.log = []

    def check(self, content: str) -> Optional[str]:
        """检查内容是否合规，返回处理后的内容或None（表示不合规）"""
        if not content:
            return None

        # 1. 长度检查
        content = self._check_length(content)
        if content is None:
            return None

        # 2. 敏感词检查
        if not self._check_sensitive_keywords(content):
            self._log_filter("rejected", "包含敏感词")
            return None

        # 3. 危险模式检查
        if not self._check_dangerous_patterns(content):
            self._log_filter("rejected", "匹配危险模式")
            return None

        # 4. 过度重复内容检查
        if self._is_excessive_repetition(content):
            self._log_filter("rejected", "内容过度重复")
            return None

        # 5. HTML/脚本注入检查
        content = self._sanitize_html(content)

        self.pass_count += 1
        self._log_filter("passed", "")
        return content

    def _check_length(self, content: str) -> Optional[str]:
        """内容长度检查"""
        content = content.strip()

        if len(content) < MIN_CONTENT_LENGTH:
            self._log_filter("rejected", "内容过短")
            return None

        if len(content) > MAX_CONTENT_LENGTH:
            # 截断超长内容
            content = content[:MAX_CONTENT_LENGTH]
            self._log_filter("truncated", "内容过长已截断")

        return content

    def _check_sensitive_keywords(self, content: str) -> bool:
        """敏感词检查"""
        content_lower = content.lower()
        for keyword in SENSITIVE_KEYWORDS:
            if keyword.lower() in content_lower:
                return False
        return True

    def _check_dangerous_patterns(self, content: str) -> bool:
        """危险模式检查"""
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        return True

    def _is_excessive_repetition(self, content: str) -> bool:
        """检查内容是否过度重复"""
        if len(content) < 50:
            return False

        # 检查单个字符的重复率
        char_count = {}
        for char in content:
            if char.isalnum():
                char_count[char] = char_count.get(char, 0) + 1

        if not char_count:
            return False

        max_count = max(char_count.values())
        total_chars = sum(char_count.values())

        # 如果单个字符占比超过50%，认为是过度重复
        return max_count / total_chars > 0.5

    def _sanitize_html(self, content: str) -> str:
        """清理HTML标签和脚本"""
        # 移除HTML标签
        content = re.sub(r'<[^>]+>', '', content)

        # 移除JavaScript代码
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # 移除事件处理器
        content = re.sub(r'\bon\w+\s*=', '', content, flags=re.IGNORECASE)

        # 移除data: URI
        content = re.sub(r'data:[^,]*;base64,', '', content)

        return content

    def get_stats(self) -> Dict[str, int]:
        """获取过滤统计"""
        return {
            "total_checks": self.filter_count + self.pass_count,
            "passed": self.pass_count,
            "rejected": self.filter_count,
            "pass_rate": f"{(self.pass_count / max(1, self.filter_count + self.pass_count)) * 100:.1f}%"
        }

    def _log_filter(self, action: str, reason: str):
        """记录过滤日志"""
        if action == "rejected":
            self.filter_count += 1

        self.log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "reason": reason
        })

        # 保持日志大小合理
        if len(self.log) > 1000:
            self.log = self.log[-500:]

    def check_batch(self, contents: List[str]) -> List[Optional[str]]:
        """批量内容检查"""
        return [self.check(content) for content in contents]
