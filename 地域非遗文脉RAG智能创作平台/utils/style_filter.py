"""
地域非遗文脉RAG智能创作平台 - 非遗风格校验过滤工具
基础工程代码由 Vibe Coding 智能体生成

【人工重构区：非遗风格强制校验过滤引擎】
本文件包含非遗风格判定规则插槽，由项目负责人独立人工重构开发
"""
import re
from typing import Optional, List, Dict
from datetime import datetime

# ============================================
# 风险词库（基础版）
# ============================================
# 低俗、暴力、政治敏感等违规内容关键词
RISK_KEYWORDS = [
    # 政治敏感词（示例，实际应更全面）
    "颠覆", "分裂", "恐怖", "极端",
    # 低俗内容
    "色情", "赌博", "毒品",
    # 违法内容
    "枪支", "爆炸", "制毒",
    # 商业违规
    "传销", "诈骗", "非法集资",
]

# 非遗风格正向关键词
HERITAGE_STYLE_KEYWORDS = [
    "传统", "古典", "古朴", "典雅", "雅致",
    "民族", "民间", "民俗", "乡土", "田园",
    "手工", "匠人", "匠心", "工艺", "技艺",
    "文化", "历史", "传承", "经典", "非遗",
    "自然", "生态", "环保", "可持续",
    "吉祥", "喜庆", "祝福", "福禄寿",
    "山水", "花鸟", "人物", "故事",
    "国风", "中国风", "中式", "东方",
]

# 风格类别定义
STYLE_CATEGORIES = {
    "古典雅致": ["古典", "雅致", "传统", "古朴", "典雅", "庄重"],
    "民族民俗": ["民族", "民间", "民俗", "乡土", "田园", "地域"],
    "自然生态": ["自然", "生态", "环保", "山水", "花鸟", "植物"],
    "吉祥喜庆": ["吉祥", "喜庆", "祝福", "福禄寿", "新春", "婚庆"],
    "工艺匠心": ["手工", "匠人", "匠心", "工艺", "技艺", "精细"],
    "文化传承": ["文化", "历史", "传承", "经典", "非遗", "遗产"],
    "国风时尚": ["国风", "中国风", "中式", "东方", "现代国潮"],
}


class StyleFilter:
    """非遗风格校验过滤器"""

    def __init__(self):
        self.filter_log = []

    def validate_style(self, style_description: str) -> Optional[str]:
        """校验风格描述是否符合非遗文化规范

        【人工重构区：非遗风格强制校验过滤引擎】
        本函数为临时实现，由项目负责人重构为更智能的风格判定规则

        重构方向建议：
        1. 基于NLP技术进行语义级别的风格分析
        2. 引入非遗专家知识图谱进行风格匹配
        3. 实现多维度风格评分机制
        4. 支持风格混合和创新度评估
        """
        # 1. 基础内容安全检查
        if not self._check_content_safety(style_description):
            self._log_filter("rejected", style_description, "包含违规内容")
            return None

        # 2. 空内容检查
        if not style_description or not style_description.strip():
            self._log_filter("rejected", style_description, "内容为空")
            return None

        # 3. 长度检查
        if len(style_description) > 200:
            style_description = style_description[:200]
            self._log_filter("truncated", style_description, "内容过长已截断")

        # 4. 风格相关性检查
        style_score = self._calculate_style_relevance(style_description)
        if style_score < 0.3:
            # 风格相关性较低，添加非遗风格引导
            style_description = self._enhance_with_heritage_style(style_description)
            self._log_filter("enhanced", style_description, f"风格相关性较低({style_score:.2f})，已添加非遗风格引导")

        self._log_filter("accepted", style_description, f"风格相关性: {style_score:.2f}")
        return style_description

    def _check_content_safety(self, text: str) -> bool:
        """内容安全检查"""
        text_lower = text.lower()
        for keyword in RISK_KEYWORDS:
            if keyword in text_lower:
                return False
        return True

    def _calculate_style_relevance(self, text: str) -> float:
        """计算风格与非遗的相关性

        【人工重构区：非遗风格强制校验过滤引擎】
        本函数为临时实现，由项目负责人重构为更精确的风格相关性算法

        重构方向建议：
        1. 使用词向量计算语义相似度
        2. 引入TF-IDF权重
        3. 考虑非遗专业术语的权重
        4. 基于上下文的动态权重调整
        """
        if not text:
            return 0.0

        match_count = 0
        for keyword in HERITAGE_STYLE_KEYWORDS:
            if keyword in text:
                match_count += 1

        # 计算匹配比例
        relevance = min(1.0, match_count / 3)  # 至少匹配3个关键词为满分
        return relevance

    def _enhance_with_heritage_style(self, text: str) -> str:
        """为描述添加非遗风格元素"""
        heritage_suffixes = [
            "，体现传统非遗工艺特色",
            "，融入民族民间文化元素",
            "，展现古典东方美学韵味",
            "，传承非物质文化遗产精髓",
        ]

        # 随机选择一个后缀（简化实现）
        import random
        suffix = random.choice(heritage_suffixes)

        return text + suffix

    def detect_style_category(self, text: str) -> List[str]:
        """检测风格类别

        【人工重构区：非遗风格强制校验过滤引擎】
        本函数为临时实现，由项目负责人重构为更精确的风格分类算法

        重构方向建议：
        1. 使用机器学习模型进行风格分类
        2. 引入多标签分类机制
        3. 支持自定义风格类别
        4. 基于用户反馈的动态调整
        """
        detected = []
        for category, keywords in STYLE_CATEGORIES.items():
            if any(kw in text for kw in keywords):
                detected.append(category)
        return detected if detected else ["通用风格"]

    def generate_style_suggestions(self, category: str) -> List[str]:
        """生成风格建议"""
        suggestions = {
            "古典雅致": [
                "采用深色木纹背景，搭配金色线条装饰",
                "使用书法字体，体现传统文化底蕴",
                "融入水墨画元素，营造意境美感",
            ],
            "民族民俗": [
                "使用鲜艳的民族配色，如红、黄、蓝",
                "融入少数民族图腾和纹样",
                "展现地方特色手工艺品元素",
            ],
            "自然生态": [
                "采用绿色、棕色等自然色调",
                "融入山水、花鸟等自然元素",
                "体现环保、可持续理念",
            ],
            "吉祥喜庆": [
                "使用红色、金色等喜庆配色",
                "融入龙凤、福字等吉祥图案",
                "展现节日庆典氛围",
            ],
            "工艺匠心": [
                "突出手工制作的质感和细节",
                "展现匠人精神和精湛技艺",
                "使用特写镜头展示工艺过程",
            ],
            "文化传承": [
                "讲述非遗背后的历史故事",
                "展现文化传承的脉络",
                "体现传统与现代的融合",
            ],
            "国风时尚": [
                "将传统元素与现代设计结合",
                "使用国潮风格的视觉语言",
                "展现东方美学的时尚表达",
            ],
        }
        return suggestions.get(category, ["请根据具体需求选择合适的风格"])

    def _log_filter(self, action: str, text: str, reason: str):
        """记录过滤日志"""
        self.filter_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "reason": reason
        })

        # 保持日志大小合理
        if len(self.filter_log) > 100:
            self.filter_log = self.filter_log[-50:]
