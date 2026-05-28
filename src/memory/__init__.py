"""
结构化对话记忆模块

跨轮追踪对话中提到的关键信息：
- 论文 (papers): 用户提过的论文标题、ID
- 方法 (methods): 讨论过的方法/模型名
- 指标 (metrics): 提及的评估指标和数值
- 表格 (tables): 引用过的表格
- 最后话题 (last_topic): 最近讨论的主题

用途：
1. 支持模糊引用（"那个方法" → 自动匹配最近提到的方法）
2. 让 LLM 生成时可以引用历史上下文
3. 长对话的摘要压缩
"""
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ── 实体提取模式 ──
_PAPER_PATTERN = re.compile(
    r'(?:论文|paper|work|method)\s*[：:]\s*["""]?([^""""\n]{5,60})["""]?', re.IGNORECASE
)
_METHOD_PATTERN = re.compile(
    r'[A-Z][A-Za-z0-9/-]*(?:Net|Former|Model|GNN|CNN|RNN|GAN|VAE|SAM|DETR|YOLO|BERT|GPT|LoRA|Caps)',
)
_METRIC_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*%?\s*(?:准确率|精度|召回率|F1|sensitivity|specificity|accuracy|precision|recall|mIoU|AUC|NDCG|MRR)',
    re.IGNORECASE,
)
_TABLE_PATTERN = re.compile(r'Table[_\s](\d+)', re.IGNORECASE)


class ConversationMemory:
    """单次对话的结构化记忆。"""

    def __init__(self):
        self.papers: List[str] = []         # 提到的论文
        self.methods: List[str] = []         # 讨论的方法
        self.metrics: List[str] = []         # 指标（含数值）
        self.tables: List[str] = []          # 引用过的表
        self.last_topic: str = ""            # 最后讨论主题
        self.turn_count: int = 0             # 对话轮次计数

    def update(self, user_query: str, assistant_answer: str = ""):
        """从一轮对话中提取并更新记忆。"""
        combined = f"{user_query} {assistant_answer}"
        self.turn_count += 1

        # 论文
        for m in _PAPER_PATTERN.finditer(combined):
            p = m.group(1).strip()
            if p and p not in self.papers:
                self.papers.append(p)

        # 方法（大写驼峰学术术语）
        for m in _METHOD_PATTERN.finditer(combined):
            method = m.group(0)
            if method and method not in self.methods:
                self.methods.append(method)

        # 指标
        for m in _METRIC_PATTERN.finditer(combined):
            metric = m.group(0).strip()
            if metric and metric not in self.metrics:
                self.metrics.append(metric)

        # 表格
        for m in _TABLE_PATTERN.finditer(combined):
            t = f"Table_{m.group(1)}"
            if t not in self.tables:
                self.tables.append(t)

        # 最后话题（取用户问题的前 60 字）
        self.last_topic = user_query[:60]

        # 裁剪：最多保留 10 个
        self.papers = self.papers[-10:]
        self.methods = self.methods[-10:]
        self.metrics = self.metrics[-10:]
        self.tables = self.tables[-5:]

        logger.debug(f"记忆更新: {len(self.papers)}论文, {len(self.methods)}方法, "
                     f"{len(self.metrics)}指标, {len(self.tables)}表格")

    def to_context_prompt(self) -> str:
        """将记忆格式化为 LLM 可用的上下文提示片段。"""
        parts = []
        if self.papers:
            parts.append(f"之前提到的论文: {'、'.join(self.papers[-5:])}")
        if self.methods:
            parts.append(f"讨论过的方法: {'、'.join(self.methods[-5:])}")
        if self.metrics:
            parts.append(f"提及的指标: {'、'.join(self.metrics[-5:])}")
        if self.tables:
            parts.append(f"引用过的表格: {'、'.join(self.tables[-3:])}")
        if self.last_topic:
            parts.append(f"上轮话题: {self.last_topic}")
        return " | ".join(parts) if parts else ""

    def resolve_reference(self, query: str) -> str:
        """
        尝试解析模糊引用。

        例如：
        - "那个方法" → "Iv3"（最近提到的方法）
        - "这张表" → "Table_2"（最近引用的表）
        - "和前面比" → "与 D-Caps 比较"
        """
        lower = query.lower()

        # 方法引用
        if any(kw in lower for kw in ["这个方法", "该模型", "那个方法", "上一个", "上述"]):
            if self.methods:
                logger.info(f"模糊引用解析: '{query[:30]}' → 方法 '{self.methods[-1]}'")
                return query.replace("这个方法", self.methods[-1]).replace("该模型", self.methods[-1])
        # 表格引用
        if any(kw in lower for kw in ["这张表", "该表", "上表", "表格"]):
            if self.tables:
                logger.info(f"模糊引用解析: '{query[:30]}' → 表格 '{self.tables[-1]}'")
                return query.replace("这张表", self.tables[-1]).replace("该表", self.tables[-1])
        # 论文引用
        if any(kw in lower for kw in ["该论文", "这篇文章", "这篇论文"]):
            if self.papers:
                logger.info(f"模糊引用解析: '{query[:30]}' → 论文 '{self.papers[-1]}'")
                return query.replace("该论文", self.papers[-1])

        return query  # 无匹配，原文返回
