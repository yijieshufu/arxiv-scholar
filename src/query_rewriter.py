"""
Query 改写模块 — 将自然语言问题改写为学术检索查询

参考：
- HyDE (Hypothetical Document Embeddings)：先假设一篇论文再检索
- MultiQueryRetriever：生成多个改写查询提高召回率
- Step-Back Prompting：先抽象问题再检索

课程关联：第 13 章 Query 改写、第 15 章 MultiQueryRetriever
"""
import re
import logging
from typing import List, Optional
from openai import OpenAI

from src.config import config

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def has_cjk(text: str) -> bool:
    """检测是否包含中日韩字符（中文查询需改写为英文学术术语）"""
    return bool(_CJK_RE.search(text))


class QueryRewriter:
    """
    学术场景 Query 改写器。

    策略（按优先级）：
    1. 学术术语标准化：LLM 改写成 ArXiv 友好的搜索词
    2. HyDE 假设论文：生成假想的论文摘要来检索
    3. MultiQuery 多路改写：生成 3-5 个不同角度的查询
    4. 关键词提取：提取关键术语用于 BM25 精确匹配
    """

    def __init__(self, llm_client: OpenAI = None):
        self.llm = llm_client
        if self.llm is None:
            from src.config import get_llm_client
            self.llm = get_llm_client()

    def rewrite(self, query: str, strategy: str = "multi") -> List[str]:
        """
        改写查询。

        Args:
            query: 用户原始查询
            strategy: 改写策略 ("academic", "hyde", "multi", "keywords", "auto")

        Returns:
            改写后的查询列表（原始查询排第一）
        """
        if strategy == "auto":
            # 中文优先走学术术语+关键词；长句走 multi
            if has_cjk(query):
                strategy = "chinese_academic"
            elif len(query) > 100:
                strategy = "multi"
            else:
                strategy = "academic"

        rewrites = [query]  # 保留原始查询（BGE-M3 支持跨语言向量检索）

        if strategy == "chinese_academic":
            rewritten = self._academic_rewrite(query)
            if rewritten and rewritten != query:
                rewrites.append(rewritten)
            keywords = self._keyword_rewrite(query)
            rewrites.extend([k for k in keywords if k and k != query])
            multi = self._multi_query_rewrite(query)
            rewrites.extend([q for q in multi if q != query])
            # 意图膨胀：追加行业术语查询
            intent = self._intent_expansion(query)
            rewrites.extend([q for q in intent if q not in rewrites])

        elif strategy == "academic":
            rewritten = self._academic_rewrite(query)
            if rewritten and rewritten != query:
                rewrites.append(rewritten)

        elif strategy == "hyde":
            hyde_text = self._hyde_rewrite(query)
            if hyde_text:
                rewrites.append(hyde_text)

        elif strategy == "multi":
            multi = self._multi_query_rewrite(query)
            rewrites.extend([q for q in multi if q != query])

        elif strategy == "keywords":
            keywords = self._keyword_rewrite(query)
            rewrites.extend([k for k in keywords if k != query])

        # 去重保序 + 过滤过短改写（"ResNet-50"/"ImageNet" 等无助于 BM25/向量检索的单词）
        seen = set()
        unique = []
        for q in rewrites:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                # 过滤纯数字/短词/逗号分隔的短列表（如 "ResNet-50, ImageNet"）
                stripped = q.strip()
                word_count = len(stripped.split())
                if word_count < 3:
                    continue  # 少于 3 个词的改写无效
                unique.append(stripped)
        rewrites = unique

        logger.info(f"Query 改写 [{strategy}]: {query[:50]}... → {len(rewrites)} 个查询")
        return rewrites[:6]  # 最多 6 个

    def rewrite_for_arxiv(self, query: str) -> str:
        """将查询改写为 ArXiv API 友好的英文学术搜索词。

        策略优先级：
        1. 仅用 _academic_rewrite (LLM 直翻 2-5 个核心术语，不加限定词)
        2. 若 academic 失败/返回中文/过长，降级到 keyword 提取
        3. 最后才用 multi_query（取最短的英文候选）
        """
        if not has_cjk(query):
            return query

        # Strategy 1: academic rewrite (single, tight, 2-5 terms)
        academic = self._academic_rewrite(query)
        if academic and not has_cjk(academic):
            # 如果太长(>8词)或含"and"串联过多概念，切取前 2 个关键词组
            words = academic.split()
            if len(words) > 8 or academic.count(" and ") >= 2:
                # 尝试只保留前 3-5 个最核心的词
                core = " ".join(words[:5])
                logger.info(f"ArXiv 搜索改写 (裁剪): '{query}' → '{core}'")
                return core
            logger.info(f"ArXiv 搜索改写: '{query}' → '{academic}'")
            return academic

        # Strategy 2: keyword fallback (comma-separated → join as AND)
        keywords = self._keyword_rewrite(query)
        english_kw = [k for k in keywords if k and not has_cjk(k)]
        if english_kw:
            # 取前 3 个关键词用空格拼接
            candidate = " ".join(english_kw[:3])
            if len(candidate.split()) <= 8:
                logger.info(f"ArXiv 搜索改写 (keywords): '{query}' → '{candidate}'")
                return candidate

        # Strategy 3: multi_query last resort (shortest English candidate)
        multi = self._multi_query_rewrite(query)
        english_multi = [q for q in multi if q and not has_cjk(q)]
        if english_multi:
            # 选最短的（最聚焦）
            candidate = min(english_multi, key=lambda x: len(x.split()))
            logger.info(f"ArXiv 搜索改写 (multi最短): '{query}' → '{candidate}'")
            return candidate

        return academic or query

    def _academic_rewrite(self, query: str) -> Optional[str]:
        """学术术语标准化改写"""
        system = """你是一个学术搜索专家。将用户的自然语言问题改写为适合在 ArXiv 上搜索的学术查询。
规则：
1. 提取核心研究问题，去掉口语化表达
2. 使用领域术语（如 "Transformer" → 保留, "图片识别" → "image recognition"）
3. 中文问题必须翻译为英文医学/计算机/学术术语（如 "肠息肉" → "colorectal polyp detection"）
4. 中英文混合时优先使用英文术语
5. **严格限制 2-5 个核心词，不要用 "and" 串联多个概念**。ArXiv 搜索越短越精准
6. 示例：中文"肠息肉检测的深度学习方法" → "colorectal polyp detection"（不要写成 "colorectal polyp detection and segmentation and deep learning"）
7. 只返回改写后的查询，不要任何解释。"""

        try:
            resp = self.llm.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=100,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"学术改写失败: {e}")
            return None

    def _hyde_rewrite(self, query: str) -> Optional[str]:
        """
        HyDE 策略：生成一段假想的论文摘要作为检索查询。

        参考：Precise Zero-Shot Dense Retrieval without Relevance Labels
        适用场景：用户问题太短，缺乏上下文
        """
        system = """你是一个学术研究者。根据用户的问题，写一段假想的论文摘要（约 100-150 词），
描述一篇可能回答这个问题的论文。包含：方法名称、关键发现、实验设置。
只返回摘要内容，不要任何前缀。"""

        try:
            resp = self.llm.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"问题：{query}"},
                ],
                temperature=0.5,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"HyDE 生成失败: {e}")
            return None

    def _multi_query_rewrite(self, query: str) -> List[str]:
        """
        MultiQuery 策略：从多个角度改写查询。

        参考：LangChain MultiQueryRetriever
        生成 3-4 个不同版本的查询以提高召回率
        """
        system = """将用户的研究问题改写成 3-4 个不同角度的 ArXiv 搜索查询。
每个查询应该从不同维度切入：
- 技术/方法角度
- 应用/场景角度
- 对比/综述角度
- 最新进展角度

若原问题含中文，所有改写必须使用英文术语。
返回格式：每行一个查询，不要编号。"""

        try:
            resp = self.llm.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            lines = resp.choices[0].message.content.strip().split("\n")
            return [line.strip().lstrip("0123456789.-) ") for line in lines if line.strip()]
        except Exception as e:
            logger.warning(f"MultiQuery 生成失败: {e}")
            return []

    def _keyword_rewrite(self, query: str) -> List[str]:
        """提取关键词用于 BM25 精确匹配"""
        system = """从用户的研究问题中提取 3-5 个核心关键词/短语。
这些关键词应该适合在论文标题和摘要中进行精确匹配。
只返回关键词，用逗号分隔。"""

        try:
            resp = self.llm.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=100,
            )
            keywords = resp.choices[0].message.content.strip()
            return [k.strip() for k in keywords.split(",") if k.strip()]
        except Exception as e:
            logger.warning(f"关键词提取失败: {e}")
            return []

    def _intent_expansion(self, query: str) -> List[str]:
        """
        意图膨胀：当用户查询涉及数据集/实验/参数配置时，
        追加带标准章节关键词的查询，使 section_title 命中加分生效。
        """
        lower = query.lower()
        expansions = []

        # 数据/基准意图
        dataset_kw = {"数据集", "数据", "训练集", "测试集", "datasets", "data",
                      "benchmarks", "benchmark", "evaluation", "kvasir",
                      "cvc", "polyp"}
        if any(kw in lower for kw in dataset_kw):
            expansions.append(f"Datasets and Benchmarks and Evaluation {query}")
            expansions.append(f"Experimental Settings and Datasets {query}")

        # 实验/对比意图
        experiment_kw = {"实验", "对比", "结果", "performance", "results",
                         "experiment", "ablation", "comparison"}
        if any(kw in lower for kw in experiment_kw):
            expansions.append(f"Experiments and Results {query}")
            expansions.append(f"Ablation Study and Comparison {query}")

        # 参数/配置意图
        config_kw = {"参数", "配置", "setting", "implementation",
                     "parameter", "hyperparameter", "setup"}
        if any(kw in lower for kw in config_kw):
            expansions.append(f"Implementation Details and Parameters {query}")
            expansions.append(f"Experimental Settings and Configuration {query}")

        return expansions[:4]  # 最多膨胀 4 个


class ContextualQueryRewriter:
    """
    对话上下文 Query 改写器。

    将多轮对话中的新问题 + 历史上下文 → 独立可搜索的学术查询。
    解决代词消解 / 隐式引用 / 主题延续。
    """

    def __init__(self, llm_client: OpenAI = None):
        self.llm = llm_client
        if self.llm is None:
            from src.config import get_llm_client
            self.llm = get_llm_client()

    def rewrite(self, query: str, chat_history: List[dict]) -> str:
        """将 query + 最近 3 轮对话改写成独立搜索词。"""
        if not chat_history or len(chat_history) < 2:
            return query

        recent = chat_history[-6:]  # 3 轮 = 6 条
        history_text = "\n".join([
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:300]}"
            for m in recent
        ])

        system = """你是一个对话理解助手。基于对话历史和用户最新问题，生成一个**独立可搜索**的学术查询。

规则：
1. 如果最新问题已经足够独立（不含代词或隐式引用），直接原文返回
2. 如果包含代词（它、这个、那个、它们、这些、该方法、上述等），用历史中的具体术语替换
3. 如果包含隐式引用（和这个比、上一个、还有什么、还有其他吗等），补全上下文
4. 输出必须是一个完整的英文学术搜索短语，简洁（≤40词）
5. 只返回改写后的查询，不要任何解释或前缀"""

        try:
            resp = self.llm.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"对话历史：\n{history_text}\n\n最新问题：{query}"},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            rewritten = resp.choices[0].message.content.strip()
            logger.info(f"上下文改写: '{query[:50]}...' → '{rewritten[:80]}...'")
            return rewritten
        except Exception as e:
            logger.warning(f"上下文改写失败，使用原始查询: {e}")
            return query


class StepBackRewriter:
    """
    Step-Back 改写器。

    参考：Take a Step Back (Zheng et al., 2023)
    先生成更抽象/更宽泛的问题，用它的检索结果来辅助原始问题的回答。
    """

    def __init__(self, llm_client: OpenAI = None):
        self.llm = llm_client
        if self.llm is None:
            from src.config import get_llm_client
            self.llm = get_llm_client()

    def step_back(self, query: str) -> Optional[str]:
        """生成 Step-Back 问题"""
        system = """给定一个具体的研究问题，生成一个更抽象、更宽泛的 Step-Back 问题。
Step-Back 问题应该：
1. 移除具体细节，提取核心概念
2. 更宽泛但相关
3. 帮助找到背景知识

示例：
- 具体: "GPT-4 的 RLHF 训练用了多少人类反馈数据？"
- Step-Back: "RLHF 训练方法概述"

只返回 Step-Back 问题。"""

        try:
            resp = self.llm.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"具体问题：{query}"},
                ],
                temperature=0.1,
                max_tokens=100,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Step-Back 生成失败: {e}")
            return None
