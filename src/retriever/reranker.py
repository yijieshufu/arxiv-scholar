"""
Rerank 模块 — 从 resume-ai 迁移，适配 arxiv-scholar

支持 BGE-Reranker 和 LLM Rerank 两种引擎。
"""
import logging
import gc
from typing import List, Dict

from src.config import config

logger = logging.getLogger(__name__)

# 全局 Reranker 缓存（跨 session_state / 跨点击复用，避免反复加载模型）
_RERANKER_CACHE: Dict[str, object] = {}


def _get_cached_cross_encoder(model_name: str, device: str = "cpu"):
    """获取缓存的 CrossEncoder 实例（全局共享，只加载一次）"""
    cache_key = f"{model_name}:{device}"
    if cache_key not in _RERANKER_CACHE:
        from sentence_transformers import CrossEncoder
        logger.info(f"加载 Reranker: {model_name} on {device}")
        _RERANKER_CACHE[cache_key] = CrossEncoder(model_name, device=device)
    return _RERANKER_CACHE[cache_key]


class BGEReranker:
    """CrossEncoder 重排序（使用全局缓存，只在首次加载模型）

    默认使用 BAAI/bge-reranker-v2-m3，与 BGE-M3 embedding 同系列，
    在学术论文多语言场景下比通用 ms-marco-MiniLM 精度显著更高。
    """

    def __init__(self, model_name: str = None, max_length: int = 512):
        self.model_name = model_name or config.reranker.model_name
        self.max_length = max_length

    def rerank(self, query: str, candidates: List[Dict], top_k: int = None):
        if not candidates:
            return []
        top_k = top_k or config.retrieval.top_k_rerank

        # ── Phase B: 文本规范化清洗 + 安全截断 ──
        # 过滤 IEEE/CJK 特殊排版符号，防止 tokenizer 崩溃
        _NOISE_CHARS = str.maketrans({
            '×': 'x', '–': '-', '—': '-', '•': '.', '·': '.',
            '●': '*', '◆': '*', '▶': '>', '◆': '*', '◦': '.',
            '©': '(c)', '®': '(r)', '™': '(tm)',
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'λ': 'lambda', 'μ': 'mu', 'σ': 'sigma',
            'θ': 'theta', 'φ': 'phi', 'ψ': 'psi', 'ω': 'omega',
            '≤': '<=', '≥': '>=', '≠': '!=', '±': '+-',
            '≈': '~=', '∞': 'inf', '∀': 'for all', '∃': 'exists',
            '→': '->', '←': '<-', '⇒': '=>', '⇔': '<=>',
        })
        cleaned = []
        for c in candidates:
            raw = c.get("text", "")
            # 强转 UTF-8 剔除畸形字节
            raw = str(raw).encode('utf-8', 'ignore').decode('utf-8')
            # 替换特殊排版符
            raw = raw.translate(_NOISE_CHARS)
            # 安全截断至 1200 字符（~300 token），防止超 max_length
            raw = raw[:1200]
            c["text"] = raw
            cleaned.append(c)
        candidates = cleaned

        model = _get_cached_cross_encoder(self.model_name)
        pairs = [(query, c["text"]) for c in candidates]
        try:
            scores = model.predict(
                pairs,
                batch_size=16,
                show_progress_bar=len(pairs) > 10,
            )
        except Exception as e:
            import traceback, sys
            for i, c in enumerate(candidates):
                meta = c.get("metadata", {})
                idx_str = meta.get("_idx", "?")
                # 打印首个出错 block 的上下文
                try:
                    _ = model.predict([(query, c["text"][:200])])
                except Exception as e2:
                    print(f"[RERANK CRITICAL ERROR] 块 idx={idx_str} 实际报错原因: {e2}", flush=True)
                    traceback.print_exc()
                    break
            print(f"[RERANK CRITICAL ERROR] 全部共 {len(pairs)} 对送入失败", flush=True)
            traceback.print_exc()
            raise RuntimeError(f"Reranker 重排失败: {e}") from e

        for i, c in enumerate(candidates):
            c["rerank_score"] = round(float(scores[i]), 4)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        gc.collect()
        return candidates[:top_k]


class LLMReranker:
    """LLM Listwise 重排序（RankGPT 风格）。

    将所有候选一次性送入 LLM，让模型在全局比较后返回排序结果。
    相比 Pointwise（逐条打分再排序），Listwise 有两个优势：
    1. 延迟低：1 次 API 调用 vs N 次（15 条 → 从 ~45s 降到 ~5s）
    2. 效果好：LLM 能看到全局做相对比较，而非独立打分
    """

    LISTWISE_PROMPT = (
        "你是一个专业的论文检索排序助手。请根据用户查询，"
        "对以下论文片段按相关性从高到低排序。\n\n"
        "用户查询: {query}\n\n"
        "论文片段列表（共 {n} 条）:\n{context}\n\n"
        "请严格按照从最相关到最不相关的顺序，"
        "仅返回一个 JSON 格式的整数列表，格式如 [2,0,1,...]。\n"
        "列表中应包含所有 {n} 个 ID，每个 ID 出现且仅出现一次。\n"
        "不要输出任何解释或其他内容。"
    )

    def __init__(self, llm_model: str = None):
        self.llm_model = llm_model or config.reranker.llm_model

    def rerank(self, query: str, candidates: List[Dict], top_k: int = None):
        if not candidates:
            return []
        top_k = top_k or config.retrieval.top_k_rerank

        # 限制候选数，避免 context 过长
        pool = candidates[:15]
        if len(pool) < 2:
            return pool[:top_k]

        # 序列化为 RankGPT 格式
        lines = []
        for i, c in enumerate(pool):
            meta = c.get("metadata", {})
            title = meta.get("section_title", str(meta.get("section_id", "")))
            full_html = meta.get("full_html_content", "")
            if full_html:
                txt = full_html[:1200].replace("\n", " ").replace("\r", " ")
            else:
                txt = (c.get("text") or "")[:500].replace("\n", " ").replace("\r", " ")
            lines.append(f"[ID: {i}] 章节: {title}\n内容: {txt}")

        context = "\n\n".join(lines)
        prompt = self.LISTWISE_PROMPT.format(
            query=query, n=len(pool), context=context,
        )

        from src.config import get_llm_client
        client = get_llm_client()

        try:
            resp = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system",
                     "content": "你是一个排序专家，只返回 JSON 数组，不输出任何其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            import json
            import re
            raw = resp.choices[0].message.content.strip()
            array_match = re.search(r'\[[\d,\s]+\]', raw)
            if not array_match:
                logger.warning(
                    "[LLM RERANK] LLM 未返回合法 JSON 数组，降级为原始排序: %s...",
                    raw[:200],
                )
                candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
                return candidates[:top_k]

            ranked_ids = json.loads(array_match.group(0))

            if len(ranked_ids) != len(pool):
                logger.warning(
                    "[LLM RERANK] 返回 %d 个 ID（预期 %d），触发降级",
                    len(ranked_ids), len(pool),
                )
                candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
                return candidates[:top_k]

            # 按 LLM 排序重排 pool
            id_map = {i: pool[i] for i in range(len(pool))}
            reranked = [id_map[i] for i in ranked_ids if i in id_map]

            # pool 之外的候选保留原始排序
            extra = [c for c in candidates if c not in pool]
            extra.sort(key=lambda x: x.get("score", 0), reverse=True)

            result = reranked + extra
            return result[:top_k]

        except Exception as e:
            import traceback
            logger.error(
                "[CRITICAL RERANK ERROR] LLM Listwise Rerank 失败: %s", e
            )
            traceback.print_exc()
            # 降级：保留原始融合分数排序
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            return candidates[:top_k]


def get_reranker(engine: str = None):
    engine = engine or config.reranker.engine
    if engine == "bge_reranker":
        return BGEReranker()
    elif engine == "llm_rerank":
        return LLMReranker()
    else:
        raise ValueError(f"未知 Reranker: {engine}")
