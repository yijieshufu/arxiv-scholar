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
    """CrossEncoder 重排序（使用全局缓存，只在首次加载模型）"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.reranker.model_name

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
    """LLM 打分重排序"""

    RERANK_PROMPT = """评分: 查询与文本块的相关性 (0-1, 步长0.1)。只返回 JSON: {"reasoning": "...", "score": 0.X}"""

    def __init__(self, llm_model: str = None):
        self.llm_model = llm_model or config.reranker.llm_model

    def _score_one(self, query: str, text: str) -> float:
        from src.config import get_llm_client
        client = get_llm_client()
        try:
            resp = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.RERANK_PROMPT},
                    {"role": "user", "content": f"查询: {query}\n文本: {text[:1500]}"},
                ],
                temperature=0, max_tokens=150,
            )
            import json
            content = resp.choices[0].message.content
            s = content.find('{')
            e = content.rfind('}') + 1
            if s >= 0 and e > s:
                parsed = json.loads(content[s:e])
                return float(parsed.get("score", 0.5))
            else:
                logger.warning(
                    "[LLM RERANK WARN] JSON not found in response: %s...",
                    content[:100],
                )
        except Exception as e:
            import traceback
            logger.error(
                "[CRITICAL RERANK ERROR] LLMReranker._score_one 失败"
            )
            traceback.print_exc()
            raise RuntimeError(f"LLM Rerank 重排失败: {e}") from e
        return 0.5

    def rerank(self, query: str, candidates: List[Dict], top_k: int = None):
        if not candidates:
            return []
        top_k = top_k or config.retrieval.top_k_rerank
        for c in candidates:
            llm_s = self._score_one(query, c["text"])
            vec_s = c.get("score", 0.5)
            c["rerank_score"] = round(0.7 * llm_s + 0.3 * vec_s, 4)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


def get_reranker(engine: str = None):
    engine = engine or config.reranker.engine
    if engine == "bge_reranker":
        return BGEReranker()
    elif engine == "llm_rerank":
        return LLMReranker()
    else:
        raise ValueError(f"未知 Reranker: {engine}")
