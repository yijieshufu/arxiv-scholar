"""
评估模块 — Langfuse 追踪 + 自定义指标

指标：
- Recall@K: 检索到的相关文档比例
- MRR: 第一个相关文档的倒数排名均值
- NDCG: 归一化折损累计增益
- Latency: 端到端延迟
"""
import time
import logging
from typing import List, Dict
import numpy as np

from src.config import config

logger = logging.getLogger(__name__)


class EvaluationTracker:
    """评估追踪器"""

    def __init__(self):
        self.traces: List[Dict] = []
        self._langfuse = None
        if config.evaluation.langfuse_enabled:
            self._init_langfuse()

    def _init_langfuse(self):
        try:
            import langfuse
            self._langfuse = langfuse.Langfuse(
                public_key=config.evaluation.langfuse_public_key,
                secret_key=config.evaluation.langfuse_secret_key,
            )
            logger.info("Langfuse 追踪已启用")
        except Exception as e:
            logger.warning(f"Langfuse 初始化失败: {e}")
            self._langfuse = None

    def trace_query(self, query: str, results: List[Dict], latency_ms: float,
                    relevant_ids: List[str] = None):
        """追踪一次查询"""
        trace = {
            "query": query,
            "num_results": len(results),
            "latency_ms": latency_ms,
            "top_scores": [r.get("rerank_score", r.get("score", 0)) for r in results[:5]],
            "timestamp": time.time(),
        }

        if relevant_ids:
            metrics = calculate_metrics(results, relevant_ids)
            trace.update(metrics)

        self.traces.append(trace)

        if self._langfuse:
            try:
                self._langfuse.trace(
                    name="arxiv-query",
                    input=query,
                    output={"count": len(results)},
                    metadata=trace,
                )
            except Exception:
                pass

    def get_summary(self) -> Dict:
        """获取评估摘要"""
        if not self.traces:
            return {"total_queries": 0}

        latencies = [t["latency_ms"] for t in self.traces]

        summary = {
            "total_queries": len(self.traces),
            "avg_latency_ms": round(np.mean(latencies), 1),
            "p95_latency_ms": round(np.percentile(latencies, 95), 1),
            "avg_results": round(np.mean([t["num_results"] for t in self.traces]), 1),
        }

        # 聚合指标
        metrics_keys = ["recall@5", "recall@10", "mrr@10", "ndcg@10"]
        for key in metrics_keys:
            values = [t.get(key, 0) for t in self.traces if key in t]
            if values:
                summary[f"avg_{key}"] = round(np.mean(values), 4)

        return summary


def calculate_metrics(results: List[Dict], relevant_ids: List[str]) -> Dict:
    """
    计算检索指标。

    Args:
        results: 检索结果 [{text, source, score, ...}, ...]
        relevant_ids: 相关文档 ID 列表（人工标注的 ground truth）

    Returns:
        {"recall@K": float, "mrr@K": float, "ndcg@K": float}
    """
    if not relevant_ids or not results:
        return {}

    relevant_set = set(relevant_ids)
    metrics = {}

    for k in [5, 10]:
        top_k = results[:k]
        retrieved_ids = [r.get("source", "") for r in top_k]

        # Recall@K
        hits = sum(1 for rid in retrieved_ids if rid in relevant_set)
        metrics[f"recall@{k}"] = hits / len(relevant_set) if relevant_set else 0

        # MRR@K
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_set:
                metrics[f"mrr@{k}"] = 1.0 / (i + 1)
                break
        else:
            metrics[f"mrr@{k}"] = 0.0

        # NDCG@K
        dcg = 0
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_set:
                dcg += 1.0 / np.log2(i + 2)
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0

    return metrics


# 全局评估追踪器
tracker = EvaluationTracker()
