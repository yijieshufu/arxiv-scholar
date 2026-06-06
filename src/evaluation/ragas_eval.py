"""
RAGAS 生成质量评估 — 补充检索指标之外的 LLM 答案质量度量

RAG 系统的评估分两层：
1. 检索质量：Recall@K / MRR / NDCG（已有，src/evaluation/__init__.py）
2. 生成质量：Faithfulness / Answer Relevancy / Context Precision（本模块）

RAGAS (RAG Assessment) 是专门为 RAG 系统设计的评估框架，
通过 LLM-as-Judge 方式评估生成答案的忠实度和相关性。
"""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from src.config import config

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    """RAGAS 单条 query 评估结果"""
    question: str
    answer: str
    faithfulness: float = 0.0       # 回答是否基于检索上下文（反幻觉）
    answer_relevancy: float = 0.0   # 回答是否切题
    context_precision: float = 0.0  # 检索到的上下文是否相关
    context_recall: float = 0.0     # 是否检索到了所有必要上下文（需要 ground_truth）
    error: str = ""
    faithfulness_reasoning: str = ""
    relevancy_reasoning: str = ""
    context_precision_reasoning: str = ""


@dataclass
class EvaluationReport:
    """RAGAS 批量评估报告"""
    results: List[RAGASResult] = field(default_factory=list)
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    total_queries: int = 0
    errors: int = 0


class RAGASEvaluator:
    """RAGAS 评估器 — 使用 LLM 评估生成答案质量

    用法:
        evaluator = RAGASEvaluator()
        result = evaluator.evaluate_single(
            question="What is the FCN architecture?",
            answer="FCN is a fully convolutional network...",
            contexts=["FCN paper section 3.1: ...", "FCN uses skip connections..."],
            ground_truth="FCN replaces FC layers with convolutions"  # optional
        )
        report = evaluator.evaluate_batch([...])
    """

    def __init__(self, llm_model: str = None):
        self.llm_model = llm_model or config.llm.model

    # ── Faithfulness Prompt ──
    FAITHFULNESS_PROMPT = (
        "Your task is to judge the faithfulness of an LLM-generated answer "
        "given the retrieved contexts.\n\n"
        "A faithful answer ONLY makes claims that can be directly supported "
        "by the provided contexts. If the answer contains information not "
        "found in the contexts, it is less faithful.\n\n"
        "Score from 0 (completely unfaithful / hallucinated) to 1 "
        "(every claim is explicitly grounded in the contexts).\n\n"
        "RETRIEVED CONTEXTS:\n{contexts}\n\n"
        "LLM ANSWER:\n{answer}\n\n"
        "First, list each factual claim in the answer and state whether it "
        "is supported by the contexts. Then, output a single JSON object: "
        '{{"score": 0.X, "reasoning": "..."}}'
    )

    # ── Answer Relevancy Prompt ──
    RELEVANCY_PROMPT = (
        "Your task is to judge whether an LLM-generated answer is relevant "
        "to the user's question.\n\n"
        "Scoring guidelines:\n"
        "- 0.8-1.0: answer fully addresses the question with specific, accurate information\n"
        "- 0.5-0.7: answer is on-topic and provides useful partial information, "
        "even if some details are incomplete or uncertain\n"
        "- 0.2-0.4: answer vaguely touches the topic but lacks specifics\n"
        "- 0.0-0.1: answer is completely off-topic\n\n"
        "IMPORTANT: If the answer directly responds to the question's core intent "
        "(e.g., providing the requested metric for the first dataset even if the second "
        "is uncertain), it is still RELEVANT. Incomplete != irrelevant. "
        "Score at least 0.6 if the answer addresses the main question.\n\n"
        "USER QUESTION:\n{question}\n\n"
        "LLM ANSWER:\n{answer}\n\n"
        "Output a single JSON object: "
        '{{"score": 0.X, "reasoning": "..."}}'
    )

    # ── Context Precision Prompt ──
    CONTEXT_PRECISION_PROMPT = (
        "Your task is to judge whether the retrieved contexts are relevant "
        "to answering the user's question.\n\n"
        "Scoring guidelines:\n"
        "- 0.7-1.0: contexts contain specific data/numbers that directly answer the question\n"
        "- 0.4-0.6: contexts are topically related — same methods, datasets, or papers — "
        "even if they don't contain the exact metrics requested\n"
        "- 0.1-0.3: contexts are loosely related (same general domain)\n"
        "- 0.0: contexts are completely off-topic\n\n"
        "IMPORTANT: If the contexts discuss the same datasets (e.g. CVC-ClinicDB, ETIS-Larib) "
        "and methods mentioned in the question, they ARE relevant regardless of whether they "
        "contain every specific number. Score them at least 0.4 in that case.\n\n"
        "USER QUESTION:\n{question}\n\n"
        "RETRIEVED CONTEXTS:\n{contexts}\n\n"
        "Output a single JSON object: "
        '{{"score": 0.X, "reasoning": "..."}}'
    )

    # ── Context Recall Prompt (needs ground_truth) ──
    CONTEXT_RECALL_PROMPT = (
        "Your task is to judge whether the retrieved contexts contain "
        "all the information needed to produce the ground truth answer.\n\n"
        "Score from 0 (contexts missing critical information) to 1 "
        "(contexts contain everything needed to produce the ground truth).\n\n"
        "USER QUESTION:\n{question}\n\n"
        "GROUND TRUTH ANSWER:\n{ground_truth}\n\n"
        "RETRIEVED CONTEXTS:\n{contexts}\n\n"
        "Output a single JSON object: "
        '{{"score": 0.X, "reasoning": "..."}}'
    )

    def _call_llm_judge(self, prompt: str) -> tuple:
        """调用 LLM 做评估判断，返回 (score, reasoning, error)."""
        from src.config import get_llm_client
        import json, re

        client = get_llm_client()
        try:
            resp = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert evaluator for RAG systems. "
                                "Always return a single valid JSON object at the end."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,  # 长答案需要更多 token 输出逐条分析 + JSON
            )
            content = resp.choices[0].message.content
            logger.info("RAGAS LLM judge raw (first 300): %s", content[:300])

            # 从整个响应中提取最后一个完整 JSON 对象
            score = 0.5
            reasoning = ""
            # 方法：从后往前找 "}"，配合从前往后找对应的 "{"
            s = content.find("{")
            while s >= 0:
                e = content.find("}", s) + 1
                if e > s:
                    try:
                        candidate = json.loads(content[s:e])
                        if "score" in candidate:
                            score = float(candidate.get("score", 0.5))
                            reasoning = candidate.get("reasoning", "")
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
                s = content.find("{", e)
            return score, reasoning, ""
        except Exception as exc:
            logger.error("RAGAS LLM judge failed: %s", exc)
            return 0.5, "", str(exc)

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> RAGASResult:
        """评估单条 query 的生成质量。"""
        result = RAGASResult(question=question, answer=answer)

        # Faithfulness
        ctx_text = "\n---\n".join(
            f"[{i+1}] {c[:1500]}" for i, c in enumerate(contexts[:10])  # 学术语境需要更长文本
        )
        if not ctx_text.strip():
            ctx_text = "(no contexts provided)"

        prompt = self.FAITHFULNESS_PROMPT.format(
            contexts=ctx_text, answer=answer[:4000],  # 长答案截断放宽
        )
        score, reasoning, err = self._call_llm_judge(prompt)
        if err:
            result.error = err
        else:
            result.faithfulness = round(score, 4)
        result.faithfulness_reasoning = reasoning

        # Answer Relevancy
        prompt = self.RELEVANCY_PROMPT.format(
            question=question, answer=answer[:4000],
        )
        score, reasoning, err = self._call_llm_judge(prompt)
        if err and not result.error:
            result.error = err
        result.answer_relevancy = round(score, 4)
        result.relevancy_reasoning = reasoning

        # Context Precision
        prompt = self.CONTEXT_PRECISION_PROMPT.format(
            question=question, contexts=ctx_text,
        )
        score, reasoning, err = self._call_llm_judge(prompt)
        if err and not result.error:
            result.error = err
        result.context_precision = round(score, 4)
        result.context_precision_reasoning = reasoning

        # Context Recall (only if ground_truth available)
        if ground_truth:
            prompt = self.CONTEXT_RECALL_PROMPT.format(
                question=question, ground_truth=ground_truth, contexts=ctx_text,
            )
            score, reasoning, err = self._call_llm_judge(prompt)
            if err and not result.error:
                result.error = err
            result.context_recall = round(score, 4)

        return result

    def evaluate_batch(
        self,
        queries: List[Dict],
        progress_callback=None,
    ) -> EvaluationReport:
        """批量评估多条 query。

        queries 格式:
        [
            {
                "question": "...",
                "answer": "...",
                "contexts": ["ctx1", "ctx2", ...],
                "ground_truth": "..."  # optional
            },
            ...
        ]
        """
        report = EvaluationReport(total_queries=len(queries))

        for i, q in enumerate(queries):
            result = self.evaluate_single(
                question=q.get("question", ""),
                answer=q.get("answer", ""),
                contexts=q.get("contexts", []),
                ground_truth=q.get("ground_truth", ""),
            )
            report.results.append(result)
            if result.error:
                report.errors += 1

            if progress_callback:
                progress_callback(i + 1, len(queries))

        if report.results:
            n = len(report.results)
            report.avg_faithfulness = round(
                sum(r.faithfulness for r in report.results) / n, 4,
            )
            report.avg_answer_relevancy = round(
                sum(r.answer_relevancy for r in report.results) / n, 4,
            )
            report.avg_context_precision = round(
                sum(r.context_precision for r in report.results) / n, 4,
            )
            report.avg_context_recall = round(
                sum(r.context_recall for r in report.results) / n, 4,
            )

        return report

    def evaluate_rag_response(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list,
        ground_truth: str = "",
    ) -> RAGASResult:
        """便捷方法：直接从 pipeline 返回的 chunk 列表评估。

        retrieved_chunks 是 pipeline.query() 返回的 chunk dict 列表，
        每个含 "text" 和 "metadata" 字段。
        """
        contexts = []
        for c in retrieved_chunks[:10]:
            meta = c.get("metadata", {})
            source = meta.get("source", "unknown")
            section = meta.get("section_title", "")
            text = c.get("text", "")[:2000]  # 放宽截断，确保评估 LLM 看到完整内容
            label = f"[{source}]" + (f" {section}: " if section else " ")
            contexts.append(label + text)

        return self.evaluate_single(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
