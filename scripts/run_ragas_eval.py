"""
ArXiv Scholar — RAGAS 批量生成质量评估

用法:
    python scripts/run_ragas_eval.py              # 跑全部 benchmark
    python scripts/run_ragas_eval.py --quick      # 仅跑前 3 条
    python scripts/run_ragas_eval.py --retrieval-only  # 仅跑检索指标

覆盖:
    1. 检索质量: Recall@K, MRR@K, NDCG@K（需 ground_truth 标注）
    2. 生成质量: Faithfulness, Answer Relevancy, Context Precision（RAGAS LLM-as-Judge）
    """
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,  # 抑制 INFO 噪音
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
# Benchmark 测试集 (query, ground_truth_label, expected_answer_keywords)
# ═════════════════════════════════════════════════════════════
#
# ground_truth_label:
#   - "source" 标注期望命中的论文 PDF 文件名
#   - "section" 标注期望命中的章节关键词
#   - "keywords" 标注回答中应包含的关键信息
#
BENCHMARK_QUERIES = [
    {
        "id": "B001",
        "query": "D-Caps 模型在 WL 白光模式下分类息肉的准确率是多少？",
        "source": "Diagnosing_Colorectal_Polyps_in_the_Wild_with_Capsule_Networks.pdf",
        "ground_truth": "D-Caps 在 WL 模式下准确率为 60.36%",
        "keywords": ["D-Caps", "60.36%", "WL", "白光"],
    },
    {
        "id": "B002",
        "query": "U-Net with Dilation Convolution 论文中使用了哪些数据集？",
        "source": "Colorectal_Polyp_Segmentation_by_U-Net_with_Dilation_Convolution.pdf",
        "ground_truth": "使用了 Kvasir-SEG 和 CVC-ClinicDB 数据集",
        "keywords": ["Kvasir", "CVC-ClinicDB", "数据集"],
    },
    {
        "id": "B003",
        "query": "HGNet 论文中 Table 1 的对比实验结果是什么？",
        "source": "HGNet_High-Order_Spatial_Awareness_Hypergraph_and_Multi-Scale_Context_Attention_Network_for_Colorectal_Polyp_Detection.pdf",
        "ground_truth": "Table 1 展示了不同方法在 Kvasir-SEG 和 CVC-ClinicDB 上的 mDice 和 mIoU 对比",
        "keywords": ["mDice", "mIoU", "Table 1", "Kvasir"],
    },
    {
        "id": "B004",
        "query": "结直肠息肉分割的综述论文中提到了哪些主流方法？",
        "source": "Colorectal_Polyp_Segmentation_in_the_Deep_Learning_Era_A_Comprehensive_Survey.pdf",
        "ground_truth": "综述中提到了 U-Net 及其变体、FCN、Transformer-based 方法等",
        "keywords": ["U-Net", "FCN", "深度学习", "分割"],
    },
    {
        "id": "B005",
        "query": "Polyp-SAM 2 模型的创新点是什么？",
        "source": "Polyp_SAM_2_Advancing_Zero_shot_Polyp_Segmentation_in_Colorectal_Cancer_Detection.pdf",
        "ground_truth": "Polyp-SAM 2 基于 SAM 2 的 zero-shot 分割能力进行息肉检测",
        "keywords": ["SAM", "zero-shot", "分割"],
    },
    {
        "id": "B006",
        "query": "Dysplasia grading 论文中 CNN 模型在 WSI 上的分级准确率如何？",
        "source": "Dysplasia_grading_of_colorectal_polyps_through_CNN_analysis_of_WSI.pdf",
        "ground_truth": "使用 CNN 对全切片图像（WSI）进行异型增生分级",
        "keywords": ["CNN", "WSI", "分级", "dysplasia"],
    },
    {
        "id": "B007",
        "query": "EndoSight AI 模型的实时检测性能如何？",
        "source": "EndoSight_AI_Deep_Learning-Driven_Real-Time_Gastrointestinal_Polyp_Detection_and_Segmentation_for_Enhanced_Endoscopic_Di.pdf",
        "ground_truth": "EndoSight AI 实现了实时息肉检测和分割",
        "keywords": ["实时", "检测", "分割", "内窥镜"],
    },
    {
        "id": "B008",
        "query": "A Lightweight and Robust Framework for Real-Time Colorectal Polyp Detection 论文中使用了什么 YOLO 版本？",
        "source": "A_Lightweight_and_Robust_Framework_for_Real-Time_Colorectal_Polyp_Detection_Using_LOF-Based_Preprocessing_and_YOLO-v11n.pdf",
        "ground_truth": "使用了 YOLOv11n 并结合 LOF 预处理",
        "keywords": ["YOLOv11n", "YOLO", "LOF", "预处理"],
    },
    {
        "id": "B009",
        "query": "Compare the accuracy of different methods for polyp detection",
        "source": None,  # 跨论文对比
        "ground_truth": "The answer should compare detection methods across multiple papers",
        "keywords": ["accuracy", "comparison", "detection"],
    },
    {
        "id": "B010",
        "query": "结直肠息肉检测的深度学习方法中，哪些使用了注意力机制？",
        "source": None,
        "ground_truth": "应提及 HGNet 等使用了注意力机制的模型",
        "keywords": ["注意力", "attention", "HGNet"],
    },
]


def benchmark_retrieval(pipeline, queries: List[Dict]) -> Dict:
    """评估检索质量（Recall@K, MRR, NDCG）。"""
    from src.evaluation import calculate_metrics

    all_metrics = []
    per_query = []

    for item in queries:
        qid = item["id"]
        query = item["query"]
        expected_source = item.get("source")

        t0 = time.time()
        results = pipeline.query(query, top_k=10, use_rerank=True, rewrite=True)
        latency_ms = (time.time() - t0) * 1000

        retrieved_sources = [r.get("source", "") for r in results]
        relevant_ids = [expected_source] if expected_source else []

        metrics = calculate_metrics(
            [{"source": s} for s in retrieved_sources],
            relevant_ids,
        )
        metrics["latency_ms"] = round(latency_ms, 1)
        metrics["num_results"] = len(results)

        # 额外：关键词命中检查
        retrieved_text = " ".join(r.get("text", "")[:500] for r in results[:5])
        kw_hits = sum(
            1 for kw in item.get("keywords", [])
            if kw.lower() in retrieved_text.lower()
        )
        metrics["keyword_hits"] = kw_hits
        metrics["keyword_total"] = len(item.get("keywords", []))

        all_metrics.append(metrics)
        per_query.append({
            "id": qid,
            "query": query,
            "num_results": len(results),
            "latency_ms": latency_ms,
            "top_sources": retrieved_sources[:5],
            "keyword_hits": f"{kw_hits}/{len(item.get('keywords', []))}",
            **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
        })

    # 聚合
    import numpy as np
    summary = {
        "total": len(all_metrics),
        "avg_latency_ms": round(np.mean([m["latency_ms"] for m in all_metrics]), 1),
        "avg_num_results": round(np.mean([m["num_results"] for m in all_metrics]), 1),
    }
    for k in ["recall@5", "recall@10", "mrr@10", "ndcg@10"]:
        vals = [m[k] for m in all_metrics if k in m]
        if vals:
            summary[f"avg_{k}"] = round(np.mean(vals), 4)

    return {"summary": summary, "per_query": per_query}


def benchmark_generation(pipeline, queries: List[Dict], max_samples: int = None) -> Dict:
    """评估生成质量（RAGAS Faithfulness / Answer Relevancy / Context Precision）。"""
    from src.evaluation.ragas_eval import RAGASEvaluator

    evaluator = RAGASEvaluator()
    results = []

    items = queries[:max_samples] if max_samples else queries

    print(f"\n{'─'*60}")
    print(f"RAGAS 生成质量评估 ({len(items)} queries)")
    print(f"{'─'*60}")

    for i, item in enumerate(items):
        qid = item["id"]
        query = item["query"]
        ground_truth = item.get("ground_truth", "")

        print(f"  [{i+1}/{len(items)}] {qid}: {query[:60]}...", end=" ", flush=True)

        try:
            # Step 1: 检索
            results = pipeline.query(query, top_k=10, use_rerank=True, rewrite=True)
            if not results:
                print("(no results)")
                continue

            # Step 2: LLM 生成答案
            from src.config import get_llm_client, config
            from src.prompts import QA_SYSTEM_PROMPT

            llm = get_llm_client()
            ctx_parts = []
            for r in results[:5]:
                meta = r.get("metadata", {})
                src = meta.get("source", "?")
                sec = meta.get("section_title", "")
                ctx_parts.append(f"[{src}] {sec}\n{r.get('text', '')[:800]}")
            ctx = "\n\n---\n\n".join(ctx_parts)

            resp = llm.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": f"问题：{query}\n\n论文内容：\n{ctx}"},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            answer = resp.choices[0].message.content

            # Step 3: RAGAS 评估
            ragas_result = evaluator.evaluate_rag_response(
                question=query,
                answer=answer,
                retrieved_chunks=results,
                ground_truth=ground_truth,
            )
            results.append({
                "id": qid,
                "query": query,
                "answer": answer[:500],
                "faithfulness": ragas_result.faithfulness,
                "answer_relevancy": ragas_result.answer_relevancy,
                "context_precision": ragas_result.context_precision,
                "context_recall": ragas_result.context_recall,
                "error": ragas_result.error,
            })
            print(f"F={ragas_result.faithfulness:.2f} R={ragas_result.answer_relevancy:.2f} P={ragas_result.context_precision:.2f}")
        except Exception as e:
            print(f"FAIL: {e}")
            results.append({
                "id": qid,
                "query": query,
                "error": str(e),
            })

    import numpy as np
    valid = [r for r in results if "faithfulness" in r]
    summary = {
        "total": len(results),
        "valid": len(valid),
        "avg_faithfulness": round(np.mean([r["faithfulness"] for r in valid]), 4) if valid else 0,
        "avg_answer_relevancy": round(np.mean([r["answer_relevancy"] for r in valid]), 4) if valid else 0,
        "avg_context_precision": round(np.mean([r["context_precision"] for r in valid]), 4) if valid else 0,
        "avg_context_recall": round(np.mean([r["context_recall"] for r in valid]), 4) if valid else 0,
    }

    return {"summary": summary, "per_query": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ArXiv Scholar RAGAS Benchmark")
    parser.add_argument("--quick", action="store_true", help="仅跑前 3 条")
    parser.add_argument("--retrieval-only", action="store_true", help="仅跑检索指标")
    parser.add_argument("--generation-only", action="store_true", help="仅跑生成质量")
    parser.add_argument("--output", type=str, default="data/test_bench/ragas_results.json",
                        help="输出 JSON 路径")
    args = parser.parse_args()

    print("=" * 60)
    print("ArXiv Scholar RAGAS Benchmark")
    print("=" * 60)

    # 加载索引
    from src.retriever.pipeline import RetrievalPipeline
    pipeline = RetrievalPipeline()
    ok, msg = pipeline.ensure_index()
    if not ok:
        print(f"❌ 索引加载失败: {msg}")
        print("请先在 data/papers/ 下放置 PDF 文件，或运行 python -m src.cli search '...' --download")
        sys.exit(1)
    print(f"✅ 索引加载完成: {pipeline.vector_store.count} 向量, {len(pipeline._all_chunks)} chunks")

    queries = BENCHMARK_QUERIES[:3] if args.quick else BENCHMARK_QUERIES

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(queries),
        "index_info": {
            "vector_count": pipeline.vector_store.count,
            "chunk_count": len(pipeline._all_chunks),
        },
    }

    # ── 检索指标 ──
    if not args.generation_only:
        print(f"\n{'═'*60}")
        print("Phase 1: 检索质量评估")
        print(f"{'═'*60}")
        retrieval_report = benchmark_retrieval(pipeline, queries)
        report["retrieval"] = retrieval_report

        print(f"\n  📊 检索指标汇总:")
        for k, v in retrieval_report["summary"].items():
            print(f"     {k}: {v}")

    # ── 生成质量 ──
    if not args.retrieval_only:
        print(f"\n{'═'*60}")
        print("Phase 2: 生成质量评估 (RAGAS LLM-as-Judge)")
        print(f"{'═'*60}")
        generation_report = benchmark_generation(pipeline, queries, max_samples=3 if args.quick else None)
        report["generation"] = generation_report

        print(f"\n  📊 生成质量汇总:")
        for k, v in generation_report["summary"].items():
            print(f"     {k}: {v}")

    # ── 保存报告 ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ 报告已保存: {output_path}")

    print(f"\n{'═'*60}")
    print("Benchmark 完成")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
