"""
ArXiv Scholar — 综合性能基准测试

测试项：
1. PDF 解析时间（小/大 PDF）
2. 索引构建时间（增量）
3. 检索延迟（向量/混合/Rerank）
4. 端到端查询延迟
5. 表格提取验证
"""
import sys, time, json, os
sys.path.insert(0, '.')
from pathlib import Path

RESULTS = {}

def log(name, value, unit="ms"):
    RESULTS[name] = {"value": round(value, 2), "unit": unit}
    print(f"  {name}: {value:.2f} {unit}")

# ═══════════════════════════════════════════════
# 0. 准备
# ═══════════════════════════════════════════════
print("=" * 60)
print("ArXiv Scholar 性能基准测试")
print("=" * 60)

# 确保 test_data 目录
test_dir = Path("data/test_bench")
test_dir.mkdir(parents=True, exist_ok=True)

# 小 PDF
small_pdf = "data/papers/Diagnosing_Colorectal_Polyps_in_the_Wild_with_Capsule_Networks.pdf"
# 建索引用的 PDF 列表
bench_pdfs = [
    small_pdf,
    "data/papers/Monodense_Deep_Neural_Model_for_Determining_Item_Price_Elasticity.pdf",
    "data/papers/Polyp_SAM_2_Advancing_Zero_shot_Polyp_Segmentation_in_Colorectal_Cancer_Detection.pdf",
    "data/papers/HGNet_High-Order_Spatial_Awareness_Hypergraph_and_Multi-Scale_Context_Attention_Network_for_Colorectal_Polyp_Detection.pdf",
]

# ═══════════════════════════════════════════════
# 1. PDF 解析时间
# ═══════════════════════════════════════════════
print(f"\n{'─'*60}")
print("1. PDF 解析性能")
print(f"{'─'*60}")

from src.parser.paper_parser import PaperParser

# 1a. 小 PDF (576K, ~5 pages)
t0 = time.time()
doc_small = PaperParser(engine='docling').parse(small_pdf)
t_small = time.time() - t0
log("parse_small_pdf", t_small * 1000)
tables = doc_small.get('tables', [])
log("parse_small_tables", len(tables), "count")

# 1b. 验证表格 colspan/rowspan
colspan_ok = sum(1 for t in doc_small.get('tables', []) if 'colspan' in t)
rowspan_ok = sum(1 for t in doc_small.get('tables', []) if 'rowspan' in t)
log("tables_with_colspan", colspan_ok, "count")
log("tables_with_rowspan", rowspan_ok, "count")

# 1c. PDF 页数获取速度
t0 = time.time()
pages = PaperParser._get_pdf_page_count(Path(small_pdf))
t_pages = time.time() - t0
log("page_count_speed", t_pages * 1000)

# ═══════════════════════════════════════════════
# 2. 索引构建
# ═══════════════════════════════════════════════
print(f"\n{'─'*60}")
print("2. 索引构建性能")
print(f"{'─'*60}")

# 重建索引前清空
import shutil
vd = Path("data/vector_store")
if vd.exists():
    shutil.rmtree(str(vd))
vd.mkdir(parents=True, exist_ok=True)

from src.retriever.pipeline import RetrievalPipeline

p = RetrievalPipeline()

t0 = time.time()
p.build_index(bench_pdfs, max_workers=1)
t_build = time.time() - t0
log("index_build_total", t_build * 1000)
log("index_chunks", p.vector_store.count, "count")
log("index_table_chunks", sum(1 for m in p.vector_store.metadata if m.get("is_table")), "count")

# 索引加载速度
t0 = time.time()
p2 = RetrievalPipeline()
p2.load_index()
t_load = time.time() - t0
log("index_load", t_load * 1000)

# ═══════════════════════════════════════════════
# 3. 检索延迟
# ═══════════════════════════════════════════════
print(f"\n{'─'*60}")
print("3. 检索延迟")
print(f"{'─'*60}")

test_queries = [
    "比较 D-Caps 和 Iv3 在 WL 模式下的灵敏度",
    "Diagnosing Colorectal Polyps 论文中 D-Caps 在 NBI 模式准确率",
    "HGNet 表格 Table 1 的实验结果",
    "polyp segmentation methods comparison accuracy",
    "what is the best specificity for hyperplastic vs adenoma",
]

# 预热（首次调用含模型加载）
print("  [预热] 首次查询...")
_ = p.query("warmup", top_k=3, use_rerank=False, rewrite=False)

# 3a. 无 Rerank 延迟
latencies = []
for q in test_queries:
    t0 = time.time()
    r = p.query(q, top_k=5, use_rerank=False, rewrite=True)
    latencies.append((time.time() - t0) * 1000)
avg_no_rerank = sum(latencies) / len(latencies)
min_no_rerank = min(latencies)
max_no_rerank = max(latencies)
log("search_no_rerank_avg", avg_no_rerank)
log("search_no_rerank_min", min_no_rerank)
log("search_no_rerank_max", max_no_rerank)

# 3b. 带 Rerank 延迟
print("  [Rerank 测试]")
latencies_rr = []
for q in test_queries:
    t0 = time.time()
    r = p.query(q, top_k=5, use_rerank=True, rewrite=True)
    latencies_rr.append((time.time() - t0) * 1000)
avg_rerank = sum(latencies_rr) / len(latencies_rr)
min_rerank = min(latencies_rr)
max_rerank = max(latencies_rr)
log("search_rerank_avg", avg_rerank)
log("search_rerank_min", min_rerank)
log("search_rerank_max", max_rerank)

# 3c. 纯向量 vs 混合对比
print("  [检索模式对比]")
for mode, alpha_val in [("vector_only", 0.0), ("bm25_only", 1.0), ("hybrid_default", 0.6)]:
    lt = []
    for q in test_queries[:3]:
        t0 = time.time()
        r = p.query(q, top_k=5, use_rerank=False, rewrite=False, alpha=alpha_val)
        lt.append((time.time() - t0) * 1000)
    log(f"search_{mode}", sum(lt) / len(lt))

# 3d. 表格精确搜索
print("  [表格精确搜索]")
t0 = time.time()
r = p.query("Table_2", top_k=5, use_rerank=False, rewrite=False)
log("search_table_exact", (time.time() - t0) * 1000)
tables_found = sum(1 for c in r if c.get("metadata", {}).get("is_table"))
log("search_table_exact_hits", tables_found, "count")

# ═══════════════════════════════════════════════
# 4. Embedding 编码速度
# ═══════════════════════════════════════════════
print(f"\n{'─'*60}")
print("4. Embedding 编码速度")
print(f"{'─'*60}")

t0 = time.time()
_ = p.embedder.encode(["test query for embedding speed benchmark"])
t_enc1 = (time.time() - t0) * 1000
log("embed_single", t_enc1)

# 批量编码
t0 = time.time()
_ = p.embedder.encode(test_queries)
t_enc5 = (time.time() - t0) * 1000
log("embed_batch_5", t_enc5)

# ═══════════════════════════════════════════════
# 5. 端到端：Agent quick_answer
# ═══════════════════════════════════════════════
print(f"\n{'─'*60}")
print("5. Agent quick_answer (不含 LLM API)")
print(f"{'─'*60}")

t0 = time.time()
from src.agent import ArxivAgent
agent = ArxivAgent()
t_agent_init = (time.time() - t0) * 1000
log("agent_init", t_agent_init)

# 只测检索阶段（不含 LLM API 调用）
t0 = time.time()
result = agent.tools["rag_query"]["function"]("D-Caps sensitivity", top_k=3)
t_rag_tool = (time.time() - t0) * 1000
log("rag_tool_call", t_rag_tool)

# ═══════════════════════════════════════════════
# 6. 报告
# ═══════════════════════════════════════════════
print(f"\n{'═'*60}")
print("性能基准测试报告")
print(f"{'═'*60}")
print(f"\n{'指标':<35} {'数值':>12}")
print(f"{'─'*48}")
for name, data in RESULTS.items():
    val = data["value"]
    unit = data["unit"]
    if unit == "ms":
        print(f"  {name:<33} {val:>8.1f} ms")
    elif unit == "count":
        print(f"  {name:<33} {val:>8.0f}")
    else:
        print(f"  {name:<33} {val:>8.2f} {unit}")

# Save to JSON
output_path = test_dir / "benchmark_results.json"
with open(output_path, "w") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: {output_path}")

print(f"\n{'═'*60}")
print("测试完成")
print(f"{'═'*60}")
