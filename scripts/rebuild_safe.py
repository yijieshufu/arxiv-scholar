"""
rebuild_safe.py — 逐篇串行重建索引（HTML 表块使用 Caption 轻量向量 + 完整 HTML 落盘元数据）
"""
import os, sys, shutil, logging, pickle
os.environ["HF_HUB_OFFLINE"] = "1"
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.config import config, get_vector_store_dir
from src.arxiv_client import ArxivClient
from src.parser.paper_parser import PaperParser
from src.chunker.section_chunker import SectionChunker
from src.retriever.pipeline import RetrievalPipeline

# ── Step 1: 清空旧索引 ──
vs_dir = get_vector_store_dir()
if vs_dir.exists():
    shutil.rmtree(str(vs_dir))
    print(f"Cleared: {vs_dir}")
vs_dir.mkdir(parents=True, exist_ok=True)

# ── Step 2: 逐篇解析 + 入库 ──
client = ArxivClient()
pdfs = sorted(client.get_local_papers(), key=lambda p: p.name)
config.parser.pdf_engine = "docling"
print(f"Papers: {len(pdfs)} (engine=docling)\n")

pipeline = RetrievalPipeline.for_indexing()
total_tables = 0
first_table_meta = None

for i, pdf in enumerate(pdfs):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(pdfs)}] {pdf.name}")
    print(f"{'='*60}")

    # 解析
    parser = PaperParser(engine=config.parser.pdf_engine)
    doc = parser.parse(str(pdf))
    chunker = SectionChunker()
    chunks = chunker.chunk(doc["full_text"], {
        "source": pdf.name,
        "paper_title": doc.get("title", ""),
    })
    print(f"  解析完成: {len(chunks)} chunks")

    # 统计表格块
    tbl_in_this = sum(1 for c in chunks if "[TABLE_HTML:" in c.text)
    total_tables += tbl_in_this
    if tbl_in_this:
        print(f"  表格块: {tbl_in_this}")

    # 入库
    n = pipeline.append_parsed_paper(
        {"source": pdf.name, "paper_title": doc.get("title", "")},
        chunks,
    )
    print(f"  入库: {n} 个向量")

    # 记录首个表格的元数据
    if first_table_meta is None and tbl_in_this:
        for c in chunks:
            if "[TABLE_HTML:" in c.text:
                first_table_meta = {
                    "table_id": c.metadata.get("table_id", "?"),
                    "is_table": c.metadata.get("is_table", False),
                    "caption_embedded": c.text.split("\n")[0] if "\n" in c.text else c.text[:200],
                }
                break

# ── Step 3: 校验 ──
print(f"\n{'='*60}")
print("Rebuild complete")
print(f"   总向量: {pipeline.vector_store.count}")
pipeline.vector_store.load()
tbl_in_store = sum(
    1 for m in pipeline.vector_store.metadata
    if m.get("is_table") or "[TABLE_HTML:" in str(m.get("text", ""))
)
print(f"   表格块: {tbl_in_store}")

# ── 展示首个表格块元数据 ──
if first_table_meta:
    print(f"\n{'='*60}")
    print("First table block metadata:")
    print(f"{'='*60}")
    # 从 FAISS metadata 中找出该表块的完整落盘数据
    for m in pipeline.vector_store.metadata:
        if m.get("is_table"):
            print(f"  表 ID:         {m.get('table_id', 'N/A')}")
            print(f"  来源 PDF:      {m.get('source', 'N/A')}")
            print(f"  所在章节:      {m.get('section_title', 'N/A')}")
            print(f"  标记:          is_table={m.get('is_table')}")
            print(f"  向量用文本:    {str(m.get('text',''))[:200]}")
            print(f"  完整 HTML 长度: {len(str(m.get('full_html_content','')))} chars")
            print(f"  完整 HTML 前 500 字:")
            print(f"    {str(m.get('full_html_content',''))[:500]}")
            break

print(f"\n{'='*60}")
from collections import Counter
id_pairs = []
for m in pipeline.vector_store.metadata:
    if m.get("is_table"):
        fl = str(m.get("text", "")).split("\n", 1)[0]
        id_pairs.append((m.get("table_id"), fl[:80]))
dup_ids = [k for k, v in Counter(m.get("table_id") for m in pipeline.vector_store.metadata if m.get("is_table")).items() if v > 1]
print(f"Unique table_id values: {len(set(p[0] for p in id_pairs))}")
print(f"Table blocks with duplicate table_id across corpus: {len(dup_ids)} (expected: many papers)")
# per-source table count for Polyp SAM paper if present
for name in sorted({m.get("source") for m in pipeline.vector_store.metadata if m.get("is_table")}):
    n = sum(1 for m in pipeline.vector_store.metadata if m.get("is_table") and m.get("source") == name)
    if "Polyp_SAM" in name or n >= 5:
        print(f"  {name}: {n} table chunks")
print(f"Index saved: {vs_dir}")
