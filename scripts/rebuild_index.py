"""增量重建全量索引，每篇 PDF 完成后打印进度"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['PYTHONUNBUFFERED'] = '1'
from pathlib import Path

import shutil

# 清空旧索引
vd = Path('data/vector_store')
if vd.exists():
    shutil.rmtree(str(vd))
vd.mkdir(parents=True, exist_ok=True)
print("=== 全量索引重建 ===", flush=True)

from src.retriever.pipeline import RetrievalPipeline, _parse_and_chunk_paper
from src.config import config

p = RetrievalPipeline()
all_pdfs = sorted(Path("data/papers").glob("*.pdf"))
total = len(all_pdfs)
print(f"共 {total} 篇 PDF", flush=True)

t_start = time.time()
chunk_id = 0
total_chunks = 0

for i, pdf_path in enumerate(all_pdfs, 1):
    t_pdf = time.time()
    try:
        p.build_index([str(pdf_path)], rebuild=(i == 1), max_workers=1)
        total_chunks = p.vector_store.count
        elapsed = time.time() - t_pdf
        print(f"  [{i}/{total}] {pdf_path.stem[:50]:<50} {elapsed:.0f}s (idx={p.vector_store.count})", flush=True)
    except Exception as e:
        print(f"  [{i}/{total}] {pdf_path.stem[:50]:<50} ERROR: {e}", flush=True)

elapsed = time.time() - t_start
table_ct = sum(1 for m in p.vector_store.metadata if m.get("is_table"))
print(f"\n=== 完成 ===", flush=True)
print(f"耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
print(f"总chunks: {total_chunks}", flush=True)
print(f"表格chunks: {table_ct}", flush=True)
