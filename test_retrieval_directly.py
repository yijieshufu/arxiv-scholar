"""
Standalone retrieval test — bypasses Streamlit entirely.
Cold-starts RetrievalPipeline, queries, prints chunk structure.
"""
import os, logging, sys

os.environ["HF_HUB_OFFLINE"] = "1"
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

from src.retriever.pipeline import RetrievalPipeline

pipeline = RetrievalPipeline()
if not pipeline.load_index():
    print("❌ Index load failed", flush=True)
    sys.exit(1)

print(f"✅ Index loaded: {pipeline.vector_store.count} vectors", flush=True)

QUERIES = [
    "Colorectal Polyp Segmentation by U-Net with Dilation Convolution 使用了什么数据集？",
    "What datasets are used in the U-Net polyp segmentation paper?",
]

for q in QUERIES:
    print(f"\n{'='*70}", flush=True)
    print(f"QUERY: {q}", flush=True)
    print(f"{'='*70}", flush=True)

    results = pipeline.query(q, top_k=5, rewrite=True)

    for i, r in enumerate(results):
        meta = r.get("metadata", {})
        text = r.get("text", "")
        print(f"\n--- Chunk #[{i}] ---", flush=True)
        print(f"  score={r.get('score', 0):.4f}", flush=True)
        print(f"  source={meta.get('source', '?')}", flush=True)
        print(f"  section_id={meta.get('section_id', '?')}", flush=True)
        print(f"  section_title={meta.get('section_title', '?')[:60]}", flush=True)
        # Print first 300 chars of content
        content_preview = text[:300].replace("\n", "↵ ")
        print(f"  content[:300]={content_preview}", flush=True)

print("\n✅ Test complete", flush=True)
