"""Index a single local PDF (for low-memory serial re-index)."""
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 必须在 import torch / sentence_transformers 之前设置
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.config import config

config.embedding.device = "cpu"
config.embedding.batch_size = 8

from src.embedding import _clear_model_cache
from src.arxiv_client import ArxivClient
from src.parser.paper_parser import PaperParser
from src.chunker.section_chunker import SectionChunker, Chunk
from src.retriever.pipeline import RetrievalPipeline


def parse_and_chunk_pdf(pdf_path: Path, meta: Dict) -> Tuple[Dict, List[Chunk]] | None:
    """使用 PaperParser 解析 PDF，再用 SectionChunker 做罗马数字感知切片。"""
    source_name = pdf_path.name
    try:
        parser = PaperParser(engine=config.parser.pdf_engine)
        doc = parser.parse(str(pdf_path))

        paper_info = {
            "paper_title": doc.get("title", meta.get("title", "")),
            "arxiv_id": meta.get("arxiv_id", pdf_path.stem),
            "source": source_name,
            "authors": meta.get("authors", []),
            "year": meta.get("year", ""),
            "abstract": doc.get("abstract", meta.get("abstract", "")),
        }

        chunker = SectionChunker(
            chunk_size=config.chunker.chunk_size,
            chunk_overlap=config.chunker.chunk_overlap,
        )
        chunker.min_chunk_size = config.chunker.min_chunk_size
        chunks = chunker.chunk(doc["full_text"], paper_info)
        return paper_info, chunks
    except Exception as e:
        logging.error("解析失败 %s: %s", source_name, e)
        return None


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/reindex_one.py <pdf_index> <rebuild:0|1>")
        sys.exit(1)
    idx = int(sys.argv[1])
    rebuild = sys.argv[2] == "1"
    client = ArxivClient()
    pdfs = client.get_local_papers()
    if idx < 0 or idx >= len(pdfs):
        print(f"Invalid index {idx}, have {len(pdfs)} PDFs")
        sys.exit(1)
    pdf = pdfs[idx]
    metas = client.metadata_for_pdfs([pdf])
    meta = metas[0] if metas else {}

    parsed = parse_and_chunk_pdf(pdf, meta)
    if parsed is None:
        sys.exit(1)

    paper_info, chunks = parsed
    logging.info(
        "PaperParser 完成: %s, engine=%s -> %s chunks",
        pdf.name,
        config.parser.pdf_engine,
        len(chunks),
    )

    _clear_model_cache()
    pipeline = RetrievalPipeline.for_indexing()
    try:
        added = pipeline.append_parsed_paper(paper_info, chunks, rebuild=rebuild)
    except Exception as e:
        logging.exception("向量化或落盘失败: %s", e)
        sys.exit(2)

    print(
        f"OK idx={idx} added={added} total_chunks={len(pipeline._all_chunks)} "
        f"vectors={pipeline.vector_store.count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
