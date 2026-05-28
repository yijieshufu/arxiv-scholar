"""检索优化相关单元测试（无需 LLM / Embedding 模型）"""
import pickle
import tempfile
from pathlib import Path

import pytest

from src.chunker.section_chunker import Chunk
from src.query_rewriter import has_cjk
from src.retriever.bm25 import BM25Retriever


class TestChineseDetection:
    def test_has_cjk_chinese(self):
        assert has_cjk("肠息肉检测")
        assert has_cjk("LLM 对齐技术")

    def test_has_cjk_english_only(self):
        assert not has_cjk("transformer attention")
        assert not has_cjk("colorectal polyp detection")


class TestBM25Persistence:
    def _make_chunks(self):
        return [
            Chunk(
                text="Colorectal polyp detection using deep learning.",
                metadata={"chunk_id": "0", "source": "paper1.pdf", "section_title": "Abstract"},
            ),
            Chunk(
                text="Transformer attention mechanism for vision tasks.",
                metadata={"chunk_id": "1", "source": "paper2.pdf", "section_title": "Method"},
            ),
        ]

    def test_save_load_restores_chunks(self):
        bm25 = BM25Retriever()
        chunks = self._make_chunks()
        bm25.index(chunks)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "bm25.pkl")
            bm25.save(path)

            loaded = BM25Retriever()
            loaded.load(path)

            assert len(loaded._chunks) == 2
            assert loaded._chunks[0].text.startswith("Colorectal")
            assert loaded._chunks[0].metadata.get("chunk_id") == "0"

    def test_search_works_after_load(self):
        bm25 = BM25Retriever()
        bm25.index(self._make_chunks())

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "bm25.pkl")
            bm25.save(path)

            loaded = BM25Retriever()
            loaded.load(path)
            results = loaded.search("colorectal polyp", top_k=2)

            assert len(results) >= 1
            assert loaded._chunks  # 加载后 chunk 对象可用

    def test_backward_compat_old_pickle_format(self):
        chunks_text = ["First chunk about NLP.", "Second chunk about CV."]
        payload = {
            "corpus": chunks_text,
            "tokenized_corpus": [BM25Retriever.tokenize(t) for t in chunks_text],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bm25_old.pkl"
            with open(path, "wb") as f:
                pickle.dump(payload, f)

            loaded = BM25Retriever()
            loaded.load(str(path))
            assert len(loaded._chunks) == 2
            results = loaded.search("NLP", top_k=1)
            assert len(results) == 1
