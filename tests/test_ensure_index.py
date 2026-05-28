"""ensure_index — 本地 PDF 自动建索引（mock 解析/向量化）"""
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.retriever.pipeline import RetrievalPipeline


class TestEnsureIndex:
    def test_builds_when_pdfs_exist_but_no_index(self, tmp_path):
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        pdf = papers_dir / "sample.pdf"
        pdf.write_bytes(b"%PDF")

        pipeline = RetrievalPipeline()
        mock_client = MagicMock()
        mock_client.get_local_papers.return_value = [pdf]
        mock_client.metadata_for_pdfs.return_value = [
            {"arxiv_id": "sample", "title": "Sample", "authors": [], "year": "", "abstract": ""}
        ]

        with patch.object(pipeline, "load_index", return_value=False), patch.object(
            pipeline, "build_index"
        ) as mock_build:
            pipeline.vector_store = MagicMock()
            pipeline.vector_store.count = 10
            ok, msg = pipeline.ensure_index(mock_client)

        assert ok is True
        assert "自动构建" in msg
        mock_build.assert_called_once()

    def test_no_pdfs_and_no_index(self, tmp_path):
        pipeline = RetrievalPipeline()
        mock_client = MagicMock()
        mock_client.get_local_papers.return_value = []

        with patch.object(pipeline, "load_index", return_value=False):
            ok, msg = pipeline.ensure_index(mock_client)

        assert ok is False
        assert "本地尚无 PDF" in msg
