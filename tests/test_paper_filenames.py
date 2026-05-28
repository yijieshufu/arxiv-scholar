"""PDF 文件名消毒与 manifest"""
from pathlib import Path

import pytest

from src.arxiv_client import ArxivClient, PaperMeta
from src.paper_files import (
    load_manifest,
    metadata_for_pdf_paths,
    register_paper_file,
    sanitize_pdf_stem,
    unique_pdf_path,
)


class TestSanitizePdfStem:
    def test_title_to_safe_stem(self):
        title = "DIAGNOSING COLORECTAL POLYPS IN THE WILD WITH CAPSULE NETWORKS"
        stem = sanitize_pdf_stem(title)
        assert "/" not in stem
        assert stem == "DIAGNOSING_COLORECTAL_POLYPS_IN_THE_WILD_WITH_CAPSULE_NETWORKS"

    def test_truncates_long_title(self):
        stem = sanitize_pdf_stem("A" * 200, max_length=40)
        assert len(stem) <= 40

    def test_empty_title_fallback(self):
        assert sanitize_pdf_stem("") == "paper"


class TestUniquePdfPath:
    def test_appends_suffix_on_collision(self, tmp_path):
        (tmp_path / "Paper.pdf").write_bytes(b"x")
        path = unique_pdf_path(tmp_path, "Paper.pdf")
        assert path.name == "Paper_2.pdf"


class TestManifestAndDownloadNaming:
    def test_download_uses_title_filename(self, tmp_path, monkeypatch):
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        client = ArxivClient(download_dir=str(papers_dir))

        paper = PaperMeta(
            arxiv_id="2001.03305v1",
            title="Diagnosing Colorectal Polyps in the Wild with Capsule Networks",
            authors=["A Author"],
            abstract="We study polyps.",
            published="2020-01-10T00:00:00",
            updated="2020-01-10T00:00:00",
            categories=["cs.CV"],
            pdf_url="https://example.com/paper.pdf",
        )

        def fake_urlretrieve(url, dest):
            Path(dest).write_bytes(b"%PDF-1.4")

        monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

        path = client.download_pdf(paper)
        assert path is not None
        assert path.name.startswith("Diagnosing_Colorectal")
        assert path.name.endswith(".pdf")
        assert not path.name.startswith("2001.03305")

        manifest = load_manifest(papers_dir)
        assert manifest["by_arxiv_id"]["2001.03305v1"] == path.name

    def test_resolve_legacy_arxiv_id_file(self, tmp_path):
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        legacy = papers_dir / "2001.03305v1.pdf"
        legacy.write_bytes(b"%PDF")

        client = ArxivClient(download_dir=str(papers_dir))
        paper = PaperMeta(
            arxiv_id="2001.03305v1",
            title="Some Title",
            authors=[],
            abstract="",
            published="2020-01-10T00:00:00",
            updated="2020-01-10T00:00:00",
            categories=[],
            pdf_url="https://example.com/paper.pdf",
        )
        assert client.resolve_local_pdf(paper) == legacy


class TestMetadataForPdfs:
    def test_reads_manifest_entry(self, tmp_path):
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        pdf = papers_dir / "My_Paper.pdf"
        pdf.write_bytes(b"%PDF")
        register_paper_file(
            papers_dir,
            pdf.name,
            {
                "arxiv_id": "1234.5678",
                "title": "My Paper",
                "authors": ["X"],
                "year": 2021,
                "abstract": "abs",
            },
        )
        metas = metadata_for_pdf_paths(papers_dir, [pdf])
        assert metas[0]["arxiv_id"] == "1234.5678"
        assert metas[0]["title"] == "My Paper"
