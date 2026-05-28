"""项目路径解析 — 不依赖进程 CWD"""
import os
from pathlib import Path

from src.arxiv_client import ArxivClient
from src.config import PROJECT_ROOT, get_data_dir, get_papers_dir, get_vector_store_dir, resolve_project_path
from src.retriever.vector_store import VectorStore


class TestResolveProjectPath:
    def test_relative_path_uses_project_root(self):
        resolved = resolve_project_path("./data/papers")
        assert resolved == (PROJECT_ROOT / "data" / "papers").resolve()

    def test_absolute_path_unchanged(self):
        abs_path = PROJECT_ROOT / "data" / "papers"
        assert resolve_project_path(abs_path) == abs_path.resolve()

    def test_get_papers_dir_matches_default(self):
        assert get_papers_dir() == (PROJECT_ROOT / "data" / "papers").resolve()
        assert get_data_dir() == (PROJECT_ROOT / "data").resolve()
        assert get_vector_store_dir() == (PROJECT_ROOT / "data" / "vector_store").resolve()


class TestLocalPapersDiscovery:
    def test_finds_pdfs_regardless_of_cwd(self, monkeypatch, tmp_path):
        papers_dir = PROJECT_ROOT / "data" / "papers"
        if not papers_dir.exists():
            papers_dir.mkdir(parents=True)
        sample = papers_dir / "_path_test_sample.pdf"
        sample.write_bytes(b"%PDF-1.4 test")
        try:
            monkeypatch.chdir(tmp_path)
            client = ArxivClient()
            assert client.download_dir == papers_dir.resolve()
            assert sample.name in [p.name for p in client.get_local_papers()]
        finally:
            sample.unlink(missing_ok=True)

    def test_lists_all_pdfs_in_directory(self, tmp_path):
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        names = ["2001.03305v1.pdf", "2101.09991v2.pdf", "custom_title.pdf"]
        for name in names:
            (papers_dir / name).write_bytes(b"%PDF")
        client = ArxivClient(download_dir=str(papers_dir))
        assert [p.name for p in client.get_local_papers()] == sorted(names)

    def test_refresh_download_dir_picks_up_new_files(self, tmp_path, monkeypatch):
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        (papers_dir / "a.pdf").write_bytes(b"%PDF")
        client = ArxivClient(download_dir=str(papers_dir))
        assert len(client.get_local_papers()) == 1

        (papers_dir / "b.pdf").write_bytes(b"%PDF")
        client.download_dir = tmp_path / "wrong"
        assert len(client.get_local_papers()) == 2

    def test_env_override_download_dir(self, monkeypatch, tmp_path):
        custom = tmp_path / "custom_papers"
        custom.mkdir()
        (custom / "env.pdf").write_bytes(b"%PDF")
        monkeypatch.setenv("ARXIV_DOWNLOAD_DIR", str(custom))
        from importlib import reload
        import src.config as config_mod
        reload(config_mod)
        try:
            client = ArxivClient()
            assert client.download_dir.resolve() == custom.resolve()
            assert [p.name for p in client.get_local_papers()] == ["env.pdf"]
        finally:
            monkeypatch.delenv("ARXIV_DOWNLOAD_DIR", raising=False)
            reload(config_mod)
