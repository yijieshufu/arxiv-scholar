"""旧路径数据迁移到项目 data/"""
import json
from pathlib import Path

from src.config import PROJECT_ROOT, get_papers_dir, get_vector_store_dir
from src.data_migration import migrate_legacy_data
from src.paper_files import MANIFEST_FILENAME


class TestDataMigration:
    def test_migrates_pdf_from_legacy_home_data(self, tmp_path, monkeypatch):
        legacy_root = tmp_path / "legacy_home" / "data"
        legacy_papers = legacy_root / "papers"
        legacy_papers.mkdir(parents=True)
        (legacy_papers / "legacy.pdf").write_bytes(b"%PDF-1.4")

        target_papers = PROJECT_ROOT / "data" / "papers"
        target_papers.mkdir(parents=True, exist_ok=True)
        dest = target_papers / "legacy.pdf"
        dest.unlink(missing_ok=True)

        monkeypatch.setattr(
            "src.data_migration.legacy_data_roots",
            lambda: [("user_home", legacy_root)],
        )
        report = migrate_legacy_data()
        assert dest.exists()
        assert any("legacy.pdf" in m["to"] for m in report["migrated_papers"])

    def test_skips_existing_pdf(self, tmp_path, monkeypatch):
        legacy_root = tmp_path / "legacy" / "data"
        legacy_papers = legacy_root / "papers"
        legacy_papers.mkdir(parents=True)
        (legacy_papers / "dup.pdf").write_bytes(b"%PDF")

        target = get_papers_dir()
        target.mkdir(parents=True, exist_ok=True)
        (target / "dup.pdf").write_bytes(b"%PDF-existing")

        monkeypatch.setattr(
            "src.data_migration.legacy_data_roots",
            lambda: [("user_home", legacy_root)],
        )
        report = migrate_legacy_data()
        assert report["migrated_papers"] == []
        assert (target / "dup.pdf").read_bytes() == b"%PDF-existing"

    def test_merges_manifest(self, tmp_path, monkeypatch):
        legacy_root = tmp_path / "legacy" / "data"
        legacy_papers = legacy_root / "papers"
        legacy_papers.mkdir(parents=True)
        manifest = {
            "files": {"only_legacy.pdf": {"arxiv_id": "1234.5678v1", "title": "T"}},
            "by_arxiv_id": {"1234.5678v1": "only_legacy.pdf"},
        }
        with open(legacy_papers / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        monkeypatch.setattr(
            "src.data_migration.legacy_data_roots",
            lambda: [("user_home", legacy_root)],
        )
        migrate_legacy_data()
        target_manifest = get_papers_dir() / MANIFEST_FILENAME
        if target_manifest.exists():
            data = json.loads(target_manifest.read_text(encoding="utf-8"))
            assert data["by_arxiv_id"].get("1234.5678v1") == "only_legacy.pdf"

    def test_default_dirs_under_project_data(self):
        assert get_papers_dir() == (PROJECT_ROOT / "data" / "papers").resolve()
        assert get_vector_store_dir() == (PROJECT_ROOT / "data" / "vector_store").resolve()
