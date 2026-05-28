"""
将误放在项目外（如用户主目录、错误 CWD）的数据迁移到 {PROJECT_ROOT}/data/。
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import ensure_data_layout, get_data_dir, get_papers_dir, get_vector_store_dir
from src.paper_files import MANIFEST_FILENAME, load_manifest, save_manifest

logger = logging.getLogger(__name__)

PAPER_GLOBS = ("*.pdf",)
VECTOR_FILES = ("faiss_*.index", "metadata_*.pkl", "bm25.pkl")


def _same_dir(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def legacy_data_roots() -> List[Tuple[str, Path]]:
    """可能因旧版 CWD 逻辑产生的数据根目录（不含项目 data/）。"""
    project_data = get_data_dir()
    seen = {project_data.resolve()}
    roots: List[Tuple[str, Path]] = []

    for label, root in [
        ("user_home", Path.home() / "data"),
        ("process_cwd", Path.cwd() / "data"),
    ]:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        if not root.is_dir():
            continue
        seen.add(resolved)
        roots.append((label, root))

    return roots


def _merge_manifests(target_dir: Path, source_manifest_path: Path) -> bool:
    if not source_manifest_path.exists():
        return False
    try:
        with open(source_manifest_path, encoding="utf-8") as f:
            incoming = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    target = load_manifest(target_dir)
    for key in ("files", "by_arxiv_id"):
        target.setdefault(key, {})
        incoming.setdefault(key, {})
        for k, v in incoming[key].items():
            if k not in target[key]:
                target[key][k] = v
    save_manifest(target_dir, target)
    return True


def _copy_if_missing(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    shutil.copy2(src, dst)
    return True


def migrate_legacy_data() -> Dict:
    """
    将旧路径下的 PDF 与索引复制到项目 data/（不覆盖已有文件）。

    Returns:
        报告 dict: migrated_papers, migrated_index_files, skipped, sources
    """
    papers_dir = get_papers_dir()
    vector_dir = get_vector_store_dir()
    ensure_data_layout()

    report = {
        "project_data_dir": str(get_data_dir()),
        "papers_dir": str(papers_dir),
        "vector_store_dir": str(vector_dir),
        "sources": [],
        "migrated_papers": [],
        "migrated_index_files": [],
        "skipped": [],
    }

    for label, root in legacy_data_roots():
        src_papers = root / "papers"
        src_vector = root / "vector_store"
        if not src_papers.is_dir() and not src_vector.is_dir():
            continue

        source_info = {"label": label, "root": str(root)}
        moved_any = False

        if src_papers.is_dir() and not _same_dir(src_papers, papers_dir):
            for pattern in PAPER_GLOBS:
                for pdf in src_papers.glob(pattern):
                    dest = papers_dir / pdf.name
                    if _copy_if_missing(pdf, dest):
                        report["migrated_papers"].append(
                            {"from": str(pdf), "to": str(dest)}
                        )
                        moved_any = True
                    elif dest.exists():
                        report["skipped"].append(str(pdf))

            manifest_src = src_papers / MANIFEST_FILENAME
            if manifest_src.exists():
                if _merge_manifests(papers_dir, manifest_src):
                    moved_any = True

        if src_vector.is_dir() and not _same_dir(src_vector, vector_dir):
            for pattern in VECTOR_FILES:
                for f in src_vector.glob(pattern):
                    dest = vector_dir / f.name
                    if _copy_if_missing(f, dest):
                        report["migrated_index_files"].append(
                            {"from": str(f), "to": str(dest)}
                        )
                        moved_any = True
                    elif dest.exists():
                        report["skipped"].append(str(f))

        if moved_any:
            report["sources"].append(source_info)

    if report["migrated_papers"] or report["migrated_index_files"]:
        logger.info(
            "数据迁移完成: %d PDF, %d 索引文件",
            len(report["migrated_papers"]),
            len(report["migrated_index_files"]),
        )
    return report

