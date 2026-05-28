"""Safe PDF filenames and local paper manifest helpers."""
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

MANIFEST_FILENAME = "manifest.json"
DEFAULT_STEM_MAX_LEN = 120


def sanitize_pdf_stem(title: str, max_length: int = DEFAULT_STEM_MAX_LEN) -> str:
    """Turn a paper title into a safe filesystem stem (no extension)."""
    if not title or not str(title).strip():
        return "paper"

    text = unicodedata.normalize("NFKC", str(title).strip())
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text)
    text = re.sub(r"[\s\u3000]+", "_", text)
    text = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("._")

    if not text:
        text = "paper"
    if len(text) > max_length:
        text = text[:max_length].rstrip("._")
    return text or "paper"


def unique_pdf_path(directory: Path, filename: str) -> Path:
    """Return a non-existing path under directory, appending _2, _3, ... if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".pdf"
    n = 2
    while True:
        candidate = directory / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def looks_like_arxiv_id(stem: str) -> bool:
    return bool(re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", stem, re.IGNORECASE))


def load_manifest(papers_dir: Path) -> Dict:
    path = papers_dir / MANIFEST_FILENAME
    if not path.exists():
        return {"files": {}, "by_arxiv_id": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "files" not in data:
            data["files"] = {}
        if "by_arxiv_id" not in data:
            data["by_arxiv_id"] = {}
        return data
    except (json.JSONDecodeError, OSError):
        return {"files": {}, "by_arxiv_id": {}}


def save_manifest(papers_dir: Path, manifest: Dict) -> None:
    path = papers_dir / MANIFEST_FILENAME
    papers_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def register_paper_file(
    papers_dir: Path,
    filename: str,
    meta: Dict,
) -> None:
    manifest = load_manifest(papers_dir)
    entry = {
        "arxiv_id": meta.get("arxiv_id", ""),
        "title": meta.get("title", ""),
        "authors": meta.get("authors", []),
        "year": meta.get("year", ""),
        "abstract": meta.get("abstract", ""),
    }
    manifest["files"][filename] = entry
    if entry["arxiv_id"]:
        manifest["by_arxiv_id"][entry["arxiv_id"]] = filename
    save_manifest(papers_dir, manifest)


def metadata_for_pdf_paths(papers_dir: Path, paths: List[Path]) -> List[Dict]:
    """Build paper_metas for build_index from manifest or filename heuristics."""
    manifest = load_manifest(papers_dir)
    metas: List[Dict] = []
    for path in paths:
        name = path.name
        entry = manifest.get("files", {}).get(name)
        if entry:
            metas.append(dict(entry))
            continue
        stem = path.stem
        if looks_like_arxiv_id(stem):
            metas.append({"arxiv_id": stem, "title": "", "authors": [], "year": "", "abstract": ""})
        else:
            title_guess = stem.replace("_", " ")
            metas.append({
                "arxiv_id": stem,
                "title": title_guess,
                "authors": [],
                "year": "",
                "abstract": "",
            })
    return metas
