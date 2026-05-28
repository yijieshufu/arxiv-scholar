"""
ArXiv API 客户端 — arxiv 4.0 API 适配

封装 ArXiv 论文搜索、下载、元数据提取。
参考：https://github.com/Future-House/paper-qa（论文全文问答）
"""
import logging
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Dict, Literal, Union
from dataclasses import dataclass, field

import arxiv
from tqdm import tqdm

from src.config import config, get_papers_dir, resolve_project_path
from src.query_rewriter import QueryRewriter, has_cjk
from src.paper_files import (
    load_manifest,
    metadata_for_pdf_paths,
    register_paper_file,
    sanitize_pdf_stem,
    unique_pdf_path,
)

logger = logging.getLogger(__name__)


def _tqdm_enabled() -> bool:
    """Disable tqdm under Streamlit or when stderr cannot be flushed."""
    if "streamlit" in sys.modules:
        return False
    try:
        from streamlit.runtime import exists as streamlit_runtime_exists

        if streamlit_runtime_exists():
            return False
    except Exception:
        pass

    stream = sys.stderr
    if stream is None:
        return False
    try:
        isatty = getattr(stream, "isatty", None)
        if callable(isatty) and not isatty():
            return False
        flush = getattr(stream, "flush", None)
        if callable(flush):
            flush()
    except OSError:
        return False
    return True


DateLike = Union[date, datetime, str]


def _coerce_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _arxiv_datetime_str(value: DateLike, *, end_of_day: bool = False) -> str:
    """ArXiv API 日期格式：YYYYMMDDHHMM（GMT）。"""
    d = _coerce_date(value)
    suffix = "2359" if end_of_day else "0000"
    return f"{d.strftime('%Y%m%d')}{suffix}"


def build_date_query_clause(
    date_from: Optional[DateLike] = None,
    date_to: Optional[DateLike] = None,
    date_field: Literal["submitted", "updated"] = "submitted",
) -> str:
    """构建 arXiv 日期范围查询子句，如 submittedDate:[202401010000 TO 202412312359]。"""
    if not date_from and not date_to:
        return ""
    field = "submittedDate" if date_field == "submitted" else "lastUpdatedDate"
    lo = _arxiv_datetime_str(date_from) if date_from else "*"
    hi = _arxiv_datetime_str(date_to, end_of_day=True) if date_to else "*"
    return f"{field}:[{lo} TO {hi}]"


def _append_query_clause(query: str, clause: str) -> str:
    if not clause:
        return query
    q = query.strip()
    if q:
        return f"({q}) AND {clause}"
    return clause


def _paper_matches_date_range(
    paper: "PaperMeta",
    date_from: Optional[DateLike],
    date_to: Optional[DateLike],
    date_field: Literal["submitted", "updated"] = "submitted",
) -> bool:
    """客户端二次过滤（API 返回粒度或时区差异时的兜底）。"""
    if not date_from and not date_to:
        return True
    raw = paper.updated if date_field == "updated" else paper.published
    if not raw:
        return True
    try:
        paper_day = date.fromisoformat(raw[:10])
    except ValueError:
        return True
    if date_from and paper_day < _coerce_date(date_from):
        return False
    if date_to and paper_day > _coerce_date(date_to):
        return False
    return True


@dataclass
class PaperMeta:
    """论文元数据"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    updated: str
    categories: List[str]
    pdf_url: str
    doi: Optional[str] = None
    comment: Optional[str] = None
    journal_ref: Optional[str] = None

    @property
    def year(self) -> int:
        return int(self.published[:4]) if self.published else 0

    @property
    def first_author(self) -> str:
        return self.authors[0] if self.authors else "Unknown"


class ArxivClient:
    """ArXiv API 客户端（适配 arxiv 4.0）"""

    def __init__(self, download_dir: str = None):
        self._download_dir_setting = download_dir or config.arxiv.download_dir
        self.download_dir = self._resolve_download_dir()
        self.max_results = config.arxiv.max_results
        self._client = arxiv.Client()

    def _resolve_download_dir(self) -> Path:
        """Resolve papers dir from config (PROJECT_ROOT), not process CWD."""
        path = (
            resolve_project_path(self._download_dir_setting)
            if self._download_dir_setting != config.arxiv.download_dir
            else get_papers_dir()
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def refresh_download_dir(self) -> Path:
        """Re-read config/env and sync download_dir (e.g. after path fix or env change)."""
        self._download_dir_setting = config.arxiv.download_dir
        self.download_dir = self._resolve_download_dir()
        return self.download_dir

    def search(
        self,
        query: str,
        max_results: int = None,
        categories: List[str] = None,
        sort_by: str = None,
        sort_order: str = None,
        date_from: Optional[DateLike] = None,
        date_to: Optional[DateLike] = None,
        date_field: Literal["submitted", "updated"] = "submitted",
        auto_rewrite: bool = True,
    ) -> List[PaperMeta]:
        """搜索 ArXiv 论文（中文查询自动改写为英文学术术语）"""
        max_results = max_results or self.max_results
        search_query = query

        if auto_rewrite and has_cjk(query):
            try:
                rewriter = QueryRewriter()
                search_query = rewriter.rewrite_for_arxiv(query)
                if search_query != query:
                    logger.info(f"ArXiv 搜索改写: '{query}' → '{search_query}'")
            except Exception as e:
                logger.warning(f"ArXiv 查询改写失败，使用原查询: {e}")

        date_clause = build_date_query_clause(date_from, date_to, date_field)
        api_query = _append_query_clause(search_query, date_clause)

        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        order_map = {
            "descending": arxiv.SortOrder.Descending,
            "ascending": arxiv.SortOrder.Ascending,
        }
        resolved_sort = sort_by or config.arxiv.sort_by
        resolved_order = sort_order or "descending"

        def _run_search(q: str) -> List[PaperMeta]:
            search = arxiv.Search(
                query=q,
                max_results=max_results,
                sort_by=sort_map.get(resolved_sort, arxiv.SortCriterion.Relevance),
                sort_order=order_map.get(resolved_order, arxiv.SortOrder.Descending),
            )
            found: List[PaperMeta] = []
            for result in tqdm(
                self._client.results(search),
                desc=f"搜索: {q[:50]}...",
                total=max_results,
                disable=not _tqdm_enabled(),
            ):
                found.append(
                    PaperMeta(
                        arxiv_id=result.entry_id.split("/")[-1],
                        title=result.title,
                        authors=[str(a) for a in result.authors],
                        abstract=result.summary.replace("\n", " "),
                        published=result.published.isoformat() if result.published else "",
                        updated=result.updated.isoformat() if result.updated else "",
                        categories=list(result.categories),
                        pdf_url=result.pdf_url,
                        doi=result.doi,
                        comment=result.comment,
                        journal_ref=result.journal_ref,
                    )
                )
            return found

        papers: List[PaperMeta] = []
        try:
            papers = _run_search(api_query)

            if categories:
                papers = [p for p in papers if any(cat in p.categories for cat in categories)]

            if date_from or date_to:
                papers = [
                    p for p in papers
                    if _paper_matches_date_range(p, date_from, date_to, date_field)
                ]

            # 改写后仍无结果，尝试用 all: 前缀扩大匹配
            if not papers and search_query != query:
                fallback_query = _append_query_clause(f"all:{search_query}", date_clause)
                papers = _run_search(fallback_query)
                if categories:
                    papers = [p for p in papers if any(cat in p.categories for cat in categories)]
                if date_from or date_to:
                    papers = [
                        p for p in papers
                        if _paper_matches_date_range(p, date_from, date_to, date_field)
                    ]

        except Exception as e:
            logger.error(f"ArXiv 搜索失败: {e}")
            raise

        logger.info(f"搜索完成: {len(papers)} 篇匹配 '{api_query[:60]}'")
        return papers

    def resolve_local_pdf(self, paper: PaperMeta) -> Optional[Path]:
        """查找已下载的 PDF（标题命名、manifest 或旧版 arxiv_id 文件名）。"""
        manifest = load_manifest(self.download_dir)
        mapped = manifest.get("by_arxiv_id", {}).get(paper.arxiv_id)
        if mapped:
            path = self.download_dir / mapped
            if path.exists():
                return path

        legacy = self.download_dir / f"{paper.arxiv_id}.pdf"
        if legacy.exists():
            return legacy

        stem = sanitize_pdf_stem(paper.title)
        titled = self.download_dir / f"{stem}.pdf"
        if titled.exists():
            return titled

        for path in self.download_dir.glob(f"{stem}_*.pdf"):
            if path.is_file():
                return path
        return None

    def pdf_path_for_paper(self, paper: PaperMeta) -> Path:
        """目标保存路径（论文标题命名，必要时去重）。"""
        existing = self.resolve_local_pdf(paper)
        if existing:
            return existing
        filename = f"{sanitize_pdf_stem(paper.title)}.pdf"
        return unique_pdf_path(self.download_dir, filename)

    def metadata_for_pdfs(self, paths: List[Path]) -> List[Dict]:
        """为 build_index 提供与 PDF 列表对应的元数据。"""
        return metadata_for_pdf_paths(self.download_dir, paths)

    def download_pdf(self, paper: PaperMeta, force: bool = False) -> Optional[Path]:
        """下载论文 PDF（保存为消毒后的论文标题文件名）。"""
        existing = self.resolve_local_pdf(paper)
        if existing and not force:
            return existing

        if existing:
            filepath = existing
        else:
            filepath = unique_pdf_path(
                self.download_dir,
                f"{sanitize_pdf_stem(paper.title)}.pdf",
            )
        legacy = self.download_dir / f"{paper.arxiv_id}.pdf"

        try:
            urllib.request.urlretrieve(paper.pdf_url, filepath)
            register_paper_file(
                self.download_dir,
                filepath.name,
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "abstract": paper.abstract,
                },
            )
            if legacy.exists() and legacy.resolve() != filepath.resolve():
                try:
                    legacy.unlink()
                except OSError as e:
                    logger.warning(f"无法删除旧文件名 {legacy.name}: {e}")
            logger.info(f"下载完成: {paper.title[:60]} -> {filepath.name}")
            return filepath
        except Exception as e:
            logger.error(f"下载失败 {paper.arxiv_id}: {e}")
            return None

    def download_papers(self, papers: List[PaperMeta], max_downloads: int = None,
                        force: bool = False) -> List[Path]:
        """批量下载"""
        if max_downloads:
            papers = papers[:max_downloads]
        paths = []
        for paper in tqdm(papers, desc="下载论文", disable=not _tqdm_enabled()):
            path = self.download_pdf(paper, force=force)
            if path:
                paths.append(path)
        logger.info(f"批量下载: {len(paths)}/{len(papers)}")
        return paths

    def get_paper_info(self, arxiv_id: str) -> Optional[PaperMeta]:
        """根据 arxiv_id 获取元数据"""
        try:
            result = next(self._client.results(arxiv.Search(id_list=[arxiv_id])))
            return PaperMeta(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=[str(a) for a in result.authors],
                abstract=result.summary.replace("\n", " "),
                published=result.published.isoformat() if result.published else "",
                updated=result.updated.isoformat() if result.updated else "",
                categories=list(result.categories),
                pdf_url=result.pdf_url,
                doi=result.doi,
                comment=result.comment,
                journal_ref=result.journal_ref,
            )
        except Exception as e:
            logger.error(f"获取论文信息失败 {arxiv_id}: {e}")
            return None

    def get_local_papers(self) -> List[Path]:
        """List every PDF file in the resolved papers directory."""
        papers_dir = self._resolve_download_dir()
        self.download_dir = papers_dir
        return sorted(p for p in papers_dir.glob("*.pdf") if p.is_file())
