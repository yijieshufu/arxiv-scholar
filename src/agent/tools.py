"""
Agent 工具集 — ArXiv Scholar 专用工具

提供 Agent 可以调用的工具函数，每个工具返回结构化 JSON。
"""
import json
import logging
from pathlib import Path

from src.arxiv_client import ArxivClient
from src.parser import PaperParser
from src.retriever.pipeline import RetrievalPipeline
from src.query_rewriter import QueryRewriter
from src.prompts import SURVEY_SYSTEM_PROMPT, COMPARE_SYSTEM_PROMPT, QA_SYSTEM_PROMPT
from src.config import config

logger = logging.getLogger(__name__)

arxiv_client = ArxivClient()


def search_papers_tool(query: str, max_results: int = 10) -> str:
    """搜索 ArXiv 论文"""
    try:
        papers = arxiv_client.search(query, max_results=max_results)
        return json.dumps({
            "count": len(papers),
            "papers": [
                {"arxiv_id": p.arxiv_id, "title": p.title,
                 "authors": p.authors[:3], "year": p.year,
                 "abstract": p.abstract[:200]}
                for p in papers
            ]
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def download_paper_tool(arxiv_id: str) -> str:
    """下载论文 PDF 并解析入库"""
    try:
        paper = arxiv_client.get_paper_info(arxiv_id)
        if not paper:
            return json.dumps({"error": f"未找到论文: {arxiv_id}"})

        pdf_path = arxiv_client.download_pdf(paper)
        if not pdf_path:
            return json.dumps({"error": "下载失败"})

        # 解析并增量入库
        pipeline = RetrievalPipeline()
        pipeline.build_index(
            [str(pdf_path)],
            [{"arxiv_id": paper.arxiv_id, "title": paper.title,
              "authors": paper.authors, "year": paper.year,
              "abstract": paper.abstract}],
            rebuild=False,
        )

        return json.dumps({
            "success": True,
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "path": str(pdf_path),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def rag_query_tool(query: str, top_k: int = 5) -> str:
    """RAG 检索本地论文库"""
    try:
        pipeline = RetrievalPipeline()
        ok, msg = pipeline.ensure_index(arxiv_client)
        if not ok:
            return json.dumps({"error": msg or "索引未构建，请先下载论文或放入 data/papers"})

        results = pipeline.query(query, top_k=top_k, use_rerank=True, rewrite=True)

        def _format_result(r):
            meta = r.get("metadata", {})
            result = {
                "text": r["text"][:500],
                "source": r["source"],
                "score": r.get("rerank_score", r["score"]),
                "paper_title": meta.get("paper_title", ""),
                "section": meta.get("section_title", ""),
                "is_table": meta.get("is_table", False),
            }
            # 表格 chunk 返回完整 HTML 供 LLM 后处理
            if meta.get("is_table"):
                result["full_html"] = meta.get("full_html_content", "")
                result["table_id"] = meta.get("table_id", "")
            return result

        return json.dumps({
            "count": len(results),
            "results": [_format_result(r) for r in results]
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def rewrite_query_tool(query: str, strategy: str = "multi") -> str:
    """Query 改写"""
    try:
        rewriter = QueryRewriter()
        rewrites = rewriter.rewrite(query, strategy=strategy)
        return json.dumps({"original": query, "rewrites": rewrites, "count": len(rewrites)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# Agent 工具注册表
AGENT_TOOLS = {
    "search_papers": {
        "function": search_papers_tool,
        "description": "搜索 ArXiv 论文（从 ArXiv 在线搜索，不依赖本地库）",
        "signature": "(query: str, max_results: int = 10) -> str",
    },
    "download_paper": {
        "function": download_paper_tool,
        "description": "下载论文 PDF 并自动解析入库到本地 RAG 索引",
        "signature": "(arxiv_id: str) -> str",
    },
    "rag_query": {
        "function": rag_query_tool,
        "description": "在本地论文库中进行 RAG 检索（混合检索 + Rerank）",
        "signature": "(query: str, top_k: int = 5) -> str",
    },
    "rewrite_query": {
        "function": rewrite_query_tool,
        "description": "将自然语言改写为学术查询（MultiQuery / HyDE / 学术术语标准化）",
        "signature": "(query: str, strategy: str = 'multi') -> str",
    },
}
