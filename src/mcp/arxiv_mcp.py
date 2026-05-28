"""
ArXiv MCP Server — 将 ArXiv API 封装为 MCP 工具

参考：Anthropic Cookbook 的 arxiv-mcp-server 设计
提供工具：search_papers, download_paper, get_paper_info, list_local_papers

MCP 协议：Model Context Protocol，让 LLM 可以直接调用外部工具
"""
import json
import logging
from typing import Any

from src.arxiv_client import ArxivClient, PaperMeta
from src.config import config

logger = logging.getLogger(__name__)

# MCP Server 实例（单例）
arxiv_client = ArxivClient()


# ---- MCP 工具函数 ----

def search_papers(query: str, max_results: int = 10, categories: list = None) -> str:
    """
    搜索 ArXiv 论文。

    Args:
        query: 搜索关键词
        max_results: 最大返回数
        categories: 限定分类

    Returns:
        JSON 格式的论文列表
    """
    try:
        papers = arxiv_client.search(query, max_results=max_results, categories=categories)
        results = [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors[:3],
                "year": p.year,
                "abstract": p.abstract[:300],
                "categories": p.categories[:3],
                "pdf_url": p.pdf_url,
            }
            for p in papers
        ]
        return json.dumps({"count": len(results), "papers": results}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def download_paper(arxiv_id: str, force: bool = False) -> str:
    """
    下载论文 PDF。

    Args:
        arxiv_id: ArXiv ID（如 "2301.12345"）
        force: 是否强制重新下载

    Returns:
        下载结果
    """
    try:
        paper = arxiv_client.get_paper_info(arxiv_id)
        if not paper:
            return json.dumps({"error": f"未找到论文: {arxiv_id}"})

        path = arxiv_client.download_pdf(paper, force=force)
        if path:
            return json.dumps({"success": True, "path": str(path), "title": paper.title})
        return json.dumps({"error": "下载失败"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_paper_info(arxiv_id: str) -> str:
    """
    获取论文详细信息。

    Args:
        arxiv_id: ArXiv ID

    Returns:
        JSON 格式的论文元数据
    """
    try:
        paper = arxiv_client.get_paper_info(arxiv_id)
        if not paper:
            return json.dumps({"error": f"未找到: {arxiv_id}"})
        return json.dumps({
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "published": paper.published,
            "categories": paper.categories,
            "pdf_url": paper.pdf_url,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def list_local_papers() -> str:
    """列出已下载的本地论文"""
    papers = arxiv_client.get_local_papers()
    return json.dumps({
        "count": len(papers),
        "files": [str(p.name) for p in papers],
    }, ensure_ascii=False)


# MCP 工具注册表
MCP_TOOLS = {
    "search_papers": {
        "function": search_papers,
        "description": "搜索 ArXiv 论文，返回论文列表（含标题、作者、摘要、链接）",
        "parameters": {
            "query": "搜索关键词（英文）",
            "max_results": "最大返回数（默认 10）",
            "categories": "限定分类列表，如 ['cs.AI', 'cs.CL']",
        },
    },
    "download_paper": {
        "function": download_paper,
        "description": "下载指定论文的 PDF 到本地",
        "parameters": {
            "arxiv_id": "ArXiv ID",
            "force": "是否强制重新下载",
        },
    },
    "get_paper_info": {
        "function": get_paper_info,
        "description": "获取论文的详细元数据（完整摘要、所有作者、分类等）",
        "parameters": {
            "arxiv_id": "ArXiv ID",
        },
    },
    "list_local_papers": {
        "function": list_local_papers,
        "description": "列出本地已下载的所有论文 PDF",
        "parameters": {},
    },
}


async def start_mcp_server():
    """
    启动 MCP Server（stdio/SSE 模式）。

    参考：Anthropic MCP SDK
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server

        server = Server("arxiv-scholar")

        @server.list_tools()
        async def list_tools():
            return [
                {
                    "name": name,
                    "description": info["description"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            k: {"type": "string" if k in ("query", "arxiv_id") else "integer" if k == "max_results" else "boolean" if k == "force" else "array", "description": v}
                            for k, v in info.get("parameters", {}).items()
                        },
                    },
                }
                for name, info in MCP_TOOLS.items()
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list:
            tool = MCP_TOOLS.get(name)
            if not tool:
                return [{"type": "text", "text": f"未知工具: {name}"}]
            result = tool["function"](**arguments)
            return [{"type": "text", "text": result}]

        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    except ImportError:
        logger.warning("MCP SDK 未安装，MCP Server 不可用")
    except Exception as e:
        logger.error(f"MCP Server 启动失败: {e}")
