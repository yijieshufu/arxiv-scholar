"""
ArXiv Scholar CLI — 命令行入口

用法：
    # 搜索并下载论文
    python -m src.cli search "transformer attention mechanism" --max 10 --download

    # 构建索引（已有 PDF）
    python -m src.cli build-index --data-dir ./data/papers

    # 查询
    python -m src.cli query "What is LoRA?"

    # 综述生成
    python -m src.cli survey "latest advances in LLM alignment"

    # 启动 MCP Server
    python -m src.cli mcp-server
"""
import sys
import json
import logging
from pathlib import Path

import click

from src.config import config, ensure_data_layout, get_papers_dir, resolve_project_path
from src.data_migration import migrate_legacy_data
from src.arxiv_client import ArxivClient
from src.retriever.pipeline import RetrievalPipeline
from src.agent import ArxivAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
def main():
    """ArXiv Scholar — 智能学术论文助手"""
    ensure_data_layout()


@main.command("migrate-data")
def migrate_data():
    """将用户主目录等旧路径下的 PDF/索引迁移到项目 data/"""
    report = migrate_legacy_data()
    click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    n_pdf = len(report.get("migrated_papers", []))
    n_idx = len(report.get("migrated_index_files", []))
    if n_pdf or n_idx:
        click.echo(f"\n完成: 迁移 {n_pdf} 个 PDF, {n_idx} 个索引文件 → {report['project_data_dir']}")
    else:
        click.echo(f"\n无需迁移；数据目录: {report['papers_dir']}")


@main.command()
@click.argument("query")
@click.option("--max", "max_results", default=10, help="最大返回数")
@click.option("--download/--no-download", default=False, help="是否下载 PDF")
@click.option("--categories", default=None, help="限定分类，逗号分隔")
def search(query, max_results, download, categories):
    """搜索 ArXiv 论文"""
    client = ArxivClient()
    cats = categories.split(",") if categories else None

    papers = client.search(query, max_results=max_results, categories=cats)

    click.echo(f"\n找到 {len(papers)} 篇论文:\n")
    for i, p in enumerate(papers):
        click.echo(f"{i+1}. {p.title}")
        click.echo(f"   ID: {p.arxiv_id} | {', '.join(p.authors[:3])}")
        click.echo(f"   年份: {p.year} | 分类: {', '.join(p.categories[:3])}")
        click.echo(f"   摘要: {p.abstract[:150]}...\n")

    if download:
        client.download_papers(papers, max_downloads=5)
        click.echo("下载完成！运行 build-index 构建索引。")


@main.command()
@click.option("--data-dir", default=None, help="PDF 目录（默认项目 data/papers）")
@click.option("--rebuild/--no-rebuild", default=False, help="是否重建索引")
def build_index(data_dir, rebuild):
    """构建论文检索索引"""
    data_path = resolve_project_path(data_dir) if data_dir else get_papers_dir()
    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        click.echo(f"在 {data_dir} 中未找到 PDF 文件")
        return

    click.echo(f"找到 {len(pdf_files)} 个 PDF 文件，开始构建索引...")

    client = ArxivClient()
    pipeline = RetrievalPipeline()
    metas = client.metadata_for_pdfs(pdf_files)
    pipeline.build_index([str(f) for f in pdf_files], metas, rebuild=rebuild)

    click.echo(f"索引构建完成！{len(pipeline._all_chunks)} 个 chunks")


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="返回数量")
@click.option("--no-rerank", is_flag=True, help="禁用 Rerank")
@click.option("--alpha", type=float, default=None, help="混合检索权重")
def query(query, top_k, no_rerank, alpha):
    """RAG 检索查询"""
    pipeline = RetrievalPipeline()
    client = ArxivClient()
    ok, msg = pipeline.ensure_index(client)
    if not ok:
        click.echo(msg or "索引未构建，请先运行 build-index 或将 PDF 放入 data/papers")
        return
    if msg:
        click.echo(msg)

    results = pipeline.query(query, top_k=top_k, use_rerank=not no_rerank, alpha=alpha)

    click.echo(f"\n查询: {query}")
    click.echo(f"找到 {len(results)} 个结果:\n")

    for i, r in enumerate(results):
        click.echo(f"--- #{i+1} [score={r.get('rerank_score', r['score']):.4f}] ---")
        click.echo(f"来源: {r['source']}")
        click.echo(f"论文: {r.get('metadata', {}).get('paper_title', 'N/A')}")
        click.echo(f"章节: {r.get('metadata', {}).get('section_title', 'N/A')}")
        click.echo(f"{r['text'][:300]}\n")


@main.command()
@click.argument("topic")
@click.option("--max-papers", default=5, help="最多下载论文数")
def survey(topic, max_papers):
    """生成文献综述"""
    agent = ArxivAgent(mode="deliberative")
    click.echo(f"🔍 开始研究: {topic}")
    click.echo("=" * 60)

    result = agent.execute(topic, max_papers=max_papers)

    click.echo("\n📋 执行步骤:")
    for step in result["steps"]:
        status = "✅" if step["status"] == "done" else "❌"
        click.echo(f"  {status} {step['step']}")

    click.echo(f"\n📚 引用论文 ({len(result['papers'])}):")
    for p in result["papers"]:
        click.echo(f"  • {p['title']} ({p['arxiv_id']})")

    click.echo("\n" + "=" * 60)
    click.echo("📄 综述报告:\n")
    click.echo(result["answer"])


@main.command()
@click.argument("question")
def ask(question):
    """快速问答（基于本地论文库）"""
    agent = ArxivAgent()
    click.echo(f"❓ {question}\n")
    answer = agent.quick_answer(question)
    click.echo(answer)


@main.command()
def mcp_server():
    """启动 ArXiv MCP Server（stdio 模式）"""
    import asyncio
    from src.mcp import start_mcp_server
    click.echo("启动 ArXiv MCP Server...")
    asyncio.run(start_mcp_server())


if __name__ == "__main__":
    main()
