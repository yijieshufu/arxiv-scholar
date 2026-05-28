"""
ArXiv Agent — 学术论文智能助手

支持两种模式：
1. 深思熟虑（deliberative）— 先规划再逐步执行
2. 反应式（reactive）— 即时决策调用工具

Agent 工作流：
用户问题 → Query 改写 → 搜索 ArXiv → 下载论文 → 构建索引 → RAG 检索 → LLM 生成综述
"""
import json
import logging
from typing import List, Dict, Optional

from src.config import config
from src.agent.tools import AGENT_TOOLS
from src.prompts import SURVEY_SYSTEM_PROMPT, COMPARE_SYSTEM_PROMPT, QA_SYSTEM_PROMPT
from src.retriever.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


class ArxivAgent:
    """
    ArXiv 学术助手 Agent。

    核心能力：
    - 搜索论文（ArXiv API）
    - 下载并解析论文（PDF → RAG 索引）
    - RAG 检索（混合检索 + Rerank）
    - 生成文献综述 / 论文对比 / 论文问答
    """

    def __init__(self, mode: str = None):
        self.mode = mode or config.agent.mode
        self.max_iterations = config.agent.max_iterations
        self.tools = AGENT_TOOLS
        self.pipeline = RetrievalPipeline()
        self._llm_client = None
        self._history: List[Dict] = []

    @property
    def llm(self):
        if self._llm_client is None:
            from src.config import get_llm_client
            self._llm_client = get_llm_client()
        return self._llm_client

    def plan(self, user_query: str) -> List[str]:
        """
        深思熟虑模式：先规划执行步骤。

        Returns:
            步骤描述列表
        """
        plan_prompt = f"""你是一个学术研究助手的规划器。根据用户的问题，制定一个逐步执行计划。

可用工具：
1. search_papers — 搜索 ArXiv 论文
2. download_paper — 下载论文 PDF 并入库
3. rag_query — 检索本地论文库
4. rewrite_query — 改写查询

用户问题：{user_query}

返回 JSON 格式的执行步骤列表：
{{"steps": ["步骤1描述", "步骤2描述", ...], "reasoning": "总体思路"}}"""

        try:
            resp = self.llm.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": "你是任务规划器。只返回 JSON。"},
                    {"role": "user", "content": plan_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            content = resp.choices[0].message.content
            s, e = content.find('{'), content.rfind('}') + 1
            if s >= 0 and e > s:
                plan = json.loads(content[s:e])
                return plan.get("steps", [])
        except Exception as e:
            logger.warning(f"规划失败，使用默认流程: {e}")

        return [
            "改写查询为学术搜索词",
            "搜索 ArXiv 相关论文",
            "下载最相关的 3-5 篇论文",
            "在本地论文库中 RAG 检索",
            "生成综述回答",
        ]

    def execute(self, user_query: str,
                max_papers: int = 5,
                stream: bool = False) -> Dict:
        """
        执行 Agent 流程。

        Args:
            user_query: 用户问题
            max_papers: 最多下载论文数
            stream: 是否流式返回中间步骤

        Returns:
            {"answer": str, "steps": [...], "papers": [...]}
        """
        steps_log = []
        downloaded = []

        # Step 1: Query 改写
        try:
            steps_log.append({"step": "query_rewrite", "status": "running"})
            result = self.tools["rewrite_query"]["function"](user_query)
            qr = json.loads(result)
            search_query = qr.get("rewrites", [user_query])[1] if len(qr.get("rewrites", [])) > 1 else user_query
            from src.query_rewriter import has_cjk, QueryRewriter
            if has_cjk(user_query):
                search_query = QueryRewriter().rewrite_for_arxiv(user_query)
            steps_log[-1]["status"] = "done"
            steps_log[-1]["result"] = search_query
        except Exception as e:
            search_query = user_query
            steps_log[-1]["status"] = "failed"
            steps_log[-1]["error"] = str(e)

        # Step 2: 搜索 ArXiv
        try:
            steps_log.append({"step": "search_arxiv", "status": "running"})
            result = self.tools["search_papers"]["function"](search_query, max_results=15)
            search_data = json.loads(result)
            papers = search_data.get("papers", [])
            steps_log[-1]["status"] = "done"
            steps_log[-1]["count"] = len(papers)
        except Exception as e:
            papers = []
            steps_log[-1]["status"] = "failed"
            steps_log[-1]["error"] = str(e)

        if not papers:
            return {"answer": "未找到相关论文，请尝试修改搜索词。", "steps": steps_log, "papers": []}

        # Step 3: 下载论文
        try:
            steps_log.append({"step": "download_papers", "status": "running"})
            for paper in papers[:max_papers]:
                result = self.tools["download_paper"]["function"](paper["arxiv_id"])
                dl = json.loads(result)
                if dl.get("success"):
                    downloaded.append(dl)

            steps_log[-1]["status"] = "done"
            steps_log[-1]["downloaded"] = len(downloaded)
        except Exception as e:
            steps_log[-1]["status"] = "failed"
            steps_log[-1]["error"] = str(e)

        # Step 4: RAG 检索
        try:
            steps_log.append({"step": "rag_retrieve", "status": "running"})
            result = self.tools["rag_query"]["function"](user_query, top_k=8)
            retrieved = json.loads(result)
            chunks = retrieved.get("results", [])
            steps_log[-1]["status"] = "done"
            steps_log[-1]["count"] = len(chunks)
        except Exception as e:
            chunks = []
            steps_log[-1]["status"] = "failed"
            steps_log[-1]["error"] = str(e)

        if not chunks:
            # 如果本地索引没有（新下载的论文还没索引），用摘要回答
            context = "\n\n---\n\n".join([
                f"**{p.get('title', 'Unknown')}** ({', '.join(p.get('authors', [])[:3])})\n{p.get('abstract', '')}"
                for p in papers[:max_papers]
            ])
            chunks = [{"text": context, "source": "arxiv_abstracts"}]

        # Step 5: LLM 生成综述
        try:
            steps_log.append({"step": "generate_survey", "status": "running"})

            def _format_chunk(c):
                """格式化单个 chunk：表格块展示完整 HTML，普通块截断文本。"""
                source = c.get("source", "?")
                title = c.get("paper_title", "")
                section = c.get("section", "")
                header = f"[{source}] {title} | {section}"
                if c.get("is_table"):
                    html = c.get("full_html", c.get("text", ""))
                    return (
                        f"{header} [TABLE: {c.get('table_id', '')}]\n"
                        f"下面是该论文表格的 HTML 内容，请根据用户问题从表格中提取关键数据回答：\n"
                        f"```html\n{html}\n```"
                    )
                else:
                    return f"{header}\n{c['text'][:1500]}"

            context_text = "\n\n---\n\n".join([
                _format_chunk(c) for c in chunks[:8]
            ])

            resp = self.llm.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": SURVEY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"用户问题：{user_query}\n\n论文内容：\n{context_text}"},
                ],
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )
            answer = resp.choices[0].message.content
            steps_log[-1]["status"] = "done"
        except Exception as e:
            answer = f"生成回答失败: {e}"
            steps_log[-1]["status"] = "failed"

        return {
            "answer": answer,
            "steps": steps_log,
            "papers": [
                {"title": p.get("title", ""), "arxiv_id": p.get("arxiv_id", ""),
                 "authors": p.get("authors", [])}
                for p in papers[:max_papers]
            ],
        }

    def quick_answer(self, query: str) -> str:
        """
        反应式模式：直接 RAG + 问答（不搜索新论文）。
        支持对话历史（self._history），滑动窗口保留最近 5 轮。
        """
        try:
            result = self.tools["rag_query"]["function"](query, top_k=5)
            retrieved = json.loads(result)
            chunks = retrieved.get("results", [])

            if not chunks:
                err = retrieved.get("error")
                if err:
                    return err
                return "本地暂无相关论文。请下载 PDF 到 data/papers，或在「论文搜索」中下载并构建索引。"

            def _fmt_q(c):
                header = f"[{c.get('source', '?')}]"
                if c.get("is_table"):
                    html = c.get("full_html", c.get("text", ""))
                    return (
                        f"{header} [TABLE: {c.get('table_id', '')}]\n"
                        f"下面是论文表格 HTML，请从中提取数据回答：\n"
                        f"```html\n{html}\n```"
                    )
                return f"{header}\n{c['text'][:1000]}"

            context_text = "\n\n---\n\n".join([
                _fmt_q(c) for c in chunks[:5]
            ])

            # ── 多轮对话历史：保留最近 5 轮（10 条消息）──
            self._history.append({"role": "user", "content": query})
            history_msgs = []
            for m in self._history[-10:]:
                history_msgs.append({"role": m["role"], "content": m["content"]})

            messages = [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                *history_msgs,
                {"role": "user", "content": f"问题：{query}\n\n论文内容：\n{context_text}"},
            ]

            resp = self.llm.chat.completions.create(
                model=config.llm.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
            )
            answer = resp.choices[0].message.content
            self._history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            return f"查询失败: {e}"
