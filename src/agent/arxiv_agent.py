"""
ArXiv Agent — 学术论文智能助手

支持两种模式：
1. 深思熟虑（deliberative）— 先规划再逐步执行，plan → execute_plan
2. 反应式（reactive）— ReAct 循环，LLM 即时决策调用工具

Agent 工作流：
用户问题 → Plan(LLM生成工具执行计划) → 动态执行工具链 → LLM 生成回答
"""
import json
import logging
from typing import List, Dict, Optional

from src.config import config
from src.agent.tools import AGENT_TOOLS
from src.prompts import SURVEY_SYSTEM_PROMPT, COMPARE_SYSTEM_PROMPT, QA_SYSTEM_PROMPT, SurveyOutput, CompareOutput, parse_structured_output
from src.retriever.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)

# ── 工具映射：plan 中的 tool 名 → (tool_fn, 参数映射规则) ──
_PLAN_TOOL_MAP = {
    "search_papers": {
        "fn_key": "search_papers",
        "arg_map": {"query": "query", "max_results": "max_results"},
    },
    "download_paper": {
        "fn_key": "download_paper",
        "arg_map": {"arxiv_id": "arxiv_id"},
    },
    "rag_query": {
        "fn_key": "rag_query",
        "arg_map": {"query": "query", "top_k": "top_k"},
    },
    "rewrite_query": {
        "fn_key": "rewrite_query",
        "arg_map": {"query": "query", "strategy": "strategy"},
    },
}

# ── 当 Plan 失败或未生成足够步骤时的兜底流程 ──
# ── 当 Plan 失败时的兜底流程（本地论文库已有索引时跳过搜索下载）──
_FALLBACK_PLAN = [
    {"tool": "rewrite_query", "args": {"query": "__QUERY__", "strategy": "chinese_academic"}, "description": "改写查询为学术搜索词"},
    {"tool": "rag_query", "args": {"query": "__QUERY__", "top_k": 8}, "description": "在本地论文库中 RAG 检索（兜底）"},
]


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

    def plan(self, user_query: str) -> List[Dict]:
        """
        深思熟虑模式：LLM 生成结构化的工具执行计划。

        Returns:
            [{"tool": "search_papers", "args": {...}, "description": "..."}, ...]
            空列表 = 规划失败，触发兜底流程
        """
        tools_desc = []
        for name, info in self.tools.items():
            tools_desc.append(f"- {name}: {info['description']}  {info['signature']}")

        plan_prompt = f"""你是一个学术研究助手的规划器。根据用户问题，生成一个工具调用计划。

可用工具：
{chr(10).join(tools_desc)}

用户问题：{user_query}

生成一个 JSON 格式的执行计划。计划是一个步骤列表，每个步骤包含：
- "tool": 工具名（必须是上面列出的工具之一）
- "args": 工具参数字典
- "description": 这一步在做什么（中文）

重要规则：
1. 如果用户用中文提问，第一步必须是 "rewrite_query" 将中文改写为英文学术搜索词
2. 如果用户已经在问本地论文的问题（如"这篇论文的方法是什么""对比一下"），直接用 "rag_query"
3. 如果用户要了解某个领域的最新进展，需要先 search_papers → download_paper → rag_query
4. 计划 3-5 步，不要多余步骤

返回格式（纯 JSON）：{{"steps": [{{"tool": "...", "args": {{}}, "description": "..."}}], "reasoning": "总体思路"}}"""

        try:
            resp = self.llm.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": "你是任务规划器。只返回 JSON，不要任何其他内容。"},
                    {"role": "user", "content": plan_prompt},
                ],
                temperature=0.1,
                max_tokens=800,
            )
            content = resp.choices[0].message.content
            s, e = content.find('{'), content.rfind('}') + 1
            if s >= 0 and e > s:
                plan = json.loads(content[s:e])
                steps = plan.get("steps", [])
                reasoning = plan.get("reasoning", "")
                if steps and reasoning:
                    logger.info("LLM Plan (%s): %d steps", reasoning, len(steps))
                return steps
        except Exception as e:
            logger.warning(f"LLM 规划失败，触发兜底流程: {e}")

        return []

    def execute(self, user_query: str,
                max_papers: int = 5,
                stream: bool = False) -> Dict:
        """
        执行 Agent 流程（plan → execute_plan → generate）。

        Args:
            user_query: 用户问题
            max_papers: 最多下载论文数
            stream: 是否流式返回中间步骤

        Returns:
            {"answer": str, "steps": [...], "papers": [...], "plan": [...]}
        """
        # ── Phase 1: Plan（LLM 生成工具执行计划）──
        if self.mode == "deliberative":
            plan_steps = self.plan(user_query)
        else:
            plan_steps = []

        if not plan_steps:
            # 兜底：使用预定义的默认流程
            plan_steps = _FALLBACK_PLAN[:]
            logger.info("使用兜底执行计划 (%d 步)", len(plan_steps))

        # ── Phase 2: Execute Plan（动态执行工具链 + 占位符插值 + 中间失败容错）──
        steps_log = []
        context = {
            "__QUERY__": user_query,
            "__REWRITTEN__": user_query,
            "__FIRST_RESULT_ID__": "",
            "papers": [],
            "max_papers": max_papers,
            "downloaded": [],
            "chunks": [],
        }

        for i, step in enumerate(plan_steps):
            tool_name = step.get("tool", "")
            raw_args = step.get("args", {})
            description = step.get("description", f"Step {i+1}: {tool_name}")

            # ═══ 插值替换占位符（__QUERY__, __FIRST_RESULT_ID__ 等）═══
            resolved_args = {}
            skip_this_step = False
            for k, v in raw_args.items():
                str_v = str(v)
                for ctx_key, ctx_val in context.items():
                    if isinstance(ctx_val, str):
                        str_v = str_v.replace(ctx_key, ctx_val)
                # 检测未解析的占位符（如 __FIRST_RESULT_ID__ 未找到论文）
                if "__FIRST_RESULT_ID__" in str_v and not context.get("__FIRST_RESULT_ID__"):
                    logger.warning(f"  ⏭️ 跳过 {tool_name}：无搜索结果可供下载")
                    skip_this_step = True
                    break
                resolved_args[k] = str_v

            if skip_this_step:
                steps_log.append({"step": tool_name, "status": "skipped",
                                  "description": f"跳过: {description} (无可下载论文)"})
                continue

            # 类型转换
            if "max_results" in resolved_args:
                try: resolved_args["max_results"] = int(resolved_args["max_results"])
                except (ValueError, TypeError): resolved_args["max_results"] = 15
            if "top_k" in resolved_args:
                try: resolved_args["top_k"] = int(resolved_args["top_k"])
                except (ValueError, TypeError): resolved_args["top_k"] = 8

            logger.info(f"执行 Step {i+1}/{len(plan_steps)}: {tool_name} | {description}")

            step_record = {"step": tool_name, "status": "running", "description": description}
            steps_log.append(step_record)

            try:
                tool_info = self.tools.get(tool_name)
                if not tool_info:
                    step_record["status"] = "failed"
                    step_record["error"] = f"未知工具: {tool_name}"
                    continue

                fn = tool_info["function"]
                result = fn(**resolved_args)
                parsed = json.loads(result)

                # ── 根据工具类型更新上下文 ──
                if tool_name == "rewrite_query":
                    rewrites = parsed.get("rewrites", [user_query])
                    # 取第一个非原文的英文改写
                    rewritten = user_query
                    for rw in rewrites[1:]:
                        if rw != user_query:
                            rewritten = rw
                            break
                    context["__REWRITTEN__"] = rewritten
                    step_record["result"] = rewritten
                    step_record["status"] = "done"
                    logger.info(f"  Query 改写: '{user_query[:40]}...' → '{rewritten[:40]}...'")

                elif tool_name == "search_papers":
                    papers = parsed.get("papers", [])
                    context["papers"] = papers
                    if papers:
                        context["__FIRST_RESULT_ID__"] = papers[0].get("arxiv_id", "")
                    step_record["count"] = len(papers)
                    step_record["status"] = "done"
                    logger.info(f"  搜索到 {len(papers)} 篇论文")

                elif tool_name == "download_paper":
                    if parsed.get("success"):
                        context["downloaded"].append(parsed)
                        step_record["result"] = parsed.get("title", "")
                    step_record["status"] = "done"
                    logger.info(f"  下载: {parsed.get('title', parsed.get('error', ''))}")

                elif tool_name == "rag_query":
                    chunks = parsed.get("results", [])
                    context["chunks"] = chunks
                    step_record["count"] = len(chunks)
                    step_record["status"] = "done"
                    logger.info(f"  RAG 检索: {len(chunks)} 个片段")

                else:
                    step_record["status"] = "done"

            except Exception as e:
                step_record["status"] = "failed"
                step_record["error"] = str(e)
                logger.warning(f"  Step {tool_name} 失败: {e}")

        # ── Phase 3: Generate Answer（LLM 生成最终回答）──
        papers = context["papers"]
        chunks = context["chunks"]
        downloaded = context["downloaded"]

        if not chunks and not papers:
            return {
                "answer": "未能找到相关内容。请尝试修改搜索词后重试。",
                "steps": steps_log,
                "papers": [],
                "plan": plan_steps,
            }

        # 如果 RAG 没检索到但搜到论文了，用摘要兜底
        if not chunks and papers:
            context_text = "\n\n---\n\n".join([
                f"**{p.get('title', 'Unknown')}** ({', '.join(p.get('authors', [])[:3])})\n{p.get('abstract', '')}"
                for p in papers[:max_papers]
            ])
            chunks = [{"text": context_text, "source": "arxiv_abstracts"}]

        try:
            steps_log.append({"step": "generate_answer", "status": "running", "description": "LLM 生成最终回答"})

            def _format_chunk(c):
                source = c.get("source", "?")
                title = c.get("paper_title", "")
                section = c.get("section", "")
                header = f"[{source}] {title} | {section}"
                if c.get("is_table"):
                    html = c.get("full_html", c.get("text", ""))
                    return (
                        f"{header} [TABLE: {c.get('table_id', '')}]\n"
                        f"以下是表格 HTML 内容，请从表格中提取关键数据回答：\n"
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
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content

            # 尝试结构化输出
            validated = parse_structured_output(raw, SurveyOutput)
            if validated:
                answer = (
                    f"**综述概览**\n\n{validated.overview}\n\n"
                    f"**论文分析**\n\n" + "\n\n".join(
                        f"### {p.get('title', '')}\n- 方法: {p.get('method', '')}\n- 贡献: {p.get('contribution', '')}\n- 亮点: {p.get('highlights', '')}"
                        for p in validated.papers
                    ) + "\n\n" +
                    f"**研究趋势**\n\n" + "\n".join(f"- {t}" for t in validated.trends) + "\n\n" +
                    f"**结论**\n\n{validated.conclusion}\n\n" +
                    f"**参考文献**\n\n" + "\n".join(f"- {r}" for r in validated.references)
                )
            else:
                answer = raw

            steps_log[-1]["status"] = "done"
        except Exception as e:
            answer = f"生成回答失败: {e}"
            steps_log[-1]["status"] = "failed"
            steps_log[-1]["error"] = str(e)

        return {
            "answer": answer,
            "steps": steps_log,
            "papers": [
                {"title": p.get("title", ""), "arxiv_id": p.get("arxiv_id", ""),
                 "authors": p.get("authors", [])}
                for p in (papers[:max_papers] if papers else [])
            ],
            "plan": plan_steps,
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
                return f"{header}\n{c['text'][:1500]}"

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
