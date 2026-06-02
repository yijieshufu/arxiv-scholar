"""
ArXiv Scholar — 智能学术论文助手 (Streamlit 界面)

参考 paper-qa 的功能设计 + resume-ai 的界面架构。

功能 Tab：
1. 🔍 论文搜索：搜索 ArXiv → 下载 → 自动入库
2. 📚 文献综述：Agent 全程托管（搜索→下载→RAG→生成综述）
3. 💬 论文问答：基于本地论文库的 RAG 问答
4. 📊 评估面板：检索效果评估
"""
import os
import sys
import time
import json
import logging
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src.config import config, ensure_data_layout, get_papers_dir, resolve_project_path
from src.data_migration import migrate_legacy_data
from src.arxiv_client import ArxivClient
from src.retriever.pipeline import RetrievalPipeline
from src.agent import ArxivAgent
from src.mcp import MCP_TOOLS
from src.evaluation import tracker

logging.basicConfig(level=logging.INFO if not config.debug else logging.DEBUG)

ensure_data_layout()
_migrate_report = migrate_legacy_data()
if _migrate_report.get("migrated_papers") or _migrate_report.get("migrated_index_files"):
    logging.getLogger(__name__).info(
        "已从旧路径迁移数据: %d PDF, %d 索引文件",
        len(_migrate_report["migrated_papers"]),
        len(_migrate_report["migrated_index_files"]),
    )

# ---- 页面配置 ----
st.set_page_config(
    page_title="ArXiv Scholar — 智能学术论文助手",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ArXiv Scholar — 智能学术论文助手")
st.caption("融合 RAG + Agent + MCP | 搜索 · 下载 · 解析 · 综述 · 问答")


# ---- 侧边栏 ----
with st.sidebar:
    st.header("⚙️ 配置")

    agent_mode = st.selectbox(
        "Agent 模式",
        ["deliberative", "reactive"],
        index=0,
        help="深思熟虑：先规划再执行 | 反应式：即时决策"
    )

    st.divider()
    st.subheader("检索参数")
    top_k = st.slider("Top-K", 1, 20, 5)
    alpha = st.slider("混合权重 α (向量 vs BM25)", 0.0, 1.0, 0.6, 0.05)
    use_rerank = st.checkbox("启用 Rerank", value=True)
    use_rewrite = st.checkbox("启用 Query 改写", value=True)

    st.divider()
    st.subheader("Embedding")
    config.embedding.use_api = st.checkbox("使用 API Embedding（免加载，首查秒回）", value=False, help="开启后使用 DeepSeek Embedding API，无需加载本地 BGE 模型")
    if config.embedding.use_api:
        st.caption(f"API: {config.embedding.api_model} | 维度: {config.embedding.dimension}")

    st.divider()
    st.subheader("论文设置")
    max_papers = st.slider("最多下载论文数", 1, 15, 5)
    categories = st.multiselect(
        "限定分类",
        ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.MA", "stat.ML"],
        default=["cs.AI", "cs.CL", "cs.LG"],
    )

    st.divider()
    st.caption("💡 面试亮点：RAG 全链路 + Agent 自主规划 + MCP + 混合检索 + Query 改写")


# ---- Tab 页面 ----
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 论文搜索", "📚 文献综述", "💬 论文问答", "📊 评估面板"
])

# ---- 辅助：同步 session 中的 ArxivClient 与当前 papers 目录 ----
def _get_arxiv_client() -> ArxivClient:
    expected = get_papers_dir()
    client = st.session_state.get("arxiv_client")
    if client is None or client.download_dir.resolve() != expected.resolve():
        st.session_state.arxiv_client = ArxivClient()
    else:
        client.refresh_download_dir()
    return st.session_state.arxiv_client


# 初始化 session_state
if "arxiv_client" not in st.session_state:
    st.session_state.arxiv_client = ArxivClient()
else:
    _get_arxiv_client()
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RetrievalPipeline()

# 当 API Embedding 开关变化时重建 Pipeline
_last_api_setting = st.session_state.get("_last_api_setting")
if _last_api_setting is not None and _last_api_setting != config.embedding.use_api:
    st.session_state.pipeline = RetrievalPipeline()
    st.session_state._model_loaded = False
    st.rerun()
st.session_state._last_api_setting = config.embedding.use_api
if "agent" not in st.session_state:
    st.session_state.agent = ArxivAgent(mode=agent_mode)
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []   # [{"role": "user"/"assistant", "content": str, "tables": [...]}]
if "conversation_memory" not in st.session_state:
    from src.memory import ConversationMemory
    st.session_state.conversation_memory = ConversationMemory()

# 预加载 Embedding 模型 + Reranker（启动时加载，用户点击按钮时模型已在内存）
# 避免首次查询需要等待 60+ 秒加载模型
if not st.session_state.get("_model_loaded"):
    with st.spinner("🚀 加载 AI 模型中（首次约 20 秒，后续秒回）..."):
        st.session_state.pipeline.load_index()
        if st.session_state.pipeline.vector_store.count > 0:
            _ = st.session_state.pipeline.embedder.encode(["warmup"])
        # 预热 Reranker（新模型 ~2s 加载，后续秒回）
        from src.retriever.reranker import _get_cached_cross_encoder
        _get_cached_cross_encoder(config.reranker.model_name)
        st.session_state._model_loaded = True


# ============================================================
# Tab 1: 论文搜索
# ============================================================
with tab1:
    st.header("🔍 搜索 ArXiv 论文")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "搜索关键词",
            placeholder="例如：transformer attention mechanism, LoRA fine-tuning, RLHF...",
        )
    with col2:
        search_btn = st.button("🔍 搜索", type="primary", use_container_width=True)

    st.caption("筛选条件（点击搜索时生效；分类可在左侧边栏设置）")
    filter_row1 = st.columns([1, 1, 1, 1])
    with filter_row1[0]:
        enable_date_filter = st.checkbox("按日期筛选", value=False)
    with filter_row1[1]:
        date_from = st.date_input(
            "起始日期",
            value=date.today() - timedelta(days=365),
            disabled=not enable_date_filter,
        )
    with filter_row1[2]:
        date_to = st.date_input(
            "结束日期",
            value=date.today(),
            disabled=not enable_date_filter,
        )
    with filter_row1[3]:
        date_field_label = st.selectbox(
            "日期依据",
            ["提交日期", "更新日期"],
            disabled=not enable_date_filter,
        )
    filter_row2 = st.columns([1, 1, 1])
    with filter_row2[0]:
        sort_by_label = st.selectbox(
            "排序方式",
            ["相关度", "提交日期", "更新日期"],
        )
    with filter_row2[1]:
        sort_order_label = st.selectbox("排序方向", ["降序", "升序"])
    with filter_row2[2]:
        search_max_results = st.number_input("最多返回", min_value=5, max_value=50, value=15, step=5)

    sort_by_map = {
        "相关度": "relevance",
        "提交日期": "submittedDate",
        "更新日期": "lastUpdatedDate",
    }
    date_field_map = {"提交日期": "submitted", "更新日期": "updated"}

    if search_btn and search_query:
        if enable_date_filter and date_from > date_to:
            st.error("起始日期不能晚于结束日期")
        else:
            with st.spinner(f"搜索中: {search_query}..."):
                client = _get_arxiv_client()
                papers = client.search(
                    search_query,
                    max_results=int(search_max_results),
                    categories=categories if categories else None,
                    sort_by=sort_by_map[sort_by_label],
                    sort_order="descending" if sort_order_label == "降序" else "ascending",
                    date_from=date_from if enable_date_filter else None,
                    date_to=date_to if enable_date_filter else None,
                    date_field=date_field_map[date_field_label],
                )

                # 标注哪些论文本地已存在
                for p in papers:
                    p._local_exists = client.resolve_local_pdf(p) is not None

                st.session_state.search_results = papers

            filter_hint = ""
            if enable_date_filter:
                filter_hint = f"（{date_from} ~ {date_to}，按{date_field_label}）"
            existing = sum(1 for p in papers if getattr(p, '_local_exists', False))
            st.success(f"找到 {len(papers)} 篇论文（{existing} 篇本地已有）{filter_hint}")

    # 显示搜索结果
    papers = st.session_state.search_results
    if papers:
        st.divider()

        # 批量下载按钮（跳过已有论文）
        need_download = [p for p in papers[:max_papers] if not getattr(p, '_local_exists', False)]
        skipped = len(papers[:max_papers]) - len(need_download)

        if skipped > 0:
            st.info(f"ℹ️ {skipped} 篇论文本地已存在，将跳过")

        if st.button("📥 下载全部并构建索引", type="primary"):
            if not need_download:
                st.info("所有论文已在本地，无需下载")
            else:
                progress = st.progress(0)
                status = st.empty()

                client = _get_arxiv_client()
                pipeline = st.session_state.pipeline

                pdf_paths = []
                paper_metas = []

                for i, paper in enumerate(need_download):
                    status.text(f"下载中 ({i+1}/{len(need_download)}): {paper.title[:60]}...")
                    pdf_path = client.download_pdf(paper)
                    if pdf_path:
                        pdf_paths.append(str(pdf_path))
                        paper_metas.append({
                            "arxiv_id": paper.arxiv_id,
                            "title": paper.title,
                            "authors": paper.authors,
                            "year": paper.year,
                            "abstract": paper.abstract,
                        })
                    progress.progress((i + 1) / max(1, len(need_download)))

                if pdf_paths:
                    status.text("构建索引中...")
                    pipeline.build_index(pdf_paths, paper_metas, rebuild=False)
                    status.text("")
                    st.success(f"✅ 已下载 {len(pdf_paths)} 篇论文并构建索引")
                    st.metric("Chunk 数", len(pipeline._all_chunks))
                else:
                    st.error("下载失败")

        for i, paper in enumerate(papers):
            pub_day = paper.published[:10] if paper.published else "未知"
            exists = getattr(paper, '_local_exists', False)
            badge = " ✅ 本地已有" if exists else ""
            with st.expander(f"📄 {i+1}. {paper.title} ({pub_day}){badge}", expanded=(i < 3)):
                if exists:
                    st.success("✅ 该论文已在本地论文库中，无需重复下载")
                st.markdown(f"**ArXiv ID:** `{paper.arxiv_id}`")
                st.markdown(f"**作者:** {', '.join(paper.authors[:5])}")
                st.markdown(f"**提交:** {pub_day} | **更新:** {paper.updated[:10] if paper.updated else '—'}")
                st.markdown(f"**分类:** {', '.join(paper.categories[:5])}")
                st.markdown(f"**摘要:** {paper.abstract[:500]}...")
                st.markdown(f"[📄 PDF]({paper.pdf_url})")

    # 本地论文状态（每次渲染重新扫描目录，避免 session 缓存旧路径）
    st.divider()
    st.subheader("本地论文库")
    papers_dir = get_papers_dir()
    st.caption(f"扫描目录: `{papers_dir}`")
    client = _get_arxiv_client()
    local = client.get_local_papers()
    if local:
        st.success(f"已发现 {len(local)} 篇本地 PDF")
        with st.expander("查看文件列表", expanded=len(local) <= 10):
            for f in local:
                st.text(f"  • {f.name}")
    else:
        st.warning("暂无本地论文（可将 PDF 放入上述目录）")

    if st.button("🔄 刷新本地列表"):
        st.rerun()

    if st.button("🔧 从本地 PDF 构建/更新索引"):
        pipeline = st.session_state.pipeline
        client = _get_arxiv_client()
        with st.spinner("构建索引中（首次可能较慢）..."):
            ok, msg = pipeline.ensure_index(client)
        if ok:
            st.success(msg or f"索引就绪，共 {pipeline.vector_store.count} 条向量片段")
            st.metric("Chunk 数", len(pipeline._all_chunks))
        else:
            st.warning(msg)


# ============================================================
# Tab 2: 文献综述
# ============================================================
with tab2:
    st.header("📚 文献综述生成")
    st.markdown("""
    Agent 会自动完成以下流程：
    1. 🔄 Query 改写（自然语言 → 学术搜索词）
    2. 🔍 搜索 ArXiv 相关论文
    3. 📥 下载最相关的论文
    4. 📖 解析并构建 RAG 索引
    5. 🔎 混合检索（向量 + BM25 + Rerank）
    6. ✍️ 生成结构化综述报告
    """)

    survey_topic = st.text_area(
        "研究主题",
        placeholder="例如：最新的大语言模型对齐技术进展、扩散模型在图像生成中的应用综述...",
        height=80,
    )

    if st.button("🚀 生成综述", type="primary") and survey_topic:
        agent = ArxivAgent(mode=agent_mode)

        with st.spinner("Agent 工作中..."):
            # 步骤可视化
            step_container = st.container()

            with step_container:
                st.info("🔄 Step 1/5: Query 改写...")

            result = agent.execute(survey_topic, max_papers=max_papers)

            # 显示步骤
            with step_container:
                st.success("✅ 完成！")
                for step in result["steps"]:
                    icon = "✅" if step["status"] == "done" else "❌"
                    detail = ""
                    if step.get("count"):
                        detail = f" ({step['count']} 条)"
                    elif step.get("result"):
                        detail = f" → {step['result']}"
                    st.text(f"  {icon} {step['step']}{detail}")

        # 显示结果
        st.divider()

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📄 综述报告")
        with col2:
            st.metric("引用论文", len(result["papers"]))

        st.markdown(result["answer"])

        with st.expander("📋 引用的论文列表"):
            for p in result["papers"]:
                st.markdown(f"- **{p['title']}** (`{p['arxiv_id']}`)")

        # 追踪
        tracker.trace_query(survey_topic, [], 0)


# ============================================================
# Tab 3: 论文问答（Messenger 聊天风格）
# ============================================================
with tab3:
    _qa_client = _get_arxiv_client()
    _local_count = len(_qa_client.get_local_papers())

    t3c1, t3c2, t3c3 = st.columns([3, 1, 1])
    with t3c1:
        st.header("💬 论文问答")
    with t3c2:
        st.caption(f"📄 {_local_count} 篇论文")
    with t3c3:
        if st.button("🗑️ 清除历史", use_container_width=True):
            st.session_state.chat_messages = []
            from src.memory import ConversationMemory
            st.session_state.conversation_memory = ConversationMemory()
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 显示表格 HTML 内容
            if msg.get("tables"):
                for ti, meta in enumerate(msg["tables"]):
                    html_raw = meta.get("full_html_content", "")
                    if html_raw:
                        html_clean = html_raw.replace("[TABLE_HTML:","").replace("[/TABLE_HTML]","").strip()
                        st.caption(f"📊 {meta.get('table_id',f'表格{ti+1}')}")
                        st.markdown(html_clean, unsafe_allow_html=True)

    question = st.chat_input("输入关于论文的问题…")

    if question:
        pipeline = st.session_state.pipeline
        client = _get_arxiv_client()

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            log = st.status("🚀 处理中…", expanded=False)

            log.write("📂 检查索引…")
            ok, msg = pipeline.ensure_index(client)
            if not ok:
                log.write(f"❌ {msg}"); log.update(state="error"); st.warning(msg); st.stop()
            log.write(f"✅ {pipeline.vector_store.count} 条向量")

            log.write("🔍 检索…"); log.update(label="检索中…")
            start = time.time()
            results = pipeline.query(
                question,
                top_k=config.retrieval.top_k_rerank if use_rerank else top_k,
                use_rerank=use_rerank, alpha=alpha, rewrite=use_rewrite,
                chat_history=st.session_state.chat_messages,
            )
            log.write(f"✅ {len(results)} 条, {(time.time()-start)*1000:.0f}ms")

            log.write("🤖 生成回答…"); log.update(label="生成中…")
            table_chunks = []; figure_chunks = []; answer = None

            # ── 额外搜图床：Figure/Table 原图匹配 ──
            try:
                import re
                fig_match = re.search(r'(?P<type>Figure|Fig\.?|Table|图|表)\s*(?P<num>\d+)', question, re.IGNORECASE)
                if fig_match:
                    t = fig_match.group("type").lower()
                    want_type = "table" if t.startswith("table") or t == "表" else "figure"
                    fig_num = fig_match.group("num")
                    from src.parser.figure_extractor import search_figures_fts, get_page_images, get_paper_figures
                    paper_src = ""
                    # 优先从向量搜索结果的 source 确定论文
                    for r in (results or []):
                        s = r.get("source","") or (r.get("metadata",{}) if isinstance(r,dict) else {}).get("source","")
                        if s and any(s == p.name for p in Path("data/papers").glob("*.pdf")):
                            paper_src = s
                            break
                    # 兜底：从问题中匹配 PDF 文件名（取命中词最多的）
                    if not paper_src:
                        best_score, best_match = 0, ""
                        for pdf_path in sorted(Path("data/papers").glob("*.pdf")):
                            name_key = pdf_path.stem.replace("_", " ").lower()
                            if name_key in question.lower():
                                best_score, best_match = 999, pdf_path.name; break
                            words = [w for w in name_key.split() if len(w) > 3]
                            score = sum(1 for w in words if w in question.lower())
                            if score > best_score:
                                best_score, best_match = score, pdf_path.name
                        if best_score >= 3:
                            paper_src = best_match
                        for r in (results or []):
                            s = r.get("source","") or (r.get("metadata",{}) if isinstance(r,dict) else {}).get("source","")
                            if s: paper_src = s; break

                    # 1. FTS5 全文搜图床（caption + page_text，支持 BM25 排序）
                    fig_results = search_figures_fts(f"Figure {fig_num}", paper_source=paper_src)
                    if not fig_results:
                        fig_results = search_figures_fts(f"Fig_{fig_num}", paper_source=paper_src)
                    # 按查询类型过滤（问 Figure 只显示 figure，问 Table 只显示 table）
                    if fig_results:
                        fig_results = [r for r in fig_results if r.get("figure_type", "") == want_type]

                    # 2. 找到了 → 关联该页 ±3 页的同类图 + 整页截图
                    if fig_results and paper_src:
                        pages_with_fig = set(r["page_no"] for r in fig_results if "page_no" in r)
                        expanded_pages = set()
                        for p in pages_with_fig:
                            for dp in range(-3, 4):
                                if p + dp >= 1: expanded_pages.add(p + dp)
                        for pg in sorted(expanded_pages):
                            all_on_page = get_page_images(paper_src, pg)
                            for img in all_on_page:
                                itype = img.get("figure_type", "")
                                # 只加同类 (figure/table) 和 page 截图
                                if itype in (want_type, "page") and \
                                   img["figure_id"] not in [f.get("figure_id","") for f in fig_results]:
                                    fig_results.append(img)
                        print(f"[图床] caption命中: Figure {fig_num} → p{list(pages_with_fig)} 展开 p{list(expanded_pages)} ({len(fig_results)} 张)")

                    # 3. 没搜到 caption → 搜所有页的整页截图（取文本含 Figure X 的页）
                    if not fig_results and paper_src:
                        import pickle
                        try:
                            with open('data/vector_store/metadata_papers.pkl','rb') as f:
                                ml = pickle.load(f)
                            target_pages = set()
                            for m in ml:
                                if m.get("source") == paper_src:
                                    txt = (m.get("text") or '') + (m.get("full_html_content") or '')
                                    if f"Figure {fig_num}" in txt or f"Fig. {fig_num}" in txt:
                                        target_pages.add(m.get("page_no", 0))
                            for pg in sorted(target_pages):
                                pg_imgs = get_page_images(paper_src, pg)
                                fig_results.extend(pg_imgs)
                        except Exception:
                            pass
                        if fig_results:
                            print(f"[图床] 文本定位: {fig_num} 在 p{sorted(target_pages)}")

                    # 4. 依然没找到 → 列出该论文所有独立图表作为兜底
                    if not fig_results and paper_src:
                        all_figs = get_paper_figures(paper_src)
                        fig_results = all_figs[:10]
                        print(f"[图床] 兜底: 展示该论文前 {len(fig_results)} 张图表")

                    if fig_results:
                        figure_chunks = [{
                            "source": r.get("paper_source", r.get("source", "")),
                            "figure_id": r.get("figure_id", ""),
                            "table_id": r.get("figure_id", ""),
                            "figure_type": r.get("figure_type", r.get("figure_type", "figure")),
                            "caption": r.get("caption", ""),
                            "full_html_content": "",
                            "image_blob": r.get("image", r.get("image_blob", b"")),
                        } for r in fig_results]
                        print(f"[图床] 最终: {len(figure_chunks)} 张")
            except Exception as e:
                print(f"[图床] 搜索失败: {e}")

            if results:
                from src.prompts import QA_SYSTEM_PROMPT, TABLE_SYSTEM_PROMPT
                from src.config import get_llm_client
                llm = get_llm_client()
                # 检测是否显式问章节（结论/方法/实验/Introduction/数字节号）→ 不显示表格
                import re as _re
                _asks_section = bool(_re.search(r'(?:Conclusion|Introduction|Method|Experiment|Related\s*Work|Background|Discussion)|\d+\.\s*(?:Conclusion|Introduction|Method)', question, _re.IGNORECASE))
                for r in results[:5]:
                    meta = r.get("metadata", {}) if isinstance(r, dict) else {}
                    if meta.get("is_table") and meta.get("full_html_content"):
                        if _asks_section:
                            pass  # 问章节时跳过表格
                        else:
                            table_chunks.append(meta)

                def _shorten(s):
                    name = s.replace('.pdf','').replace('_',' ')
                    for p in ['a ','an ','the ']:
                        if name.lower().startswith(p): name = name[len(p):]
                    words = name.split()
                    short = ' '.join(words[:4])
                    return short if len(short) <= 40 else ' '.join(words[:3])

                def _extract_text(r):
                    meta = r.get("metadata", {}) if isinstance(r, dict) else {}
                    full = meta.get("full_html_content", "")
                    return full[:2000] if full else (r.get("text","") if isinstance(r,dict) else "")[:1000]

                prompt = TABLE_SYSTEM_PROMPT if table_chunks else QA_SYSTEM_PROMPT
                ctx = "\n\n---\n\n".join([f"[{_shorten(r.get('source','?'))}]\n{_extract_text(r)}" for r in results[:5]])
                # 往 LLM 上下文注入图床命中图表的 caption（让 LLM 知道实际有什么图）
                if figure_chunks:
                    fig_notes = []
                    for fc in figure_chunks:
                        cap = fc.get("caption", "").strip()
                        fid = fc.get("figure_id", "")
                        if cap and fid:
                            fig_notes.append(f"【原图可用】{fid}: {cap}")
                    if fig_notes:
                        ctx += "\n\n" + "\n".join(fig_notes[:5])
                try:
                    hist = [{"role":"user" if m["role"]=="user" else "assistant","content":m["content"]}
                            for m in st.session_state.chat_messages[-10:]]
                    mem = st.session_state.conversation_memory.to_context_prompt()
                    mem = f"\n\n【历史上下文】\n{mem}" if mem else ""
                    # ── 构造用户消息（纯文本 or 图文混合） ──
                    text_part = f"问题：{question}\n\n论文内容：\n{ctx}{mem}"
                    is_vision = config.llm.provider == "kimi"
                    if is_vision and figure_chunks:
                        import base64
                        user_content = [{"type": "text", "text": text_part}]
                        for fc in figure_chunks[:3]:
                            blob = fc.get("image_blob")
                            if blob:
                                b64 = base64.b64encode(blob).decode()
                                user_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                })
                    else:
                        user_content = text_part

                    resp = llm.chat.completions.create(
                        model=config.llm.model,
                        messages=[{"role":"system","content":prompt}, *hist,
                                  {"role":"user","content":user_content}],
                        temperature=config.llm.temperature, max_tokens=2048)
                    answer = resp.choices[0].message.content
                except Exception as e:
                    answer = f"生成失败: {e}"

            if answer:
                log.update(label="✅ 完成", state="complete", expanded=False)
                st.markdown(answer)

                # 显示表格 HTML 内容
                for ti, meta in enumerate(table_chunks):
                    html_raw = meta.get("full_html_content", "")
                    if html_raw:
                        html_clean = html_raw.replace("[TABLE_HTML:","").replace("[/TABLE_HTML]","").strip()
                        st.caption(f"📊 {meta.get('table_id',f'表格{ti+1}')}")
                        st.markdown(html_clean, unsafe_allow_html=True)
            else:
                log.update(label="⚠️ 无结果", state="error"); answer = "未检索到相关内容。"; st.markdown(answer)

            st.session_state.chat_messages.append({"role":"user","content":question})
            st.session_state.chat_messages.append({"role":"assistant","content":answer,"tables":table_chunks if table_chunks else []})
            st.session_state.conversation_memory.update(question, answer)
            st.rerun()


# ============================================================
# Tab 4: 评估面板
# ============================================================
with tab4:
    st.header("📊 效果评估")

    summary = tracker.get_summary()

    if summary.get("total_queries", 0) == 0:
        st.info("尚未有查询记录。在「论文问答」Tab 中进行查询后，数据会显示在这里。")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总查询数", summary["total_queries"])
        col2.metric("平均延迟", f"{summary.get('avg_latency_ms', 0):.0f} ms")
        col3.metric("P95 延迟", f"{summary.get('p95_latency_ms', 0):.0f} ms")
        col4.metric("平均结果数", summary.get("avg_results", 0))

        st.divider()
        st.subheader("检索指标")

        metrics_keys = ["recall@5", "recall@10", "mrr@10", "ndcg@10"]
        if any(f"avg_{k}" in summary for k in metrics_keys):
            cols = st.columns(4)
            for i, k in enumerate(metrics_keys):
                val = summary.get(f"avg_{k}", "N/A")
                cols[i].metric(k, f"{val:.4f}" if isinstance(val, float) else val)
        else:
            st.info("指标计算需要标注的相关文档 ID（ground truth），可通过 evaluate 模块设置。")

    st.divider()
    st.subheader("Langfuse 集成")
    if config.evaluation.langfuse_enabled:
        st.success("✅ Langfuse 追踪已启用")
        st.markdown(f"- Public Key: `{config.evaluation.langfuse_public_key[:8]}...`")
        st.markdown("查看 [Langfuse Dashboard](https://cloud.langfuse.com) 获取完整追踪数据")
    else:
        st.info("配置 LANGFUSE_PUBLIC_KEY 和 LANGFUSE_SECRET_KEY 环境变量以启用 Langfuse")

    st.divider()
    st.subheader("技术栈一览")
    st.json({
        "RAG 链路": "PDF 解析 → 论文章节切片 → BGE-M3 向量化 → FAISS + BM25 混合检索 → BGE-Reranker 精排",
        "Agent 模式": ["深思熟虑（先规划再执行）", "反应式（即时决策）"],
        "MCP 工具": list(MCP_TOOLS.keys()),
        "评估指标": config.evaluation.metrics,
        "LLM": config.llm.model,
        "Embedding": config.embedding.model_name,
    })
