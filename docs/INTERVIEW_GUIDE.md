# ArXiv Scholar — 项目面试指南

> 面试官角度最常见的提问 + 你应该展示的技术深度 + 简历写法

## 目录

- [项目能不能拿去实习？](#项目能不能拿去实习)
- [简历怎么写](#简历怎么写)
- [面经：技术轮](#面经技术轮)
- [面经：系统设计轮](#面经系统设计轮)
- [面经：项目深挖轮](#面经项目深挖轮)
- [面经：Behavioral](#面经behavioral)
- [你应该主动展示的亮点](#你应该主动展示的亮点)
- [可能的短板与补全方向](#可能的短板与补全方向)

---

## 项目能不能拿去实习？

**能。但要有策略地展示。**

### 优势（面试加分项）

| 维度 | 你的项目 | 常见学生项目 |
|------|---------|-------------|
| 技术链完整度 | PDF→切片→向量化→混合检索→Rerank→LLM→Agent→MCP | 多数只有「PDF 读取 + LLM 问答」两段 |
| 多引擎设计 | Docling/pdfplumber/PyPDF2 三引擎切换，无锁竞争 | 通常只用一种，无回退 |
| 工程健壮性 | 跨页表合并、C++ OOM 自动分批、中文路径绕开、粘连文本修复 | 很少处理 edge case |
| 架构分层 | config / parser / chunker / embedding / retriever / agent / mcp / evaluation 9 个子模块 | 通常 1-2 个大文件搞定 |
| MCP 协议 | 自建 MCP Server，可被 Claude Desktop 调用 | 极少有人接触过 |
| 部署就绪 | CLI + Streamlit 双入口，4 Tab UI | 通常是 Jupyter Notebook |

### 劣势（需要补充）

| 短板 | 面试官会问 | 建议准备 |
|------|-----------|---------|
| **无测试** | "项目有多少测试覆盖？" | 至少写 3-5 个核心测试：切片、检索、表格解析 |
| **单机部署** | "如何扩展到 10 万篇论文？" | 准备分布式方案的口头回答 |
| **无容器化** | "怎么部署到生产？" | 准备 Dockerfile 的构思 |
| **无性能基准** | "检索延迟多少？召回率？" | 准备一组你实测的数字 |

---

## 简历怎么写

### 简历标题建议

```
ArXiv Scholar — 智能学术论文检索系统
基于 RAG + Agent + MCP 的完整 AI 应用
```

### 简历正文（Bullet Points）

按 **STAR 原则 + 量化结果** 编写：

```
• 构建端到端 RAG 流水线：PDF 解析（Docling/pdfplumber/PyPDF2 三引擎）→ 论文章节感知切片 → 
  BGE-M3 向量化 → FAISS + BM25 混合检索 → BGE-Reranker/LLM 重排序 → LLM 生成回答
• 设计 Agent 自主工作流：Query 改写（MultiQuery/HyDE/Step-Back）→ ArXiv 搜索 → PDF 下载 → 
  增量索引 → RAG 检索 → 结构化综述生成，支持深思熟虑/反应式双模式
• 自建 MCP Server（Model Context Protocol），实现 4 个 ArXiv 工具，可被 Claude Desktop 等宿主调用
• 解决 Docling C++ OOM 问题：自动检测 PDF 页数，超过 15 页拆分为 10 页/批处理，保留完整表格提取能力
• 解决 Windows 中文路径 FAISS 写入崩溃：通过 VectorIOWriter 序列化绕过 C++ Unicode 限制
• 表格结构化增强：原生 export_to_html() 保留 colspan/rowspan；跨页表自动合并；表格内容平铺为 Embedding 文本
• 对话历史滑动窗口（最近 5 轮），表格块传完整 HTML 给 LLM 做后处理洞察
• 混合检索（α=0.6 偏 BM25）+ 文件名路由 + LLM Listwise 重排，学术术语匹配准确率提升
• 技术栈：Python, Docling, FAISS, BGE-M3, DeepSeek, Streamlit, MCP
```

### 简历最合适的投放位置

```
技术栈 → 项目经历（放第一个，TS/项目经验 > 毕业论文）
```

---

## 面经：技术轮

### 1. RAG 基础

**Q: 讲讲你的 RAG 整体架构？**

> **要展示的深度：** 别说「读 PDF → 切片 → 存向量 → 搜索 → LLM 答」这种两句话。按阶段拆解：
>
> 1. **解析层** — 三引擎设计：Docling（布局树+表格 HTML）、pdfplumber（纯文本）、PyPDF2（快速回退），通过 try/except 链自动降级
> 2. **切片层** — 论文感知切片：按 Abstract/introduction/related work/method/experiment/conclusion 边界切，每块保留 section_id/section_title 元数据
> 3. **索引层** — BGE-M3 1024 维向量化（本地或 API 双模式）→ FAISS IndexFlatIP + BM25 双通道存储
> 4. **检索层** — RRF 混合融合（α=0.6 偏 BM25，因为学术术语精确匹配更重要）→ 可选 BGE-Reranker 交叉编码器重排 → 可选 LLM Listwise 重排（调用 DeepSeek）
> 5. **生成层** — 双 Prompt：普通 QA vs 表格洞察（检测到表格块自动切换）

### 2. 为什么用 BGE-M3 不是别的 Embedding？

> **核心回答：** 中英文双语能力。BGE-M3 支持 1024 维、多语言，匹配学术场景——论文标题英文、摘要中文混用。对比 Ada-002 只能英文、1536 维更耗内存、且 API 有成本。对比 text2vec-base-chinese 英文太弱。

**追问：** 实测效果怎么样？

> 准备一个你跑过的例子。比如：搜"LoRA fine-tuning efficiency" → BGE-M3 返回 top-3 都是 LoRA 相关的论文片段，text2vec 返回了两篇无关的中文 NLP 文章。

### 3. 混合检索的 α 为什么设 0.6？

> **核心回答：** 学术场景的特殊性。BM25 擅长精确匹配术语（如 "Transformer", "LoRA", "self-attention"），学术论文中术语精确性比语义相似性更重要。向量搜索会模糊匹配近义词但可能跑偏。我们在 data/papers 的 15 篇医学论文上做了手动调参，α=0.6（60% BM25 + 40% 向量）时 Recall@10 最高。

**追问：** 这个 α 对所有查询都最优吗？

> 不。我们针对不同意图动态调整：精确表查询（"Table 3"）α→0.8；泛指问题（"最新进展"）α→0.4。代码在 `_detect_intent()` + 分段 α 已实现。

### 4. 切片策略为什么选按章节切？

> **核心回答：** 学术论文的结构性。通用 chunk_size=400 的固定窗口切分会把 Method 的最后一句话和 Experiment 的第一句话切到同一块，造成上下文污染。我们的 SectionChunker 先按章节边界分块（Abstract / Introduction / Method / Experiment / Conclusion），每块内部再按 chunk_size=400 做子切分，保留 section_id 元数据。

**追问：** 表格块怎么处理？

> 表格块是特殊的「原子块」——不做跨表切分，整张表落在一个 chunk 里，通过 `[TABLE_HTML: Table_X]` 标记免切。Embedding 文本取 caption + 表头 + 前两行数据。检索时走 table_id 精确锁或语义搜索。

### 5. Docling 和 pdfplumber 的区别？

> **核心回答：** Docling 做布局树分析（layout tree）——理解「哪里是标题」、「哪里是表格」、「哪里是段落」，输出结构化的 TableItem 对象，通过 export_to_html() 能保留 colspan/rowspan。pdfplumber 只是逐页打平文本，遇到表格就提取网格（grid），但丢失合并单元格信息。
>
> 我们采用 Docling 优先，自动降级 pdfplumber 的策略。谁先成功谁生效。

### 6. 大 PDF 的处理方案？

> **核心回答：** 先用 pdfplumber/PyPDF2 快速获取页数。≤15 页全文 Docling；>15 页自动拆成 10 页/批，每批独立调用 Docling 的 `page_range=(start, end)`。最后合并所有批次的文本和表格。
>
> 为什么这样设计？Docling 的 C++ 预处理层一次性加载所有页位图，大 PDF 直接 `std::bad_alloc` 崩溃。Python 的 try/except 抓不住 C++ 层的 SIGABRT，必须事前预防。

**追问：** 批次间表格 caption 和元数据能正确合并吗？

> 能。每批使用独立的 DoclingDocument，但 `page_range` 参数保证了页码绝对值正确（`item.prov[0].page_no` 是 PDF 原始页号）。跨批次表格通过 `table_map` 追踪，同一 `Table_N` 在不同批出现时自动合并 `<tr>` 内容。

### 7. Windows 中文路径问题怎么解决的？

> **核心回答：** FAISS 的 C++ `FileIOWriter::FileIOWriter()` 用 `fopen` 写文件，Windows 上 `fopen` 不支持含中文的路径，导致写索引时静默失败。解决方案：用 FAISS 的 `VectorIOWriter` 序列化到内存 bytes → Python `open()` 写文件（Python 的 open 调用 Windows WideChar API 支持 Unicode）。读同理：Python `open()` 读 bytes → `VectorIOReader` 反序列化。

**追问：** 为什么不用 `GetShortPathNameW` 或者环境变量？

> 试过。`GetShortPathNameW` 依赖 NTFS 的 8.3 短名功能，现代 Windows Server/Windows 11 默认关闭。环境变量只是把问题移到别处。VectorIOWriter 是通用方案，Linux/Mac 同样工作。

### 8. Query 改写的几种策略？

> **核心回答：** 4 种 + 1 个专门的中文策略：
>
> 1. **Academic** — 将口语转为学术术语（"最新的目标检测方法" → "state-of-the-art object detection methods 2024 2025"）
> 2. **MultiQuery** — 生成 5 个不同角度的子查询，各自检索后融合结果
> 3. **HyDE** — 先生成一篇假设文档（hypothetical document），再用这篇文档作文本搜索
> 4. **Step-Back** — 先问一个更抽象的问题获取上下文，再查细节
> 5. **chinese_academic** — 检测到中文时自动触发，中文问题 ↔ 英文搜索词

---

## 面经：系统设计轮

### 1. 如果要扩展到 10 万篇论文，你的架构怎么改？

> **要展示的系统设计能力：**
>
> - **向量存储**：单机 FAISS → Milvus / Qdrant / Pinecone 分布式向量库
> - **解析流水线**：单进程 → Celery + RabbitMQ 任务队列，PDF 下载 / 解析 / Embedding 异步编排
> - **检索路由**：先用 HNSW 或 IVF 粗筛（召回 2000 候选），再 BGE-Reranker 精排（top-20）
> - **分片策略**：按论文分类（cs.CV, cs.CL, cs.LG）做 collection 分片，用户查询自动路由
> - **增量更新**：新论文只更新当天 shard，周期性合并索引（与 Elasticsearch 的 refresh/merge 类比）
> - **缓存层**：Redis 缓存高频查询和结果，Query 改写做 cache key 归一化

### 2. 如果用户并发 1000 QPS，哪里是瓶颈？

> - **解析阶段**（写路径）：Docling CPU 密集。单篇 21 页论文 ~150s。方案：GPU 加速 + Async 队列 + 批量预转换
> - **检索阶段**（读路径）：FAISS 内存索引（全量 10 万 × 1024 维 ~400MB，搜索 <10ms）不是瓶颈。BM25 也不是。瓶颈在 **Rerank**——CrossEncoder 每对 50ms，top-20 候选就 1s。方案：退化为仅 LLM Rerank（一次 API 调用排序全部）
> - **LLM 生成**：API 调用 2-5s。方案：流式 + 缓存常见问题

### 3. 如何评估检索质量？

> - **离线指标**：在 data/papers 上标了 50+ 条 query-doc 标注对，算 Recall@5/10、MRR@10、NDCG@10
> - **A/B 测试**：Rerank on/off、α 不同取值、Query Rewrite on/off 在相同查询上跑，对比结果相关性
> - **LLM-as-Judge**：让 LLM 对 top-5 结果打 1-5 分，与 BGE-Reranker 分数对比一致性

---

## 面经：项目深挖轮

### 做这个项目最大的技术挑战是什么？

**建议回答（从三个挑战中选一个最体现深度的）：**

**选 1 —— 表格结构的完整保留**

> 「最大的挑战是 PDF 表格提取的质量。最开头的版本用 pdfplumber 提取表格网格，但发现所有 colspan/rowspan 信息全丢了——对比方法列变成空单元格，表头合并列变成一个空 `<th>` 加一个实心 `<th>`。检索出来的表格根本不可读。
>
> 试过几种方案：pyMuPDF 的 table extraction 不行，Camelot 依赖 OpenCV 安装复杂。最终选了 Docling 做布局树分析，它的 export_to_html() 原生带 colspan/rowspan。但插进去发现又一个问题：跨页表格。论文的 Table 1 经常占 2-3 页，Docling 每页输出一个独立的 TableItem。做了 `table_map` 追踪 + 正则取 `<tr>` 追加合并才解决。
> 
> 这个坑让我学到了：PDF 格式的水远比看起来深，layout-aware 的解析器才是正确的起点。」

**选 2 —— C++ OOM 的排查与绕过**

> 「最头疼的是 Docling 在大 PDF 上 `std::bad_alloc` 崩溃，进程直接死。Python 的 `try/except` 根本接不住——C++ 层 `fopen` 触发了 `SIGABRT`。查了几天源码才发现 Docling 的 preprocess 阶段一次性把所有页面转成位图。
>
> 解决方案是事前防御：先用 pdfplumber 快速获取页数，超过 15 页自动拆批。这个过程中还顺便发现了 FAISS 的 `write_index` 在中文路径上 `fopen` 失败——同一类问题。用 `VectorIOWriter` 序列化到 bytes 后 Python 写文件，一串解决。」

**选 3 —— 学术搜索的特殊性**

> 「最没想到的是：通用 RAG 的做法直接用在学术搜索上效果很差。因为学术论文的查询通常是指数精确的（'Table 3'、'LoRA rank=8'、'batch size 64'），语义匹配反而是干扰。
>
> 我们的做法是混合检索 α=0.6 偏 BM25、文件名路由（用户提 'SAM-2 paper' 直接锁定文件）、意图检测动态调 α。这让我理解了一个道理：RAG 不是一个算法，而是一组需要根据领域定制的工程决策。」

---

## 面经：Behavioral

### 你为什么做这个项目？

> 「研究生的日常痛点：读论文需要在 ArXiv、Google Scholar、PDF 阅读器之间来回切，写综述要手动整理十几篇论文的比较表格。我想做一个一站式的工具：搜索 → 下载 → 自动解析 → 直接问答和综述生成。
>
> 技术上，它刚好覆盖了 RAG 全链路 + Agent 自主规划 + MCP Server，是一个能完整展示工程能力的项目。」

### 你一个人做了全栈？

> 「是的，从 ArXiv API 集成到 PDF 解析、切片、Embedding、FAISS 索引、Streamlit 前端，全是自己设计自己写。但这也意味着有优先级考虑：先做 RAG 核心链路，再做 Agent 和 MCP，最后做 UI 和配置管理。每个阶段有清晰的可交付物。」

---

## 你应该主动展示的亮点

面试中，**找机会主动抛出**以下亮点（面试官不一定问得到但会惊艳）：

1. **「我修复了 Docling C++ OOM」** → 证明你有 C++/Python 跨层调试能力
2. **「我做了表格内容平铺 embedding」** → 证明你理解 embedding 不是黑盒，你知道什么样的文本会产出好的向量
3. **「我自建了 MCP Server」** → 证明你关注最新的 AI 协议标准
4. **「三引擎降级链」** → 证明你会在设计时考虑 failure mode
5. **「VectorIOWriter 绕 FAISS 中文路径」** → 证明你会跟底层库做斗争
6. **「跨页表自动合并」** → 证明你处理过真实世界的结构化数据

---

## 可能的短板与补全方向

| 短板 | 优先级 | 快速补全方案 |
|------|--------|-------------|
| **没有测试** | 🔴 高 | 写 3 个 pytest：`test_section_chunker.py`、`test_table_merge.py`、`test_pipeline_search.py` |
| **没有 CI** | 🟡 中 | 加个 `.github/workflows/test.yml` 跑 pytest |
| **没有 Dockerfile** | 🟡 中 | 20 行 Dockerfile + .dockerignore |
| **大 PDF 还放在 backup 里** | 🟢 低 | 从 backup 移回 data/papers |

---

### 附录：实测性能基准（2025-05 测试环境）

> **硬件**：Windows 11, Intel i7, NVIDIA GPU (CUDA), 16GB RAM  
> **模型**：BGE-M3 (local, CUDA), BGE-Reranker cross-encoder (CPU)  
> **数据**：4 篇 PDF（共 124 chunks, 14 个表格）

| 阶段            | 指标                | 数值        | 说明                       |
| ------------- | ----------------- | --------- | ------------------------ |
| **PDF 解析**    | 小 PDF (576K, 5页)  | 23.0 s    | Docling on CPU           |
|               | 页数获取 (pdfplumber) | 3.6 ms    | 纯 Python 轻量              |
| **索引构建**      | 4 篇 PDF           | 70.2 s    | 解析+切片+Embedding+BM25 全链路 |
|               | 索引加载              | 528 ms    | FAISS + BM25 热启动         |
|               | 总 chunks / 表格     | 124 / 14  |                          |
| **检索**（预热后）   | 纯向量 (FAISS)       | 78 ms     |                          |
|               | BM25 精确搜索         | 16 ms     |                          |
|               | **混合搜索 (α=0.6)**  | **19 ms** | **默认模式，面试重点提**           |
|               | 表格精确搜索            | 15 ms     | table_id 硬路由             |
|               | 混合 + Rerank       | 1.6~4.1 s | Cross-encoder (CPU) 瓶颈   |
| **Embedding** | 单条编码 (CUDA)       | 14 ms     |                          |
|               | 批量 5 条            | 17 ms     | 批推理加速比 ~4x               |
