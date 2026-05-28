# ArXiv Scholar — 智能学术论文助手

> 融合 **RAG + Agent + MCP** 的完整 AI 应用，用于 ArXiv 学术论文搜索、文献综述生成与论文问答。

## 为什么选这个项目？

- ✅ **数据完全公开**：ArXiv 是全球最大的免费学术预印本库，250 万+ 论文，API 免费无限制
- ✅ **真实需求**：研究人员/研究生每天都需要查论文、写综述、对比方法
- ✅ **完整技术链**：从 ArXiv API → PDF 解析 → 章节切片 → 向量化 → 混合检索 → Rerank → Agent 自主规划 → LLM 生成
- ✅ **面试加分**：AI + 科研场景，深度和广度都远超普通 CRUD 项目

## 参考开源项目

| 项目 | 借鉴了什么 | 我们的改进 |
|------|-----------|-----------|
| [paper-qa](https://github.com/Future-House/paper-qa) | 论文全文 RAG + 引用追踪 | 加 Agent 规划、MCP Server、Streamlit 界面 |
| [arxiv-mcp-server](https://github.com/anthropics/anthropic-cookbook) | ArXiv API MCP 封装 | 加混合检索、Rerank、Query 改写 |
| LangChain ArxivLoader | 论文搜索下载接口 | 加论文感知切片 + 章节元数据 |

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 数据源 | ArXiv API (`arxiv` 包) | 搜索、下载论文，免费无限制 |
| 文档解析 | pdfplumber / PyPDF2 / Docling | 三引擎可切换，pdfplumber 支持表格提取 |
| 切片策略 | 论文感知章节切片 | 按 Abstract→Introduction→Method→Result→Conclusion 边界切分 |
| 向量化 | BGE-M3 | 1024 维，支持中英文 |
| 向量存储 | FAISS (IndexFlatIP) | 内积索引 |
| 检索引擎 | 向量 + BM25 混合融合 | α 可调，学术场景 BM25 权重偏高 |
| 重排序 | BGE-Reranker / LLM Rerank | 交叉编码器 + LLM 打分 |
| Query 改写 | MultiQuery / HyDE / Step-Back | 三种策略可切换 |
| Agent | 深思熟虑 + 反应式双模式 | 自主规划工具调用链 |
| MCP | 自建 ArXiv MCP Server | 4 个工具：search/download/get_info/list_local |
| 评估 | Langfuse + 自定义指标 | Recall@K, MRR, NDCG, Latency |
| 界面 | Streamlit + CLI | 双入口：Web 界面 + 命令行 |

## 项目架构

```
arxiv-scholar/
├── src/
│   ├── arxiv_client.py      # ArXiv API 客户端（搜索/下载/元数据）
│   ├── query_rewriter.py    # Query 改写（学术术语/MultiQuery/HyDE/Step-Back）
│   ├── config.py            # 全局配置
│   ├── cli.py               # CLI 入口
│   ├── parser/              # 论文 PDF 解析器
│   │   └── paper_parser.py  # pdfplumber/PyPDF2/Docling 三引擎
│   ├── chunker/             # 切片策略
│   │   └── section_chunker.py  # 论文感知章节切片
│   ├── embedding/           # Embedding 模型
│   │   └── __init__.py      # BGE-M3 / Sentence-Transformers
│   ├── retriever/           # 检索引擎
│   │   ├── vector_store.py  # FAISS 向量库
│   │   ├── bm25.py          # BM25 关键词检索
│   │   ├── reranker.py      # BGE-Reranker / LLM Rerank
│   │   └── pipeline.py      # 检索流水线（混合检索 + Rerank）
│   ├── agent/               # Agent 主控
│   │   ├── arxiv_agent.py   # 深思熟虑 + 反应式双模式
│   │   └── tools.py         # 4 个工具函数
│   ├── mcp/                 # MCP Server
│   │   └── arxiv_mcp.py     # ArXiv MCP 工具封装
│   ├── evaluation/          # 效果评估
│   │   └── __init__.py      # Langfuse 追踪 + 指标计算
│   └── prompts/             # 提示词模板
│       └── prompts.py       # 综述/对比/问答 Prompt
├── app.py                   # Streamlit 主界面
├── requirements.txt
├── .env.example
├── tests/
├── data/                    # 所有运行时数据（固定于项目根，不依赖启动目录）
│   ├── papers/              # 下载的论文 PDF + manifest.json
│   └── vector_store/        # FAISS + BM25 索引持久化
└── README.md
```

## 核心流水线

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 搜索论文  │ →  │ 下载 PDF │ →  │ 论文解析  │ →  │ 章节切片  │ →  │ 向量化   │
│ ArXiv API│    │  本地缓存│    │ pdfplumber│    │ Section   │    │ BGE-M3  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                      │
                                                                      ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ LLM 生成  │ ←  │ 混合检索  │ ←  │  Rerank  │ ←  │ FAISS入库│ ←  │ BM25索引│
│ 综述/问答 │    │ α 融合   │    │ 交叉编码器│    │ + 元数据  │    │ 关键词   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## Agent 工作流

```
用户提问（"LLM 对齐技术最新进展"）
        │
        ▼
┌──────────────────┐
│ Query 改写        │ ← 自然语言 → 学术搜索词
│ "LLM alignment   │
│  techniques      │
│  RLHF DPO survey"│
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 搜索 ArXiv        │ ← MCP search_papers
│ 找到 15 篇论文    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 下载 + 入库       │ ← MCP download_paper × 5
│ PDF → 解析 → 索引 │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ RAG 混合检索      │ ← 向量 + BM25 + Rerank
│ Top-K 最相关片段  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ LLM 生成综述      │ ← 结构化 Markdown
│ 论文分析 + 对比   │
│ + 趋势 + 展望     │
└──────────────────┘
```

## 快速开始

```bash
# 1. 克隆项目
cd RAG/projects/arxiv-scholar

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（复制 .env.example 为 .env，填入 API Key）
# DASHSCOPE_API_KEY=sk-xxx  （通义千问）
# 或者 OPENAI_API_KEY=sk-xxx  （OpenAI）

# 4. 数据目录（默认 {项目根}/data/，可用环境变量覆盖）
# ARXIV_DATA_DIR=./data
# ARXIV_DOWNLOAD_DIR=  （默认 data/papers）
# ARXIV_VECTOR_STORE_DIR=  （默认 data/vector_store）
# 若曾在错误目录下载过 PDF，可执行：
# python -m src.cli migrate-data

# 5. 运行 Streamlit 应用
streamlit run app.py

# 6. 或使用命令行
# 搜索论文
python -m src.cli search "transformer attention mechanism" --max 5

# 搜索并下载
python -m src.cli search "LoRA fine-tuning" --max 5 --download

# 构建索引
python -m src.cli build-index --data-dir ./data/papers

# RAG 查询
python -m src.cli query "What is the difference between LoRA and QLoRA?"

# 生成文献综述
python -m src.cli survey "recent advances in LLM alignment"

# 快速问答
python -m src.cli ask "How does RLHF work?"

# 启动 MCP Server（供其他 Agent 调用）
python -m src.cli mcp-server
```

## 使用流程（Web 界面）

### Tab 1: 论文搜索
1. 输入搜索关键词（如 "transformer attention mechanism"）
2. 浏览搜索结果（标题、作者、摘要、年份）
3. 点击「下载全部并构建索引」→ 自动下载 PDF → 解析 → 切片 → 向量化 → 入库

### Tab 2: 文献综述
1. 输入研究主题（如 "最新的大语言模型对齐技术进展"）
2. 点击「生成综述」
3. Agent 自动执行：Query 改写 → 搜索 → 下载 → RAG → 生成结构化综述
4. 获得 Markdown 格式的综述报告 + 引用论文列表

### Tab 3: 论文问答
1. 输入具体问题（如 "LoRA 和 QLoRA 的主要区别是什么？"）
2. 查看 RAG 检索到的论文片段（含分数、来源、章节）
3. 查看 LLM 基于检索片段生成的回答

### Tab 4: 评估面板
- 查看检索指标（平均延迟、P95、Recall@K、MRR、NDCG）
- Langfuse 集成状态

## 面试亮点

本项目覆盖以下 AI Agent 面试核心考点：

| 考点 | 本项目体现 |
|------|-----------|
| **RAG 全链路** | "ArXiv API → PDF 解析 → 章节切片 → Embedding → FAISS + BM25 → Rerank → LLM 生成"，每个环节都能展开讲设计理由 |
| **混合检索 + α 调参** | 向量语义检索 + BM25 关键词检索双路融合，学术场景 α=0.6（偏 BM25），解释了"为什么学术场景 BM25 更重要" |
| **Query 改写** | 3 种策略（学术术语标准化 / MultiQuery / HyDE）+ Step-Back，覆盖课程第 13 章 + 第 15 章 |
| **论文感知切片** | 按 Abstract/Introduction/Method/Result/Conclusion 切分，不是简单固定长度，保留章节元数据 |
| **Agent 双模式** | 深思熟虑（先 LLM 规划再执行）vs 反应式（即时工具调用），覆盖课程第 19 章 |
| **MCP 协议** | 自建 ArXiv MCP Server，4 个工具，支持 stdio 模式可被其他 Agent 调用，覆盖课程第 18 章 |
| **效果评估** | Langfuse 追踪 + Recall@K/MRR/NDCG/Latency，覆盖课程第 20 章 |
| **工程化** | 模块化解耦、Pydantic Schema、提示词集中管理、CLI + Web 双入口 |

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 是* | 通义千问 API Key（推荐） |
| `OPENAI_API_KEY` | 是* | OpenAI API Key |
| `OPENAI_BASE_URL` | 否 | 自定义 API 地址 |
| `LANGFUSE_PUBLIC_KEY` | 否 | Langfuse 追踪公钥 |
| `LANGFUSE_SECRET_KEY` | 否 | Langfuse 追踪私钥 |

*二选一即可

## License

MIT
"# arxiv-scholar" 
