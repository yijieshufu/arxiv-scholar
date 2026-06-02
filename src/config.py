"""
ArXiv Scholar — 全局配置管理
参考 paper-qa 的参数设计 + resume-ai 的模块化结构
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal, Union
from dotenv import load_dotenv

load_dotenv()

# 项目根目录（src/ 的上一级），用于解析 ./data 等相对路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 所有运行时数据默认位于 {PROJECT_ROOT}/data/（可用 ARXIV_DATA_DIR 覆盖根目录）
_DEFAULT_DATA_DIR = os.getenv("ARXIV_DATA_DIR", "./data")


def resolve_project_path(path: Union[str, Path]) -> Path:
    """将相对路径解析为基于项目根目录的绝对路径（不依赖进程 CWD）。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def get_data_dir() -> Path:
    """项目数据根目录（绝对路径）：默认 {PROJECT_ROOT}/data。"""
    return resolve_project_path(_DEFAULT_DATA_DIR)


def get_papers_dir() -> Path:
    """解析当前配置的本地 PDF 目录（绝对路径）。"""
    override = os.getenv("ARXIV_DOWNLOAD_DIR")
    if override:
        return resolve_project_path(override)
    return get_data_dir() / "papers"


def get_vector_store_dir() -> Path:
    """FAISS / BM25 持久化目录（绝对路径）。

    Windows 上 FAISS C++ 层不支持含中文的路径，
    因此自动使用 Windows 短路径（8.3 格式）来绕开此限制。
    """
    override = os.getenv("ARXIV_VECTOR_STORE_DIR")
    path = resolve_project_path(override) if override else (get_data_dir() / "vector_store")

    # Windows: 获取短路径（绕开 FAISS C 层对 Unicode 路径的限制）
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            buf = ctypes.create_unicode_buffer(260)
            ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 260)
            short_path = buf.value
            if short_path:
                path = Path(short_path)
        except Exception:
            pass

    return path


def ensure_data_layout() -> None:
    """创建 data/papers 与 data/vector_store（若不存在）。"""
    get_papers_dir().mkdir(parents=True, exist_ok=True)
    get_vector_store_dir().mkdir(parents=True, exist_ok=True)


@dataclass
class ArxivConfig:
    """ArXiv 数据获取配置"""
    max_results: int = 20              # 单次搜索最多返回论文数
    download_dir: str = field(
        default_factory=lambda: os.getenv(
            "ARXIV_DOWNLOAD_DIR",
            str(Path(_DEFAULT_DATA_DIR) / "papers"),
        )
    )  # PDF 下载目录（相对路径基于 PROJECT_ROOT）
    categories: List[str] = field(default_factory=lambda: [
        "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.MA"
    ])
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance"
    # 参考 paper-qa：支持 Semantic Scholar 作为补充数据源
    use_semantic_scholar: bool = False
    semantic_scholar_api_key: str = ""


@dataclass
class ParserConfig:
    """文档解析配置 — 参考 paper-qa 使用 pdfplumber + PyPDF2 双引擎"""
    pdf_engine: Literal["docling", "mineru", "pypdf2", "pdfplumber", "auto"] = "mineru"
    mineru_upload_url: str = field(default_factory=lambda: os.getenv("MINERU_UPLOAD_URL", ""))
    mineru_api_key: str = field(default_factory=lambda: os.getenv("MINERU_API_KEY", ""))
    table_format: Literal["html", "markdown"] = "html"
    max_pages: int = 50                # 论文通常 < 50 页
    extract_references: bool = True     # 提取参考文献


@dataclass
class ChunkerConfig:
    """切片策略 — 论文感知：按章节边界切分"""
    strategy: Literal["semantic", "sentence", "section", "hierarchical"] = "section"
    chunk_size: int = 512              # 学术段落长，512 tokens 更完整
    chunk_overlap: int = 128            # 25% 重叠，避免关键句被切分到两个 chunk
    min_chunk_size: int = 100
    # 论文章节标记
    section_headers: List[str] = field(default_factory=lambda: [
        "abstract", "introduction", "related work", "method",
        "experiment", "result", "discussion", "conclusion",
        "reference", "appendix"
    ])
    # 层次切片（父 chunk = 整个章节）
    parent_chunk_size: int = 2000


@dataclass
class EmbeddingConfig:
    """Embedding 配置 — BGE-M3 多语言优势"""
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"
    batch_size: int = 32
    dimension: int = 1024
    normalize: bool = True
    use_api: bool = False                          # True = 用线上 API，False = 本地模型
    api_model: str = "deepseek-embedding-v2"       # DeepSeek Embedding 模型名


@dataclass
class VectorStoreConfig:
    """FAISS 向量库"""
    index_type: Literal["FlatIP", "IVF"] = "FlatIP"
    nlist: int = 100
    metric: Literal["ip", "l2"] = "ip"
    persist_dir: str = field(
        default_factory=lambda: os.getenv(
            "ARXIV_VECTOR_STORE_DIR",
            str(Path(_DEFAULT_DATA_DIR) / "vector_store"),
        )
    )


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k_vector: int = 30
    top_k_bm25: int = 30
    top_k_hybrid: int = 20
    top_k_rerank: int = 10
    alpha: float = 0.6                # 学术场景 BM25 更重要（术语精确匹配）
    use_parent_retrieval: bool = True
    max_parent_pages: int = 3


@dataclass
class RerankerConfig:
    """Rerank 配置"""
    engine: Literal["bge_reranker", "llm_rerank"] = "bge_reranker"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # 放大模型（质量更高但更慢）："BAAI/bge-reranker-v2-m3"（~1.5GB, ~15s CPU）
    llm_model: str = "qwen-max"
    llm_score_weight: float = 0.7
    vector_score_weight: float = 0.3


@dataclass
class LLMConfig:
    """LLM 配置 — 支持 DeepSeek / Ollama / DashScope / OpenAI"""
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat"))
    api_key: str = field(default_factory=lambda: (
        os.getenv("KIMI_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
    ))
    base_url: str = field(default_factory=lambda: os.getenv(
        "LLM_BASE_URL", "https://api.deepseek.com/v1"
    ))
    temperature: float = field(default_factory=lambda: (
        1.0 if os.getenv("LLM_PROVIDER") == "kimi" else 0.1
    ))
    max_tokens: int = 4096
    timeout: int = 120


@dataclass
class AgentConfig:
    """Agent 配置 — ArXiv 学术助手专用工具"""
    mode: Literal["reactive", "deliberative"] = "deliberative"
    max_iterations: int = 15
    tools: List[str] = field(default_factory=lambda: [
        "search_papers",        # 搜索论文（ArXiv API）
        "download_paper",       # 下载论文 PDF
        "parse_paper",          # 解析论文内容
        "rag_query",            # RAG 知识库检索
        "compare_papers",       # 多论文对比分析
        "generate_survey",      # 生成文献综述
        "find_related_work",    # 找相关工作
        "rewrite_query",        # Query 改写
    ])


@dataclass
class EvaluationConfig:
    """评估配置"""
    langfuse_enabled: bool = True
    langfuse_public_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    langfuse_secret_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    metrics: List[str] = field(default_factory=lambda: [
        "recall@k", "mrr", "ndcg", "precision@k", "latency_ms"
    ])


@dataclass
class AppConfig:
    """全局配置聚合"""
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data_dir: str = field(default_factory=lambda: _DEFAULT_DATA_DIR)
    debug: bool = False


config = AppConfig()


def get_llm_client():
    """
    LLM 客户端工厂 — 统一创建 OpenAI 兼容客户端。

    支持：
    - DeepSeek:   base_url=https://api.deepseek.com/v1  model=deepseek-chat
    - Ollama:     base_url=http://localhost:11434/v1     model=qwen2.5:7b
    - DashScope:  base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
    - OpenAI:     base_url=https://api.openai.com/v1
    - Kimi:       base_url=https://api.moonshot.cn/v1  model=kimi-k2.5
    """
    base_url = config.llm.base_url
    api_key = config.llm.api_key

    # 未显式设置时，按 provider 给默认值
    if config.llm.provider == "kimi":
        base_url = "https://api.moonshot.cn/v1"
        api_key = api_key or os.getenv("KIMI_API_KEY", "")
    elif config.llm.provider == "deepseek":
        base_url = base_url or "https://api.deepseek.com/v1"
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")

    from openai import OpenAI
    return OpenAI(
        api_key=api_key or "sk-placeholder",
        base_url=base_url,
        timeout=config.llm.timeout,
    )
