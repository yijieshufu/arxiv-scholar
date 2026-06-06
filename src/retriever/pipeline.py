"""
检索流水线 — ArXiv Scholar 完整 RAG 链路

参考 paper-qa 的设计：
1. 搜索论文 → 下载 PDF → 解析 → 切片 → 向量化 → 入库
2. 查询时：Query 改写 → 混合检索 → Rerank → 生成回答
"""
import logging
import math
import re
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Set, Tuple

from src.config import config, get_vector_store_dir, resolve_project_path
from src.arxiv_client import ArxivClient
from src.parser import PaperParser
from src.chunker import SectionChunker
from src.embedding import Embedder, create_embedder
from src.retriever.vector_store import VectorStore
from src.retriever.bm25 import BM25Retriever
from src.retriever.reranker import get_reranker
from src.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


def _parse_and_chunk_paper(
    path: str, engine: str,
    chunk_size: int, chunk_overlap: int, min_chunk_size: int,
    meta: Dict,
) -> tuple | None:
    """
    Worker 函数（被 ThreadPoolExecutor 调用）。
    解析一篇 PDF → 切片 → 返回 (paper_info, [Chunk])。
    chunk_id 由主进程统一分配，此处不赋值。
    """
    source_name = Path(path).name
    try:
        parser = PaperParser(engine=engine)
        doc = parser.parse(path)

        paper_info = {
            "paper_title": doc.get("title", meta.get("title", "")),
            "arxiv_id": meta.get("arxiv_id", Path(path).stem),
            "source": source_name,
            "authors": meta.get("authors", []),
            "year": meta.get("year", ""),
            "abstract": doc.get("abstract", meta.get("abstract", "")),
        }

        chunker = SectionChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunker.min_chunk_size = min_chunk_size

        chunks = chunker.chunk(doc["full_text"], paper_info)
        return (paper_info, chunks)
    except Exception as e:
        logger.error(f"Worker 解析失败 {source_name}: {e}")
        return None


class RetrievalPipeline:
    """ArXiv 论文检索流水线。"""

    # 单条送入 Embedding 模型的最大字符数（防止异常 PDF 块触发底层崩溃）
    _MAX_EMBED_CHARS = 4096

    # 使用方式:
    #   pipeline = RetrievalPipeline()
    #   pipeline.build_index(pdf_paths, paper_metas)
    #   results = pipeline.query("Transformer attention mechanism")

    def __init__(self):
        self.parser = PaperParser(engine=config.parser.pdf_engine)
        self.chunker = SectionChunker()
        self.embedder = create_embedder()
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()
        self.reranker = None
        self.rewriter = QueryRewriter()
        self._all_chunks = []
        self._paper_metas = []

    @classmethod
    def for_indexing(cls) -> "RetrievalPipeline":
        """轻量实例：仅用于建索引，避免加载 QueryRewriter / 重复 Parser。"""
        pipeline = cls.__new__(cls)
        pipeline.parser = None
        pipeline.chunker = None
        pipeline.embedder = create_embedder()
        pipeline.vector_store = VectorStore()
        pipeline.bm25 = BM25Retriever()
        pipeline.reranker = None
        pipeline.rewriter = None
        pipeline._all_chunks = []
        pipeline._paper_metas = []
        return pipeline

    def load_index(self) -> bool:
        """从磁盘加载已有索引"""
        bm25_path = get_vector_store_dir() / "bm25.pkl"
        if not self.vector_store.load():
            return False
        if bm25_path.exists():
            self.bm25.load(str(bm25_path))
        self._all_chunks = list(self.bm25._chunks)
        logger.info(f"从磁盘加载已有索引: {self.vector_store.count} 向量, {len(self._all_chunks)} BM25 chunks")
        return True

    def ensure_index(self, arxiv_client: ArxivClient = None) -> tuple:
        """
        加载已有索引；若无索引或本地有新 PDF，则从 data/papers 增量构建。

        Returns:
            (ok: bool, message: str) — message 为空表示已就绪；非空为提示或错误说明。
        """
        client = arxiv_client or ArxivClient()
        local_pdfs = client.get_local_papers()

        if not local_pdfs:
            if self.load_index():
                return True, ""
            return (
                False,
                "本地尚无 PDF。请先在「论文搜索」中下载论文，或将 PDF 放入 data/papers 后重试。",
            )

        loaded = self.load_index()
        indexed_sources = set()
        if loaded:
            indexed_sources = {
                m.get("source") for m in self.vector_store.metadata if m.get("source")
            }

        pending = [p for p in local_pdfs if p.name not in indexed_sources]
        if loaded and not pending:
            return True, ""

        to_index = pending if loaded else local_pdfs
        paths = [str(p) for p in to_index]
        metas = client.metadata_for_pdfs(to_index)

        try:
            self.build_index(paths, metas, rebuild=False)
        except Exception as e:
            logger.exception("从本地 PDF 构建索引失败")
            return False, f"索引构建失败: {e}"

        if self.vector_store.count == 0:
            return (
                False,
                "未能从本地 PDF 建立索引（可能文件损坏或解析失败）。请查看终端日志。",
            )

        if not loaded:
            return True, f"已从 {len(to_index)} 篇本地 PDF 自动构建检索索引。"
        return True, f"已为 {len(pending)} 篇新 PDF 更新索引。"

    def build_index(self, pdf_paths: List[str],
                    paper_metas: List[Dict] = None,
                    rebuild: bool = False,
                    max_workers: int = 4):
        """
        构建论文检索索引（支持增量追加 + 并行 PDF 解析）。

        Args:
            pdf_paths: PDF 文件路径列表
            paper_metas: 论文元数据列表 [{"arxiv_id": ..., "title": ..., ...}, ...]
            rebuild: 是否强制重建（False 时在已有索引上追加新论文）
            max_workers: 并行解析线程数（默认 4，设为 1 回到串行模式）
        """
        paper_metas = paper_metas or []
        existing_sources = set()

        if rebuild:
            self.vector_store = VectorStore()
            self.bm25 = BM25Retriever()
            self._all_chunks = []
            chunk_id = 0
        else:
            if self.load_index():
                existing_sources = {m.get("source") for m in self.vector_store.metadata}
                chunk_id = len(self.vector_store.metadata)
            else:
                chunk_id = 0

        # --- Step 1: 筛选需要解析的 PDF ---
        pending_paths = []
        pending_metas = []
        for i, path in enumerate(pdf_paths):
            source_name = Path(path).name
            if source_name in existing_sources and not rebuild:
                logger.info(f"跳过已索引论文: {source_name}")
                continue
            pending_paths.append(path)
            pending_metas.append(paper_metas[i] if i < len(paper_metas) else {})

        if not pending_paths:
            if self.vector_store.count > 0:
                logger.info("无新论文需索引，使用已有索引")
            else:
                logger.warning("没有成功解析任何论文")
            return

        # --- Step 2: 并行解析 PDF ---
        logger.info(f"并行解析 {len(pending_paths)} 篇 PDF (max_workers={max_workers})...")
        all_paper_chunks: List[tuple] = []  # [(paper_info, [chunks_without_id])]

        try:
            if max_workers > 1 and len(pending_paths) > 1:
                with ThreadPoolExecutor(max_workers=min(max_workers, len(pending_paths))) as executor:
                    future_map = {}
                    for path, meta in zip(pending_paths, pending_metas):
                        future = executor.submit(
                            _parse_and_chunk_paper,
                            path, config.parser.pdf_engine,
                            config.chunker.chunk_size, config.chunker.chunk_overlap,
                            config.chunker.min_chunk_size, meta,
                        )
                        future_map[future] = path

                    for future in as_completed(future_map):
                        path = future_map[future]
                        try:
                            result = future.result()
                            if result is not None:
                                all_paper_chunks.append(result)
                        except Exception as e:
                            logger.error(f"解析失败 {path}: {e}")
            else:
                # 串行回退（PDF 少时避免线程开销）
                for path, meta in zip(pending_paths, pending_metas):
                    try:
                        result = _parse_and_chunk_paper(
                            path, config.parser.pdf_engine,
                            config.chunker.chunk_size, config.chunker.chunk_overlap,
                            config.chunker.min_chunk_size, meta,
                        )
                        if result is not None:
                            all_paper_chunks.append(result)
                    except Exception as e:
                        logger.error(f"解析失败 {path}: {e}")
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"并行解析失败 ({e})，回退串行模式...")
            all_paper_chunks = []
            for path, meta in zip(pending_paths, pending_metas):
                try:
                    result = _parse_and_chunk_paper(
                        path, config.parser.pdf_engine,
                        config.chunker.chunk_size, config.chunker.chunk_overlap,
                        config.chunker.min_chunk_size, meta,
                    )
                    if result is not None:
                        all_paper_chunks.append(result)
                except Exception as e2:
                    logger.error(f"串行解析失败 {path}: {e2}")
                    logger.error(f"解析失败 {path}: {e}")

        if not all_paper_chunks:
            if self.vector_store.count > 0:
                logger.info("无新论文需索引，使用已有索引")
            else:
                logger.warning("没有成功解析任何论文")
            return

        new_chunks = []
        for paper_info, paper_chunks in all_paper_chunks:
            packed, chunk_id = self._pack_paper_chunks(paper_info, paper_chunks, chunk_id)
            new_chunks.extend(packed)

        self._ingest_new_chunks(new_chunks)

    def append_parsed_paper(
        self,
        paper_info: Dict,
        paper_chunks: List,
        rebuild: bool = False,
    ) -> int:
        """
        将 PaperParser + SectionChunker 的解析结果写入索引（供 reindex_one 等脚本调用）。
        """
        source_name = paper_info.get("source", "")
        if rebuild:
            self.vector_store = VectorStore()
            self.bm25 = BM25Retriever()
            self._all_chunks = []
            chunk_id = 0
        else:
            if self.vector_store.count == 0 and not self.load_index():
                chunk_id = 0
            else:
                indexed = {m.get("source") for m in self.vector_store.metadata}
                if source_name and source_name in indexed:
                    logger.info(f"跳过已索引论文: {source_name}")
                    return 0
                chunk_id = len(self.vector_store.metadata)

        new_chunks, _ = self._pack_paper_chunks(paper_info, paper_chunks, chunk_id)
        if not new_chunks:
            logger.warning("无有效 chunk，跳过: %s", source_name)
            return 0
        logger.info("开始向量化 %s 个 chunk（device=%s）...", len(new_chunks), config.embedding.device)
        self._ingest_new_chunks(new_chunks)
        return len(new_chunks)

    def _pack_paper_chunks(
        self, paper_info: Dict, paper_chunks: List, chunk_id: int
    ) -> tuple:
        """组装单篇论文的 chunk 列表并分配全局 chunk_id。"""
        from src.chunker.section_chunker import Chunk

        new_chunks = []
        source_name = paper_info["source"]

        abstract = paper_info.get("abstract", "").strip()
        if abstract:
            abstract_chunk = Chunk(
                text=abstract,
                metadata={
                    **paper_info,
                    "section_title": "Abstract",
                    "section_id": "Abstract",
                    "section_number": 0,
                    "section_level": 0,
                    "chunk_type": "abstract",
                },
            )
            abstract_chunk.metadata["chunk_id"] = str(chunk_id)
            abstract_chunk.metadata["source"] = source_name
            abstract_chunk.metadata["text"] = abstract
            chunk_id += 1
            new_chunks.append(abstract_chunk)

        for c in paper_chunks:
            c.metadata["chunk_id"] = str(chunk_id)
            c.metadata["source"] = source_name
            c.metadata["text"] = c.text
            chunk_id += 1
            new_chunks.append(c)

        return new_chunks, chunk_id

    def _ingest_new_chunks(self, new_chunks: List) -> None:
        """
        向量化并持久化一批 chunk。

        时序铁律：先以 raw_text 完成 100% 表格元数据检测与打标，
        后做 Caption 截断与向量轻量化 —— 严禁"先截断、后检测"。
        """
        texts = []
        metadatas = []
        import re

        for i, c in enumerate(new_chunks):
            meta = dict(c.metadata)
            raw_text = str(c.text)                       # 锁死原始未截断完整文本
            section = meta.get("section_title", "")
            paper = meta.get("paper_title", "")

            # ══════════════════════════════════════════════════════
            # Phase 1: Header Line Lock — 仅首行 [TABLE_HTML: Table_X] 打标
            # ══════════════════════════════════════════════════════
            first_line = raw_text.split("\n", 1)[0] if "\n" in raw_text else raw_text

            if "[TABLE_HTML:" in first_line:
                meta["is_table"] = True
                meta["contains_table"] = True
                meta["chunk_type"] = "table"
                tag_match = re.search(r"Table_(\d+)", first_line, re.IGNORECASE)
                meta["table_id"] = (
                    f"Table_{tag_match.group(1)}" if tag_match else ""
                )
                is_table_html = True
            else:
                meta["is_table"] = False
                meta["table_id"] = ""
                meta["contains_table"] = False
                is_table_html = False

            # Phase 2: Embedding input - context enrichment + prefix
            # ==========================================================
            if is_table_html:
                table_text = self._extract_table_text_for_embed(raw_text)
                prefix = f"[{paper} {meta.get('table_id','')}] " if paper else ""
                safe_text = (prefix + table_text)[:self._MAX_EMBED_CHARS]
            else:
                # Late Chunking enrichment:
                # embed paper title + section header + previous chunk first sentence
                parts = []
                if paper:
                    parts.append(f"[{paper}]")
                if section:
                    parts.append(f"Section: {section}")
                prev_chunk = new_chunks[i - 1] if i > 0 else None
                if prev_chunk and not prev_chunk.metadata.get("is_table"):
                    prev_text = str(prev_chunk.text or "")
                    prev_first_sentence = prev_text.split(".")[0][:150].replace("\n", " ")
                    if prev_first_sentence.strip():
                        parts.append(f"Prev: {prev_first_sentence.strip()}")
                prefix = " | ".join(parts) + "\n" if parts else ""
                safe_text = (prefix + raw_text)[:self._MAX_EMBED_CHARS]
                # normalize null bytes and unicode replacement characters
                safe_text = safe_text.replace(chr(0), " ").replace(chr(0xfffd), " ")

            texts.append(safe_text)

            # ══════════════════════════════════════════════════════
            # Phase 3: 元数据构造 — 表块 Caption 检索 + 完整 HTML 落盘
            # ══════════════════════════════════════════════════════
            if is_table_html:
                meta["full_html_content"] = raw_text
                # BM25 文本也带论文前缀，使跨论文同名表可区分
                first_line = raw_text.split("\n")[0] if "\n" in raw_text else raw_text[:200]
                prefix = f"[{paper}] " if paper else ""
                meta["text"] = (prefix + first_line)[:500]
            else:
                meta["text"] = raw_text
                meta["full_html_content"] = ""

            metadatas.append(meta)

        embeddings = self._encode_texts_batched(texts)

        assert len(embeddings) == len(new_chunks) == len(metadatas), (
            f"⛔ 索引构建对齐失败: "
            f"embeddings={len(embeddings)}, chunks={len(new_chunks)}, metadatas={len(metadatas)}"
        )

        self.vector_store.add(embeddings, metadatas)
        self.vector_store.save()
        self.bm25.append(new_chunks)
        self.bm25.save(str(get_vector_store_dir() / "bm25.pkl"))
        self._all_chunks = list(self.bm25._chunks)

        # ── 也为每篇论文保存独立索引 ──
        self._save_per_paper_index(new_chunks, metadatas, embeddings)

        logger.info(
            f"索引构建完成: +{len(new_chunks)} chunks, 总计 {len(self._all_chunks)} chunks"
        )

    def _save_per_paper_index(
        self, new_chunks: List, metadatas: List[dict], embeddings
    ) -> None:
        """为每篇新入库的论文保存独立索引到 vector_store/papers/{slug}/。"""
        from src.retriever.paper_registry import PaperRegistry, PAPERS_DIR
        from src.retriever.vector_store import VectorStore

        paper_groups: Dict[str, tuple] = {}
        for c, meta, emb in zip(new_chunks, metadatas, embeddings):
            src = meta.get("source", "unknown")
            paper_groups.setdefault(src, {"chunks": [], "metas": [], "embs": []})
            paper_groups[src]["chunks"].append(c)
            paper_groups[src]["metas"].append(meta)
            paper_groups[src]["embs"].append(emb)

        reg = PaperRegistry()
        for source, group in paper_groups.items():
            reg.register(source)
            slug = reg._routes[source]
            # 用 slug 作 persist_dir + name 避免同名冲突
            vs = VectorStore(persist_dir=str(PAPERS_DIR / slug))
            emb_array = np.array(group["embs"], dtype=np.float32)
            if emb_array.ndim == 1:
                emb_array = emb_array.reshape(1, -1)
            vs.dimension = emb_array.shape[1]
            vs.add(emb_array, group["metas"])
            vs.save(name="paper")

            b = BM25Retriever()
            b.append(group["chunks"])
            b.save(str(PAPERS_DIR / slug / "bm25.pkl"))

            logger.info(f"独立索引: {source} → {slug} ({len(group['chunks'])} chunks)")

    def load_paper_index(self, source: str) -> bool:
        """加载某篇论文的独立索引（替换当前全局索引）。"""
        from src.retriever.paper_registry import PaperRegistry

        reg = PaperRegistry()
        paper_dir = reg.get_paper_dir(source)
        if not paper_dir:
            logger.warning(f"未找到论文独立索引: {source}")
            return False

        bm25_path = paper_dir / "bm25.pkl"
        if not bm25_path.exists():
            logger.warning(f"独立索引文件不存在: {bm25_path}")
            return False

        self.vector_store = VectorStore(persist_dir=str(paper_dir))
        if not self.vector_store.load(name="paper"):
            return False

        self.bm25 = BM25Retriever()
        if bm25_path.exists():
            self.bm25.load(str(bm25_path))
        self._all_chunks = list(self.bm25._chunks)
        logger.info(f"加载论文独立索引: {source} ({self.vector_store.count} chunks)")
        return True

    @staticmethod
    def _extract_table_text_for_embed(raw_text: str) -> str:
        """
        从 [TABLE_HTML] 包裹的 HTML 表格中提取可读文本用于 Embedding。

        提取顺序：
        1. Caption 行
        2. 表头行（第一个 <tr> 的 <th> 文本）
        3. 前 2 个数据行的 <td> 文本
        4. 左侧列（方法名/指标名）的独特值（去重，最多 5 个）
        """
        import re

        parts = []

        # ── 1. Caption ──
        first_line = raw_text.split("\n", 1)[0] if "\n" in raw_text else raw_text
        caption = (
            first_line
            .replace("[TABLE_HTML:", "")
            .replace("[TABLE_MD:", "")
            .replace("[TABLE ", "")
            .replace("[/TABLE_HTML]", "")
            .replace("[/TABLE_MD]", "")
            .replace("[/TABLE]", "")
            .rstrip("]")
            .strip()
        )
        if caption:
            parts.append(f"[TABLE] {caption}")

        # ── 2-4. 解析 HTML 中的 <tr> ──
        # 提取所有 <tr>...</tr> 内容
        tr_blocks = re.findall(r'<tr>(.*?)</tr>', raw_text, re.IGNORECASE | re.DOTALL)
        if not tr_blocks:
            # 回退：整个文本去标签
            plain = re.sub(r'<[^>]+>', ' ', raw_text)
            plain = re.sub(r'\s+', ' ', plain).strip()
            parts.append(plain[:500])
            return " | ".join(parts)

        # 表头行（第一个 tr）
        header_cells = re.findall(r'<th[^>]*>(.*?)</th>', tr_blocks[0], re.IGNORECASE | re.DOTALL)
        if header_cells:
            header_text = " | ".join(re.sub(r'<[^>]+>', ' ', c).strip() for c in header_cells if c.strip())
            if header_text:
                parts.append(f"Columns: {header_text}")

        # 数据行：取前 2 个有 <td> 的行
        data_rows = []
        left_col_values = []
        for tr in tr_blocks[1:4]:  # 最多取 3 个数据行
            td_cells = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.IGNORECASE | re.DOTALL)
            if td_cells:
                row_text = " | ".join(
                    re.sub(r'<[^>]+>', ' ', c).strip() for c in td_cells if c.strip()
                )
                if row_text:
                    data_rows.append(row_text)
            # 也取第一个非空 cell 作为左侧列的值
            all_cells_in_row = re.findall(r'<(?:th|td)[^>]*>(.*?)</(?:th|td)>', tr, re.IGNORECASE | re.DOTALL)
            first_val = ""
            for cell in all_cells_in_row:
                val = re.sub(r'<[^>]+>', ' ', cell).strip()
                if val and val not in ("", "."):
                    first_val = val
                    break
            if first_val and first_val not in left_col_values:
                left_col_values.append(first_val)

        if data_rows:
            parts.append("Rows: " + " || ".join(data_rows))

        if left_col_values:
            parts.append(f"Keys: {', '.join(left_col_values[:5])}")

        result = " | ".join(parts)
        return result if result else f"[TABLE] {caption}"

    def _encode_texts_batched(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """逐条向量化（稳定优先，避免 Windows 下大批次 native crash）。"""
        if not texts:
            return np.array([])
        parts = []
        for i, line in enumerate(texts):
            if i and i % 20 == 0:
                logger.info("Embedding 进度: %s / %s", i, len(texts))
            parts.append(np.asarray(self.embedder.encode([line]), dtype=np.float32))
        return np.vstack(parts)

    @staticmethod
    def _extract_refs(query: str) -> tuple:
        """
        从查询中提取(paper_filter, section_filter)。
        paper_filter: {"source__keyword": match_str} 或 None
        section_filter: {"section_id__startswith": match_str} 或 None
        """
        import re
        paper_filter = None
        section_filter = None

        # ── 提取论文引用 ──
        # arxiv ID 模式: 2507.10864v3
        m = re.search(r'\b(\d{4}\.\d{4,5}v?\d*)\b', query)
        if m:
            paper_filter = {"source__keyword": m.group(1)}
        # "the X paper" / "X 论文" 模式
        if not paper_filter:
            m = re.search(r'(?:the\s+)?([A-Z][A-Za-z0-9-]+)\s+(?:paper|论文|model)', query)
            if m:
                kw = m.group(1).strip()
                if len(kw) >= 3:
                    paper_filter = {"source__keyword": kw}
            # 中文 "论文X" / "X论文"
            m2 = re.search(r'(?:论文|论[文著])\s*["\x27]?([^"\x27]{2,30}?)["\x27]?\s*(?:的|中|里|第)', query)
            if m2:
                kw = m2.group(1).strip()
                if len(kw) >= 2:
                    paper_filter = {"source__keyword": kw}

        # ── 提取章节引用 ──
        # 中文：第2.3节、2.3节、2.3章
        m = re.search(r'(?:第\s*)?(\d+(?:\.\d+)*)\s*[节章]', query)
        if m:
            section_filter = {"section_id__startswith": m.group(1)}
        # 英文：Section 2.3, section 2.3.1
        if not section_filter:
            m = re.search(r'(?:Section|section|CHAPTER|Chapter)\s+(\d+(?:\.\d+)*)', query)
            if m:
                section_filter = {"section_id__startswith": m.group(1)}
        # 纯数字开头：2.3 transformer attention
        if not section_filter:
            m = re.match(r'(\d+(?:\.\d+)*)\s', query)
            if m:
                section_filter = {"section_id__startswith": m.group(1)}
        # 关键词章节
        if not section_filter:
            keywords = {"abstract", "introduction", "method", "experiment", "conclusion",
                         "related work", "background", "discussion", "result"}
            lower_q = query.lower()
            for kw in keywords:
                if kw in lower_q:
                    section_filter = {"section_id__startswith": kw.title()}
                    break

        # ── 提取表格引用（模糊容错：可识别 Table4, Tab.4, able 4 等变体）──
        table_filter = None
        m = re.search(r'(?:(?:[Tt]able|[Tt]ab)\b|表)\s*[-_.]?\s*(\d+)', query)
        if m:
            table_id = f"Table_{m.group(1)}"
            table_filter = {"table_id": table_id}

        has_section = section_filter is not None
        return paper_filter, section_filter, table_filter, has_section

    # ── Filename-Based Paper Router ─────────────────────────────────────

    _QUERY_STOPWORDS: Set[str] = {
        "the", "a", "an", "in", "on", "of", "for", "to", "and", "or", "is", "are",
        "what", "how", "which", "paper", "section", "chapter", "about", "using",
        "with", "from", "this", "that", "does", "do", "can", "请", "介绍", "分析",
        "总结", "对比", "方法", "模型", "论文", "章节", "部分", "哪些", "什么",
        "目前", "主流", "有", "是", "的", "了", "吗", "呢",
    }

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        """低噪文本：小写、统一 SAM-2 变体、去除标点。"""
        text = text.lower()
        text = re.sub(r"\bsam\s*[-_]?\s*2\b", "sam_2", text, flags=re.IGNORECASE)
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _filename_tokens(source: str) -> Set[str]:
        """从 PDF 文件名 stem 提取可匹配 token（含 sam_2 等合成词）。"""
        stem = Path(source).stem
        parts = [p.lower() for p in stem.split("_") if p]
        tokens: Set[str] = set()
        for p in parts:
            if len(p) >= 2 or p.isdigit():
                tokens.add(p)
        for i in range(len(parts) - 1):
            a, b = parts[i], parts[i + 1]
            if len(a) >= 2 and len(b) >= 1:
                tokens.add(f"{a}_{b}")
            if a == "sam" and b == "2":
                tokens.add("sam_2")
                tokens.add("sam2")
        return tokens

    @staticmethod
    def _query_match_tokens(query: str) -> Set[str]:
        """从用户 Query 提取与文件名对齐的 token 集合。"""
        norm = RetrievalPipeline._normalize_match_text(query)
        if not norm:
            return set()
        tokens: Set[str] = set()
        parts = norm.replace("_", " ").split()
        for p in parts:
            if len(p) >= 2 and p not in RetrievalPipeline._QUERY_STOPWORDS:
                tokens.add(p)
        for i in range(len(parts) - 1):
            a, b = parts[i], parts[i + 1]
            if len(a) >= 2 and a not in RetrievalPipeline._QUERY_STOPWORDS:
                tokens.add(f"{a}_{b}")
                if b.isdigit():
                    tokens.add(f"{a}{b}")
        return tokens

    @staticmethod
    def detect_paper_by_filename(query: str, all_sources: List[str]) -> Optional[str]:
        """
        Filename-Based Paper Router：在混合检索前将 Query 与全部 source 文件名打分，
        锁定唯一 target_pdf_name；泛指问题或歧义时返回 None（允许全库盲搜）。
        """
        if not query or not all_sources:
            return None

        query_tokens = RetrievalPipeline._query_match_tokens(query)
        if not query_tokens:
            return None

        source_tokens: Dict[str, Set[str]] = {
            src: RetrievalPipeline._filename_tokens(src) for src in all_sources
        }
        token_df: Dict[str, int] = {}
        for tokens in source_tokens.values():
            for t in tokens:
                token_df[t] = token_df.get(t, 0) + 1
        n_sources = len(all_sources)

        def _idf(tok: str) -> float:
            df = token_df.get(tok, 0)
            return 1.0 + math.log((n_sources + 1) / (df + 1))

        scores: Dict[str, float] = {}
        for src, ftokens in source_tokens.items():
            stem_norm = RetrievalPipeline._normalize_match_text(
                Path(src).stem.replace("_", " ")
            )
            stem_compact = stem_norm.replace(" ", "")
            score = 0.0
            hits = 0
            for qt in query_tokens:
                wt = _idf(qt)
                if qt in ftokens:
                    score += 2.0 * wt
                    hits += 1
                elif len(qt) >= 3 and (qt in stem_norm or qt in stem_compact):
                    score += 1.2 * wt
                    hits += 1
            if hits >= 2:
                score *= 1.15
            scores[src] = score

        ranked: List[Tuple[str, float]] = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )
        if not ranked or ranked[0][1] <= 0:
            return None

        best_src, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        min_score = 2.8
        min_margin = 1.2
        if best_score < min_score:
            return None
        if second_score > 0 and (best_score - second_score) < min_margin:
            if second_score >= min_score * 0.8:
                logger.info(
                    "Filename router: ambiguous match, skip lock "
                    f"({best_src}={best_score:.2f}, 2nd={second_score:.2f})"
                )
                return None

        logger.info(
            f"Filename router locked: {best_src} "
            f"(score={best_score:.2f}, 2nd={second_score:.2f})"
        )
        return best_src

    def _get_all_sources(self) -> List[str]:
        """索引中全部唯一 PDF source 文件名。"""
        seen: Set[str] = set()
        sources: List[str] = []
        for meta in self.vector_store.metadata:
            src = meta.get("source")
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        return sources

    def _resolve_paper_filter(self, paper_filter: Dict) -> Dict:
        """将 paper_filter 中的 keyword 匹配到实际 source 文件名"""
        if not paper_filter or "source__keyword" not in paper_filter:
            return None
        kw = paper_filter["source__keyword"].lower()
        # 扫描 metadata 中的 source 字段
        sources_seen = set()
        for meta in self.vector_store.metadata:
            src = meta.get("source", "").lower()
            if src and src not in sources_seen:
                sources_seen.add(src)
                if kw in src or kw in meta.get("paper_title", "").lower():
                    return {"source": meta["source"]}
        # arxiv ID 精确匹配
        for meta in self.vector_store.metadata:
            src = meta.get("source", "").lower()
            aid = meta.get("arxiv_id", "").lower()
            if kw in aid or kw in src:
                return {"source": meta["source"]}
        return None

    def query(self, query_text: str,
              top_k: int = None,
              use_rerank: bool = True,
              alpha: float = None,
              rewrite: bool = True,
              chat_history: List[dict] = None,
              paper_filter: Dict = None) -> List[Dict]:
        """
        完整检索查询（含论文双重锁定 + 按论文域降级）。

        Args:
            chat_history: 可选。多轮对话历史，用于上下文感知查询改写。
                         格式 [{"role":"user"/"assistant", "content":...}]
            paper_filter: 可选。外部传入的论文锁定，如 {"source__keyword": "xxx.pdf"}。
                         若传入，将覆盖 filename router 推断结果。
        """
        top_k = top_k or config.retrieval.top_k_rerank
        alpha = alpha if alpha is not None else config.retrieval.alpha

        if self.vector_store.count == 0 and not self.load_index():
            logger.warning("索引为空，请先 build_index 或下载论文")
            return []

        # ── Step 0: 上下文感知改写（代词消解 + 隐式引用补全）──
        if chat_history and len(chat_history) >= 2:
            from src.query_rewriter import ContextualQueryRewriter
            contextualizer = ContextualQueryRewriter()
            contextualized = contextualizer.rewrite(query_text, chat_history)
            if contextualized != query_text:
                logger.info("上下文改写: '%s' → '%s'", query_text[:60], contextualized[:80])
                query_text = contextualized

        # ── Step 0.5: 外部 paper_filter 优先解析 ──
        _external_paper_filter = None
        if paper_filter and "source__keyword" in paper_filter:
            _external_paper_filter = self._resolve_paper_filter(paper_filter)
            if _external_paper_filter:
                logger.info("外部论文锁定: %s", _external_paper_filter["source"])

        # ── Step 1: 文件名前置硬判定（Boundless Paper Lock）──
        all_sources = self._get_all_sources()
        filename_locked = self.detect_paper_by_filename(query_text, all_sources)
        paper_locked = bool(filename_locked)

        # ── Step 1: 从查询中提取论文 + 章节 + 表格引用 ──
        paper_ref, section_ref, table_ref, has_section = self._extract_refs(query_text)
        table_val = table_ref.get("table_id") if table_ref else None
        paper_filter_resolved = self._resolve_paper_filter(paper_ref) if paper_ref else None
        if filename_locked:
            paper_filter_resolved = {"source": filename_locked}
        # ⚠️ 外部 paper_filter 优先级最高
        if _external_paper_filter:
            paper_filter_resolved = _external_paper_filter

        # ── Step 2: 多级降级检索 ──
        queries = [query_text]
        # ⚠️ 如果已有 paper_filter，不启用 Query Rewrite（改写会冲淡论文锁定信号）
        if rewrite and not section_ref and not paper_ref and not paper_filter_resolved:
            from src.query_rewriter import has_cjk
            strategy = "chinese_academic" if has_cjk(query_text) else "auto"
            queries = self.rewriter.rewrite(query_text, strategy=strategy)

        candidates = []

        # 2a-table) 单表死锁：仅在目标 table_id 的 HTML 表块内检索
        if table_val:
            table_filter = {"table_id": table_val, "is_table": True}
            if paper_filter_resolved:
                table_filter = {**paper_filter_resolved, **table_filter}
            all_results = self._hybrid_search(
                queries[:5], alpha, filter_dict=table_filter
            )
            candidates = self._fuse_scores(all_results, alpha)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            if not candidates and paper_filter_resolved:
                all_results = self._hybrid_search(
                    queries[:5],
                    alpha,
                    filter_dict={"table_id": table_val, "is_table": True},
                )
                candidates = self._fuse_scores(all_results, alpha)
                candidates.sort(key=lambda x: x["score"], reverse=True)
            if candidates:
                logger.info(
                    "Exact table lock (search): %s → %d hit(s)",
                    table_val,
                    len(candidates),
                )

        # 2a) 双锁：paper + section（单表查询不走泛搜降级）
        if not table_val and not candidates and paper_filter_resolved and section_ref:
            combined = {**paper_filter_resolved, **section_ref}
            all_results = self._hybrid_search(queries[:5], alpha, filter_dict=combined)
            candidates = self._fuse_scores(all_results, alpha)
            candidates.sort(key=lambda x: x["score"], reverse=True)

        # 2b) 论文硬锁（无章节号也强制单篇 PDF 内检索）
        if not table_val and not candidates and paper_filter_resolved:
            if section_ref:
                logger.info(f"双锁无结果，降级为论文内检索: {paper_filter_resolved}")
            elif paper_locked:
                logger.info(f"Filename boundless lock: {paper_filter_resolved}")
            all_results = self._hybrid_search(queries[:5], alpha, filter_dict=paper_filter_resolved)
            candidates = self._fuse_scores(all_results, alpha)
            candidates.sort(key=lambda x: x["score"], reverse=True)

        # 2c) 仅章节过滤（无论文锁时）
        if not table_val and not candidates and section_ref and not paper_filter_resolved:
            all_results = self._hybrid_search(queries[:5], alpha, filter_dict=section_ref)
            candidates = self._fuse_scores(all_results, alpha)
            candidates.sort(key=lambda x: x["score"], reverse=True)

        # 2d) 全库泛搜：仅当文件名路由与论文锁均未命中
        if not table_val and not candidates and not paper_filter_resolved:
            all_results = self._hybrid_search(queries[:5], alpha, filter_dict=None)
            candidates = self._fuse_scores(all_results, alpha)
            candidates.sort(key=lambda x: x["score"], reverse=True)

        # 纯向量回退（单表查询仅回退到同 table_id 块）
        if not candidates:
            fb_filter = None
            if table_val:
                fb_filter = {"table_id": table_val, "is_table": True}
                if paper_filter_resolved:
                    fb_filter = {**paper_filter_resolved, **fb_filter}
            elif paper_filter_resolved:
                fb_filter = paper_filter_resolved
            candidates = self._vector_fallback(
                query_text, top_k, filter_dict=fb_filter
            )

        candidates = [c for c in candidates if c.get("text", "").strip()]

        # ── Phase A: Pre-Bundle 硬去重（消除 Query 扩展膨胀导致的重复 _idx）──
        seen_idx = set()
        deduped = []
        for c in candidates:
            meta = c.get("metadata", {})
            idx = meta.get("_idx")
            if idx is not None:
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    deduped.append(c)
            else:
                deduped.append(c)  # 无 _idx 的保底保留
        if len(deduped) != len(candidates):
            logger.info("Pre-bundle dedup: %d → %d", len(candidates), len(deduped))
            candidates = deduped

        # ── [TRACE] 混合检索融合后 ──

        # 逻辑章节连续块打包（单表查询时禁用，防止正文 sibling 混入）
        intent = self._detect_intent(query_text)
        if not table_val and intent in ("dataset", "experiment", "config"):
            candidates = self._bundle_section_siblings(candidates)

        # ── Phase B: Post-Bundle 去重（消除 _bundle_section_siblings 引入的文本重复）──
        seen_text = set()
        deduped2 = []
        for c in candidates:
            text = c.get("text", "")
            text_hash = hash(text[:500]) if text else 0
            if text_hash not in seen_text:
                seen_text.add(text_hash)
                deduped2.append(c)
        if len(deduped2) != len(candidates):
            logger.info("Post-bundle dedup: %d → %d", len(candidates), len(deduped2))
            candidates = deduped2

        # ── [TRACE] _bundle_section_siblings + Post-Dedup 后 ──

        if use_rerank and candidates and not table_val:
            try:
                prev_count = len(candidates)
                candidates = self._llm_rerank(query_text, candidates, top_k=top_k)
                logger.info("LLM 重排: %d → %d", prev_count, len(candidates))
            except Exception as e:
                import traceback
                logger.error("[CRITICAL] LLM Reranker 调用失败, 使用原始 fusion 排序保底")
                traceback.print_exc()
                candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
                candidates = candidates[:top_k]

        # ── [TRACE] LLM Reranker 重排后 ──

        # ── Exact Table Lock：交付前硬过滤，有且仅返回单表 ──
        if table_val:
            locked = [
                c for c in candidates
                if str(c.get("metadata", {}).get("table_id", "")).rstrip(".")
                == table_val
            ]
            if locked:
                locked.sort(key=lambda x: x.get("score", 0), reverse=True)
                candidates = locked[:1]
                logger.info("Exact table lock (final): %s → 1 chunk", table_val)
            else:
                logger.warning(
                    "Exact table lock: no chunk with table_id=%s in candidates",
                    table_val,
                )
                candidates = []

        return candidates[:top_k] if not table_val else candidates[:1]

    @staticmethod
    def _detect_intent(query: str) -> str:
        """从查询中检测学术意图类型：dataset / experiment / config / architecture / general"""
        lower = query.lower()
        dataset_kw = {"数据集", "数据", "训练集", "测试集", "datasets", "data",
                      "benchmarks", "benchmark", "evaluation", "kvasir",
                      "cvc", "polyp"}
        experiment_kw = {"实验", "对比", "结果", "performance", "results",
                         "experiment", "ablation", "comparison"}
        config_kw = {"参数", "配置", "setting", "implementation",
                     "parameter", "hyperparameter", "setup"}
        architecture_kw = {"骨干", "backbone", "架构", "encoder", "decoder", "网络结构",
                           "architecture", "module", "block", "head", "neck",
                           "编码器", "解码器", "resnet", "vgg", "efficientnet",
                           "transformer", "vit", "层次", "layer", "组件"}
        if any(kw in lower for kw in architecture_kw):
            return "architecture"
        if any(kw in lower for kw in dataset_kw):
            return "dataset"
        if any(kw in lower for kw in experiment_kw):
            return "experiment"
        if any(kw in lower for kw in config_kw):
            return "config"
        return "general"

    @staticmethod
    def _section_intent_keywords(intent: str) -> set:
        """根据意图返回该类型论文章节常见的 section_title 关键词"""
        if intent == "architecture":
            return {"method", "model", "architecture", "network", "encoder",
                    "decoder", "design", "approach", "framework", "implementation",
                    "proposed", "pipeline", "ablation", "component"}
        if intent == "dataset":
            return {"dataset", "data", "benchmark", "evaluation", "experimental",
                    "setup", "material", "kvasir", "cvc", "colon"}
        if intent == "experiment":
            return {"experiment", "result", "ablation", "comparison", "evaluation",
                    "performance", "analysis", "discussion"}
        if intent == "config":
            return {"implementation", "parameter", "setting", "setup", "configuration",
                    "detail", "training", "loss"}
        return set()

    def _hybrid_search(self, queries: List[str], alpha: float,
                        filter_dict: Dict = None) -> Dict:
        """多查询混合检索（含意图感知章节暴击）"""
        all_results = {}
        top_k_vec = config.retrieval.top_k_vector
        top_k_bm = config.retrieval.top_k_bm25

        # 检测意图（从首个查询）
        intent = self._detect_intent(queries[0])
        intent_section_keywords = self._section_intent_keywords(intent)
        # 非 general 意图时启用暴击权重
        base_boost = 1.8 if intent in ("dataset", "experiment", "config") else 1.3

        # 批量编码所有改写查询
        q_vecs = self.embedder.encode(queries)
        for i, q in enumerate(queries):
            q_vec = q_vecs[i]
            for idx, score, meta in self.vector_store.search(q_vec, top_k=top_k_vec, filter_dict=filter_dict):
                cid = meta.get("chunk_id", f"v_{idx}")
                text = meta.get("text") or self._text_from_meta(meta, idx)
                if cid not in all_results:
                    all_results[cid] = {
                        "vector_score": 0, "bm25_score": 0,
                        "meta": meta, "text": text,
                    }
                # 意图感知 section_title 暴击加分
                section_title = meta.get("section_title", "")
                boost = 1.0
                if section_title and intent_section_keywords:
                    if any(kw in section_title.lower() for kw in intent_section_keywords):
                        boost = base_boost
                elif section_title and any(kw.lower() in section_title.lower() for kw in q.split()):
                    boost = 1.3  # 原始备选
                score *= boost
                all_results[cid]["vector_score"] = max(all_results[cid]["vector_score"], score)
                if text and not all_results[cid]["text"]:
                    all_results[cid]["text"] = text

            for chunk, score in self.bm25.search(q, top_k=top_k_bm, filter_dict=filter_dict):
                cid = chunk.metadata.get("chunk_id", chunk.text[:20])
                if cid not in all_results:
                    all_results[cid] = {
                        "vector_score": 0, "bm25_score": 0,
                        "meta": chunk.metadata, "text": chunk.text,
                    }
                # BM25 意图感知暴击
                section_title = chunk.metadata.get("section_title", "")
                boost = 1.0
                if section_title and intent_section_keywords:
                    if any(kw in section_title.lower() for kw in intent_section_keywords):
                        boost = base_boost
                elif section_title and any(kw.lower() in section_title.lower() for kw in q.split()):
                    boost = 1.3
                score *= boost
                all_results[cid]["bm25_score"] = max(all_results[cid]["bm25_score"], score)

        return all_results

    def _bundle_section_siblings(self, candidates: List[Dict]) -> List[Dict]:
        """
        逻辑章节连续块强行打包（Section Bundling）。
        
        对一个被召回 chunk，如果属于可合并的 section 类型，
        将整个 section 的所有 sibling chunks 按文档顺序全部捞出。
        
        合并策略：
        - 按 (source, section_id) 分组，视为一个"逻辑文档块"
        - 收集组内所有 chunk（含未召回的）
        - 按 _idx 升序排列
        - 去重（chunk_id）
        - 保留原候选的 score，新增 chunk score=0（reranker 会重排）
        """
        if not candidates or not self.vector_store.metadata:
            return candidates

        # Step 1: 构建 (source, section_id) → [chunk_entries] 的倒排索引
        section_map: Dict[tuple, List[Dict]] = {}
        for meta in self.vector_store.metadata:
            src = meta.get("source", "")
            sid = meta.get("section_id", "")
            idx = meta.get("_idx", -1)
            cid = meta.get("chunk_id", idx)
            if src and sid:
                key = (src, sid)
                if key not in section_map:
                    section_map[key] = []
                section_map[key].append({
                    "_idx": idx,
                    "chunk_id": cid,
                    "meta": meta,
                    "text": meta.get("text") or self._text_from_meta(meta, idx),
                })

        # Step 2: 收集候选中被命中的 (source, section_id)
        hit_keys = set()
        for c in candidates:
            meta = c.get("metadata", {})
            key = (meta.get("source", ""), meta.get("section_id", ""))
            if key[0] and key[1]:
                hit_keys.add(key)

        if not hit_keys:
            return candidates

        # Step 3: 对整个逻辑块排序，并判断是否是大块连续列表章节
        def _is_list_section(sid: str, section_map: Dict, key: tuple) -> bool:
            """判断一个 section 是否包含列表特征（≥3 个 chunk 或标题含 Dataset/Experiment）"""
            entries = section_map.get(key, [])
            if len(entries) >= 3:
                return True
            if entries:
                title = entries[0]["meta"].get("section_title", "")
                list_kw = {"dataset", "experiment", "benchmark", "data", "evaluation",
                           "implementation", "paramet", "ablation", "result"}
                if any(kw in title.lower() for kw in list_kw):
                    return True
            return False

        # Step 4: 构建 bundles
        candidate_by_cid = {}
        for c in candidates:
            cid = c.get("metadata", {}).get("chunk_id", id(c))
            candidate_by_cid[cid] = c

        merged = []
        seen_cids = set()

        # 先收集所有需合并的 key 及其排序后的 entry
        all_bundled = []  # (sort_key, entry_dict)
        for key in hit_keys:
            if not _is_list_section(key[1], section_map, key):
                continue
            entries = sorted(section_map.get(key, []), key=lambda e: e["_idx"])
            for entry in entries:
                cid = entry["chunk_id"]
                if cid in seen_cids:
                    continue
                seen_cids.add(cid)
                # 优先取候选人（保留原有分数），否则新建
                if cid in candidate_by_cid:
                    merged.append(candidate_by_cid[cid])
                else:
                    text = entry["text"]
                    if text.strip():
                        merged.append({
                            "text": text,
                            "source": entry["meta"].get("source", ""),
                            "score": 0.0,
                            "metadata": entry["meta"],
                            "_bundled": True,
                        })

        # Step 5: 补充未被 bundle 的零散候选（不在任何 list section 中的）
        for c in candidates:
            cid = c.get("metadata", {}).get("chunk_id", id(c))
            meta = c.get("metadata", {})
            key = (meta.get("source", ""), meta.get("section_id", ""))
            if cid not in seen_cids and key not in hit_keys:
                merged.append(c)
                seen_cids.add(cid)
            elif cid not in seen_cids:
                # 在 hit_keys 但被 _is_list_section 判定为否（单块小节）
                merged.append(c)
                seen_cids.add(cid)

        bundled_count = sum(1 for m in merged if m.get("_bundled"))
        if bundled_count:
            logger.info(f"章节段打包: {len(hit_keys)} 个节, {bundled_count} 个额外 chunk 合并")

        return merged

    def _text_from_meta(self, meta: Dict, idx: int) -> str:
        """兼容旧索引：metadata 未存 text 时从 BM25 corpus 回填"""
        chunk_id = meta.get("chunk_id")
        if chunk_id is not None:
            for chunk in self.bm25._chunks:
                if chunk.metadata.get("chunk_id") == chunk_id:
                    return chunk.text
        if 0 <= idx < len(self.bm25._corpus):
            return self.bm25._corpus[idx]
        return ""

    def _vector_fallback(self, query_text: str, top_k: int,
                         filter_dict: Dict = None) -> List[Dict]:
        """混合融合无命中时的纯向量回退（可携带 source 硬过滤）"""
        q_vec = self.embedder.encode_single(query_text)
        results = self.vector_store.search(q_vec, top_k=top_k, filter_dict=filter_dict)
        candidates = []
        for idx, score, meta in results:
            text = meta.get("text") or self._text_from_meta(meta, idx)
            if not text.strip():
                continue
            candidates.append({
                "text": text,
                "source": meta.get("source", ""),
                "score": round(float(score), 4),
                "vector_score": round(float(score), 4),
                "bm25_score": 0.0,
                "metadata": meta,
            })
        return candidates

    def _fuse_scores(self, results: Dict, alpha: float) -> List[Dict]:
        """分数归一化 + α 加权融合"""
        if not results:
            return []

        vec_scores = {k: v["vector_score"] for k, v in results.items()}
        bm_scores = {k: v["bm25_score"] for k, v in results.items()}

        def normalize(d):
            if not d:
                return {}
            vals = list(d.values())
            vmin, vmax = min(vals), max(vals)
            if vmax == vmin:
                return {k: 1.0 for k in d}
            return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}

        vec_norm = normalize(vec_scores)
        bm_norm = normalize(bm_scores)

        candidates = []
        for cid, data in results.items():
            vs = vec_norm.get(cid, 0)
            bs = bm_norm.get(cid, 0)
            candidates.append({
                "text": data["text"],
                "source": data["meta"].get("source", ""),
                "score": round(alpha * vs + (1 - alpha) * bs, 4),
                "vector_score": round(vs, 4),
                "bm25_score": round(bs, 4),
                "metadata": data["meta"],
            })

        return candidates

    def _llm_rerank(self, query_text: str, candidates: List[Dict],
                    top_k: int = 10) -> List[Dict]:
        """
        DeepSeek Listwise 重排器（RankGPT 风格）。
        利用大模型跨语言语义理解能力，对候选列表整体重排序。
        """
        if not candidates:
            return []
        pool = candidates[:15]
        if len(pool) < 2:
            return pool[:top_k]

        # ── 序列化为 RankGPT prompt（表块使用完整 HTML，普通块使用 text）──
        lines = []
        for i, c in enumerate(pool):
            meta = c.get("metadata", {})
            title = meta.get("section_title", str(meta.get("section_id", "")))
            full_html = meta.get("full_html_content", "")
            if full_html:
                txt = full_html[:1200].replace("\n", " ").replace("\r", " ")
            else:
                txt = (c.get("text") or "")[:1500].replace("\n", " ").replace("\r", " ")
            lines.append(f"[ID: {i}] 章节: {title}\n内容: {txt}")

        context = "\n\n".join(lines)

        prompt = (
            f"你是一个专业的论文检索排序助手。请根据用户查询，"
            f"对以下论文片段按相关性从高到低排序。\n\n"
            f"用户查询: {query_text}\n\n"
            f"论文片段列表（共 {len(pool)} 条）:\n{context}\n\n"
            f"【技术细节问题 — 章节优先级规则】\n"
            f"如果用户询问具体的技术细节（如骨干网络名称、预训练数据集、参数设置、超参数、"
            f"架构设计、损失函数等），必须优先排列来自 Method/Implementation/Experiment/"
            f"Ablation Study 章节的片段。Conclusion 和 Introduction 章节通常只做概述性总结，"
            f"不包含具体技术参数，对于技术细节类问题应排在 Method/Experiment 之后。\n"
            f"例：用户问\"使用了什么骨干网络\"，Method 章节明确写明 \"Resnet-50 as backbone\" "
            f"的片段应排第一，Conclusion 章节只说 \"redesign backbone structure\" 的片段应排在后面。\n\n"
            f"请严格按照从最相关到最不相关的顺序，"
            f"仅返回一个 JSON 格式的整数列表，格式如 [2,0,1,...]。\n"
            f"列表中应包含所有 {len(pool)} 个 ID，每个 ID 出现且仅出现一次。\n"
            f"不要输出任何解释或其他内容。"
        )

        from src.config import get_llm_client, config
        client = get_llm_client()

        try:
            resp = client.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system",
                     "content": "你是一个排序专家，只返回 JSON 数组，不输出任何其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            import json, re
            array_match = re.search(r'\[[\d,\s]+\]', raw)
            if not array_match:
                raise ValueError(f"DeepSeek 未返回合法 JSON 数组: {raw[:200]}")

            ranked_ids = json.loads(array_match.group(0))

            if len(ranked_ids) != len(pool):
                logger.warning(
                    "[LLM RERANK] 返回 %d 个 ID（预期 %d），触发降级",
                    len(ranked_ids), len(pool),
                )
                candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
                return candidates[:top_k]

            # 按 ranked_ids 重排
            id_map = {i: pool[i] for i in range(len(pool))}
            reranked = [id_map[i] for i in ranked_ids if i in id_map]
            extra = [c for c in candidates if c not in pool]
            extra.sort(key=lambda x: x.get("score", 0), reverse=True)
            result = reranked + extra

            return result[:top_k]

        except Exception as e:
            import traceback
            logger.warning("[LLM RERANK WARNING] DeepSeek 重排失败 (%s)，fusion 降级", e)
            traceback.print_exc()
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            return candidates[:top_k]
