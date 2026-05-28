"""
BM25 检索器 — 适配 arxiv-scholar 的论文检索场景

从 resume-ai 迁移，修改了 import 路径和 Chunk 接口。
"""
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.config import config
from src.chunker.section_chunker import Chunk

logger = logging.getLogger(__name__)

try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


class BM25Retriever:
    """BM25 关键词检索器，与向量检索互补"""

    def __init__(self):
        self._corpus: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._chunks = []
        self._bm25 = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if HAS_JIEBA:
            tokens = jieba.lcut(text)
        else:
            import re
            tokens = []
            for part in re.split(r'(\s+)', text):
                if re.match(r'^[\u4e00-\u9fff]+$', part):
                    tokens.extend(list(part))
                else:
                    tokens.extend(part.split())
        return [t.strip().lower() for t in tokens if t.strip()]

    def index(self, chunks):
        self._chunks = chunks
        self._corpus = [chunk.text for chunk in chunks]
        self._tokenized_corpus = [self.tokenize(text) for text in self._corpus]
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 索引: {len(chunks)} 个文档")

    def search(self, query: str, top_k: int = None, filter_dict: Dict = None):
        if self._bm25 is None or not self._chunks:
            return []
        top_k = top_k or config.retrieval.top_k_bm25
        tokenized = self.tokenize(query)
        scores = self._bm25.get_scores(tokenized)

        if filter_dict:
            all_idx = np.argsort(scores)[::-1]
            results = []
            for idx in all_idx:
                chunk = self._chunks[idx]
                if self._match_filter(chunk.metadata, filter_dict):
                    results.append((chunk, float(scores[idx])))
                    if len(results) >= top_k:
                        break
            return results
        else:
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [(self._chunks[i], float(scores[i])) for i in top_idx]

    @staticmethod
    def _match_filter(meta: Dict, filter_dict: Dict) -> bool:
        for key, value in filter_dict.items():
            if key.endswith("__startswith"):
                actual_key = key[:-len("__startswith")]
                if not str(meta.get(actual_key, "")).startswith(str(value)):
                    return False
            else:
                if meta.get(key) != value:
                    return False
        return True

    def append(self, chunks):
        """增量追加文档并重建 BM25 索引"""
        if not chunks:
            return
        if self._chunks:
            self.index(self._chunks + list(chunks))
        else:
            self.index(chunks)

    def save(self, path: str):
        payload = {
            "corpus": self._corpus,
            "tokenized_corpus": self._tokenized_corpus,
            "chunks": [
                {"text": c.text, "metadata": c.metadata}
                for c in self._chunks
            ],
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"BM25 索引已保存: {path} ({len(self._chunks)} chunks)")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._corpus = data["corpus"]
        self._tokenized_corpus = data["tokenized_corpus"]
        saved_chunks = data.get("chunks")
        if saved_chunks:
            self._chunks = [
                Chunk(text=item["text"], metadata=item.get("metadata", {}))
                for item in saved_chunks
            ]
        else:
            # 兼容旧版索引：仅有 corpus，无 chunk 元数据
            self._chunks = [
                Chunk(text=text, metadata={"chunk_id": str(i)})
                for i, text in enumerate(self._corpus)
            ]
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 索引已加载: {path} ({len(self._chunks)} chunks)")
