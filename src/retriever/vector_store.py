"""
FAISS 向量存储 — 适配 arxiv-scholar 的简化版
"""
import os
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import faiss

from src.config import config, get_vector_store_dir, resolve_project_path

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS 向量存储"""

    def __init__(self, dimension: int = None, persist_dir: str = None):
        self.dimension = dimension or config.embedding.dimension
        self.persist_dir = (
            resolve_project_path(persist_dir)
            if persist_dir
            else get_vector_store_dir()
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.index = faiss.IndexFlatIP(self.dimension)  # 内积 = 余弦相似度（归一化后）
        self.metadata: List[Dict] = []
        self._id_counter = 0

    def add(self, vectors: np.ndarray, metadata_list: List[Dict]):
        """批量添加向量（含硬断言：向量数与元数据数必须一致）"""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        assert vectors.shape[0] == len(metadata_list), (
            f"⛔ VectorStore.add 对齐失败: "
            f"vectors={vectors.shape[0]}, metadata={len(metadata_list)}"
        )
        self.index.add(vectors.astype(np.float32))
        for meta in metadata_list:
            meta["_idx"] = self._id_counter
            self._id_counter += 1
        self.metadata.extend(metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = None,
                filter_dict: Dict = None) -> List[Tuple[int, float, Dict]]:
        """
        搜索最近邻，支持可选的元数据过滤。
        filter_dict 示例: {"section_id": "2.3"} 或 {"section_id__startswith": "2."}
        """
        top_k = top_k or config.retrieval.top_k_vector
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if filter_dict:
            all_count = len(self.metadata)
            scores, indices = self.index.search(query_vector.astype(np.float32), max(all_count, 1))
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                if self._match_filter(meta, filter_dict):
                    results.append((int(idx), float(scores[0][i]), meta))
                    if len(results) >= top_k:
                        break
            return results
        else:
            scores, indices = self.index.search(query_vector.astype(np.float32), top_k)
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx >= 0 and idx < len(self.metadata):
                    results.append((int(idx), float(scores[0][i]), self.metadata[idx]))
            return results

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

    def save(self, name: str = "papers"):
        """持久化（含落盘前对齐断言）

        通过 VectorIOWriter 序列化为 bytes 后用 Python 写文件，
        绕开 FAISS C 层对 Windows Unicode 路径的限制。
        """
        assert self.index.ntotal == len(self.metadata), (
            f"⛔ FAISS 落盘前对齐失败: "
            f"index.ntotal={self.index.ntotal}, metadata={len(self.metadata)}"
        )
        # 用 VectorIOWriter 序列化到内存
        writer = faiss.VectorIOWriter()
        faiss.write_index(self.index, writer)
        idx_bytes = bytes(faiss.vector_to_array(writer.data))
        idx_path = self.persist_dir / f"faiss_{name}.index"
        with open(str(idx_path), "wb") as f:
            f.write(idx_bytes)
        with open(str(self.persist_dir / f"metadata_{name}.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"向量库已保存: {name} ({len(self.metadata)} 条)")

    def load(self, name: str = "papers") -> bool:
        """加载（含加载后对齐断言）

        通过 Python 读文件后在内存中用 VectorIOReader 反序列化，
        绕开 FAISS C 层对 Windows Unicode 路径的限制。
        """
        idx_path = self.persist_dir / f"faiss_{name}.index"
        meta_path = self.persist_dir / f"metadata_{name}.pkl"
        if not idx_path.exists() or not meta_path.exists():
            return False
        with open(str(idx_path), "rb") as f:
            idx_data = f.read()
        idx_vec = faiss.UInt8Vector()
        faiss.copy_array_to_vector(
            np.frombuffer(idx_data, dtype=np.uint8), idx_vec
        )
        reader = faiss.VectorIOReader()
        reader.data = idx_vec
        self.index = faiss.read_index(reader)
        with open(str(meta_path), "rb") as f:
            self.metadata = pickle.load(f)
        self._id_counter = len(self.metadata)
        assert self.index.ntotal == len(self.metadata), (
            f"⛔ FAISS 加载后对齐失败: "
            f"index={self.index.ntotal}, metadata={len(self.metadata)}"
        )
        logger.info(f"向量库已加载: {name} ({len(self.metadata)} 条)")
        return True

    @property
    def count(self) -> int:
        return len(self.metadata)
