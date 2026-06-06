"""
BM25 检索器 — 适配 arxiv-scholar 的论文检索场景

特性：
- 学术领域词典保护（AI/ML/CV/NLP 术语不被切碎）
- camelCase/PascalCase 模型名保持完整
- jieba 中文分词（如有安装）
- 数字+单位组合保持完整
"""
import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.config import config, PROJECT_ROOT
from src.chunker.section_chunker import Chunk

logger = logging.getLogger(__name__)

try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

# ── 加载学术领域用户词典 ──
_ACADEMIC_DICT_PATH = PROJECT_ROOT / "data" / "academic_dict.txt"
if HAS_JIEBA and _ACADEMIC_DICT_PATH.exists():
    jieba.load_userdict(str(_ACADEMIC_DICT_PATH))
    logger.info("已加载学术词典: %s", _ACADEMIC_DICT_PATH)

# ── 学术术语保护模式（camelCase/PascalCase/连字符组合/数字+单位） ──
# 注意：顺序很重要！更具体的模式（连字符/下划线复合词）必须放在更泛化模式之前
_ACADEMIC_TERM_PATTERN = re.compile(
    r'(?:'
    # 1. 连字符/下划线复合词（优先级最高）：BGE-M3, CVC-ClinicDB, self-attention, Swin-Transformer
    r'[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+(?:v\d+)?(?:\+\+|\+)?'
    r'|'
    # 2. PascalCase + 版本号：DeepLabV3+, LoRA, ResNet-50, UNet++
    #    使用 * 而非 + 以容纳末尾连续大写（如 LoRA, QLoRA）
    r'[A-Z][a-z\d]*(?:[A-Z][a-z\d]*)+(?:v\d+)?(?:\+\+|\+)?(?:-\d+[a-z]?)?'
    r'|'
    # 3. 大写缩写 + 可选版本号：YOLOv8, BGE, ViT, C++, DINOv2
    r'[A-Z]{2,}(?:[-_]?[vV]?\d+[a-z]?)?(?:\+\+|\+)?(?:-\w+)?'
    r'|'
    # 4. camelCase：mIoU, batchNorm, mDice
    #    使用 * 以容纳末尾连续大写（如 mIoU 的末尾 U）
    r'[a-z]+(?:[A-Z][a-z\d]*)+'
    r'|'
    # 5. 数字+单位：300GB, 10fps, 1024px
    r'\d+(?:\.\d+)?(?:GB|MB|KB|TB|PB|ms|s|fps|px|mm|cm|m|kg|GHz|MHz|kHz|W|FPS|PPM)'
    r'|'
    # 6. 分辨率：1024x1024
    r'\d+x\d+'
    r'|'
    # 7. arXiv ID：arXiv:2507.10864
    r'arXiv:\d+\.\d+'
    r')'
)


def _preserve_academic_terms(text: str) -> list:
    """从文本中分离出需要保护的学术术语和普通文本片段。

    返回 [(片段文本, 是否被保护不被分词), ...]
    """
    segments = []
    last_end = 0
    for m in _ACADEMIC_TERM_PATTERN.finditer(text):
        if m.start() > last_end:
            segments.append((text[last_end:m.start()], False))
        segments.append((m.group(), True))
        last_end = m.end()
    if last_end < len(text):
        segments.append((text[last_end:], False))
    return segments if segments else [(text, False)]


class BM25Retriever:
    """BM25 关键词检索器，与向量检索互补

    与纯向量检索不同，BM25 对学术术语、模型名、指标名等精确字符串
    匹配天然更准确。学术词典 + 术语保护确保 BGE-M3、DeepLabV3+
    这类模式不会被切碎。
    """

    def __init__(self):
        self._corpus: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._chunks = []
        self._bm25 = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """学术感知分词器。

        策略：
        1. 先用正则保护学术术语（camelCase/连字符/数字单位）
        2. 对剩余文本：中文用 jieba，英文用空白拆分
        3. 最终全部小写并去空白
        """
        tokens = []
        segments = _preserve_academic_terms(text)

        for seg_text, is_protected in segments:
            if not seg_text.strip():
                continue
            if is_protected:
                # 保护的学术术语：保持完整，只做小写
                tokens.append(seg_text.strip().lower())
            elif HAS_JIEBA:
                # jieba 处理中英混合文本
                cut_tokens = jieba.lcut(seg_text)
                tokens.extend(t.strip().lower() for t in cut_tokens if t.strip())
            else:
                # 回退：中文按字切，英文按空白切
                for part in re.split(r'(\s+)', seg_text):
                    part = part.strip()
                    if not part:
                        continue
                    if re.match(r'^[一-鿿]+$', part):
                        tokens.extend(list(part.lower()))
                    else:
                        tokens.extend(t.lower() for t in part.split() if t)

        return [t for t in tokens if t]

    def index(self, chunks):
        self._chunks = chunks
        self._corpus = [chunk.text for chunk in chunks]
        self._tokenized_corpus = [self.tokenize(text) for text in self._corpus]
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 索引: %d 个文档", len(chunks))

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
        logger.info("BM25 索引已保存: %s (%d chunks)", path, len(self._chunks))

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
        logger.info("BM25 索引已加载: %s (%d chunks)", path, len(self._chunks))
