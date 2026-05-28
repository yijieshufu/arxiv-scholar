"""
Embedding 模块 — 文本向量化（复用 resume-ai 设计）
"""
import logging
from typing import List
import numpy as np

from src.config import config

logger = logging.getLogger(__name__)

# 设备检测缓存
_HAS_CUDA = None
_HAS_MPS = None


def _detect_cuda() -> bool:
    global _HAS_CUDA
    if _HAS_CUDA is None:
        try:
            import torch
            _HAS_CUDA = torch.cuda.is_available()
        except Exception:
            _HAS_CUDA = False
    return _HAS_CUDA


def _detect_mps() -> bool:
    global _HAS_MPS
    if _HAS_MPS is None:
        try:
            import torch
            _HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            _HAS_MPS = False
    return _HAS_MPS


# 模型缓存（跨实例复用，防止反复加载）
_MODEL_CACHE = {}


def resolve_device(device_setting: str = None) -> str:
    """解析设备设置："auto" → GPU 检测，否则原样返回"""
    device_setting = device_setting or config.embedding.device
    if device_setting != "auto":
        return device_setting
    if _detect_cuda():
        logger.info("✅ 检测到 CUDA GPU，自动启用 GPU 加速")
        return "cuda"
    if _detect_mps():
        logger.info("✅ 检测到 Apple Silicon MPS，自动启用 MPS 加速")
        return "mps"
    logger.info("ℹ️ 未检测到 GPU，使用 CPU")
    return "cpu"


def _clear_model_cache():
    """清空模型缓存（测试/重载时使用）"""
    _MODEL_CACHE.clear()


class Embedder:
    """文本 Embedding 编码器（支持 GPU 自动检测）"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding.model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            device = resolve_device()
            cache_key = f"{self.model_name}:{device}"
            if cache_key in _MODEL_CACHE:
                self._model = _MODEL_CACHE[cache_key]
                logger.info(f"复用模型缓存: {self.model_name} on {device}")
            else:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name,
                    device=device,
                )
                _MODEL_CACHE[cache_key] = self._model
                logger.info(f"首次加载 Embedding 模型: {self.model_name} on {device}")
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        """批量编码（已支持 batch_size，见 config.embedding.batch_size=32）"""
        embeddings = self.model.encode(
            texts,
            batch_size=config.embedding.batch_size,
            normalize_embeddings=config.embedding.normalize,
            show_progress_bar=len(texts) > 100,
        )
        # 清理 GPU 缓存，防止显存碎片化影响后续 reranker / 编码
        if config.embedding.device == "auto" and _detect_cuda():
            import torch
            torch.cuda.empty_cache()
        return np.array(embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """单条编码"""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return config.embedding.dimension


class APIEmbedder:
    """用线上 Embedding API 编码（无需加载本地模型，首查秒回）"""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=config.llm.api_key,
                base_url=config.llm.base_url,
            )
        return self._client

    def encode(self, texts: List[str]) -> np.ndarray:
        """调用 Embedding API 批量编码（兼容 OpenAI / 兼容接口）"""
        try:
            resp = self.client.embeddings.create(
                model=config.embedding.api_model,
                input=texts,
            )
        except Exception as e:
            raise RuntimeError(
                f"Embedding API 调用失败: {e}\n"
                f"当前 API 提供商 ({config.llm.base_url}) 可能不支持 Embedding 模型 '{config.embedding.api_model}'。\n"
                f"请尝试: ① 在 sidebar 关闭「API Embedding」开关 ② 切到本地模型 ③ 更换为支持的 API 提供商"
            ) from e
        ordered = sorted(resp.data, key=lambda x: x.index)
        embeddings = np.array([item.embedding for item in ordered], dtype=np.float32)
        if config.embedding.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return config.embedding.dimension


def create_embedder():
    """根据配置创建 Embedder（API 或本地）"""
    if config.embedding.use_api:
        logger.info(f"使用 API Embedding: {config.embedding.api_model}")
        return APIEmbedder()
    logger.info(f"使用本地 Embedding 模型: {config.embedding.model_name}")
    return Embedder()
