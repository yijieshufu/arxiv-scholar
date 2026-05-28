"""
论文索引注册表 — 管理每篇论文的独立索引路径

目录结构：
vector_store/
└── papers/
    ├── {paper_slug}/
    │   ├── faiss.index
    │   ├── bm25.pkl
    │   └── metadata.pkl
    └── routes.json          # 论文名 → 目录映射
"""
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

VECTOR_STORE = Path("data/vector_store")
PAPERS_DIR = VECTOR_STORE / "papers"
ROUTES_FILE = VECTOR_STORE / "routes.json"


def _slugify(source: str) -> str:
    """将 PDF 文件名转为合法目录名。"""
    name = source.replace(".pdf", "").replace(" ", "_")
    # 限制长度
    return name[:80]


def _get_paper_dir(source: str) -> Path:
    return PAPERS_DIR / _slugify(source)


class PaperRegistry:
    """单篇论文独立索引注册表。"""

    def __init__(self):
        self._routes: Dict[str, str] = {}  # source -> dir_name
        self._load()

    # ── 读写 ──

    def _load(self):
        if ROUTES_FILE.exists():
            try:
                self._routes = json.loads(ROUTES_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"routes.json 读取失败: {e}")
                self._routes = {}

    def _save(self):
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        ROUTES_FILE.write_text(
            json.dumps(self._routes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 增删查 ──

    def register(self, source: str):
        """注册一篇论文（如已存在则跳过）。"""
        if source in self._routes:
            return
        dir_name = _slugify(source)
        self._routes[source] = dir_name
        self._save()
        logger.info(f"注册论文: {source} → {dir_name}")

    def unregister(self, source: str):
        """注销一篇论文 + 删除索引文件。"""
        dir_name = self._routes.pop(source, None)
        if dir_name:
            paper_dir = PAPERS_DIR / dir_name
            if paper_dir.exists():
                shutil.rmtree(str(paper_dir))
            self._save()
            logger.info(f"注销论文: {source}")

    def get_paper_dir(self, source: str) -> Optional[Path]:
        """获取某篇论文的索引目录。"""
        dir_name = self._routes.get(source)
        if not dir_name:
            return None
        d = PAPERS_DIR / dir_name
        return d if d.exists() else None

    def list_papers(self) -> List[str]:
        """返回所有已注册的论文 source 列表。"""
        return list(self._routes.keys())

    def has_source(self, source: str) -> bool:
        """检查某篇论文是否已注册。"""
        return source in self._routes

    def count(self) -> int:
        return len(self._routes)
