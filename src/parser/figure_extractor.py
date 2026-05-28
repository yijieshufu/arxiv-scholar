"""
PDF 图表提取器 v2 — PyMuPDF 原生提取，不依赖 Docling 布局树

策略：
1. 每页整页截图（兜底）
2. 从 PyMuPDF 文本块提取 "Figure X." / "Table X." 图注
3. 从图注位置向上估算图表区域 → 裁剪
4. 提取页面嵌入的光栅图片（photos, 图表图片）
5. 图注匹配到最近的裁剪/图片区域
"""
import io
import re
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)

FIGURES_DB = Path("data/figures/figures.db")
FIGURES_DB.parent.mkdir(parents=True, exist_ok=True)

# 真正的图注：以 "Figure 1." 或 "Table 2:" 开头
_FIGURE_CAPTION_RE = re.compile(
    r'^(?:Figure|Fig\.?|TABLE|Table|图|表)\s+\d+[\.:]\s',
    re.IGNORECASE,
)


def _init_db():
    conn = sqlite3.connect(str(FIGURES_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS figures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_source TEXT NOT NULL,
            figure_id TEXT NOT NULL,
            figure_type TEXT NOT NULL,
            page_no INTEGER NOT NULL,
            caption TEXT DEFAULT '',
            page_text TEXT DEFAULT '',
            image BLOB NOT NULL,
            width INTEGER DEFAULT 0,
            height INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fig_paper ON figures(paper_source, figure_type)")
    conn.commit()

    # FTS5 全文索引（caption + page_text 双字段，支持 BM25 排序）
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS figures_fts
            USING fts5(caption, page_text, content='figures', content_rowid='id', tokenize='unicode61')
        """)
    except Exception:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS figures_fts
            USING fts5(caption, page_text, content='figures', content_rowid='id')
        """)
        logger.warning("FTS5 降级为默认分词器（如需中文支持请安装支持 unicode61 的 sqlite3）")

    # 触发器：figures 表更新时同步 FTS
    for trigger_sql in [
        """CREATE TRIGGER IF NOT EXISTS figures_ai AFTER INSERT ON figures BEGIN
            INSERT INTO figures_fts(rowid, caption, page_text)
            VALUES (new.id, new.caption, new.page_text);
        END""",
        """CREATE TRIGGER IF NOT EXISTS figures_ad AFTER DELETE ON figures BEGIN
            INSERT INTO figures_fts(figures_fts, rowid, caption, page_text)
            VALUES ('delete', old.id, old.caption, old.page_text);
        END""",
        """CREATE TRIGGER IF NOT EXISTS figures_au AFTER UPDATE ON figures BEGIN
            INSERT INTO figures_fts(figures_fts, rowid, caption, page_text)
            VALUES ('delete', old.id, old.caption, old.page_text);
            INSERT INTO figures_fts(rowid, caption, page_text)
            VALUES (new.id, new.caption, new.page_text);
        END""",
    ]:
        try:
            conn.execute(trigger_sql)
        except Exception:
            pass

    conn.commit()
    return conn


# ── 辅助函数 ──

def _page_text_captions(page) -> List[dict]:
    """从 PyMuPDF 页面文本块中提取 Figure/Table 图注。

    只返回真正以 "Figure X." / "Table X:" 开头的图注。
    "Figure X showed..." 等正文引用不返回（标记为 reference）。
    """
    results = []
    blocks = page.get_text("blocks")
    for b in blocks:
        text = b[4].strip()
        if not text:
            continue
        if _FIGURE_CAPTION_RE.match(text):
            # 真正的图注 → 用于裁剪
            m = re.match(r'(?:Figure|Fig\.?|TABLE|Table|图|表)\s+(\d+)', text, re.IGNORECASE)
            num = int(m.group(1)) if m else 0
            ftype = "table" if "table" in text[:10].lower() or "表" in text[0] else "figure"
            results.append({
                "num": num,
                "type": ftype,
                "y0": b[1], "y1": b[3], "x0": b[0], "x1": b[2],
                "text": text[:200],
                "is_caption": True,
            })
    return results


def _estimate_figure_rect(caption: dict, page_height: float, page_width: float) -> tuple:
    """
    从图注位置估算图表区域。使用更大的裁剪范围（60%）确保不遗漏图表内容。

    表格：图注在表格上方 → 向下载入 ~60% 页面高度
    图片：图注在图片下方 → 向上载入 ~60% 页面高度

    Returns: (x0, y0, x1, y1) in points
    """
    pad = 20
    crop_ratio = 0.60
    if caption["type"] == "table":
        est_bottom = min(page_height, caption["y0"] + page_height * crop_ratio)
        est_top = max(0, caption["y1"])
        return (max(0, caption["x0"] - pad), est_top,
                min(page_width, caption["x1"] + pad), est_bottom)
    else:
        est_top = max(0, caption["y0"] - page_height * crop_ratio)
        return (max(0, caption["x0"] - pad), est_top,
                min(page_width, caption["x1"] + pad), min(page_height, caption["y1"] + pad))


# ── 核心提取函数 ──

def extract_figures(pdf_path: Path, dpi: int = 200) -> int:
    """
    从 PDF 提取整页截图 + 图表裁剪，存入 SQLite。

    完全基于 PyMuPDF，不依赖 Docling 布局树。
    """
    import fitz

    source = pdf_path.name
    conn = _init_db()
    saved = 0
    doc = fitz.open(str(pdf_path))
    scale = dpi / 72

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_no = page_idx + 1
        pw, ph = page.rect.width, page.rect.height  # points

        # ── 该页完整文本（用于 FTS5 搜索） ──
        page_text = page.get_text("text")[:5000]

        # ── 整页截图（兜底） ──
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img_bytes = pix.tobytes("png")
        conn.execute(
            "INSERT INTO figures (paper_source, figure_id, figure_type, page_no, image, width, height, page_text) "
            "VALUES (?, ?, 'page', ?, ?, ?, ?, ?)",
            (source, f"page_{page_no}", page_no, img_bytes, pix.width, pix.height, page_text),
        )
        saved += 1

        # ── 提取嵌入光栅图片 ──
        imgs_on_page = page.get_images(full=True)
        for img_idx, img_info in enumerate(imgs_on_page):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
                img_data = base["image"]
                w, h = base["width"], base["height"]
                # 过滤小图（logo、图标）
                if len(img_data) < 20 * 1024:
                    continue
                fig_id = f"raster_{img_idx + 1}"
                conn.execute(
                    "INSERT INTO figures (paper_source, figure_id, figure_type, page_no, image, width, height) "
                    "VALUES (?, ?, 'raster', ?, ?, ?, ?)",
                    (source, fig_id, page_no, img_data, w, h),
                )
                saved += 1
            except Exception:
                pass

        # ── 从图注位置估算图表区域并裁剪 ──
        captions = _page_text_captions(page)
        page_img_pil = None  # 延迟加载

        for cap in captions:
            rect = _estimate_figure_rect(cap, ph, pw)
            r_x0, r_y0, r_x1, r_y1 = rect
            if r_x1 <= r_x0 or r_y1 <= r_y0:
                continue

            # 延迟加载 PIL 图片
            if page_img_pil is None:
                page_img_pil = Image.open(io.BytesIO(bytes(img_bytes)))

            # 裁剪：points → pixels
            cx0 = int(r_x0 * scale)
            cy0 = int(pix.height - r_y1 * scale)  # 注意 Y 轴翻转
            cx1 = int(r_x1 * scale)
            cy1 = int(pix.height - r_y0 * scale)
            if cx1 <= cx0 or cy1 <= cy0:
                continue

            crop = page_img_pil.crop((cx0, cy0, cx1, cy1))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop_bytes = buf.getvalue()

            fig_id = f"{cap['type']}_{cap['num']}"
            conn.execute(
                "INSERT INTO figures (paper_source, figure_id, figure_type, page_no, image, width, height, caption, page_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source, fig_id, cap["type"], page_no, crop_bytes, crop.width, crop.height, cap["text"], page_text),
            )
            saved += 1

    conn.commit()
    conn.close()
    doc.close()
    logger.info(f"图床 v2: {source} → {saved} 张")
    return saved


# ── 查询接口（不变） ──

# ── FTS5 全文搜索 ──

def search_figures_fts(query: str, paper_source: str = "", limit: int = 10) -> List[dict]:
    """
    FTS5 全文搜索图表，支持 BM25 排序。

    Args:
        query: 搜索词（FTS5 语法：支持 AND/OR/NOT、"短语匹配"、*前缀）
        paper_source: 可选，限定论文
        limit: 最多返回条数
    """
    conn = sqlite3.connect(str(FIGURES_DB))
    conn.row_factory = sqlite3.Row

    # FTS5 查询构造
    q = query.replace(" ", " AND ")  # 默认 AND 连接
    sql = """
        SELECT f.*, rank
        FROM figures_fts
        JOIN figures ON figures_fts.rowid = figures.id
        WHERE figures_fts MATCH ?
    """
    params = [q]
    if paper_source:
        sql += " AND f.paper_source = ?"
        params.append(paper_source)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        # 降级到 LIKE 搜索
        like_q = f"%{query}%"
        sql = "SELECT * FROM figures WHERE caption LIKE ? OR page_text LIKE ?"
        params = [like_q, like_q]
        if paper_source:
            sql += " AND paper_source = ?"
            params.append(paper_source)
        sql += " LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def get_figure(paper_source: str, figure_id: str) -> Optional[dict]:
    conn = sqlite3.connect(str(FIGURES_DB))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM figures WHERE paper_source = ? AND figure_id = ?",
        (paper_source, figure_id),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_page_images(paper_source: str, page_no: int) -> List[dict]:
    conn = sqlite3.connect(str(FIGURES_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM figures WHERE paper_source = ? AND page_no = ? ORDER BY id",
        (paper_source, page_no),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_by_caption(keyword: str) -> List[dict]:
    conn = sqlite3.connect(str(FIGURES_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM figures WHERE caption LIKE ? LIMIT 20",
        (f"%{keyword}%",),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_paper_figures(paper_source: str) -> List[dict]:
    conn = sqlite3.connect(str(FIGURES_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM figures WHERE paper_source = ? AND figure_type != 'page' ORDER BY page_no, id",
        (paper_source,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
