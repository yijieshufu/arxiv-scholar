"""论文 PDF 解析器 — 提取文本、表格、参考文献"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import re

logger = logging.getLogger(__name__)


class PaperParser:
    """
    论文 PDF 解析器。

    参考 paper-qa 的设计：
    - 默认 Docling 元素流（仅 TableItem → HTML，正文保持 Markdown）
    - 回退 pdfplumber（无 Docling 或转换失败时）
    - 回退 PyPDF2（纯文本快速路径）

    输出格式：
    - 全文纯文本（用于切片和检索）
    - 分段信息（用于章节识别）
    - 参考文献列表（用于引用追踪）
    """

    # 已知英文高频词（用于粘连修复时的边界判断）
    _COMMON_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "need",
        "dare", "ought", "used", "this", "that", "these", "those", "it", "its",
        "we", "our", "you", "your", "they", "their", "he", "she", "him", "her",
        "not", "no", "nor", "all", "each", "every", "both", "few", "more",
        "most", "some", "any", "such", "only", "own", "same", "so", "than",
        "too", "very", "just", "also", "if", "then", "else", "when", "where",
        "why", "how", "which", "who", "whom", "what", "while", "although",
        "because", "since", "until", "after", "before", "between", "under",
        "over", "through", "during", "without", "within", "about", "against",
        "between", "into", "through", "throughout", "via", "per", "among",
        "vision", "medical", "image", "images", "model", "segment",
        "segmentation",
        "detection", "classification", "network", "learning", "training",
        "testing", "validation", "dataset", "data", "feature", "features",
        "layer", "layers", "input", "output", "kernel", "filter", "pooling",
        "convolution", "attention", "transformer", "encoder", "decoder",
        "algorithm", "method", "approach", "framework", "architecture",
        "performance", "accuracy", "precision", "recall", "score",
    }

    # 无内横线学术大表：用文本列/行对齐，不依赖 PDF 矢量线
    TEXT_TABLE_SETTINGS = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 3,
        "text_x_tolerance": 3,
        "text_y_tolerance": 3,
    }

    # 带完整网格线的表：线条策略作回退
    LINES_TABLE_SETTINGS = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
    }

    _docling_converter: Any = None

    def __init__(self, engine: str = "auto"):
        self.engine = engine

    def parse(self, pdf_path: str) -> Dict[str, object]:
        """
        解析论文 PDF。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            {
                "full_text": str,           # 全文
                "pages": List[str],         # 每页文本
                "title": str,               # 标题（从第一页提取）
                "abstract": str,            # 摘要
                "references": List[str],    # 参考文献
                "tables": List[str],        # 表格（HTML 格式）
            }
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

        if self.engine == "auto":
            return self._parse_with_auto(path)
        elif self.engine == "pdfplumber":
            return self._parse_with_pdfplumber(path)
        elif self.engine == "pypdf2":
            return self._parse_with_pypdf2(path)
        elif self.engine == "docling":
            return self._parse_with_docling(path)
        elif self.engine == "mineru":
            return self._parse_with_mineru(path)
        else:
            raise ValueError(f"不支持的解析引擎: {self.engine}")

    def _parse_with_pdfplumber(self, path: Path) -> Dict:
        """
        pdfplumber 纯文本回退：不执行文本对齐表格盲猜，避免多栏页误检。
        结构化表格仅由 Docling TableItem 通道产出。
        """
        import pdfplumber

        hyperlinks = self._extract_hyperlinks(str(path))
        pages = []

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=3) or ""
                text = self._clean_glued_text(text)
                if i in hyperlinks:
                    for url in hyperlinks[i]:
                        if "github" in url.lower() or "gitlab" in url.lower():
                            text += f"\n[GitHub Code Link: {url}]"
                        else:
                            text += f"\n[Link: {url}]"
                pages.append(text)

        full_text = "\n\n".join(pages)

        title = self._extract_title(full_text)
        abstract = self._extract_abstract(full_text)
        references = self._extract_references(full_text)

        return {
            "full_text": full_text,
            "pages": pages,
            "title": title,
            "abstract": abstract,
            "references": references,
            "tables": [],
        }

    @classmethod
    def _find_page_tables(cls, page) -> list:
        """文本对齐优先；与线条策略候选去重后保留高分、可信表格。"""
        page_area = float(page.width * page.height) or 1.0
        ranked: List[tuple] = []

        for settings in (cls.TEXT_TABLE_SETTINGS, cls.LINES_TABLE_SETTINGS):
            for table_obj in page.find_tables(table_settings=settings):
                grid = table_obj.extract(table_settings=settings) or []
                if not cls._is_plausible_table(grid, table_obj.bbox, page_area):
                    continue
                ranked.append(
                    (cls._score_table_grid(grid), table_obj, settings, grid)
                )

        if not ranked:
            return []

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected: list = []
        for score, table_obj, settings, grid in ranked:
            if any(
                cls._bbox_iou(table_obj.bbox, kept.bbox) > 0.72
                for kept in selected
            ):
                continue
            table_obj._arxiv_extract_settings = settings  # noqa: SLF001
            selected.append(table_obj)
        return selected

    @classmethod
    def _extract_table_grid(cls, table_obj, page) -> List[List[str]]:
        """按发现时策略提取；对同一 bbox 比较 text/lines 取更优网格。"""
        page_area = float(page.width * page.height) or 1.0
        preferred = getattr(table_obj, "_arxiv_extract_settings", None)
        settings_chain = [preferred] if preferred else []
        settings_chain += [
            s for s in (cls.TEXT_TABLE_SETTINGS, cls.LINES_TABLE_SETTINGS)
            if s not in settings_chain
        ]

        best_grid: List[List[str]] = []
        best_score = -1.0
        for settings in settings_chain:
            if not settings:
                continue
            grid = table_obj.extract(table_settings=settings) or []
            if not cls._is_plausible_table(grid, table_obj.bbox, page_area):
                continue
            score = cls._score_table_grid(grid)
            if score > best_score:
                best_score = score
                best_grid = grid
        return best_grid

    @staticmethod
    def _bbox_iou(a, b) -> float:
        ax0, atop, ax1, abot = a
        bx0, btop, bx1, bbot = b
        ix0, iy0 = max(ax0, bx0), max(atop, btop)
        ix1, iy1 = min(ax1, bx1), min(abot, bbot)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        area_a = (ax1 - ax0) * (abot - atop)
        area_b = (bx1 - bx0) * (bbot - btop)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _is_plausible_table(
        grid: List[List[str]], bbox, page_area: float
    ) -> bool:
        """过滤整页误检（如封面）；保留 Methods / 指标类学术表。"""
        if not grid or not grid[0]:
            return False
        nrows = len(grid)
        ncols = max(len(r) for r in grid)
        if nrows < 2 or ncols < 2:
            return False

        x0, top, x1, bottom = bbox
        area_ratio = ((x1 - x0) * (bottom - top)) / page_area
        flat = " ".join(str(c or "") for row in grid for c in row)
        has_metrics = any(
            k in flat
            for k in (
                "Method", "mDice", "mIoU", "Dice", "IoU", "Publication",
                "Dataset", "Accuracy", "F1", "Recall", "Precision",
                "PraNet", "UNet",
            )
        )
        numeric_cells = sum(
            1 for row in grid for c in row
            if re.search(r"\d+\.\d{2,}", str(c or ""))
        )

        if area_ratio > 0.55:
            if has_metrics and ncols >= 5:
                return True
            if has_metrics and nrows <= 30 and numeric_cells >= 3:
                return True
            return False
        if area_ratio > 0.35 and not has_metrics and numeric_cells < 2:
            return False
        return True

    @staticmethod
    def _score_table_grid(grid: List[List[str]]) -> float:
        """启发式评分：行数多、列数多、单元格内换行少 → 更可信。"""
        if not grid or not grid[0]:
            return 0.0
        nrows = len(grid)
        ncols = max(len(r) for r in grid)
        nonempty = sum(
            1 for row in grid for c in row if str(c or "").strip()
        )
        multiline_penalty = sum(
            1 for row in grid for c in row
            if c and "\n" in str(c)
        )
        flat = " ".join(str(c or "") for row in grid for c in row)
        method_bonus = 80 if "Method" in flat else 0
        numeric_bonus = sum(
            3 for row in grid for c in row
            if re.search(r"\d+\.\d{2,}", str(c or ""))
        )
        return (
            nrows * ncols * 2 + nonempty + method_bonus + numeric_bonus
            - multiline_penalty * 5
        )

    @staticmethod
    def _split_multiline_cells(grid: List[List[str]]) -> List[List[str]]:
        """
        将仍含 \\n 的单元格按行展开（线条策略误合并时的兜底）。
        仅在首列（Methods 等标签列）多行且其它列单行时展开。
        """
        if not grid:
            return grid
        ncols = max(len(r) for r in grid)
        first_col_lines = []
        for row in grid:
            cell0 = str((row[0] if row else "") or "").strip()
            if "\n" in cell0:
                first_col_lines.append([ln.strip() for ln in cell0.split("\n") if ln.strip()])
            else:
                first_col_lines.append([cell0] if cell0 else [""])

        max_splits = max(len(parts) for parts in first_col_lines)
        if max_splits <= 1:
            return grid

        other_multiline = any(
            "\n" in str(row[ci] or "")
            for row in grid
            for ci in range(1, min(len(row), ncols))
        )
        if other_multiline:
            return grid

        expanded = []
        for row, parts in zip(grid, first_col_lines):
            pad = parts + [""] * (max_splits - len(parts))
            for label in pad:
                new_row = [label] + [
                    str((row[ci] if ci < len(row) else "") or "").strip()
                    for ci in range(1, ncols)
                ]
                expanded.append(new_row)
        return expanded if len(expanded) > len(grid) else grid

    def _parse_with_pypdf2(self, path: Path) -> Dict:
        """使用 PyPDF2 解析（回退方案，同用粘连修复）"""
        from PyPDF2 import PdfReader

        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text = self._clean_glued_text(text)  # 同样修复 PyPDF2 的粘连问题
            pages.append(text)

        full_text = "\n\n".join(pages)
        title = self._extract_title(full_text)
        abstract = self._extract_abstract(full_text)
        references = self._extract_references(full_text)

        return {
            "full_text": full_text,
            "pages": pages,
            "title": title,
            "abstract": abstract,
            "references": references,
            "tables": [],
        }

    @classmethod
    def _get_docling_converter(cls):
        """Docling 转换器（CPU + 关闭 OCR，降低显存/内存占用）。"""
        if cls._docling_converter is not None:
            return cls._docling_converter
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import (
            AcceleratorOptions,
            AcceleratorDevice,
        )

        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            accelerator_options=AcceleratorOptions(
                num_threads=4,
                device=AcceleratorDevice.CPU,
            ),
        )
        cls._docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        return cls._docling_converter

    @staticmethod
    def _is_docling_table_item(item) -> bool:
        """Type Shield：仅 Docling 布局树中的 TableItem 节点可生成 HTML 表块。"""
        try:
            from docling_core.types.doc import TableItem
        except ImportError:
            return type(item).__name__ == "TableItem"
        if not isinstance(item, TableItem):
            return False
        data = getattr(item, "data", None)
        if data is None:
            return False
        rows = int(getattr(data, "num_rows", 0) or 0)
        cols = int(getattr(data, "num_cols", 0) or 0)
        return rows >= 1 and cols >= 1

    @staticmethod
    def _docling_page_no(item) -> int:
        prov = getattr(item, "prov", None) or []
        if prov:
            return int(getattr(prov[0], "page_no", 1) or 1)
        return 0

    @classmethod
    def _docling_item_plaintext(cls, item, doc) -> str:
        """段落/标题/列表 → 纯 Markdown 文本，严禁包裹 <table>。"""
        if cls._is_docling_table_item(item):
            return ""
        try:
            from docling_core.types.doc import DocItemLabel
        except ImportError:
            DocItemLabel = None  # type: ignore

        label = getattr(item, "label", None)
        if DocItemLabel and label in (
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.PICTURE,
        ):
            return ""

        text = ""
        if hasattr(item, "export_to_markdown"):
            try:
                text = item.export_to_markdown(doc=doc) or ""
            except Exception:
                text = getattr(item, "text", "") or ""
        else:
            text = getattr(item, "text", "") or ""
        text = text.strip()
        if not text:
            return ""
        # 对象纯净性：正文中若混入表格 HTML，整段丢弃（表格由 TableItem 专通道处理）
        if re.search(r"<\s*table[\s>]", text, re.IGNORECASE):
            return ""
        return cls._clean_glued_text(text)

    @staticmethod
    def _table_data_to_grid(table_data) -> List[List[str]]:
        """Docling TableData → 二维字符串矩阵。"""
        if not table_data:
            return []
        rows = int(getattr(table_data, "num_rows", 0) or 0)
        cols = int(getattr(table_data, "num_cols", 0) or 0)
        if rows < 1 or cols < 1:
            return []
        grid = [[""] * cols for _ in range(rows)]
        for cell in getattr(table_data, "table_cells", []) or []:
            r = int(getattr(cell, "start_row_offset_idx", 0) or 0)
            c = int(getattr(cell, "start_col_offset_idx", 0) or 0)
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = str(getattr(cell, "text", "") or "").strip()
        return grid

    @staticmethod
    def _resolve_table_id(caption: str, fallback_index: int) -> str:
        """从 Caption 解析 Table_N（严格 \\bTable\\s+N，拒绝 Tabs. 1 to 3 类引用）。"""
        if caption:
            m = re.search(r"\bTable\s+(\d+)\b", caption, re.IGNORECASE)
            if m:
                return f"Table_{m.group(1)}"
            m_id = re.search(r"Table_(\d+)\b", caption, re.IGNORECASE)
            if m_id:
                return f"Table_{m_id.group(1)}"
            roman = {
                "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
                "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
            }
            m2 = re.search(r"\bTable\s+([IVX]+)\b", caption, re.IGNORECASE)
            if m2 and m2.group(1).upper() in roman:
                return f"Table_{roman[m2.group(1).upper()]}"
        return f"Table_{fallback_index + 1}"

    @classmethod
    def _docling_table_block(
        cls,
        item,
        doc,
        table_index: int,
        pending_caption: str = "",
    ) -> Tuple[str, Dict[str, object]]:
        """TableItem 专属通道 → [TABLE_HTML] 块 + 元数据断言。

        直接使用 Docling 原生 export_to_html()（保留 colspan/rowspan），
        不经过 grid 重拼以避免结构丢失。
        fallback：当 export_to_html 返回空时降级为 grid 拼 HTML。
        """
        # ── 方案 A（主路径）：原生 export_to_html() ──
        raw_html = ""
        caption = ""
        try:
            raw_html = item.export_to_html(doc=doc) or ""
            cap_m = re.search(
                r"<caption[^>]*>(.*?)</caption>",
                raw_html,
                re.IGNORECASE | re.DOTALL,
            )
            if cap_m:
                caption = re.sub(r"<[^>]+>", " ", cap_m.group(1)).strip()
        except Exception:
            raw_html = ""

        page_num = cls._docling_page_no(item)
        table_id = cls._resolve_table_id(caption or pending_caption, table_index)

        if raw_html.strip():
            # 移除 <caption>（我们单独管理 caption 元数据）
            html_no_cap = re.sub(
                r'<caption[^>]*>.*?</caption>',
                "",
                raw_html,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            meta_tag = f"[TABLE_HTML: {table_id} | Page {page_num}]"
            html_block = f"{meta_tag}\n{html_no_cap}\n[/TABLE_HTML]"
            meta = cls.assert_table_block_metadata(html_block, table_id, page_num)
            meta["caption"] = caption[:500] if caption else ""
            return html_block, meta

        # ── 方案 B（降级）：grid 重拼 ──
        grid = cls._table_data_to_grid(getattr(item, "data", None))
        if not grid:
            logger.warning(
                "Docling TableItem 无数据，跳过: %s page %s",
                table_id,
                page_num,
            )
            return "", {}

        html_block = cls._table_to_html(grid, table_id, page_num=page_num)
        meta = cls.assert_table_block_metadata(html_block, table_id, page_num)
        meta["caption"] = (caption or pending_caption)[:500] if (caption or pending_caption) else ""
        return html_block, meta

    @staticmethod
    def assert_table_block_metadata(
        block: str,
        expected_table_id: str,
        expected_page: int,
    ) -> Dict[str, object]:
        """表格原子块元数据断言（供切片/索引落盘校验）。"""
        detected = PaperParser._detect_table_metadata(block)
        meta: Dict[str, object] = {
            "is_table": True,
            "table_id": expected_table_id,
            "table_page": str(expected_page) if expected_page else "",
            "contains_table": True,
        }
        if not detected.get("is_table"):
            logger.warning("表格块缺少 [TABLE_HTML] 标记: %s", expected_table_id)
        elif detected.get("table_id") and detected["table_id"] != expected_table_id:
            logger.warning(
                "table_id 不一致: 期望 %s, 检测到 %s",
                expected_table_id,
                detected["table_id"],
            )
        else:
            meta["table_id"] = detected.get("table_id") or expected_table_id
            if detected.get("table_page"):
                meta["table_page"] = detected["table_page"]
        return meta

    @classmethod
    def _get_pdf_page_count(cls, path: Path) -> int:
        """快速获取 PDF 总页数（无需加载模型）。"""
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                return len(pdf.pages)
        except Exception:
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                return len(reader.pages)
            except Exception:
                return 0

    def _parse_with_mineru(self, path: Path) -> Dict:
        """
        MinerU 多级解析：
        1. v4 API + VLM 模型（需 COS URL，公式/表格最准）
        2. Agent API v1 + pipeline 模型（免 Key，直接上传）
        3. Docling 降级
        """
        import requests, time, json
        from src.config import config

        def _save_tables(md_text, pdf_name):
            """提取 Markdown 中的表格存入 figures.db"""
            tables = []
            for m in re.finditer(r'<table[^>]*>.*?</table>', md_text, re.DOTALL | re.IGNORECASE):
                tables.append(f"[TABLE_HTML: Table]\n{m.group(0)}")
            try:
                import sqlite3
                _db = Path('data/figures/figures.db')
                _db.parent.mkdir(parents=True, exist_ok=True)
                _conn = sqlite3.connect(str(_db))
                for ti, m in enumerate(re.finditer(r'<table[^>]*>.*?</table>', md_text, re.DOTALL | re.IGNORECASE), 1):
                    _cap = ""
                    for _l in reversed(md_text[:m.start()].split("\n")[-10:]):
                        if not _l.startswith("|"):
                            _c = re.sub(r'^#+\s*|^\*\*|\*\*$', '', _l).strip()
                            if _c and 5 < len(_c) < 150: _cap = _c; break
                    _conn.execute("INSERT OR REPLACE INTO figures(paper_source,figure_id,figure_type,caption,html_content,page_no) VALUES(?,?,?,?,?,?)",
                        (pdf_name, f"Table_{ti}", "table", _cap, m.group(0), 0))
                _conn.commit(); _conn.close()
            except: pass
            return tables

        def _parse_v4_vlm(pdf_path, pdf_name):
            """v4 API + VLM 模型（通过 COS URL）"""
            upload_base = config.parser.mineru_upload_url
            api_key = config.parser.mineru_api_key
            if not upload_base or not api_key:
                return None
            file_url = upload_base.rstrip("/") + "/" + pdf_name
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            r = requests.post("https://mineru.net/api/v4/extract/task", headers=headers, json={
                "url": file_url, "enable_formula": True, "enable_table": True,
                "is_ocr": False, "model_version": "vlm",
            }, timeout=30)
            if r.status_code != 200: return None
            task_id = r.json().get("data", {}).get("task_id", "")
            if not task_id: return None
            for _ in range(12):  # 最多等 60s，VLM 太慢就降级
                time.sleep(5)
                r = requests.get(f"https://mineru.net/api/v4/extract/task/{task_id}", headers=headers, timeout=30)
                if r.status_code != 200: continue
                d = r.json().get("data", {})
                if d.get("status") == "success":
                    md_url = d.get("markdown_url", "")
                    md = requests.get(md_url, timeout=120).text if md_url else d.get("content", "")
                    if md:
                        tables = _save_tables(md, pdf_name)
                        title = next((l[2:].strip() for l in md.split("\n") if l.startswith("# ")), "")
                        return {"full_text": md, "pages": [md], "title": title, "abstract": "", "references": [], "tables": tables, "refs": []}
                elif d.get("status") == "failed":
                    return None
            return None

        def _parse_agent(pdf_path, pdf_name):
            """Agent API v1 + pipeline 模型（直接上传）"""
            base = "https://mineru.net/api/v1/agent"
            r = requests.post(f"{base}/parse/file", json={
                "file_name": pdf_name, "enable_table": True, "enable_formula": True, "is_ocr": False,
            }, timeout=30)
            d = r.json()
            if d.get("code") != 0: return None
            tid = d["data"]["task_id"]
            furl = d["data"]["file_url"]
            import urllib3; urllib3.disable_warnings()
            with open(str(pdf_path), "rb") as f:
                up = urllib3.PoolManager().request("PUT", furl, body=f.read(),
                    headers={"Content-Type": "application/octet-stream"})
            if up.status not in (200, 201): return None
            for _ in range(100):
                time.sleep(3)
                r = requests.get(f"{base}/parse/{tid}", timeout=30)
                if r.status_code != 200: continue
                d = r.json().get("data", {})
                if d.get("state") == "done":
                    md = requests.get(d.get("markdown_url", ""), timeout=60).text if d.get("markdown_url") else ""
                    if md:
                        tables = _save_tables(md, pdf_name)
                        title = next((l[2:].strip() for l in md.split("\n") if l.startswith("# ")), "")
                        return {"full_text": md, "pages": [md], "title": title, "abstract": "", "references": [], "tables": tables, "refs": []}
                elif d.get("state") == "failed":
                    return None
            return None

        import re
        # 1. 优先 v4 + VLM（最准，需 COS + API Key）
        result = _parse_v4_vlm(path, path.name)
        if result: return result
        # 2. 其次 Agent API + pipeline（免 Key）
        result = _parse_agent(path, path.name)
        if result: return result
        # 3. 兜底 Docling
        return self._parse_with_docling(path)

    def _parse_with_docling(self, path: Path) -> Dict:
        """
        Docling 元素流解析：仅 TableItem 生成 HTML 表块；
        段落/标题/列表以 Markdown 纯文本落盘。

        自动防 OOM：若全文转换失败（大 PDF），拆成每批 10 页处理，
        合并所有批次的表格和文本结果。
        """
        from docling_core.types.doc import DocItemLabel

        converter = self._get_docling_converter()
        MAX_FULL_PAGES = 15   # 超过此页数直接走分批（避免 C++ bad_alloc 崩溃）
        PAGE_BATCH = 10       # 每批页数

        # ── 判断策略：页数少 → 全文；页数多 → 分批 ──
        total_pages = self._get_pdf_page_count(path)
        doc = None
        batch_docs = []

        if 0 < total_pages <= MAX_FULL_PAGES:
            result = converter.convert(str(path))
            doc = result.document
            logger.info("Docling 全文转换 (%d 页)", total_pages)
        elif total_pages > MAX_FULL_PAGES:
            logger.info("PDF 共 %d 页，超过 %d 页阈值，物理拆分成 %d 页/批",
                        total_pages, MAX_FULL_PAGES, PAGE_BATCH)
            # 物理拆分：page_range 不能阻止 Docling C++ 预处理加载全文
            # 需用 PyPDF2 拆成独立小 PDF 再分别处理
            try:
                from pypdf import PdfReader, PdfWriter
                reader = PdfReader(str(path))
            except Exception:
                raise RuntimeError(f"无法读取 PDF 内容: {path}")

            temp_dir = path.parent / f"._tmp_{path.stem}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            for batch_start in range(1, total_pages + 1, PAGE_BATCH):
                batch_end = min(batch_start + PAGE_BATCH, total_pages + 1)
                split_path = temp_dir / f"pages_{batch_start}_{batch_end}.pdf"
                try:
                    if not split_path.exists():
                        writer = PdfWriter()
                        for pg in range(batch_start - 1, batch_end - 1):
                            writer.add_page(reader.pages[pg])
                        with open(str(split_path), "wb") as f:
                            writer.write(f)
                    br = converter.convert(str(split_path))
                    batch_docs.append(br.document)
                    logger.info("  物理批次 %d-%d 完成 (%d 页)",
                                batch_start, batch_end - 1, len(br.document.pages))
                except Exception as e:
                    logger.warning("  物理批次 %d-%d 失败: %s", batch_start, batch_end - 1, e)

            # 清理临时文件
            import shutil
            shutil.rmtree(str(temp_dir), ignore_errors=True)

            if not batch_docs:
                raise RuntimeError(f"Docling 逐批处理全部失败: {path}")
            logger.info("物理拆批完成，共 %d 个批次", len(batch_docs))
        else:
            raise RuntimeError(f"无法获取 PDF 页数: {path}")

        hyperlinks = self._extract_hyperlinks(str(path))

        body_parts: List[str] = []
        all_tables: List[str] = []
        table_registry: List[Dict[str, object]] = []
        page_buffers: Dict[int, List[str]] = {}
        pending_caption = ""
        table_seq = 0
        seen_table_refs: set[str] = set()
        # table_id → {"all_tables_idx": int, "last_page": int} 用于跨页合并
        table_map: Dict[str, dict] = {}

        # 迭代：全文 doc 或逐批 batch_docs
        docs_to_iterate = batch_docs if doc is None else [doc]
        for current_doc in docs_to_iterate:
            for item, _level in current_doc.iterate_items():
                label = getattr(item, "label", None)

                if label == DocItemLabel.CAPTION:
                    cap = self._docling_item_plaintext(item, current_doc)
                    if cap:
                        pending_caption = cap
                    continue

                if self._is_docling_table_item(item):
                    table_ref = getattr(item, "self_ref", None)
                    if table_ref and table_ref in seen_table_refs:
                        continue
                    if table_ref:
                        seen_table_refs.add(table_ref)
                    block, tmeta = self._docling_table_block(
                        item, current_doc, table_seq
                    )
                    if not block:
                        continue

                    table_id = str(tmeta.get("table_id", ""))
                    current_page = int(tmeta.get("table_page") or 0) or self._docling_page_no(item)

                    # ── 跨页表合并检测 ──
                    prev = table_map.get(table_id)
                    is_continued = (
                        prev is not None
                        and prev["last_page"] != current_page
                        and prev["last_page"] > 0
                    )
                    # Caption 含 "(continued)" 是强信号
                    caption_text = str(tmeta.get("caption", "")).lower()
                    if "(continued" in caption_text or "(cont" in caption_text:
                        is_continued = True

                    if is_continued:
                        prev_idx = prev["all_tables_idx"]
                        existing = all_tables[prev_idx]
                        new_trs_match = re.search(
                            r"(<tr>.*?</tr>)+", block, re.IGNORECASE | re.DOTALL
                        )
                        if new_trs_match:
                            new_trs = new_trs_match.group(0)
                            merged = existing.replace("</table>", f"{new_trs}\n</table>")
                            if prev_idx < len(body_parts):
                                body_parts[prev_idx] = merged
                            all_tables[prev_idx] = merged
                            table_map[table_id]["last_page"] = current_page
                            page_buffers.setdefault(prev["last_page"], []).append(new_trs)
                            logger.info(
                                "跨页表合并: %s page %d → page %d",
                                table_id, prev["last_page"], current_page,
                            )
                        continue

                    # ── 新表 ──
                    body_parts.append(block)
                    all_tables.append(block)
                    table_registry.append(tmeta)
                    table_map[table_id] = {
                        "all_tables_idx": len(all_tables) - 1,
                        "last_page": current_page,
                    }
                    table_seq += 1
                    page_buffers.setdefault(current_page, []).append(block)
                    continue

                text = self._docling_item_plaintext(item, current_doc)
                if not text:
                    continue
                body_parts.append(text)
                pg = self._docling_page_no(item) or 1
                page_buffers.setdefault(pg, []).append(text)

        full_text = "\n\n".join(body_parts)

        # ── 图表原图提取 v2：PyMuPDF 原生提取，不依赖 Docling ──
        try:
            from src.parser.figure_extractor import extract_figures
            extract_figures(path)
        except Exception as e:
            logger.warning(f"图表提取失败 (不影响主流程): {e}")

        for pg_idx, urls in hyperlinks.items():
            link_lines = []
            for url in urls:
                if "github" in url.lower() or "gitlab" in url.lower():
                    link_lines.append(f"[GitHub Code Link: {url}]")
                else:
                    link_lines.append(f"[Link: {url}]")
            if link_lines:
                page_buffers.setdefault(pg_idx + 1, []).extend(link_lines)

        if page_buffers:
            max_pg = max(page_buffers)
            pages = []
            for p in range(1, max_pg + 1):
                pages.append("\n\n".join(page_buffers.get(p, [])))
        else:
            pages = full_text.split("\n\n") if full_text else []

        full_text = self._clean_glued_text(full_text)
        title = self._extract_title(full_text)
        abstract = self._extract_abstract(full_text)
        references = self._extract_references(full_text)

        return {
            "full_text": full_text,
            "pages": pages,
            "title": title,
            "abstract": abstract,
            "references": references,
            "tables": all_tables,
            "table_registry": table_registry,
        }

    def _parse_with_auto(self, path: Path) -> Dict:
        """Docling 布局树（Type Shield）优先；失败时仅回退纯文本，禁止表格盲猜。"""
        try:
            logger.info("Auto -> docling (layout tree): %s", path.name)
            return self._parse_with_docling(path)
        except Exception as exc:
            logger.warning(
                "Docling failed, text-only fallback: %s — %s", path.name, exc
            )

        result = self._parse_with_pypdf2(path)
        full_text = result["full_text"]
        page_count = len(result["pages"])
        avg_chars = len(full_text) / max(1, page_count)

        if avg_chars >= 100 and len(full_text) > 500:
            logger.info(
                "Auto -> PyPDF2 (%.0f chars/page): %s", avg_chars, path.name
            )
            return result

        logger.info("Auto -> pdfplumber text-only: %s", path.name)
        return self._parse_with_pdfplumber(path)

    def _extract_title(self, text: str) -> str:
        """从全文提取论文标题（通常是第一行非空内容）"""
        lines = text.strip().split("\n")
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 10 and not line.startswith("arXiv"):
                return line
        return ""

    def _extract_abstract(self, text: str) -> str:
        """提取摘要"""
        # 查找 "Abstract" 标记
        abstract_match = re.search(
            r'Abstract[:\s-]*\n?(.*?)(?=\n\s*(?:\d\.?\s+)?(?:Introduction|1\.|I\.))',
            text, re.DOTALL | re.IGNORECASE
        )
        if abstract_match:
            return abstract_match.group(1).strip()[:2000]
        return ""

    def _extract_references(self, text: str) -> List[str]:
        """提取参考文献"""
        # 找 Reference 章节
        ref_match = re.search(
            r'(?:Reference|Bibliography|REFERENCES)\s*\n(.*?)(?:\n\s*Appendix|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if not ref_match:
            return []

        ref_text = ref_match.group(1)
        # 按编号分割
        refs = re.split(r'\n\s*\[\d+\]', ref_text)
        return [r.strip() for r in refs if len(r.strip()) > 20]

    @staticmethod
    def _extract_hyperlinks(pdf_path: str) -> Dict[int, List[str]]:
        """
        从 PDF 注释(/Annots)中提取隐藏的超链接 URL。
        遍历每页的 /Link 型注释，捕获 /URI，回写到对应页面。
        """
        import pdfplumber
        page_links: Dict[int, List[str]] = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    urls = []
                    if page.annots:
                        for annot in page.annots:
                            uri = None
                            if isinstance(annot, dict):
                                # 直接键
                                uri = annot.get("uri") or annot.get("URI")
                                # 嵌套在 /A (Action) 字典里
                                if not uri:
                                    action = annot.get("A") or annot.get("action")
                                    if isinstance(action, dict):
                                        uri = action.get("URI") or action.get("uri")
                                # 递归搜索整个 dict
                                if not uri:
                                    uri = PaperParser._deep_find_uri(annot)
                            if uri and isinstance(uri, str) and uri.startswith("http"):
                                urls.append(uri)
                    if urls:
                        page_links[i] = urls
        except Exception as e:
            logger.warning(f"超链接提取失败 {Path(pdf_path).name}: {e}")
        return page_links

    @staticmethod
    def _deep_find_uri(obj, depth: int = 0) -> str:
        """递归搜索 dict 中任何含 http 的字符串值"""
        if depth > 5:
            return None
        if isinstance(obj, dict):
            for v in obj.values():
                result = PaperParser._deep_find_uri(v, depth + 1)
                if result:
                    return result
        elif isinstance(obj, str) and obj.startswith("http"):
            return obj
        elif isinstance(obj, list):
            for item in obj:
                result = PaperParser._deep_find_uri(item, depth + 1)
                if result:
                    return result
        return None

    @staticmethod
    def _clean_glued_text(text: str) -> str:
        """
        修复 PDF 文本提取中的单词粘连和 URL 失空格问题。

        策略：先用占位符保护 URL → 执行所有正则修复 → 恢复 URL。
        """
        if not text:
            return text

        # ── Step 0: 清除 arXiv 页面水印 ──
        # 匹配形如 "ar Xiv:1912.11947v 1  [eess. IV]  26 Dec 2019" 的页边水印
        text = re.sub(
            r'ar\s*Xiv:\s*\d{4}\.\s*\d{4,5}v?\s*\d*\s*\[[a-zA-Z\s\.\-]+\]\s*\d{1,2}\s+[A-Z][a-z]+\s+\d{4}',
            '',
            text,
            flags=re.IGNORECASE,
        )
        # 清除残留的 "arXiv:xxxx.xxxxx" 孤立水印片段
        text = re.sub(r'ar\s*Xiv:\s*\d{4}\.\s*\d{4,5}v?\s*\d*', '', text, flags=re.IGNORECASE)

        # ── Step A: 先修复 URL 前的空格丢失 ──
        # 必须在 URL 保护前做，否则粘连部分被占位符打包看不见
        text = re.sub(r'([a-zA-Z0-9])(github\.com|gitlab\.com)', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z0-9])(https?://)', r'\1 \2', text)

        # ── Step B: 提取并保护 URL ──
        # 把所有 URL 替换为 __URL_0__, __URL_1__ 等占位符，后续正则不碰 URL 内容
        url_pattern = r'(?:github\.com|gitlab\.com|https?://)\S+'
        url_holder: Dict[str, str] = {}
        def _capture_url(m):
            key = f"__URL_{len(url_holder)}__"
            url_holder[key] = m.group(0)
            return key
        text = re.sub(url_pattern, _capture_url, text)

        # ── Step C: 修复单词粘连 ──
        # B1) 小写→大写边界补空格（camelCase 拆分）
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # B2) 字母→数字边界补空格
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        # B3) 数字→大写边界补空格
        text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)
        # B4) 句点后大写字母前补空格（句末边界）
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        # B5) 闭合括号后、开括号前补空格
        text = re.sub(r'([\]\)}])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z0-9])([\[\(])', r'\1 \2', text)

        # B6) 已知高频词粘连拆解（仅 len >= 4，短词如 a/an/in 不参与防误伤）
        #     第一遍：全小写匹配 + 跨大小写边界（处理 ".Segmentanything..." 等）
        for w in sorted([w for w in PaperParser._COMMON_WORDS if len(w) >= 4], key=len, reverse=True):
            text = re.sub(
                r'(?<![a-zA-Z])(' + re.escape(w) + r')(?=[a-zA-Z]{4,})',
                r'\1 ',
                text,
            )
            # 额外：大写首字母版本（如 "Segment" 在句首被大写的情况）
            cap_w = w[0].upper() + w[1:]
            text = re.sub(
                r'(?<![a-zA-Z])(' + re.escape(cap_w) + r')(?=[a-z]{4,})',
                r'\1 ',
                text,
            )
            # 额外：前有大写字母相连（如 "Asegmentanything" → "A segment anything"）
            text = re.sub(
                r'([A-Z])(' + re.escape(w) + r')(?=[a-z]{4,})',
                r'\1 \2 ',
                text,
            )

        # B7) 词内宽松匹配：处理 "andmedicalimagesegmentation" → "and medical image segmentation"
        #     关键：捕获前导字母，空格插在单词前后两侧
        common_5plus = sorted([w for w in PaperParser._COMMON_WORDS if len(w) >= 5], key=len, reverse=True)
        for _ in range(2):
            for w in common_5plus:
                text = re.sub(
                    r'([a-z])(' + re.escape(w) + r')(?=[a-z]{5,})',
                    r'\1 \2 ',
                    text,
                )

        # ── Step D: 恢复 URL ──
        for placeholder, original_url in url_holder.items():
            text = text.replace(placeholder, original_url)

        return text

    def _table_to_markdown(self, table: List[List[str]], page_num: int) -> str:
        """将提取的表格转为 Markdown 表格文本（嵌入 full_text 用）"""
        if not table or not table[0]:
            return ""
        rows = []
        for ri, row in enumerate(table):
            cells = [str(c or "").strip() for c in row]
            rows.append("|" + "|".join(cells) + "|")
            if ri == 0 and len(table) > 1:
                # 表头分隔线
                rows.append("|" + "|".join(["---"] * len(row)) + "|")
        return "[TABLE Page {}]\n{}\n[/TABLE]".format(page_num, "\n".join(rows))

    @staticmethod
    def _rebuild_text_from_chars(chars: list) -> str:
        """从 pdfplumber chars 列表重建文本，保留阅读顺序"""
        if not chars:
            return ""
        # 按 top → x0 排序（逐行，行内从左到右）
        sorted_chars = sorted(chars, key=lambda c: (round(c['top'], 0), c['x0']))
        lines = []
        current_line = []
        last_top = None
        for ch in sorted_chars:
            top = round(ch['top'], 0)
            if last_top is not None and abs(top - last_top) > 3:
                lines.append("".join(current_line))
                current_line = []
            current_line.append(ch.get('text', ''))
            last_top = top
        if current_line:
            lines.append("".join(current_line))
        return "\n".join(lines)

    @staticmethod
    def _find_table_caption(page_text: str, table_index: int, page_num: int) -> str:
        """从页面文本中查找 Table Caption（如 'Table 1.', 'TABLE I.' 等）"""
        patterns = [
            rf'Table\s+{table_index + 1}[\.\:\s][^\n]{{0,100}}',
            rf'TABLE\s+{table_index + 1}[\.\:\s][^\n]{{0,100}}',
            rf'表\s*{table_index + 1}[\.\:\s][^\n]{{0,100}}',
        ]
        # 也尝试序号 I, II, III...
        roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
        if table_index < len(roman):
            patterns.append(rf'Table\s+{roman[table_index]}[\.\:\s][^\n]{{0,100}}')

        for pat in patterns:
            m = re.search(pat, page_text)
            if m:
                cap = m.group(0).strip()
                if len(cap) >= 8:
                    return cap
        return ""

    @staticmethod
    def _table_to_html(grid: List[List[str]], table_id: str,
                              page_num: int = 0) -> str:
        """
        增强型 Markdown 表格转换。

        特性:
        - 堆叠表头合并： 上下两行分别含 mDice / mIoU → 合并为 `mDice/mIoU`
        - 空列修剪： 删掉全空的首尾列
        - Metadata 标记： 放入 `is_table=True, table_id=Table_1`
        """
        if not grid or not grid[0]:
            return ""

        # ── 修剪首尾全空列 ──
        ncols = len(grid[0])
        col_has_data = [False] * ncols
        for row in grid:
            for ci in range(min(len(row), ncols)):
                val = str(row[ci] or "").strip()
                if val:
                    col_has_data[ci] = True
        kept_cols = [ci for ci, has in enumerate(col_has_data) if has]
        if not kept_cols:
            kept_cols = list(range(ncols))
        trimmed = []
        for row in grid:
            trimmed.append([row[ci] if ci < len(row) else "" for ci in kept_cols])

        # ── 堆叠表头合并 ──
        # 如果前两行都是标题行（含文字、无大段数字），合并为单行
        merged_rows = list(trimmed)
        if len(merged_rows) >= 2:
            row0_has_digits = sum(
                1 for c in merged_rows[0] if re.search(r'\d', str(c or ""))
            )
            row1_has_digits = sum(
                1 for c in merged_rows[1] if re.search(r'\d', str(c or ""))
            )
            # 如果第二行标题的数字少于两个（不是数据行），合并到第一行
            if row1_has_digits <= 1 and row0_has_digits <= 1:
                merged = []
                for ci in range(len(trimmed[0])):
                    a = str(trimmed[0][ci] or "").strip()
                    b = str(trimmed[1][ci] or "").strip()
                    if a and b and a != b:
                        merged.append(f"{a}/{b}")
                    elif a:
                        merged.append(a)
                    else:
                        merged.append(b)
                merged_rows = [merged] + trimmed[2:]

        # ── 构建原生 HTML 表格 ──
        html_rows = []
        for ri, row in enumerate(merged_rows):
            tag = "th" if ri == 0 else "td"
            cells_html = "".join(
                f"<{tag}>{str(c or '').strip().replace(chr(10), ' ')}</{tag}>"
                for c in row
            )
            html_rows.append(f"<tr>{cells_html}</tr>")

        html_table = "<table>\n" + "\n".join(html_rows) + "\n</table>"

        # ── 元数据标记 ──
        meta_tag = f'[TABLE_HTML: {table_id} | Page {page_num}]'
        return f"{meta_tag}\n{html_table}\n[/TABLE_HTML]"

    @staticmethod
    def _detect_table_metadata(text: str) -> dict:
        """Header Line Lock：仅从首行 [TABLE_HTML: Table_N] 提取元数据。"""
        meta: Dict[str, object] = {"is_table": False}
        first_line = text.split("\n", 1)[0] if "\n" in text else text
        if "[TABLE_HTML:" not in first_line:
            return meta
        if "[/TABLE_HTML]" not in text:
            return meta
        meta["is_table"] = True
        tag_m = re.search(r"Table_(\d+)", first_line, re.IGNORECASE)
        if tag_m:
            meta["table_id"] = f"Table_{tag_m.group(1)}"
        page_m = re.search(r"Page\s+(\d+)", first_line, re.IGNORECASE)
        if page_m:
            meta["table_page"] = page_m.group(1)
        return meta
