"""
论文感知切片器 — 按学术论文的章节结构进行智能切片

参考：
- paper-qa 的 SmartChunker（按语义边界而非固定长度）
- LangChain 的 SemanticChunker + RecursiveCharacterTextSplitter
- 论文常用结构：Abstract / Introduction / Related Work / Method / Experiment / Conclusion

与通用切片器的区别：
1. 识别论文章节标题（1. Introduction / 2. Method 等）
2. 保留章节元数据（第几节、章节名）
3. 摘要单独作为一个 chunk（通常是最高质量的检索目标）
4. 参考文献区域不参与切片（降噪）
"""
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from src.config import config
from src.parser.paper_parser import PaperParser

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """论文章节"""
    title: str           # 章节名（如 "Introduction"）
    number: int          # 章节序号（一级数字，兼容旧版）
    level: int           # 层级（1 = 一级标题，2 = 二级）
    content: str         # 原文内容
    start_pos: int       # 原文起始位置
    end_pos: int         # 原文结束位置
    section_id: str = "" # 完整章节路径（如 "2.3"），用于元数据过滤


@dataclass
class Chunk:
    """切片"""
    text: str
    metadata: Dict = field(default_factory=dict)
    tokens: int = 0


class SectionChunker:
    """
    论文感知切片器。

    策略：
    1. 先用正则识别论文的章节结构
    2. Abstract 整体作为一个 chunk（长度通常适中）
    3. 正文章节：按段落切分，保持语义完整性
    4. 去除 Reference/Appendix 之后的噪声内容
    5. 每个 chunk 保留章节元数据（section_title, section_number）
    """

    # IEEE/ACM 三级标题规范（按优先级匹配）
    _ROMAN_NUMERAL = r"(?:X|IX|IV|VIII|VII|VI|V|IV|III|II|I)"
    PATTERN_ARABIC = re.compile(
        r"^(?:#+\s*)?(\d+(?:\.\d+)*)\s*\.?\s+(.+)$"
    )
    PATTERN_ROMAN_MAIN = re.compile(
        rf"^(?:#+\s*)?({_ROMAN_NUMERAL})\.\s+(.+)$",
        re.IGNORECASE,
    )
    PATTERN_LETTER_SUB = re.compile(
        r"^(?:#+\s*)?([A-Z])\.\s+([A-Z][A-Za-z0-9].*)$"
    )
    PATTERN_KEYWORD = re.compile(
        r"^(?:#+\s*)?(Abstract|Introduction|Related\s*Work|Background|Method|"
        r"Experiment|Result|Discussion|Conclusion|Reference|Appendix)\s*$",
        re.IGNORECASE,
    )

    SECTION_KEYWORDS = {
        "introduction", "background", "related work", "method",
        "experiment", "result", "discussion", "conclusion",
        "implementation", "parameter", "setting", "setup",
        "architecture", "training", "evaluation", "ablation",
        "analysis", "overview", "preliminaries", "pipeline",
        "algorithm", "procedure", "approach", "framework",
        "dataset", "data", "model", "network", "loss",
        "optimization", "inference", "deployment", "pre-processing",
        "preprocessing", "augmentation", "performance", "measures",
    }

    # 需要排除的章节（不参与切片）
    SKIP_SECTIONS = {"reference", "references", "bibliography", "appendix"}

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunker.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunker.chunk_overlap
        self.min_chunk_size = config.chunker.min_chunk_size

    def chunk(self, text: str, paper_meta: Dict = None) -> List[Chunk]:
        """
        对论文文本进行感知切片。

        Args:
            text: 论文全文（纯文本）
            paper_meta: 论文元数据（标题、作者、arxiv_id 等）

        Returns:
            切片列表
        """
        paper_meta = paper_meta or {}

        # Step 1: 识别章节结构
        sections = self._detect_sections(text)

        if not sections:
            # 无法识别章节，回退到按段落切分
            logger.warning("未能识别论文章节，使用段落切片回退方案")
            return self._fallback_chunk(text, paper_meta)

        # Step 2: 按章节切分
        chunks = []
        for section in sections:
            if section.title.lower() in self.SKIP_SECTIONS:
                continue

            section_chunks = self._chunk_section(section, paper_meta)
            chunks.extend(section_chunks)

        # Step 3: 确保 Abstract 作为首个 chunk
        # （已在 _detect_sections 中保证 Abstract 在前，这里做去重）
        logger.info(f"切片完成: {len(chunks)} chunks (已按 {len(sections)} 个章节切分)")
        return chunks

    def _detect_sections(self, text: str) -> List[Section]:
        """检测论文的章节结构（阿拉伯数字 / 罗马主标题 / 字母子标题）"""
        sections: List[Section] = []
        lines = text.split("\n")
        heading_positions: List[Tuple[int, str, int, int, str]] = []

        parent_heading: Optional[str] = None
        parent_section_id: Optional[str] = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) > 100:
                continue

            parsed = self._parse_section_heading(
                line_stripped, parent_heading, parent_section_id
            )
            if parsed:
                title = parsed["title"]
                level = parsed["level"]
                number = parsed["number"]
                section_id = parsed["section_id"]
                if parsed.get("parent_heading"):
                    parent_heading = parsed["parent_heading"]
                    parent_section_id = parsed.get("parent_section_id")
                heading_positions.append((i, title, level, number, section_id))
                continue

            # 弱标题兜底：正则未匹配，但行特征像章节标题
            ws = line_stripped
            if re.search(r"[%]|\d+/\d+", ws):
                continue
            if re.search(r"(?:Fig|Table|Figure)\s*\d", ws, re.IGNORECASE):
                continue
            if (
                ws.isupper()
                or (
                    ws[0].isupper()
                    and ws[0].isalpha()
                    and not ws.endswith((".", ":", ";", ",", ")", "]"))
                )
            ):
                words = ws.lower().split()
                if 2 <= len(words) <= 8 and 5 <= len(ws) <= 60:
                    has_kw = any(kw in ws.lower() for kw in self.SECTION_KEYWORDS)
                    if has_kw:
                        too_close = any(
                            abs(i - hp[0]) < 5
                            for hp in heading_positions[-3:]
                            if heading_positions
                        )
                        if not too_close:
                            number = len(heading_positions) + 1
                            section_id = ws[:30]
                            heading_positions.append((i, ws, 1, number, section_id))
                            parent_heading = ws
                            parent_section_id = section_id

        for idx, items in enumerate(heading_positions):
            start_line, title, level, number, section_id = items
            next_start = (
                heading_positions[idx + 1][0]
                if idx + 1 < len(heading_positions)
                else len(lines)
            )

            content_lines = lines[start_line + 1 : next_start]
            content = "\n".join(content_lines).strip()

            if len(content) < self.min_chunk_size and title.lower() != "abstract":
                continue

            sections.append(
                Section(
                    title=title,
                    number=number,
                    level=level,
                    content=content,
                    start_pos=start_line,
                    end_pos=next_start,
                    section_id=section_id,
                )
            )

        return sections

    def _parse_section_heading(
        self,
        line: str,
        parent_heading: Optional[str] = None,
        parent_section_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        解析单行章节标题。字母子标题继承主标题，形成
        "IV. IMPLEMENTATION - A. Dataset" 供下游意图加权命中 Dataset。
        """
        m_kw = self.PATTERN_KEYWORD.match(line)
        if m_kw:
            title = m_kw.group(1).strip()
            section_id = title
            if title.lower() == "abstract":
                return {
                    "title": title,
                    "level": 0,
                    "number": 0,
                    "section_id": section_id,
                    "parent_heading": None,
                    "parent_section_id": None,
                }
            number = 1
            return {
                "title": title,
                "level": 1,
                "number": number,
                "section_id": section_id,
                "parent_heading": title,
                "parent_section_id": section_id,
            }

        m = self.PATTERN_ROMAN_MAIN.match(line)
        if m and self._is_valid_roman_main(m.group(1), m.group(2)):
            roman = m.group(1).upper()
            title_text = self._normalize_heading_title(m.group(2).strip())
            full_main = f"{roman}. {title_text}"
            return {
                "title": full_main,
                "level": 1,
                "number": self._roman_to_int(roman),
                "section_id": roman,
                "parent_heading": full_main,
                "parent_section_id": roman,
            }

        m = self.PATTERN_LETTER_SUB.match(line)
        if m and self._is_valid_letter_sub(m.group(1), m.group(2).strip(), line):
            letter = m.group(1).upper()
            sub_title = self._normalize_heading_title(m.group(2).strip())
            full_sub = f"{letter}. {sub_title}"
            if parent_heading:
                display = f"{parent_heading} - {full_sub}"
                section_id = (
                    f"{parent_section_id}.{letter}" if parent_section_id else letter
                )
            else:
                display = full_sub
                section_id = letter
            return {
                "title": display,
                "level": 2,
                "number": ord(letter) - ord("A") + 1,
                "section_id": section_id,
                "parent_heading": None,
                "parent_section_id": None,
            }

        m = self.PATTERN_ARABIC.match(line)
        if m:
            full_id = m.group(1).replace(" ", "")
            title_text = self._normalize_heading_title(m.group(2).strip())
            if not re.match(r"^[A-Z]", title_text):
                return None
            if self._is_noise_number(full_id.split(".")[0], title_text):
                return None
            parts = full_id.split(".")
            number = int(parts[0])
            level = len(parts)
            display = f"{full_id} {title_text}"
            section_id = full_id
            if level >= 2 and parent_heading:
                display = f"{parent_heading} - {display}"
                section_id = f"{parent_section_id}.{full_id}" if parent_section_id else full_id
            update_parent = display if level == 1 else None
            update_parent_id = full_id if level == 1 else None
            return {
                "title": display,
                "level": level,
                "number": number,
                "section_id": section_id,
                "parent_heading": update_parent,
                "parent_section_id": update_parent_id,
            }

        return None

    @staticmethod
    def _normalize_heading_title(title: str) -> str:
        """合并 PDF 提取产生的断裂全大写标题，如 I MPLEMENTATION → IMPLEMENTATION。"""
        t = re.sub(r"\s+", " ", title.strip())
        for _ in range(12):
            new_t = re.sub(r"([A-Z])\s+([A-Z])", r"\1\2", t)
            if new_t == t:
                break
            t = new_t
        return t.strip()

    @classmethod
    def _is_valid_roman_main(cls, roman: str, raw_title: str) -> bool:
        roman = roman.upper()
        if cls._roman_to_int(roman) <= 0:
            return False
        title = cls._normalize_heading_title(raw_title)
        if not title or len(title) > 80:
            return False
        letters = re.sub(r"[^A-Za-z]", "", title)
        if not letters:
            return False
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        return upper_ratio >= 0.7

    @classmethod
    def _is_valid_letter_sub(cls, letter: str, title: str, line: str) -> bool:
        """过滤参考文献作者行（H. Zhang, ...）等假阳性。"""
        if letter.upper() > "Z" or len(title) > 70:
            return False
        if re.search(r"\d{4}", line):
            return False
        if "," in line and (
            " and " in line.lower()
            or "et al" in line.lower()
            or line.count(",") >= 2
        ):
            return False
        lower = title.lower()
        if any(kw in lower for kw in cls.SECTION_KEYWORDS):
            return True
        words = title.split()
        return 1 <= len(words) <= 8 and title[0].isupper()

    def _chunk_section(self, section: Section, paper_meta: Dict) -> List[Chunk]:
        """对单个章节内容进行切片（表格块作为原子 chunk，不分割）"""
        chunks = []
        paragraphs = self._split_paragraphs(section.content)

        current_text = ""
        current_tokens = 0

        for para in paragraphs:
            # ── 表格段落作为原子块 ──
            first_para_line = para.split("\n", 1)[0] if "\n" in para else para
            is_table_block = "[TABLE_HTML:" in first_para_line

            if is_table_block:
                # 先 flush 当前累积的正文
                if current_text.strip():
                    chunks.append(self._make_chunk(
                        current_text.strip(),
                        section, paper_meta
                    ))
                # 表格作为独立原子 chunk
                chunks.append(self._make_chunk(para.strip(), section, paper_meta))
                current_text = ""
                current_tokens = 0
                continue

            para_tokens = self._estimate_tokens(para)

            if current_tokens + para_tokens > self.chunk_size and current_text:
                # 保存当前 chunk
                chunks.append(self._make_chunk(
                    current_text.strip(),
                    section, paper_meta
                ))
                # 重叠处理
                overlap_text = self._get_overlap(current_text, self.chunk_overlap)
                current_text = overlap_text + "\n\n" + para
                current_tokens = self._estimate_tokens(current_text)
            else:
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_text = para
                current_tokens += para_tokens

        # 最后一个 chunk
        if current_text.strip():
            chunks.append(self._make_chunk(current_text.strip(), section, paper_meta))

        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        按段落分割（表格原子块免切：完整保留 [TABLE_HTML: … [/TABLE_HTML] 区域）。

        策略：沿 [TABLE_HTML: 边界切分，非表块走双换行分割，表块完整保留。
        """
        paragraphs = []
        # 按 [TABLE_HTML: 边界拆分为「正文区间」和「表格区间」交替的列表
        # 先找到所有表格块边界
        parts = re.split(
            r'(\[TABLE_HTML:.*?\[/TABLE_HTML\])',
            text, flags=re.DOTALL
        )
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 完整表格块 → 原子添加
            if part.startswith('[TABLE_HTML:') and part.endswith('[/TABLE_HTML]'):
                paragraphs.append(part)
            else:
                # 正文区间 → 按双换行再分割
                for sub in re.split(r'\n\s*\n', part):
                    sub = sub.strip()
                    if len(sub) > 20:
                        paragraphs.append(sub)
        return paragraphs

    def _make_chunk(self, text: str, section: Section, paper_meta: Dict) -> Chunk:
        """构建切片的元数据（含 section_id / 表格标记 / table_id）"""
        meta = {
            **paper_meta,
            "section_title": section.title,
            "section_number": section.number,
            "section_id": section.section_id or (section.title if section.level == 0 else ""),
            "section_level": section.level,
            "chunk_type": "abstract" if section.title.lower() == "abstract" else "body",
        }
        first_line = text.split("\n", 1)[0] if "\n" in text else text
        if "[TABLE_HTML:" in first_line:
            meta["contains_table"] = True
            meta["chunk_type"] = "table"
            meta["is_table"] = True
            tag_m = re.search(r"Table_(\d+)", first_line, re.IGNORECASE)
            meta["table_id"] = (
                f"Table_{tag_m.group(1)}" if tag_m else ""
            )
        else:
            meta["is_table"] = False
            meta["table_id"] = ""
            meta.pop("contains_table", None)
        return Chunk(
            text=text,
            metadata=meta,
            tokens=self._estimate_tokens(text),
        )

    @staticmethod
    def _contains_table(text: str) -> bool:
        """仅识别解析器输出的原子表块标记，禁止启发式把正文误判为表。"""
        if "[TABLE_HTML:" in text and "[/TABLE_HTML]" in text:
            return True
        if "[TABLE_MD:" in text and "[/TABLE_MD]" in text:
            return True
        if text.strip().startswith("[TABLE ") and "[/TABLE]" in text:
            return True
        return False

    def _get_overlap(self, text: str, overlap_tokens: int) -> str:
        """获取文本尾部作为重叠部分（按 token 估算）"""
        words = text.split()
        overlap_words = min(int(len(words) * 0.3), overlap_tokens)
        return " ".join(words[-overlap_words:]) if words else ""

    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数（英文按 4 char/token，中文按 1.5 char/token）"""
        if not text:
            return 0
        # 简单估算
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def _fallback_chunk(self, text: str, paper_meta: Dict) -> List[Chunk]:
        """回退：按段落切片"""
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current = ""
        tokens = 0

        for para in paragraphs:
            pt = self._estimate_tokens(para)
            if tokens + pt > self.chunk_size and current:
                chunks.append(Chunk(
                    text=current.strip(),
                    metadata={**paper_meta, "chunk_type": "body"},
                    tokens=tokens,
                ))
                current = para
                tokens = pt
            else:
                current = current + "\n\n" + para if current else para
                tokens += pt

        if current.strip():
            chunks.append(Chunk(
                text=current.strip(),
                metadata={**paper_meta, "chunk_type": "body"},
                tokens=tokens,
            ))

        return chunks

    @staticmethod
    def _is_noise_number(num_str: str, title: str) -> bool:
        """判断数字编号是否为噪音（年份、页码、图号、引用号等）"""
        if not num_str or not num_str.isdigit():
            return False
        n = int(num_str)
        # 年份噪声
        if (1900 <= n <= 2099) or (n > 10000):
            return True
        # 纯数字无文字标题
        if not title or len(title.split()) == 0:
            return True
        # 标题以数字开头且整体像表格/图注
        if title[0].isdigit() and len(title) < 20:
            return True
        return False

    @staticmethod
    def _roman_to_int(roman: str) -> int:
        """罗马数字转整数"""
        values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
        total = 0
        prev = 0
        for char in reversed(roman.upper()):
            curr = values.get(char, 0)
            if curr < prev:
                total -= curr
            else:
                total += curr
            prev = curr
        return total
