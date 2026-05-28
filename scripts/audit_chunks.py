"""
audit_chunks.py — 底层元数据审计
排查 Colorectal Polyp Segmentation by U-Net with Dilation Convolution 中
Section IV.A (实验数据集) 是否被正确切片、落盘。

验证目标：
  1. 是否含 "GIANA" / "Train set" / "10,025" 等数据集关键词
  2. section_title / section_id 是否保留
  3. 水印是否已清除干净
"""
import os, sys, re, json

os.environ["HF_HUB_OFFLINE"] = "1"
import logging

logging.basicConfig(level=logging.ERROR)

import pickle
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import get_vector_store_dir

# ── 1. 加载 metadata ──
vs_dir = get_vector_store_dir()
meta_path = vs_dir / "metadata_papers.pkl"
if not meta_path.exists():
    print(f"❌ metadata 文件不存在: {meta_path}")
    sys.exit(1)

with open(meta_path, "rb") as f:
    all_metas: list[dict] = pickle.load(f)

print(f"📊 总 Chunks: {len(all_metas)}\n")

# ── 2. 筛选 U-Net 论文 ──
print("=" * 70)
print("🔍 Step 1: 筛选 U-Net 论文的所有 Chunks")
print("=" * 70)

unet_chunks = []
for m in all_metas:
    src = m.get("source", "")
    if "Colorectal_Polyp_Segmentation_by_U-Net" in src:
        unet_chunks.append(m)

if not unet_chunks:
    print("❌ 未找到 U-Net 论文的任何 Chunk！")
    print("  检查所有 source 值:")
    sources = sorted(set(m.get("source", "?") for m in all_metas))
    for s in sources:
        print(f"    - {s}")
    sys.exit(1)

print(f"✅ U-Net 论文共 {len(unet_chunks)} 个 Chunks")
print()

# ── 2a. 列举 section_id 分布 ──
print("=" * 70)
print("📋 Step 1a: U-Net 论文 Section ID 分布")
print("=" * 70)
sec_counts = {}
for m in unet_chunks:
    sid = m.get("section_id", "?")
    sec_counts[sid] = sec_counts.get(sid, 0) + 1
for sid, cnt in sorted(sec_counts.items(), key=lambda x: x[0]):
    # 取第一个该 section 的 chunk 的 section_title
    title = ""
    for m in unet_chunks:
        if m.get("section_id") == sid:
            title = str(m.get("section_title", ""))[:60]
            break
    print(f"  [{sid}] {title}  ({cnt} chunks)")

# ── 3. 关键词检索 ──
print()
print("=" * 70)
print("🔬 Step 2: 关键词检索 (GIANA / Train set / 10,025)")
print("=" * 70)

keywords = ["GIANA", "Train set", "10,025", "training data", "MICCAI", "dataset"]
found_any = False

for kw in keywords:
    print(f"\n--- 关键词: \"{kw}\" ---")
    matches = []
    for m in unet_chunks:
        text = m.get("text", "")
        if re.search(re.escape(kw), text, re.IGNORECASE):
            matches.append(m)

    if matches:
        found_any = True
        for m in matches:
            idx = m.get("_idx", "?")
            sid = m.get("section_id", "?")
            sec_title = m.get("section_title", "?")
            source = m.get("source", "?")
            text = m.get("text", "")
            # 定位关键词上下文
            kw_pos = text.lower().find(kw.lower())
            ctx_start = max(0, kw_pos - 80)
            ctx_end = min(len(text), kw_pos + 120)
            context = text[ctx_start:ctx_end].replace("\n", "↵ ")

            print(f"\n  ┌─ _idx={idx}  source={source}")
            print(f"  ├─ section_id={sid}")
            print(f"  ├─ section_title=\"{sec_title[:60]}\"")
            print(f"  ├─ text[:300]=\"{text[:300].replace(chr(10),'↵ ')}\"...")
            print(f"  └─ kw_context[...{context.strip()}...]")
    else:
        print(f"  (无匹配)")
        # Dump section titles for context
        print(f"  相关的 section_id 列表:")
        for sid, cnt in sorted(sec_counts.items()):
            sample_text = ""
            for m in unet_chunks:
                if m.get("section_id") == sid:
                    sample_text = m.get("text", "")[:80]
                    break
            print(f"    [{sid}] {sample_text}")

if not found_any:
    print("\n⚠️  未在任何 U-Net chunk 中找到 GIANA 关键字！")
    print("   可能原因:")
    print("    1) Section IV.A 被切出了 chunk 但 section_title 标记错误")
    print("    2) Section IV.A 被跳过了 (clean_glued_text 过滤)")
    print("    3) PDF 解析时 IV.A 标题被水印截断")
    print()
    # 列出所有 section_title 以便肉眼排查
    print("   === 全部 section_title 列表 ===")
    seen = set()
    for m in unet_chunks:
        st = str(m.get("section_title", ""))[:80]
        if st not in seen:
            seen.add(st)
            print(f"    \"{st}\"")

# ── 4. 检查是否存在 arxiv 水印残留 ──
print()
print("=" * 70)
print("🧼 Step 3: arXiv 水印残留检查")
print("=" * 70)
watermark_fragments = []
for m in unet_chunks:
    text = m.get("text", "")
    for pat in [r'ar\s*Xiv:', r'ar Xiv:']:
        for match in re.finditer(pat, text, re.IGNORECASE):
            pos = match.start()
            ctx = text[max(0, pos - 10): pos + 50]
            watermark_fragments.append((m.get("_idx"), ctx))
if watermark_fragments:
    print(f"❌ 发现 {len(watermark_fragments)} 处水印残留:")
    for idx, ctx in watermark_fragments[:5]:
        print(f"  _idx={idx}: ...{ctx}...")
else:
    print("✅ 无水印残留")

print("\n✅ 审计完成")
