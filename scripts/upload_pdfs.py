"""
PDF 上传脚本 — 将 data/papers/ 的新 PDF 自动上传到 GitHub Release。
用法：
    python scripts/upload_pdfs.py [--tag v1.0]
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="上传 PDF 到 GitHub Release")
    parser.add_argument("--tag", default="v1.0", help="GitHub Release 标签名")
    parser.add_argument("--repo", default="yijieshufu/arxiv-scholar", help="GitHub 仓库")
    args = parser.parse_args()

    papers_dir = Path("data/papers")
    if not papers_dir.exists():
        print(f"❌ data/papers/ 目录不存在")
        return

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"❌ data/papers/ 中没有 PDF 文件")
        return

    print(f"📤 准备上传 {len(pdfs)} 个 PDF 到 GitHub Release {args.tag}")
    print(f"   仓库: {args.repo}")
    print()

    # 检查 gh CLI 是否安装
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 请先安装 GitHub CLI: https://cli.github.com/")
        print("   安装后运行: gh auth login")
        return

    # 检查 Release 是否存在
    r = subprocess.run(
        ["gh", "release", "view", args.tag, "--repo", args.repo],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"⚠️  Release {args.tag} 不存在，尝试创建...")
        subprocess.run(
            ["gh", "release", "create", args.tag, "--repo", args.repo,
             "--title", f"Papers {args.tag}", "--notes", "论文 PDF 集合"],
            check=True,
        )
        print(f"✅ Release {args.tag} 已创建")

    # 上传每个 PDF
    for pdf in pdfs:
        print(f"  📄 {pdf.name}...", end=" ", flush=True)
        r = subprocess.run(
            ["gh", "release", "upload", args.tag, str(pdf),
             "--repo", args.repo, "--clobber"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print("✅")
        else:
            print(f"❌ {r.stderr.strip()}")

    print()
    print(f"✅ 全部上传完成！")
    print(f"   MINERU_UPLOAD_URL=https://github.com/{args.repo}/releases/download/{args.tag}")

if __name__ == "__main__":
    main()
