"""上传 PDF 到腾讯云 COS"""
import os, sys
from pathlib import Path

# 如果没装 cos-python-sdk，先运行: pip install cos-python-sdk-v5
try:
    from qcloud_cos import CosConfig, CosS3Client
except ImportError:
    print("请先安装: pip install cos-python-sdk-v5")
    sys.exit(1)

# 配置（从 .env 或环境变量读取）
from dotenv import load_dotenv
load_dotenv()

BUCKET = "papers-1304363349"
REGION = "ap-guangzhou"
SECRET_ID = os.getenv("COS_SECRET_ID", "")
SECRET_KEY = os.getenv("COS_SECRET_KEY", "")

if not SECRET_ID or not SECRET_KEY:
    print("请在 .env 中配置 COS_SECRET_ID 和 COS_SECRET_KEY")
    print("获取地址: https://console.cloud.tencent.com/cam/capi")
    sys.exit(1)

config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
client = CosS3Client(config)

papers_dir = Path("data/papers")
if not papers_dir.exists():
    print(f"❌ data/papers/ 不存在")
    sys.exit(1)

pdfs = sorted(papers_dir.glob("*.pdf"))
if not pdfs:
    print(f"❌ data/papers/ 中没有 PDF")
    sys.exit(1)

for pdf in pdfs:
    key = pdf.name
    print(f"  📤 {key}...", end=" ", flush=True)
    try:
        client.upload_file(
            Bucket=BUCKET,
            LocalFilePath=str(pdf),
            Key=key,
        )
        print("✅")
    except Exception as e:
        print(f"❌ {e}")

print(f"\n✅ 全部上传完成！")
print(f"   MINERU_UPLOAD_URL=https://{BUCKET}.cos.{REGION}.myqcloud.com")
