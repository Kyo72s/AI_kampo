# supa_client.py
import os
from supabase import create_client
from dotenv import load_dotenv

# ローカル実行時は .env を読み込み（Render では環境変数から自動で読む）
load_dotenv(override=True)

def get_supa():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("環境変数 SUPABASE_URL / SUPABASE_KEY が見つかりません。")
    return create_client(url, key)
