import google.generativeai as genai
import os

# 1. 外部ファイルからAPIキーを読み込む
key_file = "GOOGLE_API_KEY"

try:
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as f:
            # strip() で改行コードや余計な空白を除去
            api_key = f.read().strip()
        
        # 2. Gemini APIの設定
        genai.configure(api_key=api_key)

        print("--- 利用可能なモデル一覧 ---")
        models = genai.list_models()
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f"モデル名: {m.name}")
    else:
        print(f"エラー: {key_file} が見つかりません。ファイルを作成してください。")

except Exception as e:
    print(f"APIキーが無効、または接続エラーです: {e}")