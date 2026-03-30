import streamlit as st
import json
import asyncio
import edge_tts
import os
import pandas as pd
import tempfile
import random
import time
import re
from datetime import datetime
from google import genai
from google.genai import types

# ==========================================
# 1. 設定 & カスタムCSS（SILフォント適用）
# ==========================================
def apply_custom_css():
    # Charis SIL (発音記号に最適なSILフォント) を読み込み
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Charis+SIL:ital,wght@0,400;0,700;1,400;1,700&display=swap');
        .phonetic-text {
            font-family: 'Charis SIL', serif;
            font-size: 1.3em;
            color: #2e7d32;
            background-color: #f1f8e9;
            padding: 2px 8px;
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

def load_api_key():
    """GEMINI_API_KEYファイルからキーを読み取る"""
    try:
        with open("GEMINI_API_KEY", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error("エラー: 'GEMINI_API_KEY' ファイルが同フォルダ内に見つかりません。")
        return None

GEMINI_API_KEY = load_api_key()
BASE_CSV_DIR = "csv"
DEFAULT_EN_VOICE = "en-US-AriaNeural"

MODEL_PRIORITY = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview"
]

if not os.path.exists(BASE_CSV_DIR):
    os.makedirs(BASE_CSV_DIR)

# ==========================================
# 2. ロジック関数
# ==========================================

def clean_phonetic(text):
    """発音記号からスラッシュ / を除去する"""
    if not text: return ""
    return text.replace("/", "").strip()

async def generate_audio(text, voice, filename, folder):
    path = os.path.join(folder, filename)
    communicate = edge_tts.Communicate(text, voice)
    
    max_retries = 3
    for i in range(max_retries):
        try:
            await communicate.save(path)
            return path
        except Exception as e:
            if i < max_retries - 1:
                # 1〜3秒ランダムに待機してリトライ
                await asyncio.sleep(random.uniform(1, 3))
                continue
            else:
                st.error(f"音声生成エラー (3回試行): {e}")
                return None

def fetch_word_data_via_ai(word):
    if not GEMINI_API_KEY:
        raise Exception("APIキーが設定されていないため、AI生成を利用できません。")
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # 品詞間の読点、スラッシュなし発音記号、英語指示への翻訳を統合
    prompt = f"""
    Return ONLY a JSON object for the English word/phrase "{word}".
    
    [INSTRUCTIONS]:
    1. 'word_meaning': Provide the Japanese meanings using the following strict format:
       - Format: [Part of Speech] Meaning
       - If there are multiple meanings or parts of speech, separate them with a Japanese comma (、).
       - Ensure the space is AFTER the bracket, not before it.
       - Example: [名] 監視装置、 [動] 監視する
       - Another Example: [形] 重要な、 [名] 要点
    2. 'phonetic': Provide the accurate IPA phonetic notation. Do NOT include slashes (/).
    3. 'example_sentence': Create a natural English example sentence using the word.
    4. 'example_meaning': Provide the Japanese translation of the example sentence.
    5. If the word/phrase is invalid, return: {{"error": "not_found"}}
    6. 'category': "単語" if it's a single word, "熟語" if it's a phrase/idiom.

    JSON Schema:
    {{
        "word": "{word}",
        "phonetic": "IPA notation (no slashes)",
        "word_meaning": "[POS] meaning、 [POS] meaning",
        "example_sentence": "English example",
        "example_meaning": "Japanese translation",
        "category": "単語 or 熟語"
    }}
    """
    for model_id in MODEL_PRIORITY:
        try:
            response = client.models.generate_content(
                model=model_id, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text)
            if "error" in data: raise ValueError(f"「{word}」は見つかりませんでした。")
            
            # 発音記号の整形
            data['phonetic'] = clean_phonetic(data.get('phonetic', ''))
            return data
        except Exception:
            continue
    raise Exception("AIデータの生成に失敗しました。")

def handle_ai_generation_on_submit():
    """入力欄でEnterが押された時のコールバック"""
    word = st.session_state.get("word_input_field")
    if word:
        st.session_state.process_trigger = True
        st.session_state.target_word = word

def get_deck_info_list():
    """デッキフォルダと単語数を取得"""
    if not os.path.exists(BASE_CSV_DIR): return [], []
    decks = sorted([d for d in os.listdir(BASE_CSV_DIR) if os.path.isdir(os.path.join(BASE_CSV_DIR, d))])
    display_list = []
    raw_names = []
    for d in decks:
        count = len([f for f in os.listdir(os.path.join(BASE_CSV_DIR, d)) if f.endswith(".csv")])
        display_list.append(f"{d} ({count})")
        raw_names.append(d)
    return display_list, raw_names

def load_data_from_deck(deck_name):
    all_data = []
    deck_path = os.path.join(BASE_CSV_DIR, deck_name)
    if os.path.exists(deck_path):
        for file in os.listdir(deck_path):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(deck_path, file))
                    d = df.iloc[0].to_dict()
                    d['_filename'] = file
                    all_data.append(d)
                except: pass
    return all_data

def pick_random_meaning(text):
    """
    【テスト表示・判定専用】
    保存データから意味の候補を分割し、その中からランダムに1つを抽出する。
    例: "[名] 三角形、トライアングル" -> "トライアングル" (ランダム)
    """
    if not text: return ""
    # 1. 品詞タグ [名] などを除去
    text = re.sub(r"\[.*?\]", "", text)
    # 2. 読点、カンマ、スラッシュ、スペース等で分割
    meanings = re.split(r"[、,，/／/ \s]", text)
    # 3. 空文字を除外してリスト化
    candidates = [m.strip() for m in meanings if m.strip()]
    
    if not candidates:
        return text.strip()
    
    # 4. 候補の中からランダムに1つ返す
    return random.choice(candidates)

def extract_test_questions(deck_name, num=10):
    """
    【単熟語テストモード専用ロジック】
    1. 保存された CSV データは一切変更しない。
    2. 綴りが同じ単語を誤答候補から排除。
    3. 正解および誤答の選択肢を pick_random_meaning で単一語にする。
    """
    words = load_data_from_deck(deck_name)
    if not words: return []
    
    # 出題対象をランダムに選定
    random.shuffle(words)
    questions = words[:num]
    
    for q in questions:
        # この問題の正解スペル (例: triangle)
        correct_spelling = str(q['word']).strip().lower()
        
        # 今回のテストで「正解」として扱う単一の語をランダムに選定
        display_correct = pick_random_meaning(q['word_meaning'])
        
        # 誤答候補：スペルが異なる単語のみを対象にする
        # 各単語からもランダムに1つの意味を抽出して選択肢の種にする
        other_candidates = [pick_random_meaning(w['word_meaning']) for w in words 
                            if str(w['word']).strip().lower() != correct_spelling]
        
        # 重複を排除し、今回の正解と同じ語が表示されないようガード
        unique_others = list(dict.fromkeys(other_candidates))
        unique_others = [m for m in unique_others if m != display_correct]
        
        # 誤答を3つ選ぶ
        if len(unique_others) >= 3:
            distractors = random.sample(unique_others, 3)
        else:
            distractors = unique_others + ["(選択肢不足)"] * (3 - len(unique_others))
        
        # 最終的な4択（すべてランダム抽出された1語の状態）
        final_choices = distractors + [display_correct]
        random.shuffle(final_choices)
        
        # セッション管理用のキーにセット
        q['test_choices'] = final_choices
        q['test_correct_word'] = display_correct
        
    return questions

def get_existing_words(deck_name):
    """指定したデッキ内に既に存在する単語のリスト（小文字）をセットで返す"""
    existing_words = set()
    deck_path = os.path.join(BASE_CSV_DIR, deck_name)
    if os.path.exists(deck_path):
        for f in os.listdir(deck_path):
            if f.endswith(".csv"):
                try:
                    # 最初の1行だけ読み込んで単語を取得
                    df = pd.read_csv(os.path.join(deck_path, f))
                    word = str(df.iloc[0]['word']).strip().lower()
                    existing_words.add(word)
                except: pass
    return existing_words

def fetch_multiple_words_via_ai(words_list):
    """複数単語を1回のAPIリクエストで取得する"""
    if not GEMINI_API_KEY:
        raise Exception("APIキーが設定されていません。")
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    words_str = ", ".join(words_list)
    
    # リスト形式で返却させるプロンプト
    prompt = f"""
    Return a JSON ARRAY of objects for these English words/phrases: [{words_str}].
    
    [INSTRUCTIONS]:
    For each item, provide:
    1. 'word_meaning': Provide the Japanese meanings using the following strict format:
       - Format: [Part of Speech] Meaning
       - If there are multiple meanings or parts of speech, separate them with a Japanese comma (、).
       - Ensure the space is AFTER the bracket, not before it.
       - Example: [名] 監視装置、 [動] 監視する
       - Another Example: [形] 重要な、 [名] 要点
    2. 'phonetic': Provide the accurate IPA phonetic notation. Do NOT include slashes (/).
    3. 'example_sentence': Create a natural English example sentence using the word.
    4. 'example_meaning': Provide the Japanese translation of the example sentence.
    5. If the word/phrase is invalid, return: {{"error": "not_found"}}
    6. 'category': "単語" if it's a single word, "熟語" if it's a phrase/idiom.
    7. If an item is invalid, still include the 'word' and 'category', but set other fields to "Error: Not Found".

    JSON Schema:
    {{
        "data": [
            {{
                "word": "word1",
                "phonetic": "...",
                "word_meaning": "...",
                "example_sentence": "...",
                "example_meaning": "...",
                "category": "単語 or 熟語"
            }},
            ...
        ]
    }}
    """
    
    for model_id in MODEL_PRIORITY:
        try:
            response = client.models.generate_content(
                model=model_id, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            # --- デバッグ用：AIの生レスポンスを表示 ---
            print(f"--- Model: {model_id} ---")
            print("Response Text:", response.text) 
            
            raw_res = json.loads(response.text)
            data_list = raw_res.get("data", []) 
            
            for item in data_list:
                item['phonetic'] = clean_phonetic(item.get('phonetic', ''))
            return data_list
        except Exception as e:
            # --- デバッグ用：失敗した理由を表示 ---
            print(f"!!! Error with {model_id}: {e}")
            continue
    raise Exception("一括AIデータの生成に失敗しました。")

def generate_words_by_theme(theme, word_count, phrase_count):
    """テーマに基づいて単語と熟語のリストを生成する（1回目）"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    Based on the theme "{theme}", select {word_count} essential English words and {phrase_count} essential English idioms/phrases.
    
    [IMPORTANT CASE RULES]:
    - Use lowercase for common nouns and verbs (e.g., "apple", "run", "take a break").
    - Use Capital Letters ONLY for proper nouns (e.g., "London", "Google", "Sunday").

    Return ONLY a JSON object with a "words" key containing an array of strings.
    Combine both words and phrases into this single array.
    
    JSON Schema:
    {{
        "words": ["word1", "word2", "idiom 1", "phrase 2", ...]
    }}
    """
    
    for model_id in MODEL_PRIORITY:
        try:
            response = client.models.generate_content(
                model=model_id, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            raw_res = json.loads(response.text)
            return raw_res.get("words", [])
        except Exception as e:
            print(f"Error in theme generation ({model_id}): {e}")
            continue
    return []

def sanitize_filename(filename):
    """
    ファイル名に使用できない文字を '_' に置換する
    Windowsの禁止文字: '\\ / : * ? " < > |'
    """
    # Windowsで禁止されている文字を正規表現で指定
    # [\\/:*?"<>|] にマッチする文字を "_" に変える
    return re.sub(r'[\\/:*?"<>|]', '_', filename)

def process_and_save_words(word_list, deck_name):
    """詳細情報を生成し、指定したデッキに保存する共通ロジック"""
    deck_path = os.path.join(BASE_CSV_DIR, deck_name)
    os.makedirs(deck_path, exist_ok=True)
    
    with st.spinner(f"{len(word_list)}語を詳細解析中..."):
        try:
            results = fetch_multiple_words_via_ai(word_list)
            current_files = [f for f in os.listdir(deck_path) if f.endswith(".csv")]

            for data in results:
                if 'word' not in data: continue

                raw_word = data['word'].strip()
    
                # 【小文字化ロジック】
                if raw_word[0].isupper() and (len(raw_word) == 1 or raw_word[1:].islower()):
                    data['word'] = raw_word.lower()
                else:
                    data['word'] = raw_word

                target_word_lower = data['word'].strip().lower()
                
                # 重複ファイルの削除
                for f in current_files:
                    try:
                        file_full_path = os.path.join(deck_path, f)
                        if os.path.exists(file_full_path):
                            check_df = pd.read_csv(file_full_path)
                            if str(check_df.iloc[0]['word']).strip().lower() == target_word_lower:
                                os.remove(file_full_path)
                    except: pass

                # --- 保存ファイル名の禁則処理 ---
                # 1. スペースをアンダースコアに置換
                word_label = data['word'].replace(' ', '_')
                # 2. OSの禁止文字（?や:など）を安全な文字に置換
                word_label = sanitize_filename(word_label)
                ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                new_fn = f"{word_label}_{ts}.csv"
                pd.DataFrame([data]).to_csv(os.path.join(deck_path, new_fn), index=False, encoding="utf-8-sig")

            st.success(f"完了！ '{deck_name}' に {len(results)}語 登録しました。")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error(f"解析エラー: {e}")

def handle_mode_change():
    """モードが切り替わった際に実行されるクリーンアップ処理"""
    # 1. デッキ管理モードのフラグをすべてFalseにする
    for key in list(st.session_state.keys()):
        if key.startswith("edit_mode_"):
            st.session_state[key] = False
            
    # 2. テスト状態のリセット
    st.session_state.test_active = False
    st.session_state.test_finished = False

def save_test_stats(deck_name, score, total):
    stats_dir = "stats"
    if not os.path.exists(stats_dir): os.makedirs(stats_dir)
    stats_file = os.path.join(stats_dir, f"{deck_name}.csv")
    new_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "score": score,
        "total": total,
        "accuracy": round((score / total) * 100, 1) if total > 0 else 0
    }
    df = pd.read_csv(stats_file) if os.path.exists(stats_file) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(stats_file, index=False, encoding="utf-8-sig")

def get_stats_data(deck_name):
    stats_file = os.path.join("stats", f"{deck_name}.csv")
    if os.path.exists(stats_file): return pd.read_csv(stats_file)
    return None

# ==========================================
# 3. メイン UI
# ==========================================
def main():
    st.set_page_config(page_title="AI Anki Tool", layout="wide", page_icon="🚀")
    apply_custom_css()
    
    if "current_mode" not in st.session_state: st.session_state.current_mode = "単語登録"
    
    st.sidebar.title("🚀 AI Anki Tool")

    # 初期値の設定（エラー防止）
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "単語登録"

    # ラジオボタンの修正
    mode = st.sidebar.radio(
        "モード選択", 
        ["単語登録", "単熟語テスト", "デッキ管理", "学習統計"],
        key="current_mode",        # session_state.current_mode と同期
        on_change=handle_mode_change # モード変更時に自動実行
    )

    deck_display, deck_raw = get_deck_info_list()
    
    # --- 1. 単語登録モード ---
    if mode == "単語登録":
        if "step" not in st.session_state: st.session_state.step = "input"
        if "process_trigger" not in st.session_state: st.session_state.process_trigger = False
        
        st.sidebar.divider()
        menu_options = deck_display + ["+ 新規デッキ作成"]
        selected_idx = st.sidebar.selectbox("追加先のデッキ", range(len(menu_options)), 
                                         format_func=lambda x: menu_options[x])
        
        if menu_options[selected_idx] == "+ 新規デッキ作成":
            new_name = st.sidebar.text_input("新規デッキ名")
            if st.sidebar.button("作成"):
                if new_name:
                    os.makedirs(os.path.join(BASE_CSV_DIR, new_name), exist_ok=True)
                    st.rerun()
            return
        
        current_deck = deck_raw[selected_idx]
        st.title(f"⚡ {current_deck} に登録")

        # 【追加】登録方式の選択
        reg_type = st.radio("登録方式", ["個別登録", "一括登録"], horizontal=True)

        if reg_type == "個別登録":
            # --- 既存の個別登録ロジック ---
            if st.session_state.process_trigger:
                # (ここに元のコードの AI解析・音声生成ロジックを入れる)
                with st.spinner(f"「{st.session_state.target_word}」を解析中..."):
                    try:
                        res = fetch_word_data_via_ai(st.session_state.target_word)
                        st.session_state.word_data = res
                        tmp = tempfile.gettempdir()
                        st.session_state.word_audio = asyncio.run(generate_audio(res['word'], DEFAULT_EN_VOICE, f"w_{time.time()}.mp3", tmp))
                        st.session_state.ex_audio = asyncio.run(generate_audio(res['example_sentence'], DEFAULT_EN_VOICE, f"e_{time.time()}.mp3", tmp))
                        st.session_state.step = "confirm"
                    except Exception as e:
                        st.error(e)
                    finally:
                        st.session_state.process_trigger = False
                        st.rerun()

            if st.session_state.step == "input":
                st.text_input("英単語/熟語を入力 (Enterで生成)", key="word_input_field", on_change=handle_ai_generation_on_submit)
                st.info("💡 単語を入力してEnterキーを押すと、自動的にAIが意味を生成します。")
            elif st.session_state.step == "confirm":
                data = st.session_state.word_data

                # --- 追加: カテゴリ表示を含むプレビュー UI ---
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(f"単語: {data['word']}")
                    with col2:
                        # AIが判定した 'category' を表示
                        st.info(f"種別: {data.get('category', '不明')}")

                    st.markdown(f"**意味:** {data['word_meaning']}")
                    st.markdown(f"**発音:** <span class='phonetic-text'>{data['phonetic']}</span>", unsafe_allow_html=True)
                    st.audio(st.session_state.word_audio)
                    st.divider()
                    st.write(f"**例文:** {data['example_sentence']}")
                    st.write(f"**訳:** {data['example_meaning']}")
                    st.audio(st.session_state.ex_audio)

                c1, c2 = st.columns(2)
                if c1.button("✅ この内容で保存", type="primary", use_container_width=True):
                    deck_path = os.path.join(BASE_CSV_DIR, current_deck)
                    # 1. スペースをアンダースコアに置換
                    word_label = data['word'].replace(' ', '_')
                    # 2. 【ここを追加！】OS禁止文字（?や:など）を安全な文字に置換
                    word_label = sanitize_filename(word_label)
                    ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                    new_fn = f"{word_label}_{ts}.csv"

                    # data辞書には category が含まれているため、そのままDataFrameにして保存
                    pd.DataFrame([data]).to_csv(os.path.join(deck_path, new_fn), index=False, encoding="utf-8-sig")

                    st.success(f"「{data['word']}」を保存しました。")
                    time.sleep(1)
                    st.session_state.step = "input"
                    st.rerun()

                if c2.button("🔙 やり直し", use_container_width=True):
                    st.session_state.step = "input"
                    st.rerun()

        else:
            # --- 【新規】登録モードの切り替えタブ ---
            tab_manual, tab_ai_theme = st.tabs(["📋 手動一括入力", "🪄 AIテーマ自動生成"])

            with tab_manual:
                bulk_input = st.text_area("英単語をカンマ区切りで入力", placeholder="apple, get up, take a break...", height=200, key="manual_bulk")
                if st.button("一括AI解析 & 保存", type="primary", key="btn_manual"):
                    if bulk_input:
                        raw_list = [w.strip() for w in bulk_input.replace("、", ",").split(",") if w.strip()]
                        # 重複排除ロジック
                        unique_input_words = []
                        seen_lower = set()
                        for w in raw_list:
                            if w.lower() not in seen_lower:
                                unique_input_words.append(w)
                                seen_lower.add(w.lower())

                        words_to_query = unique_input_words 
                        
                        # 解析と保存の共通処理（後述のヘルパー関数化するとスッキリします）
                        process_and_save_words(words_to_query, current_deck)
                    else:
                        st.warning("単語を入力してください。")

            with tab_ai_theme:
                st.write("テーマを入力すると、関連する英単語をAIが選定して新規デッキを作成します。")
                # --- 注意書きの追加 ---
                st.caption("⚠️ 合計登録数は最大30項目までです（単語数 + 熟語数 ≦ 30）。")

                with st.form("ai_theme_form"):
                    t_theme = st.text_input("英語のテーマ", placeholder="例: IT業界のニュース、海外旅行")
                    t_deck_name = st.text_input("新規デッキ名 ※既存デッキ名を入力した場合はそのデッキに追加されます", placeholder="例: MyNewDeck")
                    c1, c2 = st.columns(2)
                    t_word_count = c1.number_input("英単語数", min_value=0, max_value=30, value=10)
                    t_phrase_count = c2.number_input("英熟語数", min_value=0, max_value=30, value=5)
                    
                    submit_theme = st.form_submit_button("AIでデッキを自動作成", type="primary")

                if submit_theme:
                    # 合計値の計算
                    total_count = t_word_count + t_phrase_count

                    if not t_theme or not t_deck_name:
                        st.error("テーマとデッキ名を入力してください。")
                    # --- バリデーションの追加 ---
                    elif total_count > 30:
                        st.error(f"合計登録数が制限（30）を超えています。現在は {total_count} です。数値を調整してください。")
                    elif total_count == 0:
                        st.error("単語数または熟語数を1以上に設定してください。")
                    else:
                        # 31以上の場合はここに来ないため、安心して処理を続行できる
                        # 1. デッキフォルダの準備
                        new_deck_path = os.path.join(BASE_CSV_DIR, t_deck_name)
                        os.makedirs(new_deck_path, exist_ok=True)

                        # 2. 単語リストの生成（1回目のAPI）
                        with st.spinner("AIがテーマに沿った単語を選定中..."):
                            generated_list = generate_words_by_theme(t_theme, t_word_count, t_phrase_count)
                        
                        if generated_list:
                            st.info(f"選定された単語: {', '.join(generated_list)}")
                            # 3. 詳細解析と保存（2回目のAPI）
                            process_and_save_words(generated_list, t_deck_name)
                        else:
                            st.error("単語リストの生成に失敗しました。")

    # --- 2. 単熟語テストモード ---
    elif mode == "単熟語テスト":
        st.title("📝 単熟語テスト")
        if "test_active" not in st.session_state: st.session_state.test_active = False
        if "test_finished" not in st.session_state: st.session_state.test_finished = False

        with st.expander("テスト設定", expanded=True):
            t_idx = st.selectbox("デッキを選択", range(len(deck_display)), 
                             format_func=lambda x: deck_display[x], 
                             disabled=st.session_state.test_active) # テスト中は変更不可にする
            if deck_raw:
                t_deck = deck_raw[t_idx]
                pool = load_data_from_deck(t_deck)
                c1, c2 = st.columns(2)
                with c1: test_type = st.radio("形式", ["英 → 日", "日 → 英"], horizontal=True, disabled=st.session_state.test_active)
                with c2: q_count = st.number_input("出題数", 1, max(1, len(pool)), min(10, max(1, len(pool))), disabled=st.session_state.test_active)

                if st.button("テスト開始", type="primary", disabled=st.session_state.test_active):
                    if len(pool) < 4: st.error("選択肢生成のため4語以上必要です。")
                    else:
                        st.session_state.test_q = random.sample(pool, q_count)
                        st.session_state.test_idx = 0
                        st.session_state.test_results = []
                        st.session_state.test_active, st.session_state.test_finished = True, False
                        st.session_state.start_time = time.time()
                        st.session_state.test_sid = time.time()
                        for k in list(st.session_state.keys()):
                            if k.startswith(("q_cache_", "a_cache_", "opts_cache_")): del st.session_state[k]
                        st.rerun()

        if st.session_state.test_active:
            idx = st.session_state.test_idx
            q = st.session_state.test_q[idx]
            st.write(f"**Progress: {idx + 1} / {len(st.session_state.test_q)}**")

            q_k, a_k, o_k = f"q_cache_{idx}", f"a_cache_{idx}", f"opts_cache_{idx}"

            # --- 修正箇所: 選択肢生成ロジック ---
            if test_type == "英 → 日":
                question_display = q['word']
                # 正解をランダムに1つ確定
                if a_k not in st.session_state: st.session_state[a_k] = pick_random_meaning(q['word_meaning'])
                correct_ans = st.session_state[a_k]

                if o_k not in st.session_state:
                    # 課題解決: 綴りが違う単語からのみ誤答を生成し、かつランダム1語にする
                    others = []
                    for w in pool:
                        if str(w['word']).strip().lower() != str(q['word']).strip().lower():
                            others.append(pick_random_meaning(w['word_meaning']))

                    # 重複排除と正解の除去
                    others = list(set(others))
                    if correct_ans in others: others.remove(correct_ans)

                    final_opts = random.sample(others, min(3, len(others))) + [correct_ans]
                    random.shuffle(final_opts)
                    st.session_state[o_k] = final_opts
            else:
                # 日 → 英 の場合
                if q_k not in st.session_state: st.session_state[q_k] = pick_random_meaning(q['word_meaning'])
                question_display = st.session_state[q_k]
                correct_ans = q['word']

                if o_k not in st.session_state:
                    # スペルが違うものを誤答にする
                    others = list(set([w['word'] for w in pool if str(w['word']).strip().lower() != str(q['word']).strip().lower()]))
                    final_opts = random.sample(others, min(3, len(others))) + [correct_ans]
                    random.shuffle(final_opts)
                    st.session_state[o_k] = final_opts

            st.markdown(f"<p style='font-size:32px; font-weight:bold; color:#1E88E5;'>{question_display}</p>", unsafe_allow_html=True)
            ans = st.radio("選択してください:", st.session_state[o_k], index=None, key=f"r_{st.session_state.test_sid}_{idx}")

            c1, c2 = st.columns([1, 4])
            if c1.button("回答確定") and ans:
                this_time = time.time() - st.session_state.start_time
                # 1語同士の単純比較でOK
                is_ok = (ans == correct_ans)

                st.session_state.test_results.append({
                    "q": question_display, "correct": correct_ans, "user": ans, "is_ok": is_ok,
                    "time": this_time, "full_info": q['word_meaning'] if test_type == "英 → 日" else q['word']
                })
                if idx + 1 < len(st.session_state.test_q):
                    st.session_state.test_idx += 1
                    st.session_state.start_time = time.time()
                else:
                    st.session_state.test_active, st.session_state.test_finished = False, True
                    save_test_stats(t_deck, sum(1 for r in st.session_state.test_results if r['is_ok']), len(st.session_state.test_results))
                st.rerun()
        
            if c2.button("中断"):
                st.session_state.test_active = False
                st.rerun()

        elif st.session_state.test_finished:
            st.header("🎉 テスト結果")
            results = st.session_state.test_results
            score = sum(1 for r in results if r['is_ok'])
            total = len(results)
            
            # --- 満点演出の追加 ---
            if score == total and total > 0:
                st.balloons() # 風船を飛ばす
                st.success("🎊 PERFECT! 全問正解です！ 🎊")
            # --------------------

            st.metric("Score", f"{score} / {total}", f"{round(score/total*100,1)}%" if total>0 else "")
            
            for i, r in enumerate(results):
                # expanded=True にすることで、正解・不正解に関わらず最初から中身を表示します
                # タイトルのアイコン表示はそのまま維持
                icon = "✅" if r['is_ok'] else "❌"
                
                with st.expander(f"{i+1}. {icon} {r['q']}", expanded=True):
                    # 正解とユーザーの回答を強調して表示
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.write(f"**解答:**")
                        st.info(r['correct'])
                    
                    with c2:
                        st.write(f"**あなたの答え:**")
                        if r['is_ok']:
                            st.success(r['user'])
                        else:
                            st.error(r['user'])
                    
                    # 補足情報（意味や例文など）
                    st.caption(f"💡 詳細: {r['full_info']} （回答時間: {r['time']:.1f}s）")
                    
            if st.button("終了して戻る", type="primary", use_container_width=True):
                st.session_state.test_finished = False
                # テスト終了時にアクティブフラグも一応折っておく
                st.session_state.test_active = False 
                st.rerun()

    # --- 3. デッキ管理モード ---
    elif mode == "デッキ管理":
        st.title("🛠️ デッキ管理")

        # 1. 初期化（最初のみ）
        if "selected_deck_index" not in st.session_state:
            st.session_state.selected_deck_index = 0

        # 2. セレクトボックス
        # keyをあえて使わず、indexのみで管理するのが「連続操作」には最も強いです
        e_idx = st.selectbox(
            "デッキを選択", 
            range(len(deck_display)), 
            index=st.session_state.selected_deck_index,
            format_func=lambda x: deck_display[x]
        )
        
        # 3. 選択が変わったら即座に保存
        if e_idx != st.session_state.selected_deck_index:
            st.session_state.selected_deck_index = e_idx
            st.rerun()

        if deck_raw:
            e_deck = deck_raw[e_idx]
            deck_full_path = os.path.join(BASE_CSV_DIR, e_deck)
            
            tab1, tab2 = st.tabs(["🔍 検索・編集・一括削除", "⚠️ デッキ完全削除"])
            
            with tab1:
                # --- 検索・クリアのロジック (確実にクリアされる方式) ---
                search_key_base = f"sq_{e_deck}"
                # クリアボタンが押された回数をカウントし、それをキーに含めることで入力をリセット
                if "search_reset_counter" not in st.session_state:
                    st.session_state.search_reset_counter = 0
                
                # 動的なキーを作成 (これが変わると入力欄が真っさらになる)
                current_input_key = f"in_{search_key_base}_{st.session_state.search_reset_counter}"

                c_s, c_c = st.columns([4, 1])
                
                # 検索入力欄
                search = c_s.text_input(
                    "検索...", 
                    key=current_input_key, 
                    label_visibility="collapsed",
                    placeholder="単語または意味で検索"
                )
                
                # 検索ボタンが押されたか、入力がある場合に値を保持
                query = search if search else ""

                if c_c.button("クリア", key=f"clr_btn_{e_deck}", use_container_width=True):
                    # カウンターを増やすことで、次の rerun 時に text_input のキーが変わり、強制リセットされる
                    st.session_state.search_reset_counter += 1
                    st.rerun()

                words = load_data_from_deck(e_deck)
                # フィルタリングに使用
                display_words = [w for w in words if query.lower() in w['word'].lower() or query in w['word_meaning']]
                
                # --- 一括削除ロジック ---
                if query and display_words:
                    st.info(f"検索結果: {len(display_words)} 件がヒットしています")
                    if st.button(f"🔍 検索結果の {len(display_words)} 件を一括削除する", key=f"bulk_btn_{e_deck}", type="secondary"):
                        st.session_state.confirm_bulk_delete = True
                    
                    if st.session_state.get("confirm_bulk_delete", False):
                        with st.status("一括削除の確認"):
                            st.error(f"表示されている {len(display_words)} 件をすべて削除します。よろしいですか？")
                            cb1, cb2 = st.columns(2)
                            if cb1.button("はい、一括削除を実行", key="bulk_yes", type="primary", use_container_width=True):
                                for w in display_words:
                                    os.remove(os.path.join(deck_full_path, w['_filename']))
                                st.session_state.confirm_bulk_delete = False
                                st.rerun() # rerunしてもselected_deck_indexがあるため維持される
                            if cb2.button("キャンセル", key="bulk_no", use_container_width=True):
                                st.session_state.confirm_bulk_delete = False
                                st.rerun()

                st.divider()

                # --- 個別表示・編集・削除 ---
                if not display_words:
                    st.info("該当する単語がありません。")
                else:
                    for i, w in enumerate(display_words, 1):
                        edit_key = f"edit_mode_{w['_filename']}"
                        if edit_key not in st.session_state:
                            st.session_state[edit_key] = False

                        # 通常表示時
                        if not st.session_state[edit_key]:
                            with st.container(border=True):
                                # カラム構成を5つに分割（番号, 単語, カテゴリ, 意味, ボタンx2）
                                # 比率は画面の幅に合わせて適宜調整してください
                                c0, c1, c2, c3, c4, c5 = st.columns([0.4, 2.0, 0.8, 3.8, 1.0, 1.0])
                                
                                # 1. 通し番号
                                c0.write(f"**{i}**") 
                                
                                # 2. 単語（太字）
                                c1.write(f"**{w['word']}**")
                                
                                # 3. カテゴリ（単語のすぐ右）
                                # 辞書 w に 'category' がない場合を考慮して get を使用
                                category = w.get('category', '単語')
                                c2.caption(f"[{category}]") 
                                
                                # 4. 意味（左端が揃う）
                                c3.write(f"{w['word_meaning']}")

                                # 5. 編集ボタン
                                if c4.button("編集", key=f"eb_{w['_filename']}", use_container_width=True):
                                    st.session_state[edit_key] = True
                                    st.rerun()
                                
                                # 6. 削除ボタン
                                if c5.button("削除", key=f"db_{w['_filename']}", use_container_width=True):
                                    f_path = os.path.join(deck_full_path, w['_filename'])
                                    if os.path.exists(f_path):
                                        os.remove(f_path)
                        
                                        # 【重要】削除直前に「現在のインデックス」をSession Stateに再セットする
                                        # これにより、rerun後の描画で確実に同じデッキが選択されます
                                        st.session_state.selected_deck_index = e_idx
                                        st.rerun()
                        
                        # 編集フォーム表示時
                        else:
                            with st.container(border=True):
                                st.write(f"**No.{i} を編集中...**")
                                st.caption(f"ファイル: {w['_filename']}")
                                
                                # --- 入力エリア ---
                                e_word = st.text_input("単語", value=w['word'], key=f"ew_{w['_filename']}")

                                # 【追加】カテゴリ（種別）の選択
                                current_cat = w.get('category', '単語')
                                cat_options = ["単語", "熟語"]
                                cat_idx = cat_options.index(current_cat) if current_cat in cat_options else 0
                                e_category = st.selectbox("種別", options=cat_options, index=cat_idx, key=f"ec_{w['_filename']}")

                                e_phonetic = st.text_input("発音記号", value=w.get('phonetic',''), key=f"ep_{w['_filename']}")
                                e_meaning = st.text_input("意味", value=w['word_meaning'], key=f"em_{w['_filename']}")
                                e_ex = st.text_area("例文", value=w.get('example_sentence', ''), key=f"ex_{w['_filename']}")
                                e_ex_m = st.text_area("訳", value=w.get('example_meaning', ''), key=f"exm_{w['_filename']}")
                                
                                st.divider()
                                # --- 音声プレビュー（機能維持） ---
                                st.write("🎧 **書き換えた内容で音声プレビュー**")
                                ca1, ca2 = st.columns(2)
                                with ca1:
                                    if st.button("新単語をプレビュー", key=f"p_w_{w['_filename']}", use_container_width=True):
                                        if e_word:
                                            tmp = tempfile.gettempdir()
                                            p = asyncio.run(generate_audio(e_word, DEFAULT_EN_VOICE, f"pre_w_{time.time()}.mp3", tmp))
                                            st.audio(p, autoplay=True)
                                with ca2:
                                    if st.button("新例文をプレビュー", key=f"p_e_{w['_filename']}", use_container_width=True):
                                        if e_ex:
                                            tmp = tempfile.gettempdir()
                                            p = asyncio.run(generate_audio(e_ex, DEFAULT_EN_VOICE, f"pre_e_{time.time()}.mp3", tmp))
                                            st.audio(p, autoplay=True)
                                
                                st.divider()
                                # --- 保存・キャンセル ---
                                ce1, ce2 = st.columns(2)
                                if ce1.button("更新を保存", type="primary", key=f"save_{w['_filename']}", use_container_width=True):
                                    # updated_data に category を追加
                                    updated_data = {
                                        **w, 
                                        "word": e_word, 
                                        "category": e_category, # 追加
                                        "phonetic": clean_phonetic(e_phonetic), 
                                        "word_meaning": e_meaning, 
                                        "example_sentence": e_ex, 
                                        "example_meaning": e_ex_m
                                    }
                                    
                                    # 保存前に管理用の一時キーを削除
                                    if '_filename' in updated_data: del updated_data['_filename']
                                    
                                    pd.DataFrame([updated_data]).to_csv(
                                        os.path.join(deck_full_path, w['_filename']), 
                                        index=False, 
                                        encoding="utf-8-sig"
                                    )
                                    st.session_state[edit_key] = False
                                    st.success("保存しました")
                                    st.rerun()
                                    
                                if ce2.button("キャンセル", key=f"can_{w['_filename']}", use_container_width=True):
                                    st.session_state[edit_key] = False
                                    st.rerun()

            with tab2:
                st.warning(f"デッキ '{e_deck}' を完全に削除します。この操作は取り消せません。")
                if st.button(f"🔥 デッキ '{e_deck}' を削除する", type="primary", use_container_width=True):
                    try:
                        # 1. フォルダ内の全ファイルを削除してからフォルダ自体を削除
                        import shutil
                        shutil.rmtree(deck_full_path)
                        
                        st.success(f"デッキ '{e_deck}' を削除しました。")
                        
                        # 【重要】ここが修正ポイント！
                        # デッキがなくなったので、選択位置を一番上（0）に強制リセットする
                        st.session_state.selected_deck_index = 0
                        
                        # 念のため、セレクトボックスのキーも消去して完全に初期化する
                        if "deck_selector_management" in st.session_state:
                            del st.session_state["deck_selector_management"]
                        
                        # 画面を更新して最新のデッキリストを反映させる
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"デッキの削除に失敗しました: {e}")

    # --- 4. 学習統計モード ---
    elif mode == "学習統計":
        st.title("📈 学習統計")
        s_idx = st.selectbox("統計デッキ", range(len(deck_display)), format_func=lambda x: deck_display[x])
        if deck_raw:
            df_stats = get_stats_data(deck_raw[s_idx])
            if df_stats is not None and not df_stats.empty:
                st.line_chart(df_stats.set_index('date')['accuracy'])
                st.dataframe(df_stats.sort_values('date', ascending=False), use_container_width=True)
            else:
                st.info("まだテストデータがありません。")

if __name__ == "__main__":
    main()