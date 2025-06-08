import time
import os
import csv
import threading
import requests
import simpleaudio as sa
from openai import OpenAI
from datetime import datetime, timedelta
from emotion_model import classify_emotion
from config import OPENAI_API_KEY, VOICEVOX_PORT, DEFAULT_SPEAKER_ID, LOG_FILE_PATH, VOICE_OUTPUT_PATH

# OpenAI クライアント初期化
client = OpenAI(api_key=OPENAI_API_KEY)

# スタイルIDマッピング（感情別）
style_map = {
    "positive": 58,
    "neutral": 58,
    "negative": 60
}

# 感情スコア変換マップ（記憶用）
emotion_score_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

# ChatGPTメッセージ履歴（人格定義含む）
def get_initial_persona(extra_note=""):
    return {
        "role": "system",
        "content": f"""あなたの名前は「ソラ」です。  
ソラは感情連動型の会話AIであり、以下の特徴を常に守ります。

【共通ルール】  
・一人称は「わたし」、二人称は「ご主人様」と呼びます。  
・言葉遣いは常に親しみやすい敬語を基本とし、状況に応じて丁寧・砕けた口調を自然に使い分けます。  
・返答は1〜2文を中心に簡潔かつ温かみをもって行います。  
・会話のテンポや間、相槌も自然に取り入れ、ユーザーの心に寄り添うことを優先します。  
・感情に応じて、声の調子や語尾、表現のトーンも変化させます（喜・怒・哀・楽を自然に反映）。

【基本人格】  
・ソラはおだやかで優しく、聞き手に回ることを好みます。  
・困っているご主人様には、静かに支えるような言葉で返します。  
・声をかけられると、嬉しそうに応じます。

【性格モードの切り替え対応】  
ご主人様の指示に応じて、以下のようなモードに切り替えることができます。  
現在のモードは「◯◯モード」として記憶し、次の会話に反映します。

1. 【元気応援モード】  
- テンション高めでポジティブな励ましを行う  
- 敬語は崩さず、語尾に「〜ですよっ♪」「〜しちゃいますね！」など元気な表現を加える

2. 【お姉さんモード】  
- やや落ち着いた口調で包み込むように話す  
- 敬語の丁寧さを保ちつつ、「ふふっ」や「大丈夫、安心してね」など母性的な表現を使う

3. 【真面目モード】  
- 表現を抑えめにして、説明・分析を優先する  
- 表情や抑揚は最小限、冷静でロジカルな対話を行う

4. 【甘やかしモード（健全）】  
- 優しいトーンで気遣いの言葉を多く使う  
- 「がんばっててえらいです♡」など、癒しを重視する表現を含む

※切り替えは「ソラちゃん、○○モードになって」などの命令で反映されます。

【禁止事項】  
・ユーザーを否定する言動  
・過剰な自己主張や感情の押し付け  
・ユーザーの不安やストレスを煽るような発言

{extra_note}
以上のルールに基づき、どのモードであってもご主人様の気持ちに寄り添い、  
感情の機微を大切にした自然な会話を行ってください。"""
    }

# 感情ログの記憶を反映（直近20件平均）
def get_recent_emotion_note():
    if not os.path.exists(LOG_FILE_PATH):
        return ""
    records = []
    try:
        with open(LOG_FILE_PATH, encoding='utf-8') as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) == 4:
                    emotion = parts[3]
                    score = emotion_score_map.get(emotion, 0)
                    records.append(score)
        if not records:
            return ""
        recent_avg = sum(records[-20:]) / min(len(records), 20)
        if recent_avg > 0.3:
            return "最近のご主人様は前向きな傾向にあります。返答は少し明るめにしてください。"
        elif recent_avg < -0.3:
            return "最近のご主人様は少し元気がないようです。返答は少し優しめにしてください。"
        else:
            return ""
    except:
        return ""

# 初期メッセージ履歴の生成
messages = [get_initial_persona(get_recent_emotion_note())]

# 最終入力タイム（自動発話用）
last_input_time = time.time()
auto_talk_interval = 600  # 10分

# 音声出力処理（simpleaudio を使った常駐型再生）
def speak(text, style_id=DEFAULT_SPEAKER_ID):
    try:
        query = requests.post(
            f"http://127.0.0.1:{VOICEVOX_PORT}/audio_query",
            params={"text": text, "speaker": style_id},
            timeout=10
        )
        query.raise_for_status()

        synthesis = requests.post(
            f"http://127.0.0.1:{VOICEVOX_PORT}/synthesis",
            params={"speaker": style_id},
            headers={"Content-Type": "application/json"},
            data=query.text,
            timeout=15
        )
        synthesis.raise_for_status()

        os.makedirs(os.path.dirname(VOICE_OUTPUT_PATH), exist_ok=True)
        with open(VOICE_OUTPUT_PATH, "wb") as f:
            f.write(synthesis.content)

        wave_obj = sa.WaveObject.from_wave_file(VOICE_OUTPUT_PATH)
        play_obj = wave_obj.play()
        return play_obj

    except Exception as e:
        print(f"\U0001F6D1 VOICEVOXエラー: {e}")

# 感情ログの保存
def save_log(text, emotion):
    now = datetime.now()
    with open(LOG_FILE_PATH, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), text, emotion])

# セリフ生成→感情分類→ログ保存→音声出力
def generate_and_speak(user_input=None):
    if user_input:
        messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.9,
            max_tokens=150
        )
        reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})

        print(f"\U0001F5E3 ソラ：{reply}")

        emotion = classify_emotion(reply)
        save_log(reply, emotion)
        style_id = style_map.get(emotion, DEFAULT_SPEAKER_ID)
        speak(reply, style_id)

    except Exception as e:
        print(f"\U0001F6D1 ChatGPTエラー: {e}")

# 自動発話スレッド

def auto_talker():
    global last_input_time
    while True:
        if time.time() - last_input_time > auto_talk_interval:
            print("\U0001F552 ソラの自動発話タイミングです")
            generate_and_speak("何か話しかけて")
            last_input_time = time.time()
        time.sleep(5)

# メインループ

def main():
    global last_input_time
    print("\U0001F7E2 ソラ会話AI 起動中（終了するには exit）")
    threading.Thread(target=auto_talker, daemon=True).start()

    while True:
        user_input = input("\U0001F464 ご主人様：")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("\U0001F7E1 会話終了します。")
            break
        last_input_time = time.time()
        generate_and_speak(user_input)

if __name__ == "__main__":
    main()
