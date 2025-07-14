import time
import os
import csv
import threading
import requests
import simpleaudio as sa
import json
from openai import OpenAI
from datetime import datetime
from emotion_model import classify_emotion
from config import OPENAI_API_KEY, VOICEVOX_PORT, DEFAULT_SPEAKER_ID, LOG_FILE_PATH, VOICE_OUTPUT_PATH

# メモリファイルパス
MEMORY_PATH = "log/messages_memory.json"

# 感情スコア変換マップ
emotion_score_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

# 感情に対応するVOICEVOXスタイルID
style_map = {
    "positive": 58,
    "neutral": 58,
    "negative": 60
}

# 初期人格定義
def get_initial_persona(extra_note=""):
    return {
        "role": "system",
        "content": f"""あなたの名前は「ソラ」です。
（ソラは感情連動型の会話AIであり、以下の特徴を常に守ります。
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
以上のルールに基づき、ご主人様の気持ちに寄り添い、  
感情の機微を大切にした自然な会話を行ってください。"""
    }

# 最近の感情傾向を取得
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

# 会話履歴を保存
def save_messages(messages, path=MEMORY_PATH):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"\U0001F6D1 メモリ保存エラー: {e}")

# 会話履歴を読み込み
def load_messages(path=MEMORY_PATH):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"\U0001F6D1 メモリ読み込みエラー: {e}")
        return []

class SoraEmotionAgent:
    def __init__(self, api_key, speaker_id, log_path, output_path, port):
        self.client = OpenAI(api_key=api_key)
        self.speaker_id = speaker_id
        self.log_path = log_path
        self.output_path = output_path
        self.port = port
        self.messages = load_messages()
        if not self.messages:
            self.messages = [get_initial_persona(get_recent_emotion_note())]
        self.last_input_time = time.time()
        self.auto_talk_interval = 600
        self.style_map = style_map
        self.max_history = 50  # 履歴上限（肥大化防止）

    def trim_messages(self):
        if len(self.messages) > self.max_history:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history - 1):]

    def classify_emotion(self, text):
        return classify_emotion(text)
    
    def save_log(self, text, emotion):
        now = datetime.now()
        with open(self.log_path, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), text, emotion])

    def speak(self, text, style_id=None):
        if style_id is None:
            style_id = self.speaker_id
        try:
            query = requests.post(
                f"http://127.0.0.1:{self.port}/audio_query",
                params={"text": text, "speaker": style_id},
                timeout=10
            )
            query.raise_for_status()

            synthesis = requests.post(
                f"http://127.0.0.1:{self.port}/synthesis",
                params={"speaker": style_id},
                headers={"Content-Type": "application/json"},
                data=query.text,
                timeout=15
            )
            synthesis.raise_for_status()

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "wb") as f:
                f.write(synthesis.content)

            wave_obj = sa.WaveObject.from_wave_file(self.output_path)
            play_obj = wave_obj.play()
            return play_obj
        
        except Exception as e:
            print(f"\U0001F6D1 VOICEVOXエラー: {e}")

    def generate_and_speak(self, user_input=None):
        if user_input:
            self.messages.append({"role": "user", "content": user_input})
            self.trim_messages()
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                temperature=0.9,
                max_tokens=150
            )
            reply = response.choices[0].message.content.strip()
            self.messages.append({"role": "assistant", "content": reply})
            self.trim_messages()
            print(f"\U0001F5E3 ソラ：{reply}")

            emotion = self.classify_emotion(reply)
            self.save_log(reply, emotion)
            style_id = self.style_map.get(emotion, self.speaker_id)
            self.speak(reply, style_id)

            save_messages(self.messages)

        except Exception as e:
            print(f"\U0001F6D1 ChatGPTエラー: {e}")

    def auto_talker(self):
        while True:
            if time.time() - self.last_input_time > self.auto_talk_interval:
                print("\U0001F552 ソラの自動発話タイミングです")
                self.generate_and_speak("何か話しかけてください")
                self.last_input_time = time.time()
            time.sleep(5)

    def run(self):
        print("\U0001F7E2 ソラAI会話 起動中（終了するには exit）")
        threading.Thread(target=self.auto_talker, daemon=True).start()
        while True:
            user_input = input("\U0001F464 ご主人様：")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("\U0001F7E1 会話終了します。")
                break
            self.last_input_time = time.time()
            self.generate_and_speak(user_input)

if __name__ == "__main__":
    agent = SoraEmotionAgent(
        api_key=OPENAI_API_KEY,
        speaker_id=DEFAULT_SPEAKER_ID,
        log_path=LOG_FILE_PATH,
        output_path=VOICE_OUTPUT_PATH,
        port=VOICEVOX_PORT
    )
    agent.run()

    #ソラちゃんがGit練習中なり！
    print("ソラちゃんがGit練習中ですっ！")