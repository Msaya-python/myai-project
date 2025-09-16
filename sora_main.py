# -*- coding: utf-8 -*-
import time
import os
import csv
import threading
import requests
import simpleaudio as sa
import soundfile as sf
import json
import numpy as np
import asyncio
import websockets
from openai import OpenAI
from datetime import datetime

from emotion_model import classify_emotion
from config import OPENAI_API_KEY, VOICEVOX_PORT, DEFAULT_SPEAKER_ID, LOG_FILE_PATH, VOICE_OUTPUT_PATH
from vts_lipsync import VTSLipsync  # SoraMouthProxy優先＋Form任意対応

MEMORY_PATH = "log/messages_memory.json"

# ===== 口パク調整パラメータ =====
OFFSET_MS   = 110   # 再生と口のズレ補正（90〜140ms付近で調整）
ATTACK      = 0.45  # 開く速さ（大きいほど素早く開く）
RELEASE     = 0.25  # 閉じる速さ（小さすぎると残留）
NOISE_GATE  = 0.02  # これ未満は0扱い
TARGET_FPS  = 60    # 送信フレームレート

emotion_score_map = {"positive": 1, "neutral": 0, "negative": -1}
# 女声スタイル（ご主人様指定）
style_map = {"positive": 58, "neutral": 58, "negative": 60}

# ===== VTSモーション（Hotkey送信用の軽量クライアント） =====
VTS_WS_URL = os.environ.get("VTS_WS_URL", "ws://127.0.0.1:8001")
VTS_TOKEN_PATH = os.environ.get("VTS_TOKEN_PATH", "vts_token.txt")
PLUGIN_NAME = os.environ.get("VTS_PLUGIN_NAME", "SoraMotion")
DEVELOPER   = os.environ.get("VTS_DEVELOPER",  "SoraDev")

# Hotkey名（VTS側で同名のHotkeyを作成してね）
HOTKEY_MAP = {
    "joy": "SoraJoy",
    "nod": "SoraNod",
    "think": "SoraThink",
    "surprise": "SoraSurprise",
    "sad": "SoraSad",
}

class VTSMotionClient:
    def __init__(self):
        self.ws = None
        self._req_id = 0
        self._authed = False
        self._loop = asyncio.new_event_loop()
        self._th = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._th.start()

    def _run(self, coro, timeout: float = 10.0):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    async def _send(self, message_type: str, data=None):
        self._req_id += 1
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "messageType": message_type,
            "requestID": str(self._req_id)
        }
        if data is not None:
            payload["data"] = data
        await self.ws.send(json.dumps(payload))
        resp = json.loads(await self.ws.recv())
        return resp

    async def _authenticate(self):
        token = None
        if os.path.exists(VTS_TOKEN_PATH):
            token = open(VTS_TOKEN_PATH, "r", encoding="utf-8").read().strip()
        if token:
            r = await self._send("AuthenticationRequest", {
                "pluginName": PLUGIN_NAME, "pluginDeveloper": DEVELOPER,
                "authenticationToken": token
            })
            if r.get("data", {}).get("authenticated", False):
                self._authed = True
                return
        # 初回発行
        r = await self._send("AuthenticationTokenRequest", {
            "pluginName": PLUGIN_NAME, "pluginDeveloper": DEVELOPER
        })
        token = r.get("data", {}).get("authenticationToken")
        r = await self._send("AuthenticationRequest", {
            "pluginName": PLUGIN_NAME, "pluginDeveloper": DEVELOPER,
            "authenticationToken": token
        })
        if not r.get("data", {}).get("authenticated", False):
            raise RuntimeError("VTS auth failed")
        with open(VTS_TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(token)
        self._authed = True

    def connect(self):
        async def _c():
            self.ws = await websockets.connect(VTS_WS_URL, max_size=1<<20, ping_interval=15, ping_timeout=20)
            try:
                await self._send("APIStateRequest")
            except Exception:
                pass
            await self._authenticate()
        self._run(_c())

    def trigger_hotkey(self, hotkey_name: str):
        async def _t():
            await self._send("HotkeyTriggerRequest", {"hotkeyID": hotkey_name})
        try:
            self._run(_t(), timeout=2.0)
        except Exception:
            pass

    def close(self):
        async def _x():
            if self.ws:
                await self.ws.close()
        try: self._run(_x(), timeout=3.0)
        except Exception: pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._th.join(timeout=2.0)
        except Exception: pass

# ===== Persona / 会話メモリ =====
def get_initial_persona(extra_note=""):
    return {
        "role": "system",
        "content": f"""あなたの名前は「ソラ」です。あなたは女性のメイドAIアシスタントです。
（以下略：既存の長文をそのまま使用。口調は丁寧だが感情は誇張しない）
{extra_note}
以上のルールに基づき、ご主人様の意図を汲み、簡潔に丁寧に応答してください。"""
    }

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
            return "最近の傾向は前向きです。返答は少し明るめに。"
        elif recent_avg < -0.3:
            return "最近の傾向は落ち込み気味です。返答は少し優しめに。"
        else:
            return ""
    except:
        return ""

def save_messages(messages, path=MEMORY_PATH):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"🛑 メモリ保存エラー: {e}")

def load_messages(path=MEMORY_PATH):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"🛑 メモリ読み込みエラー: {e}")
        return []

# ===== VOICEVOX TTS（クエリJSONも返す） =====
def voicevox_tts(port: int, text: str, style_id: int, out_path: str):
    query = requests.post(
        f"http://127.0.0.1:{port}/audio_query",
        params={"text": text, "speaker": style_id},
        timeout=10
    )
    query.raise_for_status()
    synthesis = requests.post(
        f"http://127.0.0.1:{port}/synthesis",
        params={"speaker": style_id},
        headers={"Content-Type": "application/json"},
        data=query.text,
        timeout=15
    )
    synthesis.raise_for_status()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(synthesis.content)
    return out_path, query.text

# ===== 母音タイムラインの構築（audio_query） =====
def build_vowel_timeline(aq_text: str):
    """
    返り値: [(start_sec, end_sec, tag)]  tag: 'a'|'i'|'u'|'e'|'o'|'cl'|'pau'
    """
    data = json.loads(aq_text)
    t = 0.0
    segs = []
    for phrase in data.get("accent_phrases", []):
        for mora in phrase.get("moras", []):
            cl = float(mora.get("consonant_length") or 0.0)
            if cl > 0:
                segs.append((t, t+cl, "cl")); t += cl
            vl = float(mora.get("vowel_length") or 0.0)
            v  = str(mora.get("vowel") or "").lower()
            tag = v if v in ("a","i","u","e","o") else "cl"  # 'N'などは閉口扱い
            segs.append((t, t+vl, tag)); t += vl
        pm = phrase.get("pause_mora")
        if pm:
            pl = float(pm.get("vowel_length") or 0.0)
            if pl > 0:
                segs.append((t, t+pl, "pau")); t += pl
    return segs

# ===== テキスト→モーションキュー =====
def build_motion_cues(text: str, emotion: str, total_duration: float):
    """
    戻り値: [(秒, hotkey名)]  — 時刻ソート済みで返す
    """
    import re
    cues = []
    # 句読点に合わせて小アクション
    for m in re.finditer(r"[。．、,！？!?]", text):
        ch = m.group()
        t = total_duration * (m.start() / max(1, len(text)))
        if ch in "！!":
            cues.append((max(0, t-0.05), HOTKEY_MAP["joy"]))
        elif ch in "？?":
            cues.append((t, HOTKEY_MAP["think"]))
        elif ch in "、,":
            cues.append((t, HOTKEY_MAP["nod"]))
        else:
            cues.append((t, HOTKEY_MAP["nod"]))
    # 冒頭と締め
    start_hotkey = {"positive": HOTKEY_MAP["joy"], "negative": HOTKEY_MAP["sad"]}.get(emotion, HOTKEY_MAP["think"])
    cues.append((0.0, start_hotkey))
    cues.append((max(0.0, total_duration - 0.15), HOTKEY_MAP["nod"]))

    # よくある感情ワード
    keymap = {
        r"(ありがとう|助かる|嬉し|よかった)": HOTKEY_MAP["joy"],
        r"(ごめん|申し訳|すまん|すみません)": HOTKEY_MAP["sad"],
        r"(了解|任せて|OK|お任せ)": HOTKEY_MAP["nod"],
        r"(えっ|え！？|まじ|本当|びっくり)": HOTKEY_MAP["surprise"],
    }
    for pat, hk in keymap.items():
        if re.search(pat, text, flags=re.IGNORECASE):
            cues.append((min(total_duration*0.3, 0.6), hk))

    # 時刻ソート＋間引き（0.15秒以内の重複を抑制）
    cues.sort(key=lambda x: x[0])
    compact = []
    last = -1.0
    for t, hk in cues:
        if last < 0 or (t - last) > 0.15:
            compact.append((max(0.0, t), hk))
            last = t
    return compact

# ===== RMS + 母音タイムラインで送信（再生同期・確実クローズ） =====
def _run_vts_lipsync_thread(vts: VTSLipsync, wav_path: str, aq_json: str,
                            start_event: threading.Event, stop_event: threading.Event):
    import traceback
    try:
        vts.connect()
        vts.send_vowel("x", 0.0)  # 初期閉口

        timeline = build_vowel_timeline(aq_json)
        seg_idx = 0

        # 再生開始を待ってオフセット補正
        start_event.wait()
        time.sleep(OFFSET_MS / 1000.0)

        with sf.SoundFile(wav_path, mode="r") as f:
            sr = int(f.samplerate or 24000)
            hop = max(1, sr // TARGET_FPS)

            ref = 0.02
            smooth = 0.0
            frames_read = 0

            t0 = time.perf_counter()
            n = 0
            while not stop_event.is_set():
                target = t0 + n / TARGET_FPS
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)

                data = f.read(hop, dtype="float32", always_2d=True)
                if data.size == 0:
                    break
                frames_read += data.shape[0]
                t_sec = frames_read / sr

                mono = data.mean(axis=1)
                peak = float(np.max(np.abs(mono))) if mono.size else 0.0

                ref = max(ref * 0.995, peak * 0.7, 0.02)
                x = min(peak / ref, 1.5)
                x = x ** 0.5
                if x < NOISE_GATE:
                    x = 0.0

                while seg_idx+1 < len(timeline) and t_sec >= timeline[seg_idx][1]:
                    seg_idx += 1
                tag = timeline[seg_idx][2] if timeline else "a"

                if tag in {"cl","pau"}:
                    x = 0.0

                alpha = ATTACK if x > smooth else RELEASE
                smooth = (1 - alpha) * smooth + alpha * x
                amp = max(0.0, min(smooth, 1.0))

                vts.send_vowel(tag if tag in {"a","i","u","e","o"} else "x", amp)
                n += 1

        vts.send_vowel("x", 0.0)

    except Exception as e:
        print("🛑 VTSリップシンクスレッドエラー:", e)
        try:
            vts.send_vowel("x", 0.0)
        except Exception:
            pass

# ===== 会話エージェント =====
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
        self.max_history = 50

    def trim_messages(self):
        if len(self.messages) > self.max_history:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history - 1):]

    def classify_emotion(self, text):
        return classify_emotion(text)

    def save_log(self, text, emotion):
        now = datetime.now()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, mode='a', encoding='utf-8', newline='') as f:
            csv.writer(f, quoting=csv.QUOTE_ALL).writerow(
                [now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), text, emotion]
            )

    # --- ChatGPT API応答生成 ---
    def _chat(self, messages):
        try:
            # openai>=1.x
            return self.client.chat_completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.9,
                max_tokens=150
            )
        except AttributeError:
            # 旧SDK
            return self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.9,
                max_tokens=150
            )

    # --- TTS & 再生 & VTS口パク + モーション ---
    def speak(self, text, style_id=None):
        if style_id is None:
            style_id = self.speaker_id
        try:
            wav_path, aq_json = voicevox_tts(self.port, text, style_id, self.output_path)

            # 口パクスレッド
            vts_lip = VTSLipsync(
                preferred_inputs=["SoraMouthProxy", "MouthOpen", "PlusMouthOpen", "VoiceVolume"],
                preferred_form_inputs=["SoraMouthFormProxy", "MouthForm", "MouthShape"],
            )
            start = threading.Event()
            stop  = threading.Event()
            th_lip = threading.Thread(
                target=_run_vts_lipsync_thread,
                args=(vts_lip, wav_path, aq_json, start, stop),
                daemon=True
            )
            th_lip.start()

            # モーションスレッド
            # 総時間推定
            tl = build_vowel_timeline(aq_json)
            total_dur = (tl[-1][1] if tl else 0.0)
            cues = build_motion_cues(text, self.classify_emotion(text), total_dur)

            def _motion_thread(start_evt, stop_evt, cues_list):
                vm = VTSMotionClient()
                try:
                    vm.connect()
                    start_evt.wait()
                    t0 = time.perf_counter()
                    i = 0
                    while not stop_evt.is_set() and i < len(cues_list):
                        now = time.perf_counter() - t0
                        t_target, hotkey = cues_list[i]
                        if now + 0.01 >= t_target:
                            vm.trigger_hotkey(hotkey)
                            i += 1
                        else:
                            time.sleep(0.01)
                finally:
                    vm.close()

            th_motion = threading.Thread(
                target=_motion_thread,
                args=(start, stop, cues),
                daemon=True
            )
            th_motion.start()

            # 再生開始
            wave_obj = sa.WaveObject.from_wave_file(wav_path)
            play = wave_obj.play()
            start.set()
            play.wait_done()

            # 終了処理
            stop.set()
            th_lip.join()
            th_motion.join()
            vts_lip.close()

        except Exception as e:
            print(f"🛑 VOICEVOX/VTSエラー: {e}")

    # --- ユーザー入力→応答→発話 ---
    def generate_and_speak(self, user_input=None):
        if user_input:
            self.messages.append({"role": "user", "content": user_input})
            self.trim_messages()
        resp = self._chat(self.messages)
        try:
            reply = resp.choices[0].message.content.strip()
        except Exception:
            reply = resp.choices[0].message["content"].strip()

        self.messages.append({"role": "assistant", "content": reply})
        self.trim_messages()
        print(f"🗣 ソラ：{reply}")

        emotion = self.classify_emotion(reply)
        self.save_log(reply, emotion)
        style_id = style_map.get(emotion, self.speaker_id)

        self.speak(reply, style_id=style_id)
        save_messages(self.messages)

    def auto_talker(self):
        while True:
            if time.time() - self.last_input_time > self.auto_talk_interval:
                print("🕐 自動発話タイミング")
                self.generate_and_speak("何か話しかけてください")
                self.last_input_time = time.time()
            time.sleep(5)

    def run(self):
        print("🟢 ソラAI会話 起動中（終了するには exit）")
        threading.Thread(target=self.auto_talker, daemon=True).start()
        while True:
            user_input = input("👤 ご主人様：")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("🟡 会話終了します。")
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