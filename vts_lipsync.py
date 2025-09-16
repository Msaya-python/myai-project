# -*- coding: utf-8 -*-
# vts_lipsync.py — SoraMouthProxy優先 / 入力注入 / 専用asyncループ / 任意Form対応

import os, json, re, asyncio, threading
from typing import Optional, List, Dict
import websockets

API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"
VTS_WS_URL = os.environ.get("VTS_WS_URL", "ws://127.0.0.1:8001")
TOKEN_PATH = os.environ.get("VTS_TOKEN_PATH", "vts_token.txt")
PLUGIN_NAME = os.environ.get("VTS_PLUGIN_NAME", "SoraLipSync")
DEVELOPER  = os.environ.get("VTS_DEVELOPER",  "SoraDev")

_VOWEL_LATIN = re.compile(r"[aiueoAIUEO]")

def text_to_vowel_stream(text: str) -> List[str]:
    vlist = _VOWEL_LATIN.findall(text)
    if vlist:
        return [v.lower() for v in vlist]
    return ["a"] * max(1, min(len(text), 60))

class _VTSClient:
    def __init__(
        self,
        preferred_inputs: Optional[List[str]] = None,
        preferred_form_inputs: Optional[List[str]] = None,
    ):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._req_id = 0
        self._authed = False
        # 口の開き（必須）
        self.amp_input: Optional[str] = None
        # 口の形（任意）
        self.form_input: Optional[str] = None

        self.preferred_inputs = preferred_inputs or [
            "SoraMouthProxy", "MouthOpen", "PlusMouthOpen", "VoiceVolume"
        ]
        # 形用の入力がある場合のみ使う
        self.preferred_form_inputs = preferred_form_inputs or [
            "SoraMouthFormProxy", "MouthForm", "MouthShape"
        ]

    async def connect(self):
        self.ws = await websockets.connect(
            VTS_WS_URL, max_size=1<<20, ping_interval=15, ping_timeout=20
        )
        try:
            await self._send("APIStateRequest")
        except Exception:
            pass
        await self._authenticate()
        await self._detect_inputs()

    async def close(self):
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def _send(self, message_type: str, data: Optional[dict]=None) -> dict:
        if not self.ws:
            raise RuntimeError("WebSocket not connected")
        self._req_id += 1
        payload = {
            "apiName": API_NAME, "apiVersion": API_VERSION,
            "messageType": message_type, "requestID": str(self._req_id)
        }
        if data is not None:
            payload["data"] = data
        await self.ws.send(json.dumps(payload))
        resp = json.loads(await self.ws.recv())
        d = resp.get("data", {})
        if isinstance(d, dict) and d.get("errorID"):
            if d.get("errorID")==8 and message_type not in (
                "AuthenticationRequest","AuthenticationTokenRequest","APIStateRequest"
            ):
                if not self._authed:
                    await self._authenticate()
                    await self.ws.send(json.dumps(payload))
                    resp = json.loads(await self.ws.recv())
                    d = resp.get("data", {})
                    if isinstance(d, dict) and d.get("errorID"):
                        raise RuntimeError(f"VTS error {d.get('errorID')}: {d.get('message')}")
                    return resp
            raise RuntimeError(f"VTS error {d.get('errorID')}: {d.get('message')}")
        return resp

    async def _authenticate(self):
        token = None
        if os.path.exists(TOKEN_PATH):
            token = open(TOKEN_PATH, "r", encoding="utf-8").read().strip()
        if token:
            try:
                r = await self._send("AuthenticationRequest", {
                    "pluginName": PLUGIN_NAME, "pluginDeveloper": DEVELOPER,
                    "authenticationToken": token,
                })
                if r.get("data", {}).get("authenticated", False):
                    self._authed = True
                    return
            except Exception:
                pass
        r = await self._send("AuthenticationTokenRequest", {
            "pluginName": PLUGIN_NAME, "pluginDeveloper": DEVELOPER
        })
        token = r.get("data", {}).get("authenticationToken")
        r = await self._send("AuthenticationRequest", {
            "pluginName": PLUGIN_NAME, "pluginDeveloper": DEVELOPER,
            "authenticationToken": token,
        })
        if not r.get("data", {}).get("authenticated", False):
            raise RuntimeError("Authentication failed")
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(token)
        self._authed = True

    async def _detect_inputs(self):
        r = await self._send("InputParameterListRequest")
        d = r.get("data", {})
        names = [p.get("name") for p in (d.get("defaultParameters", []) or []) if isinstance(p.get("name"), str)]
        names += [p.get("name") for p in (d.get("customParameters",  []) or []) if isinstance(p.get("name"), str)]
        lower = { (n or "").lower(): n for n in names }

        # 必須: 開き
        for cand in self.preferred_inputs:
            n = lower.get(cand.lower())
            if n:
                self.amp_input = n
                print(f"[VTS] selected input (amp): {self.amp_input}")
                break
        if not self.amp_input:
            raise RuntimeError(f"Preferred amp inputs not found: {self.preferred_inputs}")

        # 任意: 形
        for cand in self.preferred_form_inputs:
            n = lower.get(cand.lower())
            if n:
                self.form_input = n
                print(f"[VTS] selected input (form): {self.form_input}")
                break

    async def send_values(self, values: Dict[str, float]):
        if not values:
            return
        params = []
        for pid, val in values.items():
            val = float(val)
            if pid is None:  # 未検出の入力は無視
                continue
            params.append({"id": pid, "value": val})
        if not params:
            return
        await self._send("InjectParameterDataRequest", {
            "parameterValues": params,
            "faceFound": True,
            "mode": "set",
        })

    async def send_amplitude(self, a: float):
        a = max(0.0, min(float(a), 1.0))
        await self.send_values({self.amp_input: a})

    async def send_amp_and_form(self, a: float, form: Optional[float]):
        a = max(0.0, min(float(a), 1.0))
        vals = {self.amp_input: a}
        if form is not None and self.form_input:
            # Formは -1.0〜+1.0 を想定
            f = max(-1.0, min(float(form), 1.0))
            vals[self.form_input] = f
        await self.send_values(vals)

class VTSLipsync:
    """
    同期API（スレッド安全）。開きと任意のFormを送出する。
    send_vowel(vowel, base_amp) で母音に応じたゲイン＆Formを自動適用。
    """
    def __init__(
        self,
        preferred_inputs: Optional[List[str]] = None,
        preferred_form_inputs: Optional[List[str]] = None,
    ):
        self._client = _VTSClient(
            preferred_inputs=preferred_inputs,
            preferred_form_inputs=preferred_form_inputs
        )
        # 開きゲイン（母音ごと）
        self._vowel_gain = {"a":1.00, "i":0.70, "u":0.85, "e":0.90, "o":0.95, "x":0.00}
        # 任意Form（母音→形：-1..+1）。入力が無ければ自動的に送らない。
        self._vowel_form = {"a": 0.00, "i": -0.60, "u": -0.30, "e": +0.30, "o": +0.60, "x": 0.00}

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="VTSLipsyncLoop", daemon=True
        )
        self._loop_thread.start()

    def _run(self, coro, timeout: float = 10.0):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def connect(self):
        self._run(self._client.connect())

    def close(self):
        try: self._run(self._client.send_amplitude(0.0), timeout=2.0)
        except Exception: pass
        try: self._run(self._client.close(), timeout=3.0)
        except Exception: pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=2.0)
        except Exception: pass

    def send_vowel(self, vowel: str, base_amp: float):
        v = (vowel or "x").lower()
        gain = self._vowel_gain.get(v, 1.0)
        form = self._vowel_form.get(v, 0.0)
        a = max(0.0, min(base_amp * gain, 1.0))
        try:
            self._run(self._client.send_amp_and_form(a, form), timeout=2.0)
        except Exception:
            return  # クローズ中などは無視
