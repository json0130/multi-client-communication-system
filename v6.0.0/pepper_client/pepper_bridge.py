"""
pepper_shit/pepper_bridge.py
============================
Bridge adapter: v6.0.0 WebSocket protocol  ↔  Pepper NAOqi TCP protocol.

From the central server's perspective: a normal v6.0.0 robot client.
From Pepper's perspective: a normal chatbot_client.

Run from this directory:
    python pepper_bridge.py

Requires client_config.json in the same directory with a "pepper_config" block.
"""

import json
import logging
import queue as _queue
import re
import sys
import os
import threading
import time

# ── Add v6.0.0/client to path so we can import BasicClient ───────────────────
# pepper_shit/ lives inside v6.0.0/, so client/ is one level up
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'client'))

from client import BasicClient          # noqa: E402
from utils.connection import Connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PepperBridge(BasicClient):
    """
    Bridges the v6.0.0 WebSocket protocol and Pepper's NAOqi TCP protocol.

    Inherits from BasicClient to get:
      - WebSocket server (central server connects to us)
      - Persona update handling
      - send_to_server / send_ack helpers

    Overrides:
      - demo_step handler  → TTS via NAOqi TCP socket
      - chat_response handler → TTS via NAOqi TCP socket
      - STT background loop  → record from Pepper → Whisper → server
    """

    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)

        pepper_cfg = self.config.get("pepper_config", {})
        self._pepper_ip   = pepper_cfg.get("pepper_ip",   "172.24.192.51")
        self._pepper_port = pepper_cfg.get("pepper_port",  8001)

        self._wps             = pepper_cfg.get("words_per_second",   2.3)
        self._record_duration = pepper_cfg.get("record_duration",    5.0)
        self._pepper_rate     = pepper_cfg.get("pepper_sample_rate", 48000)
        self._pepper_channels = pepper_cfg.get("pepper_channels",    1)
        self._vad_threshold   = pepper_cfg.get("vad_threshold",      300)   # RMS energy floor (0–32768)

        # TCP connection to Pepper — starts as None, filled by connect loop
        self._pepper      = None
        self._pepper_lock = threading.Lock()

        # TTS requests from the server are queued and spoken between recording cycles
        self._tts_queue = _queue.Queue()

        # Override the handlers that BasicClient registered
        self.server_connection.register_handler("demo_step",       self._pepper_demo_step)
        self.server_connection.register_handler("chat_response",   self._pepper_chat_response)
        self.server_connection.register_handler("speech_response", self._pepper_speech_response)

        self._has_speech = "speech" in self.config.get("modules", [])

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Override BasicClient.start() to enforce connection order:
          1. Connect to Pepper TCP server (blocking retry — must succeed first)
          2. Register with central server + open WebSocket listener

        The central server cannot connect to this bridge at all until
        Pepper is reachable, so the dashboard "Connect" button only works
        once Pepper is up.
        """
        self._connect_to_pepper()       # blocks until Pepper TCP is ready
        result = super().start()        # sets self.running=True, opens WS server

        # Start STT loop here — self.running is now True so the loop stays alive
        if self._has_speech:
            threading.Thread(
                target=self._stt_loop, daemon=True, name="pepper-stt"
            ).start()
            logger.info("[Bridge] STT loop started — audio forwarded to server for processing")

        return result

    def _connect_to_pepper(self):
        """Blocking retry loop — exits only when Pepper TCP connection succeeds."""
        while True:
            logger.info(
                f"[Bridge] Connecting to Pepper at "
                f"{self._pepper_ip}:{self._pepper_port} ..."
            )
            try:
                conn = Connection(
                    ip=self._pepper_ip, port=self._pepper_port,
                    type='client', max_retry=1,
                )
                if conn.connected:
                    with self._pepper_lock:
                        self._pepper = conn
                    logger.info(
                        "[Bridge] Pepper connected — starting WebSocket server..."
                    )
                    return
            except Exception:
                pass

            logger.info("[Bridge] Pepper not reachable — retrying in 5 s...")
            time.sleep(5)

    # ── Emotion hook ──────────────────────────────────────────────────────────

    def on_emotion_detected(self, emotion_tag: str):
        """
        Called for every [TAG] found in server responses.
        Extend this to send Pepper gesture/animation commands.
        """
        logger.info(f"[Emotion] {emotion_tag.strip().upper()}")
        # TODO: map tag → Pepper animation, e.g.:
        #   self._send_pepper({"command": "animate", "content": "Hey_1"})

    # ── Demo step handler ─────────────────────────────────────────────────────

    def _pepper_demo_step(self, data: dict):
        """
        Replaces BasicClient._on_demo_step.
        Sends generated text to Pepper TTS, waits estimated duration, ACKs.
        """
        step_id  = data.get("step_id", "")
        text     = data.get("text", "")
        need_ack = data.get("require_ack", True)

        if not text:
            if need_ack:
                self.send_ack(step_id)
            return

        logger.info(f"[Demo] Step '{step_id}': {text[:80]}")

        # Dispatch emotion tag (for gestures / logging)
        match = re.search(r"\[(.*?)\]", text)
        if match:
            self.on_emotion_detected(match.group(1))

        clean = re.sub(r"\[.*?\]", "", text).strip()
        if not clean:
            if need_ack:
                self.send_ack(step_id)
            return

        # Speak via Pepper and block for estimated duration
        self._speak(clean)

        if need_ack:
            self.send_ack(step_id)

    # ── Chat response handler ─────────────────────────────────────────────────

    def _pepper_chat_response(self, data: dict):
        """
        Replaces BasicClient._default_chat_handler.
        Queues LLM response for TTS — the STT loop speaks it between recording cycles
        so TTS never blocks on the Pepper TCP lock.
        """
        text = data.get("response", "")
        if not text:
            return

        match = re.search(r"\[(.*?)\]", text)
        if match:
            self.on_emotion_detected(match.group(1))

        clean = re.sub(r"\[.*?\]", "", text).strip()
        if clean:
            logger.info(f"[TTS] Chat response queued for Pepper: {clean[:70]}")
            self._tts_queue.put(clean)

    def _pepper_speech_response(self, data: dict):
        """
        Replaces BasicClient._default_speech_handler.
        Called when the server returns a speech_response event (transcription + LLM reply).
        Logs the transcription and queues the LLM reply for Pepper TTS.
        """
        transcription = data.get("transcription", "")
        if transcription:
            logger.info(f"[STT] Transcribed: '{transcription[:100]}'")

        text = data.get("response", "")
        if not text:
            return

        match = re.search(r"\[(.*?)\]", text)
        if match:
            self.on_emotion_detected(match.group(1))

        clean = re.sub(r"\[.*?\]", "", text).strip()
        if clean:
            logger.info(f"[TTS] Speech response queued for Pepper: {clean[:70]}")
            self._tts_queue.put(clean)

    # ── TTS helpers ───────────────────────────────────────────────────────────

    def _speak(self, text: str):
        """
        Send TTS to Pepper and block for estimated speech duration.
        Sets is_speaking so the STT loop pauses during playback.
        """
        words   = len(text.split())
        est_sec = max(1.0, words / self._wps)

        self.is_speaking.set()
        try:
            ok = self._send_pepper({"command": "tts", "content": text})
            if ok:
                logger.info(f"[TTS] Speaking (~{est_sec:.1f}s): {text[:70]}")
                time.sleep(est_sec)
            else:
                logger.warning("[TTS] Send failed — sleeping anyway for timing")
                time.sleep(est_sec)
        finally:
            self.is_speaking.clear()

    def _speak_async(self, text: str):
        """Non-blocking version of _speak — runs in its own daemon thread."""
        threading.Thread(
            target=self._speak, args=(text,), daemon=True, name="pepper-tts"
        ).start()

    # ── STT loop ──────────────────────────────────────────────────────────────

    def _stt_loop(self):
        """
        Record audio from Pepper and forward the raw bytes to the central server.
        The server handles STT processing — same pipeline as all other robots.

        TTS responses are spoken HERE between recording cycles so they never
        compete with the Pepper TCP lock held during recording.
        """
        logger.info("[STT] Listening loop started")
        _last_no_server_log = 0.0
        while self.running:
            try:
                # Pause if server WebSocket is not connected — no point recording
                if not self.server_connection.is_connected():
                    now = time.time()
                    if now - _last_no_server_log >= 10:
                        logger.info("[STT] Waiting for server WebSocket connection...")
                        _last_no_server_log = now
                    time.sleep(1)
                    continue

                # Drain any pending TTS responses before starting a new recording.
                # Lock is free here — safe to send to Pepper without racing the STT receive.
                while not self._tts_queue.empty():
                    try:
                        text = self._tts_queue.get_nowait()
                        logger.info(f"[TTS] Speaking queued response: {text[:70]}")
                        self._speak(text)
                    except _queue.Empty:
                        break

                # Don't start a new recording while Pepper is still finishing speech
                if self.is_speaking.is_set():
                    self.is_speaking.wait(timeout=0.5)
                    continue

                logger.info("[STT] Conditions met — starting record cycle")
                audio_bytes = self._record()
                if audio_bytes is None:
                    logger.warning("[STT] _record() returned None — pausing 2 s")
                    time.sleep(2)
                    continue

                # Energy VAD — skip silent audio before sending to server
                if not self._has_voice(audio_bytes):
                    logger.info("[STT] No voice energy detected — skipping cycle")
                    continue

                # Mix channels to mono then wrap in WAV for server
                mono_bytes = self._mix_to_mono(audio_bytes)
                wav_bytes = self._pcm_to_wav(mono_bytes, channels=1)
                logger.info(
                    f"[STT] Forwarding {len(wav_bytes)} bytes WAV "
                    f"(mono, {self._pepper_rate}Hz) to server"
                )
                self.send_to_server("speech", wav_bytes)

            except Exception as e:
                logger.error(f"[STT] Loop error: {e}", exc_info=True)
                time.sleep(1)

    def _record(self):
        """
        Ask Pepper to record for self._record_duration seconds.
        Returns audio bytes, or None on failure.
        """
        logger.info("[STT] ── record cycle start ──")
        with self._pepper_lock:
            logger.info("[STT] Sending record-start to Pepper...")
            ok = self._send_pepper_raw({"command": "record", "content": None})
            if not ok:
                logger.warning("[STT] record-start send failed — skipping cycle")
                return None
            logger.info("[STT] Waiting for start-ACK from Pepper (timeout 5 s)...")
            self._pepper.receive()
            try:
                ack = self._pepper.queue.get(timeout=5)
                logger.info(f"[STT] Got start-ACK: {ack[:60]!r}")
            except _queue.Empty:
                logger.warning("[STT] No ACK from Pepper for record-start — timeout")
                return None

        # logger.info(f"[STT] Recording for {self._record_duration} s...")
        # time.sleep(self._record_duration)

        with self._pepper_lock:
            logger.info("[STT] Sending record-stop to Pepper...")
            ok = self._send_pepper_raw({"command": "record", "content": "stop"})
            if not ok:
                logger.warning("[STT] record-stop send failed — skipping cycle")
                return None
            logger.info("[STT] Waiting for audio from Pepper (timeout 15 s)...")
            self._pepper.receive()
            try:
                audio = self._pepper.queue.get(timeout=15)
                logger.info(f"[STT] Got {len(audio)} bytes audio from Pepper")
            except _queue.Empty:
                logger.warning("[STT] No audio received from Pepper — timeout")
                return None

        return None if audio == b'-1' else audio

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _has_voice(self, pcm_bytes: bytes) -> bool:
        """
        Simple energy-based VAD.
        Returns True if the audio RMS exceeds self._vad_threshold (0–32768).
        Filters out silence before sending to server, avoiding wasteful Whisper calls.
        """
        import numpy as np
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        logger.info(f"[VAD] RMS energy: {rms:.1f} (threshold: {self._vad_threshold})")
        return rms > self._vad_threshold

    def _mix_to_mono(self, pcm_bytes: bytes) -> bytes:
        """
        Average Pepper's 4-channel mic array down to mono int16.
        Whisper expects mono audio — 4-channel interleaved PCM causes hallucinations.
        """
        import numpy as np
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        # reshape to (n_frames, n_channels) then average across channels
        samples = samples.reshape(-1, self._pepper_channels)
        mono = samples.mean(axis=1).astype(np.int16)
        return mono.tobytes()

    def _pcm_to_wav(self, pcm_bytes: bytes, channels: int = None) -> bytes:
        """
        Wrap PCM int16 bytes in a WAV header.
        channels defaults to self._pepper_channels; pass 1 for mono-mixed audio.
        """
        import wave, io
        n_channels = channels if channels is not None else self._pepper_channels
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(2)                  # int16 = 2 bytes
            wav.setframerate(self._pepper_rate)
            wav.writeframes(pcm_bytes)
        return buf.getvalue()

    # ── Pepper socket helpers ─────────────────────────────────────────────────

    def _send_pepper(self, payload: dict) -> bool:
        """Thread-safe send to Pepper TCP socket."""
        with self._pepper_lock:
            return self._send_pepper_raw(payload)

    def _send_pepper_raw(self, payload: dict) -> bool:
        """Send to Pepper socket — caller must hold _pepper_lock."""
        if self._pepper is None or not self._pepper.connected:
            logger.warning("[Bridge] Pepper not connected — skipping send")
            return False
        try:
            self._pepper.send(json.dumps(payload).encode())
            return True
        except Exception as e:
            logger.error(f"[Bridge] Pepper send error: {e}")
            return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    try:
        bridge = PepperBridge("client_config.json")

        print("\n" + "=" * 60)
        print(f"  Pepper Bridge")
        print(f"  Robot   : {bridge.config.get('robot_name', 'Pepper')}")
        print(f"  ID      : {bridge.config.get('client_id')}")
        print(f"  Server  : {bridge.config.get('server_url')}")
        print(f"  WS port : {bridge.config.get('ws_port')}")
        print(f"  Pepper  : {bridge.config.get('pepper_config', {}).get('pepper_ip')}:"
              f"{bridge.config.get('pepper_config', {}).get('pepper_port')}")
        print(f"  Modules : {', '.join(bridge.config.get('modules', []))}")
        print("=" * 60 + "\n")

        bridge.run()
        return 0

    except FileNotFoundError:
        print("Error: client_config.json not found in pepper_shit/")
        return 1
    except KeyboardInterrupt:
        print("\nStopped")
        return 0
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
