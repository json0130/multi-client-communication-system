# gazebo_client/sim_tts_output.py
"""
SimTTSOutputModule — a "speech" output module for simulated robots.

Implements the same speak_with_callback / interrupt / clear_non_callback_items
contract as OutputModules/edge_tts_output.py (see client/client.py's
_on_demo_step / _on_chat_sentence / _on_tts_stop), but there is no speaker to
play audio on: "speaking" is a console log line held for the estimated
duration of the sentence, exactly like PepperBridge._speak() estimates
duration from word count instead of measuring real playback.
"""

import re
import threading
import queue
import logging
from typing import Any, Dict, Optional

from client import OutputModule

logger = logging.getLogger(__name__)


class SimTTSOutputModule(OutputModule):
    """Console-only stand-in for TTS, timed by words_per_second."""

    def __init__(self, name: str = "sim_tts_output", config: Dict = None):
        super().__init__(name, config)
        self._wps = self.config.get("words_per_second", 2.3)
        self._max_length = self.config.get("max_length", 500)

        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.stop_event = threading.Event()
        self._interrupt_event = threading.Event()

        # Sentence-level progress for the utterance "playing" now, so
        # interrupt() can report exactly what was never said — same
        # reasoning as edge_tts_output.py's identical fields.
        self._progress_lock = threading.Lock()
        self._current_sentences: list = []
        self._current_idx = -1
        self._current_has_callback = False

    # ── BaseModule interface ───────────────────────────────────────────────

    def initialize(self) -> bool:
        return True

    def start(self) -> bool:
        if not self.enabled:
            self.enabled = True
            self.stop_event.clear()
            self.tts_thread = threading.Thread(
                target=self._tts_worker, daemon=True, name=f"{self.name}-worker"
            )
            self.tts_thread.start()
            return True
        return False

    def stop(self):
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            self.tts_queue.put(None)
            if self.tts_thread:
                self.tts_thread.join(timeout=2)

    # ── OutputModule interface ─────────────────────────────────────────────

    def process_output(self, data: Any) -> bool:
        """Queue a plain chat_sentence for 'speech' — no callback to fire."""
        if not self.enabled:
            return False
        text = self._prepare_text(data.get("text", "") if isinstance(data, dict) else str(data))
        if not text:
            return False
        self.tts_queue.put((text, None))
        return True

    def speak_with_callback(self, text: str, callback=None) -> bool:
        """
        Queue text to be "spoken" and fire callback() once the estimated
        duration has elapsed. This is the method BasicClient._on_demo_step
        looks for via hasattr() to know it can block on real speech.
        """
        if not self.enabled:
            if callback:
                callback()
            return False
        text = self._prepare_text(text)
        if not text:
            if callback:
                callback()
            return False
        self.tts_queue.put((text, callback))
        return True

    def interrupt(self) -> str:
        """
        Stop after the sentence in progress, same as edge_tts_output.py.
        Returns the unspoken remainder (only when what was interrupted was
        a demo step, not a plain chat_sentence), so it can be resumed.
        """
        self._interrupt_event.set()
        with self._progress_lock:
            if self._current_has_callback and self._current_sentences:
                remainder = " ".join(self._current_sentences[self._current_idx + 1:])
            else:
                remainder = ""
        while True:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break
        return remainder

    def clear_non_callback_items(self):
        """Drop pending chat_sentence items, keep pending demo steps."""
        keep = []
        while True:
            try:
                item = self.tts_queue.get_nowait()
                self.tts_queue.task_done()
                if isinstance(item, tuple) and item[1] is not None:
                    keep.append(item)
            except queue.Empty:
                break
        for item in keep:
            self.tts_queue.put(item)

    # ── Internal ────────────────────────────────────────────────────────────

    def _prepare_text(self, text: str) -> str:
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if self._max_length and len(text) > self._max_length:
            text = text[: self._max_length].rsplit(" ", 1)[0] + "..."
        return text

    def _tts_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.tts_queue.get(timeout=1)
                if item is None:
                    break
            except queue.Empty:
                continue

            text, callback = item if isinstance(item, tuple) else (item, None)
            try:
                self._speak_text(text, has_callback=callback is not None)
            except Exception as e:
                logger.error(f"[SimTTS] Playback error: {e}")
            finally:
                self.tts_queue.task_done()
                if callback:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"[SimTTS] Callback error: {e}")

    def _speak_text(self, text: str, has_callback: bool = False):
        self._interrupt_event.clear()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if not sentences:
            return

        with self._progress_lock:
            self._current_sentences = sentences
            self._current_idx = -1
            self._current_has_callback = has_callback

        if self.client and hasattr(self.client, "is_speaking"):
            self.client.is_speaking.set()
        if self.client and hasattr(self.client, "tts_started_event"):
            self.client.tts_started_event.set()

        try:
            for i, sentence in enumerate(sentences):
                if self._interrupt_event.is_set():
                    break
                with self._progress_lock:
                    self._current_idx = i
                duration = max(0.3, len(sentence.split()) / self._wps)
                logger.info(f"[SimTTS] Speaking (~{duration:.1f}s): {sentence[:70]}")
                # Interruptible wait — an interrupt() mid-sentence returns
                # immediately instead of finishing out the estimated duration.
                self._interrupt_event.wait(timeout=duration)
        finally:
            if self.client and hasattr(self.client, "is_speaking"):
                self.client.is_speaking.clear()
            with self._progress_lock:
                self._current_sentences = []
                self._current_idx = -1
                self._current_has_callback = False
