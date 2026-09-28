"""
test_tts_pipeline.py — the TTS module overlaps synthesis with playback.

Run from v6.0.0/client:  python3 -m pytest test_tts_pipeline.py
(The server suite's pytest.ini only collects server/tests.)

Streamed answers arrive one sentence at a time. Each sentence used to be
synthesised only once the previous one had finished playing, so every
sentence paid a full TTS round-trip of silence before it.
"""

import threading
import time

from OutputModules.edge_tts_output import EdgeTTSOutputModule

SYNTH, PLAY = 0.2, 0.2


class _Client:
    def __init__(self):
        self.states = []

    def on_tts_state(self, speaking):
        self.states.append(speaking)


def _module():
    m = EdgeTTSOutputModule(config={})
    m.client = _Client()
    m._generate_audio = lambda s, lang: (time.sleep(SYNTH), (None, None))[1]
    m._play_audio = lambda text, mp3, wav: time.sleep(PLAY)
    m.start()
    return m


def _wait_idle(m, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if m.client.states and m.client.states[-1] is False and m.tts_queue.empty():
            return
        time.sleep(0.01)
    raise AssertionError("TTS never went idle")


def test_streamed_sentences_are_synthesised_while_the_first_plays():
    m = _module()
    started = time.time()
    for s in ("First sentence here.", "Second sentence here.", "Third sentence here."):
        m.process_output({"text": s})
        time.sleep(0.02)                      # arriving as a stream
    _wait_idle(m)
    took = time.time() - started
    # Serial would be 3 x (SYNTH + PLAY) = 1.2s; overlapped is ~SYNTH + 3 x PLAY.
    assert took < 3 * (SYNTH + PLAY) - 0.2, took
    m.stop()


def test_speaking_is_reported_once_each_way():
    m = _module()
    m.process_output({"text": "One thing to say."})
    m.process_output({"text": "And another."})
    _wait_idle(m)
    assert m.client.states == [True, False]
    m.stop()


def test_an_interrupt_drops_what_was_queued():
    m = _module()
    played = []
    m._play_audio = lambda text, mp3, wav: (played.append(text), time.sleep(PLAY))
    for s in ("Keep this one.", "Drop this.", "And this."):
        m.process_output({"text": s})
    time.sleep(SYNTH + 0.05)                  # first sentence is now playing
    m.interrupt()
    _wait_idle(m)
    assert played == ["Keep this one."]
    m.stop()


def test_a_demo_step_callback_still_fires_after_playback():
    m = _module()
    done = threading.Event()
    m.speak_with_callback("Welcome to the lab.", callback=done.set)
    assert done.wait(2)
    m.stop()
