"""
tests/test_qa_classifier.py
============================
RobotInstance.classify_qa_intent — the label the LLM answers with.

The stay-in-Q&A label used to be "continue", while the question the model is
shown ends "...or shall we continue the demonstration?". "okay thanks" came
back 'continue' — continue the demo — and was read as "more questions", so
the window never closed. The model now answers done/more; only the parsing is
testable here, the model's behaviour was measured on qwen2.5:7b (see the
comment in classify_qa_intent).
"""

from types import SimpleNamespace

import pytest

from robot.robot_instance import RobotInstance


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.system = reply, None

    def is_available(self):
        return True

    def generate(self, system, user_message):
        self.system = system
        return SimpleNamespace(text=self.reply)


def _classify(reply, message="okay thanks"):
    robot = RobotInstance("navel_01", "Navel", user_id=1, enabled_modules=set())
    robot.llm = FakeLLM(reply)
    return robot.classify_qa_intent(message), robot.llm.system


@pytest.mark.parametrize("reply,decision", [
    ("done", "done"),
    ("Done.", "done"),
    ("more", "continue"),
    ("continue", "continue"),   # an off-label answer stays: never skip a question
    ("", "continue"),
])
def test_only_done_closes_the_window(reply, decision):
    assert _classify(reply)[0] == decision


def test_the_model_is_never_offered_continue_as_a_label():
    _, system = _classify("done")
    labels = [line.split("—")[0].strip() for line in system.splitlines()
              if "—" in line]
    assert labels == ["done", "more"]

