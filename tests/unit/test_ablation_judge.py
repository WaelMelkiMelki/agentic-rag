"""The ablation's judge must survive an unreachable OpenRouter account.

``evals/run_ablation.py`` used ``router.get_llm("openrouter")`` unconditionally,
so a 401/402 from OpenRouter killed the whole run before the first question.
It now probes once and degrades to the local fallback chain.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `evals/` lives outside `src/`, so make it importable however pytest is called.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.run_ablation import build_judge  # noqa: E402
from evals.judges import judge_answer


class FailingLLM:
    def invoke(self, prompt):
        raise RuntimeError("User not found")


class WorkingLLM:
    def __init__(self, reply="4"):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Response", (), {"content": self.reply})()


class FakeRouter:
    def __init__(self, openrouter):
        self.openrouter = openrouter
        self.fallback_prompts = []

    def get_llm(self, provider):
        assert provider == "openrouter"
        return self.openrouter

    def invoke_with_fallback(self, prompt):
        self.fallback_prompts.append(prompt)
        return "5"


def test_openrouter_is_used_when_reachable():
    judge, name = build_judge(FakeRouter(WorkingLLM()))

    assert name == "openrouter"
    assert judge_answer("q", "gold", "answer", judge) == 4


def test_openrouter_failure_degrades_to_the_local_chain(capsys):
    router = FakeRouter(FailingLLM())
    judge, name = build_judge(router)

    assert "local chain" in name
    assert "OpenRouter judge unavailable" in capsys.readouterr().out
    # The chain answers "5", so the ablation still produces a score instead of crashing.
    assert judge_answer("q", "gold", "answer", judge) == 5
    assert router.fallback_prompts
