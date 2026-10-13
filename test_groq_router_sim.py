"""
Offline simulation of groq_router.py -- no network, no API key, fake clock.

Throwaway like the other test_*.py probes: proves the ROUTING LOGIC
(fallback order, return-to-primary, cooldowns, pacing, clean exhaustion)
against a fake Groq that enforces its own limits. It says nothing about
real Groq behavior; run `python groq_router.py --ping` for that.

Usage:
    cd src && python ../test_groq_router_sim.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from groq_router import (  # noqa: E402
    AllModelsExhausted,
    GroqRouter,
    ModelSpec,
    extract_json,
    parse_duration,
)

SYS, USER = "s" * 300, "u" * 300  # ~200 est. prompt tokens
ACTUAL_TOKENS = 700


class FakeResp:
    def __init__(self, status, headers=None, body=None, text=""):
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.text = text or (json.dumps(body) if body else "")

    def json(self):
        return self._body


class FakeGroq:
    """Enforces its own (possibly tighter-than-router-believes) limits."""

    def __init__(self, clock, server_limits=None):
        self.clock = clock
        self.server_limits = server_limits or {}  # model -> max calls before 429
        self.calls = {}
        self.log = []
        self.block_until = {}  # model -> ts

    def __call__(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        model = json["model"]
        now = self.clock[0]
        if now < self.block_until.get(model, 0):
            return FakeResp(429, {"retry-after": str(int(self.block_until[model] - now))}, text="rate limited")
        n = self.calls.get(model, 0)
        if model in self.server_limits and n >= self.server_limits[model]:
            self.block_until[model] = now + 300
            return FakeResp(429, {"retry-after": "300"}, text="Please try again in 5m0s")
        self.calls[model] = n + 1
        self.log.append(model)
        body = {
            "choices": [{"message": {"content": '{"agrees": true}'}, "finish_reason": "stop"}],
            "usage": {"total_tokens": ACTUAL_TOKENS},
        }
        return FakeResp(200, {"x-ratelimit-remaining-tokens": "5000", "x-ratelimit-remaining-requests": "900"}, body)


def make_router(models, server_limits=None, max_wait=120.0):
    clock = [1_000_000.0]
    server = FakeGroq(clock, server_limits)

    def sleep(s):
        clock[0] += s

    tmp = Path(tempfile.mkdtemp()) / "ledger.json"
    router = GroqRouter(
        "k", models=models, ledger_path=tmp, max_wait_s=max_wait, verbose=False,
        now_fn=lambda: clock[0], sleep_fn=sleep, post_fn=server,
    )
    return router, server, clock


def test_unit_helpers():
    assert abs(parse_duration("2m59.56s") - 179.56) < 1e-6
    assert parse_duration("250ms") == 0.25
    assert parse_duration("1h2m3s") == 3723
    assert parse_duration("7") == 7.0
    assert parse_duration("junk") is None
    assert extract_json('<think>{"x": 1} hmm</think>\n{"agrees": false}') == {"agrees": False}
    print("ok  helpers")


def test_proactive_fallback_and_return_to_primary():
    # Tiny daily budgets so exhaustion is quick: ~3 calls per model.
    specs = [ModelSpec(n, 30, 1000, 8000, 3000) for n in ("A", "B", "C")]
    router, server, clock = make_router(specs)

    used = [router.chat(SYS, USER).model for _ in range(9)]
    assert used == ["A"] * 3 + ["B"] * 3 + ["C"] * 3, used

    try:
        router.chat(SYS, USER)
        raise AssertionError("expected AllModelsExhausted")
    except AllModelsExhausted as e:
        assert e.resume_in_s > 120  # daily-bucket wait, beyond max_wait

    clock[0] += 86400 + 5  # a full day later: every bucket has drained
    assert router.chat(SYS, USER).model == "A"  # back to PRIMARY, not stuck on C
    print("ok  fallback A->B->C, clean exhaustion, back to A after reset")


def test_reactive_429_and_return():
    # Server cuts A off after 2 calls (as if another script spent the org
    # budget); router's ledger thinks A has plenty of room.
    specs = [ModelSpec(n, 30, 1000, 8000, 200_000) for n in ("A", "B")]
    router, server, clock = make_router(specs, server_limits={"A": 2})

    used = [router.chat(SYS, USER).model for _ in range(4)]
    assert used == ["A", "A", "B", "B"], used
    assert router.ledger.cooldown_left("A") > 0

    clock[0] += 400  # past A's retry-after (and past server's block)
    server.server_limits = {}
    assert router.chat(SYS, USER).model == "A"
    print("ok  server 429 -> cooldown -> B, then back to A after retry-after")


def test_tpm_pacing_waits_instead_of_failing():
    specs = [ModelSpec("A", 30, 1000, 8000, 200_000)]
    router, server, clock = make_router(specs)
    t0 = clock[0]
    for _ in range(14):  # ~9 fit in the first minute at 90% of 8K TPM
        # 600 max_completion reserved per call (Groq charges it up front)
        assert router.chat(SYS, USER, max_completion_tokens=600, est_completion_tokens=700).model == "A"
    assert clock[0] > t0, "expected the router to wait for the TPM window"
    assert clock[0] - t0 < 130
    print(f"ok  TPM pacing: 14 calls, waited {clock[0] - t0:.0f}s of fake time, no failure")


def test_ledger_survives_restart():
    specs = [ModelSpec(n, 30, 1000, 8000, 3000) for n in ("A", "B")]
    router, server, clock = make_router(specs)
    for _ in range(3):
        router.chat(SYS, USER)  # A is now spent for the day
    # "Restart": new router, same ledger file + clock.
    router2 = GroqRouter(
        "k", models=specs, ledger_path=router.ledger.path, verbose=False,
        now_fn=lambda: clock[0], sleep_fn=lambda s: None, post_fn=server,
    )
    assert router2.chat(SYS, USER).model == "B"
    print("ok  ledger persisted: restarted run skips the already-spent primary")


def test_daily_bucket_refills_gradually():
    # Confirmed from real headers: 1000 RPD refills 1 request / 86.4s,
    # not all-at-once after 24h. Drain a model, then check partial refill.
    specs = [ModelSpec("A", 30, 10, 8000, 200_000)]  # 9 usable req/day at 90%
    router, server, clock = make_router(specs, max_wait=100000)
    for _ in range(9):
        router.chat(SYS, USER, max_completion_tokens=100, est_completion_tokens=100)
        clock[0] += 61  # stay clear of the per-minute buckets
    t = clock[0]
    router.chat(SYS, USER, max_completion_tokens=100, est_completion_tokens=100)  # waits for ONE request to refill
    waited = clock[0] - t
    assert 0 <= waited < 8640, waited  # ~1 request's worth (8640s), nowhere near a full day
    print(f"ok  daily bucket refills gradually (waited {waited:.0f}s, not 86400s)")


def test_too_large_request_never_fits():
    specs = [ModelSpec("A", 30, 1000, 500, 200_000)]
    router, *_ = make_router(specs)
    try:
        router.chat(SYS, USER)
        raise AssertionError("expected AllModelsExhausted")
    except AllModelsExhausted as e:
        assert e.resume_in_s == float("inf")
    print("ok  request bigger than every model's TPM fails fast, no infinite wait")


if __name__ == "__main__":
    test_unit_helpers()
    test_proactive_fallback_and_return_to_primary()
    test_reactive_429_and_return()
    test_tpm_pacing_waits_instead_of_failing()
    test_ledger_survives_restart()
    test_daily_bucket_refills_gradually()
    test_too_large_request_never_fits()
    print("\nall simulated scenarios passed")
