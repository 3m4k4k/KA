"""
Rate-limit-aware model router for Groq's free tier.

NOT a pipeline stage by itself -- it's the transport layer the future
answer-correctness / explanation stage will call instead of talking to
Groq directly. It knows nothing about questions, QuestionRecords, or
prompts; it takes a system+user prompt and returns text plus which
model produced it.

What it does
------------
Models are tried in PRIORITY ORDER on every single call (not "stick
with the fallback until restart"). A short pacing wait (<= prefer_wait_s,
default 20s, e.g. the per-minute token limit) is waited out on the
better model; it falls to the next one only when a model is truly out
(empty daily budget, long 429 cooldown). So the moment the primary's
window frees up or its cooldown expires, the next call goes back to
it -- that's the "until the initial one resets" behavior.

Two layers of protection, because neither alone is enough:

  1. Proactive: a local usage ledger (persisted to cache/groq_usage.json)
     tracks requests/tokens per model over rolling 60s and 24h windows
     and skips a model that wouldn't fit the next call. The ledger
     survives restarts, so a re-run the next morning doesn't blindly
     hammer a model that's already spent for the day.
  2. Reactive: on a real HTTP 429 the model gets a cooldown taken from
     Groq's own `retry-after`, and the router moves to the next model.
     This is the ground truth -- limits are ORG-level on Groq, so
     anything else using the same org (earlier probes, a second
     script) spends the same budget and the local ledger can't see it.

If every usable model is out of budget, it waits when the wait is short
(per-minute limits) and raises AllModelsExhausted when it's long
(daily limits), carrying `resume_in_s`. The calling stage should treat
that as "stop cleanly, re-run later" -- the stages here are idempotent,
so nothing is lost.

Honest gaps
-----------
  - Simulated offline against a fake Groq; the --ping header check was
    run for real (header names confirmed). Limits behave as token
    buckets that refill continuously (limit/window per second), and the
    per-minute bucket is charged prompt + max_completion_tokens up
    front. The ledger models exactly that.
  - Groq doesn't expose the daily TOKEN bucket in headers, so it is
    charged with actual usage. If that's wrong the 429 path corrects it.
  - Models are NOT interchangeable. Every CallResult carries `model`;
    the caller must store it next to each verdict (same idea as
    ai_model_used) and every fallback model should pass the --corrupt
    probe in test_groq.py before it's trusted as a checker.

Usage:
    python groq_router.py --status          # budget per model, no API call
    python groq_router.py --ping            # one tiny real call + header dump
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MINUTE = 60.0
DAY = 86400.0
INF = float("inf")


# ---------------------------------------------------------------------------
# Model table
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    rpm: int
    rpd: int
    tpm: int
    tpd: int
    # Sent verbatim in the request body, e.g. {"reasoning_effort": "low"}.
    # Per-model on purpose: an unsupported param returns HTTP 400.
    extra_params: dict = field(default_factory=dict)


# Priority order. Limits are from the Groq console (org limits page).
#
# Deliberately NOT in this list:
#   - llama-prompt-guard-2-22m / -86m: prompt-injection classifiers, not
#     chat models; they can't judge an anatomy question.
#   - gpt-oss-safeguard-20b: a policy/safety classification model, and
#     only 3 requests/min -- wrong tool and too slow.
#   - allam-2-7b: has the biggest daily budget (500K TPD) but is a 7B
#     Arabic/English model; don't enable it as a checker unless it
#     passes the --corrupt probe. See OPTIONAL_MODELS.
DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec("openai/gpt-oss-120b", 30, 1000, 8000, 200_000),
    ModelSpec("qwen/qwen3.8-27b", 30, 1000, 8000, 200_000),
]

OPTIONAL_MODELS: list[ModelSpec] = [
    ModelSpec("allam-2-7b", 30, 7000, 6000, 500_000),
]


# ---------------------------------------------------------------------------
# Errors / results
# ---------------------------------------------------------------------------


class RouterError(Exception):
    pass


class FatalAPIError(RouterError):
    """Bad key / forbidden. Retrying or switching models won't help."""


class AllModelsExhausted(RouterError):
    def __init__(self, message: str, resume_in_s: float):
        super().__init__(message)
        self.resume_in_s = resume_in_s


@dataclass
class CallResult:
    text: str
    model: str
    fallback_index: int  # 0 = primary model answered
    total_tokens: int
    finish_reason: str | None
    latency_s: float
    rate_headers: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(value: str | None) -> float | None:
    """'2m59.56s' -> 179.56, '250ms' -> 0.25, '7' -> 7.0, junk -> None."""
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    parts = _DURATION_RE.findall(value)
    if not parts:
        return None
    return sum(float(n) * _UNIT[u] for n, u in parts)


def _int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def estimate_tokens(text: str) -> int:
    # Deliberately pessimistic (~3 chars/token) -- medical terms and the
    # mixed-script OCR junk in these records tokenize badly.
    return max(1, len(text) // 3)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def extract_json(text: str | None) -> dict | None:
    """Pull a JSON object out of a model reply. Strips <think>...</think>
    first -- Qwen-family models can emit reasoning inline, and braces
    inside it would otherwise poison the match."""
    if not text:
        return None
    text = _THINK_RE.sub("", text)
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Usage ledger (persisted)
# ---------------------------------------------------------------------------


class Ledger:
    """Leaky-bucket accounting per model, matching how Groq actually
    behaves (confirmed from real --ping headers: a 1000 req/day limit
    reports reset-requests of 86.4s after ONE request, i.e. a steady
    refill of limit/window per second, not a fixed reset time).

    Each of rpm/tpm/rpd/tpd is a bucket: `level` = amount used, draining
    at limit/window per second. Persisted so restarts remember it."""

    def __init__(self, path: Path, now_fn=time.time):
        self.path = path
        self.now = now_fn
        self.buckets: dict[str, dict[str, list[float]]] = {}  # model -> dim -> [level, ts]
        self.cooldowns: dict[str, float] = {}  # model -> unix ts it's usable again
        self._load()

    @staticmethod
    def _dims(spec: ModelSpec) -> dict[str, tuple[float, float]]:
        return {
            "rpm": (spec.rpm, MINUTE),
            "tpm": (spec.tpm, MINUTE),
            "rpd": (spec.rpd, DAY),
            "tpd": (spec.tpd, DAY),
        }

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self.buckets = {
                    m: {d: [float(v[0]), float(v[1])] for d, v in dims.items()}
                    for m, dims in raw.get("buckets", {}).items()
                }
                self.cooldowns = {k: float(v) for k, v in raw.get("cooldowns", {}).items()}
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError, IndexError):
                # A corrupt/old-format ledger must never block a run.
                # Start clean; the 429 path re-teaches anything forgotten.
                self.buckets, self.cooldowns = {}, {}
        self._prune()

    def _prune(self) -> None:
        now = self.now()
        self.cooldowns = {m: t for m, t in self.cooldowns.items() if t > now}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"buckets": self.buckets, "cooldowns": self.cooldowns}))
        tmp.replace(self.path)

    def level(self, spec: ModelSpec, dim: str) -> float:
        lim, win = self._dims(spec)[dim]
        b = self.buckets.get(spec.name, {}).get(dim)
        if not b:
            return 0.0
        return max(0.0, b[0] - (self.now() - b[1]) * lim / win)

    def charge(self, spec: ModelSpec, tpm_tokens: float, tpd_tokens: float) -> None:
        now = self.now()
        for dim, amt in (("rpm", 1), ("rpd", 1), ("tpm", tpm_tokens), ("tpd", tpd_tokens)):
            self.buckets.setdefault(spec.name, {})[dim] = [self.level(spec, dim) + amt, now]
        self._prune()
        self._save()

    def set_cooldown(self, model: str, seconds: float) -> None:
        until = self.now() + max(seconds, 0.0)
        self.cooldowns[model] = max(self.cooldowns.get(model, 0.0), until)
        self._save()

    def cooldown_left(self, model: str) -> float:
        return max(self.cooldowns.get(model, 0.0) - self.now(), 0.0)

    def wait_needed(self, spec: ModelSpec, need_tpm: float, need_tpd: float, margin: float) -> float:
        """Seconds until a call fits this model's budget (0 = go now,
        INF = can never fit)."""
        w = self.cooldown_left(spec.name)
        for dim, need in (("rpm", 1), ("tpm", need_tpm), ("rpd", 1), ("tpd", need_tpd)):
            lim, win = self._dims(spec)[dim]
            cap = lim * margin
            if need > cap:
                return INF
            over = self.level(spec, dim) + need - cap
            if over > 0:
                w = max(w, over * win / lim)
        return w


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class GroqRouter:
    def __init__(
        self,
        api_key: str,
        models: list[ModelSpec] | None = None,
        ledger_path: Path | None = None,
        margin: float = 0.9,
        max_wait_s: float = 120.0,
        prefer_wait_s: float = 20.0,
        verbose: bool = True,
        now_fn=time.time,
        sleep_fn=time.sleep,
        post_fn=requests.post,
    ):
        self.api_key = api_key
        self.models = models if models is not None else list(DEFAULT_MODELS)
        self.margin = margin
        self.max_wait_s = max_wait_s
        self.prefer_wait_s = prefer_wait_s
        self.verbose = verbose
        self.now = now_fn
        self.sleep = sleep_fn
        self.post = post_fn
        if ledger_path is None:
            ledger_path = Path(__file__).resolve().parent.parent / "cache" / "groq_usage.json"
        self.ledger = Ledger(ledger_path, now_fn)
        self.disabled: set[str] = set()  # models that returned a hard 400/404 this session
        self.calls_by_model: dict[str, int] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [router] {msg}")

    # -- public ---------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_completion_tokens: int = 1500,
        est_completion_tokens: int = 800,
        temperature: float = 0.1,
    ) -> CallResult:
        """est_completion_tokens is what the ledger RESERVES for budget
        checks (real usage replaces it after the call). Keep it near the
        observed average (~630 in the probe) rather than at
        max_completion_tokens, or the 8K TPM limit will look full after
        only a few calls."""
        prompt_est = estimate_tokens(system + user)
        # Groq's TPM bucket is charged prompt + max_completion_tokens up
        # front (confirmed via --ping: 284 charged for a 143-token call),
        # so budget the per-minute check on that, not on expected usage.
        need_tpm = prompt_est + max_completion_tokens
        need_tpd = prompt_est + est_completion_tokens
        max_attempts = 3 * max(len(self.models), 1)
        attempts = 0

        while True:
            live = [(i, m) for i, m in enumerate(self.models) if m.name not in self.disabled]
            if not live:
                raise FatalAPIError("every configured model was rejected by the API (see log above)")
            waits = [(i, m, self.ledger.wait_needed(m, need_tpm, need_tpd, self.margin)) for i, m in live]

            # Quality over speed: the daily budget is the scarce thing, the
            # per-minute limit is just pacing. So the highest-priority model
            # whose wait is SHORT (<= prefer_wait_s) is waited for instead of
            # dropping to a weaker model. A model only gets skipped when it
            # is genuinely out (daily bucket empty, long 429 cooldown).
            ready = [(i, m, w) for i, m, w in waits if w <= self.prefer_wait_s]
            if ready:
                idx, spec, w = ready[0]
                if w > 0:
                    self._log(f"{spec.name}: pacing, waiting {w:.0f}s")
                    self.sleep(w + 0.25)
                attempts += 1
                result = self._attempt(spec, idx, system, user, max_completion_tokens, temperature, need_tpm, need_tpd)
                if result is not None:
                    return result
                if attempts >= max_attempts:
                    raise RouterError(
                        f"gave up after {attempts} failed attempts across {len(self.models)} model(s)"
                    )
                continue

            soonest = min(w for _, _, w in waits)
            if soonest == INF:
                raise AllModelsExhausted("this request can never fit any model's limits", INF)
            if soonest > self.max_wait_s:
                raise AllModelsExhausted(
                    f"all models are out of budget; earliest one frees in {soonest / 60:.1f} min", soonest
                )
            self._log(f"all models out of budget, waiting {soonest:.0f}s")
            self.sleep(soonest + 0.25)

    # -- internals ------------------------------------------------------

    def _attempt(self, spec, idx, system, user, max_completion_tokens, temperature, need_tpm, need_tpd) -> CallResult | None:
        """One HTTP call to one model. Returns a CallResult on success,
        None if it failed in a way that's been handled (cooldown set /
        model disabled) and the caller should try the next model."""
        payload = {
            "model": spec.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            **spec.extra_params,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        start = self.now()

        try:
            resp = self.post(GROQ_CHAT_URL, json=payload, headers=headers, timeout=90)
        except requests.RequestException as exc:
            self._log(f"{spec.name}: network error ({exc}); 15s cooldown")
            self.ledger.set_cooldown(spec.name, 15)
            return None

        code = resp.status_code

        if code == 429:
            retry_after = parse_duration(resp.headers.get("retry-after"))
            if retry_after is None:
                m = re.search(r"try again in ([0-9hms.]+)", resp.text or "")
                retry_after = parse_duration(m.group(1)) if m else None
            cooldown = (retry_after if retry_after is not None else 30.0) + 1.0
            self.ledger.set_cooldown(spec.name, cooldown)
            self._log(f"{spec.name}: 429, cooling down {cooldown:.0f}s -> next model")
            return None

        if code in (401, 403):
            raise FatalAPIError(f"HTTP {code} from Groq -- check GROQ_API_KEY: {(resp.text or '')[:200]}")

        if code in (400, 404):
            self.disabled.add(spec.name)
            self._log(f"{spec.name}: HTTP {code}, disabling for this run -- {(resp.text or '')[:300]}")
            return None

        if code == 413:
            self.ledger.set_cooldown(spec.name, 60)
            self._log(f"{spec.name}: 413 request too large for this model")
            return None

        if code != 200:
            self.ledger.set_cooldown(spec.name, 20)
            self._log(f"{spec.name}: HTTP {code}, 20s cooldown -- {(resp.text or '')[:200]}")
            return None

        body = resp.json()
        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = body.get("usage") or {}
        total = int(usage.get("total_tokens") or need_tpd)
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(system + user))
        # TPM: what Groq reserved (prompt + max_completion). TPD: actual
        # use -- the daily bucket isn't exposed in headers, so this is an
        # assumption; the 429 path is the backstop if it's wrong.
        self.ledger.charge(spec, prompt_tokens + max_completion_tokens, total)
        self.calls_by_model[spec.name] = self.calls_by_model.get(spec.name, 0) + 1

        rate_headers = {
            k.lower(): v
            for k, v in resp.headers.items()
            if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"
        }
        self._sync_from_headers(spec, rate_headers, need_tpm)

        if idx > 0:
            self._log(f"answered by fallback #{idx}: {spec.name}")

        return CallResult(
            text=text,
            model=spec.name,
            fallback_index=idx,
            total_tokens=total,
            finish_reason=choice.get("finish_reason"),
            latency_s=self.now() - start,
            rate_headers=rate_headers,
        )

    def _sync_from_headers(self, spec: ModelSpec, h: dict, need: int) -> None:
        """If Groq's own remaining-budget headers say the next call
        wouldn't fit, trust them over the local ledger. Defensive: any
        missing/odd header is ignored. Groq's docs describe
        remaining-requests as the daily bucket and remaining-tokens as
        the per-minute one -- verify with --ping."""
        rem_req = _int(h.get("x-ratelimit-remaining-requests"))
        rem_tok = _int(h.get("x-ratelimit-remaining-tokens"))
        if rem_req is not None and rem_req <= 0:
            wait = parse_duration(h.get("x-ratelimit-reset-requests")) or 60.0
            self.ledger.set_cooldown(spec.name, wait + 1)
            self._log(f"{spec.name}: Groq reports 0 requests left, cooling down {wait:.0f}s")
        elif rem_tok is not None and rem_tok < need:
            wait = parse_duration(h.get("x-ratelimit-reset-tokens")) or 10.0
            self.ledger.set_cooldown(spec.name, wait + 0.5)

    # -- reporting ------------------------------------------------------

    def status_lines(self) -> list[str]:
        """'used' = bucket level now (drains continuously; TPM level
        includes the max_completion_tokens reservation)."""
        lines = []
        for spec in self.models:
            lv = lambda d: self.ledger.level(spec, d)  # noqa: E731
            cd = self.ledger.cooldown_left(spec.name)
            lines.append(
                f"{spec.name:<24} 1m: {lv('rpm'):.0f}/{spec.rpm} req, {lv('tpm'):.0f}/{spec.tpm} tok | "
                f"24h: {lv('rpd'):.0f}/{spec.rpd} req, {lv('tpd'):.0f}/{spec.tpd} tok"
                + (f" | cooldown {cd:.0f}s" if cd > 0 else "")
            )
        return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Groq router: budget status / live header check.")
    parser.add_argument("--status", action="store_true", help="Show tracked usage per model (no API call).")
    parser.add_argument("--ping", action="store_true", help="One tiny real call; prints x-ratelimit headers.")
    parser.add_argument("--include-allam", action="store_true", help="Add allam-2-7b as last resort.")
    parser.add_argument("--ledger", type=Path, default=None)
    args = parser.parse_args()

    models = list(DEFAULT_MODELS) + (list(OPTIONAL_MODELS) if args.include_allam else [])
    api_key = os.environ.get("GROQ_API_KEY", "")
    router = GroqRouter(api_key, models=models, ledger_path=args.ledger)

    if args.ping:
        if not api_key:
            print("Set GROQ_API_KEY first.")
            return
        res = router.chat("Reply with exactly: {\"ok\": true}", "ping", max_completion_tokens=200, est_completion_tokens=100)
        print(f"model: {res.model}  tokens: {res.total_tokens}  latency: {res.latency_s:.2f}s")
        print(f"reply: {res.text!r}")
        print("rate-limit headers Groq sent back:")
        for k, v in sorted(res.rate_headers.items()):
            print(f"  {k}: {v}")

    print("\n".join(router.status_lines()))


if __name__ == "__main__":
    main()
