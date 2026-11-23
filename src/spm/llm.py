"""Model backends.

Two of them:

* ChatBackend  -- elicitation against any OpenAI-compatible chat endpoint. Works
  against OpenAI itself and against gateways such as 9Router, OpenRouter, vLLM or
  LiteLLM. Retries are capped at 5; MOSAIC's Utils.generate_llm_response uses
  stop_after_attempt(1000), which on a metered API is a runaway.
* MockBackend  -- a deterministic agent that literally executes Def. 2. It exists
  so the whole pipeline can be exercised without spending a token, and so the
  calibration regression in analyze.py can be validated against known kappa and
  delta*alpha. If the regression cannot recover the mock's own coefficients, the
  regression is wrong.

Structured output is the thing that breaks when you leave OpenAI. OpenAI's
``client.beta.chat.completions.parse`` sends a strict JSON schema and guarantees a
parsed object back. Most gateways do not implement it -- 9Router's API reference
documents ``model``, ``messages``, ``temperature`` and ``max_tokens`` and says
nothing about ``response_format`` -- and a gateway that rewrites requests between
provider formats may drop the field silently, which is worse than failing. So
``ChatBackend`` has two modes: ``native`` uses .parse, ``json`` asks for a JSON
object in the prompt and parses the reply here, with a repair retry. The default,
``auto``, picks ``json`` whenever a custom base URL is set.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field


def resolve_api_key() -> str | None:
    """Where the key comes from, in order of preference.

    1. ``OPENAI_API_KEY`` in the environment. Use this one.
    2. ``src/keys.py``, MOSAIC's own convention.

    A warning about (2): that file is **tracked by git** in the MOSAIC repository,
    despite the "DO NOT UPLOAD TO GITHUB" comment at the top of it. Putting a live
    key there and committing publishes it. If you use it anyway, untrack it first:

        git rm --cached src/keys.py && echo "src/keys.py" >> .gitignore

    For a gateway (9Router, OpenRouter, vLLM, LiteLLM) the key is whatever that
    gateway issues, and it goes in the same place; set ``base_url`` in the config
    or ``OPENAI_BASE_URL`` in the environment to point at it.
    """
    for var in ("SPM_API_KEY", "OPENAI_API_KEY"):
        key = os.environ.get(var)
        if key:
            return key
    try:
        import keys  # MOSAIC's src/keys.py, on sys.path via run_spm.py
    except ImportError:
        return None
    return getattr(keys, "OPENAI_API_KEY", "") or None


class BeliefResponse(BaseModel):
    """The single structured output an agent produces per step."""

    confidence: float = Field(
        description="Your probability that the CLAIM is true, strictly between 0 and 1."
    )
    stance: str = Field(
        description="One sentence, under 200 characters, saying where you now stand and why."
    )


@dataclass
class TurnContext:
    """Numeric state passed alongside the prompt; used only by MockBackend."""

    prior_q: float
    signed_llr: float
    neighbour_psis: list[float]


class MockBackend:
    """Def. 2 in closed form, in either of two social forms.

    additive (Def. 2 exactly)
        L' = L + kappa*l(x) + delta*alpha * mean_j psi(m_j)

        In consensus every neighbour reports roughly L, so this is
        L' ~ L(1 + delta*alpha): the population's log-odds diverge geometrically
        and pin at the elicitation bound within a few tens of steps, permanently.
        Once pinned, every belief series is constant, every transfer entropy is
        zero for honest agents and manipulators alike, and the responsiveness
        detector is blind. Not an artefact of this mock -- it is the instability
        the MD flags in the old Appendix A (rho(I + delta*alpha*P_H) > 1), which
        the revised Def. 2 still carries.

    convex (log-linear pooling, as in the non-Bayesian social learning it cites)
        L' = L + kappa*l(x) + delta*alpha * (mean_j psi(m_j) - L)

        Stable, and agents stay responsive to their neighbours indefinitely. This
        is the default here because it is the behaviour a real LLM agent will
        plausibly show -- models asked for a probability do not run away to
        certainty -- and because the additive form leaves nothing to measure.
    """

    def __init__(self, kappa: float = 0.35, delta_alpha: float = 0.30,
                 noise: float = 0.05, seed: int = 0, social_form: str = "convex"):
        self.kappa = kappa
        self.delta_alpha = delta_alpha
        self.noise = noise
        self.social_form = social_form
        self.rng = random.Random(seed)
        self.calls = 0

    def elicit(self, system: str, prompt: str, ctx: TurnContext) -> BeliefResponse:
        self.calls += 1
        prior = min(max(ctx.prior_q, 1e-6), 1 - 1e-6)
        L = math.log(prior / (1 - prior))
        social = (sum(ctx.neighbour_psis) / len(ctx.neighbour_psis)
                  if ctx.neighbour_psis else 0.0)
        L += self.kappa * ctx.signed_llr
        if self.social_form == "convex":
            L += self.delta_alpha * (social - L)
        else:
            L += self.delta_alpha * social
        L += self.rng.gauss(0.0, self.noise)
        q = 1.0 / (1.0 + math.exp(-L))
        # Elicitation bounds: a person asked for a probability does not answer
        # 0.999. Saturation here is not an artefact -- Def. 2's social term is
        # positive feedback, so belief runs away and locks at the bound. That is
        # the same instability the MD flags in the old Appendix A, and it is why
        # the analysis works on the transient window rather than the whole run.
        q = min(max(q, 0.02), 0.98)
        return BeliefResponse(
            confidence=round(q, 3),
            stance=f"My meter and my neighbours put me at about {q:.2f} on this.",
        )


JSON_INSTRUCTION = (
    "\n\nReply with ONE JSON object and nothing else, in exactly this form:\n"
    '{"confidence": <number strictly between 0 and 1>, '
    '"stance": "<one sentence, under 200 characters>"}\n'
    "No markdown fences, no commentary before or after the object."
)


def extract_json(text: str) -> dict:
    """Pull the first complete JSON object out of a model reply.

    Gateways route to whatever provider is cheapest or available, so the reply may
    arrive wrapped in ```json fences, prefixed with "Sure, here you go:", or
    followed by an explanation. Scanning for the first balanced object survives all
    three; json.loads on the raw text does not.
    """
    depth, start, in_string, escaped = 0, None, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return json.loads(text[start:i + 1])
    raise ValueError(f"no JSON object in reply: {text[:200]!r}")


RESET_HINT = re.compile(r"reset after\s+(?:(\d+)m\s*)?(\d+)s", re.I)


#: substrings that mean "retrying will fail exactly the same way". Six attempts
#: against a bad key or an unknown model wastes a minute and, worse, buries the
#: real message under a generic one.
FATAL_MARKERS = (
    "invalid_api_key", "incorrect api key", "authentication",
    "no such model", "model_not_found", "does not exist",
    "insufficient_quota", "billing", "permission",
    "unsupported_parameter", "unsupported_value", "unrecognized request argument",
)


def is_fatal(exc: Exception) -> bool:
    """True when retrying cannot possibly help, so we should surface it at once."""
    if isinstance(exc, (AttributeError, ImportError, TypeError)):
        return True                      # an SDK-shape problem, not a network one
    text = str(exc).lower()
    return any(marker in text for marker in FATAL_MARKERS)


def retry_delay(exc: Exception, attempt: int) -> float:
    """How long to wait before retrying, in seconds.

    A rate limit is not a transient blip to be shrugged off with exponential
    backoff from one second: the provider tells you when the window reopens, and
    anything shorter is guaranteed to fail again. 9Router puts it in the message
    as "(reset after 3m 29s)"; some providers use a Retry-After header. Fall back
    to a minute, which beats 31 seconds of pointless retries.
    """
    text = str(exc)
    if not ("429" in text or "rate_limit" in text.lower()):
        return min(2 ** attempt, 16)
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    for key in ("retry-after", "Retry-After"):
        if key in header:
            try:
                return min(float(header[key]) + 1.0, 300.0)
            except (TypeError, ValueError):
                pass
    match = RESET_HINT.search(text)
    if match:
        return min(int(match.group(1) or 0) * 60 + int(match.group(2)) + 2.0, 300.0)
    return 60.0


class RateLimiter:
    """Global pacing across every worker thread.

    Concurrency without pacing is what trips a requests-per-minute limit: sixteen
    threads fire the instant a step begins. This spaces departures evenly, so the
    same throughput arrives without the burst.
    """

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start - now > 0:
            time.sleep(start - now)


class ChatBackend:
    """Elicitation against any OpenAI-compatible chat endpoint.

    ``structured`` selects how the object is obtained:

    native  OpenAI's strict-schema ``beta.chat.completions.parse``. Only OpenAI and
            a few gateways implement it.
    json    Ask for a JSON object in the prompt and parse the reply here. Works
            anywhere, including 9Router, and is what ``auto`` chooses whenever a
            custom base URL is set.
    auto    native when talking to OpenAI directly, json otherwise.
    """

    MAX_ATTEMPTS = 6

    def __init__(self, engine: str, temperature: float, api_key: str | None = None,
                 base_url: str = "", structured: str = "auto", max_tokens: int = 2048,
                 rpm: int = 0, reasoning_effort: str = ""):
        base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        key = api_key or resolve_api_key()
        if not key and not base_url:
            raise RuntimeError(
                f"No API key found for engine {engine!r}. Set OPENAI_API_KEY (or "
                "SPM_API_KEY) in your environment, or fill in OPENAI_API_KEY in "
                "src/keys.py (which is git-tracked -- see resolve_api_key). Use "
                'engine "mock" to run without a key.'
            )
        from openai import OpenAI  # imported lazily so the mock path needs no dependency

        self.engine = engine
        self.temperature = temperature
        self.base_url = base_url
        # A self-hosted gateway may not require a key at all; the SDK still insists
        # on a non-empty string, so send a placeholder rather than refusing to start.
        self.client = OpenAI(api_key=key or "not-required",
                             **({"base_url": base_url} if base_url else {}))
        if structured == "auto":
            structured = "json" if base_url else "native"
        if structured not in ("native", "json"):
            raise ValueError(f"unknown structured mode {structured!r}")
        self.structured = structured
        self.max_tokens = max_tokens
        # GPT-5-family and o-series models reject `max_tokens` and want
        # `max_completion_tokens`; 4o and earlier want the old name. Rather than
        # keep a list of which is which, start with the old name and switch once
        # if the endpoint objects.
        self.token_param = "max_tokens"
        self.send_temperature = True
        # Bumped whenever the request shape changes. Thirty worker threads hit the
        # same 400 at the same instant; exactly one of them performs the fix, and
        # the other twenty-nine would otherwise find nothing left to adapt, fall
        # through to is_fatal() and give up -- killing the run the moment before
        # it was about to work. A thread that sees this counter move knows the
        # request has been repaired and simply tries again.
        self._param_gen = 0
        # Sent through extra_body rather than as a named kwarg: `reasoning_effort`
        # is only a first-class parameter in recent SDKs, and a gateway that
        # forwards to a non-OpenAI provider needs it in the raw body anyway. Empty
        # means send nothing at all, so the provider default is untouched.
        self.reasoning_effort = (reasoning_effort or "").strip()
        self._lock = threading.Lock()
        self.limiter = RateLimiter(rpm)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _effort_kwargs(self) -> dict:
        if not self.reasoning_effort:
            return {}
        return {"extra_body": {"reasoning_effort": self.reasoning_effort}}

    def _limit_kwargs(self) -> dict:
        return {self.token_param: self.max_tokens} if self.max_tokens else {}

    def _temperature_kwargs(self) -> dict:
        return {"temperature": self.temperature} if self.send_temperature else {}

    def _adapt_parameters(self, exc: Exception) -> bool:
        """Rename or drop a parameter the endpoint just rejected. True if changed.

        Model families disagree about their own request schema: the GPT-5 series
        wants `max_completion_tokens` where 4o wants `max_tokens`, and reasoning
        models accept only the default temperature. Both come back as a 400 that
        names the offending field, so adapt once and retry rather than making the
        user discover it from a truncated traceback.
        """
        text = str(exc)
        with self._lock:
            if "max_tokens" in text and self.token_param == "max_tokens":
                self.token_param = "max_completion_tokens"
                self._param_gen += 1
                print("note: endpoint wants max_completion_tokens; switching.",
                      file=sys.stderr)
                return True
            if "temperature" in text and self.send_temperature:
                self.send_temperature = False
                self._param_gen += 1
                print(f"note: {self.engine} rejected temperature; using its "
                      "default.", file=sys.stderr)
                return True
        return False

    def _drop_effort_if_rejected(self, exc: Exception) -> bool:
        """Disable reasoning_effort once, if that is what the endpoint objected to.

        A non-reasoning model answers `reasoning_effort` with a 400. Without this
        the parameter is fatal, and a sweep that is twelve hours in dies on a knob
        that was never load-bearing. Drop it, say so once, carry on -- the science
        is unaffected because a model that rejects the field never had the
        behaviour it controls.
        """
        if not self.reasoning_effort:
            return False
        if "reasoning_effort" not in str(exc):
            return False
        with self._lock:
            if not self.reasoning_effort:      # another thread got here first
                return False
            dropped, self.reasoning_effort = self.reasoning_effort, ""
            self._param_gen += 1
        print(f"note: {self.engine} rejected reasoning_effort={dropped!r}; "
              "continuing without it.", file=sys.stderr)
        return True

    def _record(self, completion) -> None:
        usage = getattr(completion, "usage", None)
        with self._lock:                      # elicit() may run on many threads
            self.calls += 1
            if usage is not None:
                self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def elicit(self, system: str, prompt: str, ctx: TurnContext) -> BeliefResponse:
        if self.structured == "native":
            return self._elicit_native(system, prompt)
        return self._elicit_json(system, prompt)

    def _elicit_native(self, system: str, prompt: str) -> BeliefResponse:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        last: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            generation = self._param_gen
            try:
                self.limiter.wait()
                completion = self.client.beta.chat.completions.parse(
                    model=self.engine, messages=messages,
                    response_format=BeliefResponse,
                    timeout=60, **self._temperature_kwargs(),
                    **self._limit_kwargs(),
                    **self._effort_kwargs(),
                )
                self._record(completion)
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("model returned no parsed content")
                parsed.confidence = _clip(parsed.confidence)
                return parsed
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                last = exc
                if self._drop_effort_if_rejected(exc) or self._adapt_parameters(exc):
                    continue
                if self._param_gen != generation:
                    continue          # another thread repaired the request shape
                if is_fatal(exc):
                    break
                time.sleep(retry_delay(exc, attempt))
        raise RuntimeError(
            f"structured (native) elicitation failed against "
            f'{self.base_url or "api.openai.com"} with model {self.engine!r}.\n'
            f"  underlying error: {type(last).__name__}: {last}\n"
            "  if the endpoint does not support response_format, set "
            'SPM_STRUCTURED=json'
        ) from last

    def _elicit_json(self, system: str, prompt: str) -> BeliefResponse:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt + JSON_INSTRUCTION}]
        last: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            generation = self._param_gen
            try:
                self.limiter.wait()
                completion = self.client.chat.completions.create(
                    model=self.engine, messages=messages,
                    timeout=60, **self._temperature_kwargs(),
                    **self._limit_kwargs(),
                    **self._effort_kwargs(),
                )
                self._record(completion)
                text = completion.choices[0].message.content or ""
                data = extract_json(text)
                data["confidence"] = _clip(float(data["confidence"]))
                data["stance"] = str(data.get("stance", ""))[:400]
                return BeliefResponse(**data)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if self._drop_effort_if_rejected(exc) or self._adapt_parameters(exc):
                    continue
                if self._param_gen != generation:
                    continue          # another thread repaired the request shape
                if is_fatal(exc):
                    break
                # Show the model its own bad output once, then fall back to waiting.
                if attempt == 0 and "text" in dir():
                    messages = messages[:2] + [
                        {"role": "assistant", "content": text[:500]},
                        {"role": "user", "content":
                         "That was not a single valid JSON object." + JSON_INSTRUCTION},
                    ]
                else:
                    time.sleep(retry_delay(exc, attempt))
        raise RuntimeError(
            f"JSON elicitation failed against "
            f'{self.base_url or "api.openai.com"} with model {self.engine!r}.\n'
            f"  underlying error: {type(last).__name__}: {last}"
        ) from last


def _clip(q: float) -> float:
    return min(max(float(q), 0.01), 0.99)


def make_backend(engine: str, temperature: float, seed: int,
                 base_url: str = "", structured: str = "auto", max_tokens: int = 2048,
                 rpm: int = 0, reasoning_effort: str = ""):
    if engine == "mock":
        return MockBackend(seed=seed)
    # max_tokens and rpm used to be accepted here and then not forwarded, so the
    # RateLimiter was constructed with rpm = 0 (no pacing at all) and max_tokens
    # fell back to the constructor default no matter what SPM_MAX_TOKENS said.
    return ChatBackend(engine, temperature, base_url=base_url,
                       structured=structured, max_tokens=max_tokens, rpm=rpm,
                       reasoning_effort=reasoning_effort)
