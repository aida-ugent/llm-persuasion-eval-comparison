"""
Model clients: one interface over OpenAI-compatible APIs, Anthropic and local vLLM.
Records the served model, shares one concurrency cap, retries transient errors, detects refusals.

    client = get_client("llama-3.3-70b", cfg)
    r = client.chat(messages, max_tokens=300, temperature=0.7)
    r.text, r.served_model, r.prompt_tokens, r.completion_tokens
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MAX_CONCURRENT = 8

# Minimum completion cap for a reasoning model, whose reasoning tokens count against it.
REASONING_TOKEN_FLOOR = 512

# Largest budget an empty, budget-exhausted reply is retried at (512 -> 2048 -> 8192).
REASONING_TOKEN_CEILING = 8192

# Seconds per request before the SDK raises APITimeoutError.
REQUEST_TIMEOUT = 300

# A refusal: a first-person refusal verb followed by an assistance object, near the start.
_REFUSAL_VERBS = (
    "can't", "cannot", "can not", "won't", "will not",
    "unable to", "not able to", "don't feel comfortable", "do not feel comfortable",
)
_ASSIST_OBJECTS = (
    "help", "assist", "do that", "do this", "comply", "provide", "create", "write",
    "generate", "engage", "participate", "continue", "fulfil", "fulfill", "take part",
    "support you", "writing", "helping", "assisting", "creating", "generating",
)
_EXPLICIT_DECLINES = (
    "i must decline", "i have to decline", "i must refuse", "i have to refuse",
    "i will have to decline", "i'd rather not help", "i would rather not help",
)

_REFUSAL_WINDOW = 300


def is_refusal(text: str) -> bool:
    """True if the model declined the task."""
    if not text or not text.strip():
        return False          # empty is a parse failure, not a refusal
    head = text.strip()[:_REFUSAL_WINDOW].lower().replace("’", "'")

    if any(p in head for p in _EXPLICIT_DECLINES):
        return True

    for verb in _REFUSAL_VERBS:
        start = head.find(verb)
        while start != -1:
            # must be first person
            before = " " + head[max(0, start - 16):start]
            stripped = before.rstrip()
            first_person = (" i " in before or stripped.endswith(" i")
                            or stripped.endswith(" i'm") or stripped.endswith(" i am"))
            if first_person:
                after = head[start + len(verb): start + len(verb) + 24]
                # "I can't help but ..." is an idiom, not a decline
                if any(obj in after for obj in _ASSIST_OBJECTS) and " help but" not in after:
                    return True
            start = head.find(verb, start + 1)
    return False


@dataclass
class Response:
    text: str
    served_model: str            # as reported by the provider
    requested_model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    rate_limited: bool = False
    temperature_sent: float | None = None   # None = not sent
    temperature_honoured: bool = True       # False = rejected by the provider and dropped
    token_budget: int = 0                   # completion cap actually sent
    budget_raised: bool = False             # True = raised to REASONING_TOKEN_FLOOR
    finish_reason: str | None = None        # provider's stop/finish reason, verbatim

    # stop reasons of an API-layer refusal, which returns no text
    API_REFUSAL = ("refusal", "content_filter")

    @property
    def refused(self) -> bool:
        return is_refusal(self.text) or (self.finish_reason in self.API_REFUSAL)


class ModelChangedError(RuntimeError):
    """A provider served a different model than before."""


# The model each alias served at first contact, asserted across the whole collection.
_LOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "results" / "_model_lock.json"


def _read_lock() -> dict:
    try:
        return json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_lock(d: dict):
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_PATH.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")


class _Base:
    """Shared retry, concurrency and served-model tracking."""

    # class attribute: one cap over all in-flight calls
    _sem = threading.BoundedSemaphore(DEFAULT_MAX_CONCURRENT)
    _lock = threading.Lock()

    def __init__(self, alias: str, model_id: str, is_pinned: bool = False,
                 reasoning: str | None = None):
        self.alias = alias
        self.model_id = model_id
        self.is_pinned = is_pinned
        self.reasoning = reasoning
        self._warned_budget = False
        self._served: str | None = None
        self.stats = dict(calls=0, retries=0, rate_limits=0, connection_errors=0,
                          refusals=0)

    def _note_served(self, served: str | None, fingerprint: str | None = None):
        """Assert the served id matches first contact; record any new fingerprint."""
        if not served:
            return
        with self._lock:
            if self._served is None:
                self._served = served
            elif served != self._served:
                raise ModelChangedError(
                    f"{self.alias}: provider served '{served}' but had been serving "
                    f"'{self._served}' in this run. An alias was re-pointed; results before and "
                    f"after are different models. Pin a dated snapshot id in models.yaml.")

            lock = _read_lock()
            rec = lock.get(self.alias)
            if rec is None:
                lock[self.alias] = dict(
                    requested=self.model_id, served=served,
                    fingerprints=[fingerprint] if fingerprint else [],
                    pinned=self.is_pinned,
                    first_seen=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
                _write_lock(lock)
                return

            if rec.get("served") != served:
                scope = (f"a PINNED agent - it appears in most of the 135 runs, so this "
                         f"contaminates most of the matrix, not one row"
                         if self.is_pinned else "one tested model, so one row of the matrix")
                raise ModelChangedError(
                    f"{self.alias}: provider now serves '{served}' but served "
                    f"'{rec['served']}' at first contact ({rec.get('first_seen')}). This is "
                    f"{scope}. Runs already collected are NOT comparable with new ones. "
                    f"Delete {_LOCK_PATH.name} only if you intend to start the collection over.")

            if fingerprint and fingerprint not in (rec.get("fingerprints") or []):
                rec.setdefault("fingerprints", []).append(fingerprint)
                lock[self.alias] = rec
                _write_lock(lock)
                print(f"  [WARN] {self.alias}: new system_fingerprint '{fingerprint}' "
                      f"(model id unchanged). The serving backend rotated mid-collection; "
                      f"recorded in {_LOCK_PATH.name}.")

    @property
    def served_model(self) -> str | None:
        return self._served

    # reasoning settings that get no token floor
    _NO_REASONING = ("none", "off", "false", "disabled")

    def _exhausted(self, resp) -> bool:
        """True if the reply is empty and used the whole token budget."""
        return (not resp.text.strip()
                and resp.token_budget > 0
                and resp.completion_tokens >= resp.token_budget)

    def _escalate(self, budget: int) -> int | None:
        """A strictly larger budget to retry at, or None at the ceiling."""
        if budget >= REASONING_TOKEN_CEILING:
            return None
        return min(max(budget * 4, REASONING_TOKEN_FLOOR), REASONING_TOKEN_CEILING)

    def _budget(self, max_tokens: int, respect_cap: bool = False) -> tuple[int, bool]:
        """The completion cap to send; respect_cap=True keeps an upstream cap unraised."""
        if respect_cap or str(self.reasoning or "none").lower() in self._NO_REASONING or max_tokens >= REASONING_TOKEN_FLOOR:
            return max_tokens, False
        if not self._warned_budget:
            self._warned_budget = True
            print(f"  [note] {self.alias}: reasoning={self.reasoning}, so the completion cap is "
                  f"raised {max_tokens} -> {REASONING_TOKEN_FLOOR} (reasoning tokens count "
                  f"against it; at {max_tokens} this model returns empty text).")
        return REASONING_TOKEN_FLOOR, True

    def _call_with_retry(self, fn, max_attempts=8):
        """Retry rate limits, timeouts, connection errors and 5xx with backoff; raise on 4xx."""
        # anthropic's exception classes do not subclass openai's, so both families are caught
        import anthropic
        import openai
        RATE = (openai.RateLimitError, openai.APITimeoutError,
                anthropic.RateLimitError, anthropic.APITimeoutError)
        CONN = (openai.APIConnectionError, anthropic.APIConnectionError)
        ANY = (openai.APIError, anthropic.APIError)
        last, retries, limited = None, 0, False
        for attempt in range(max_attempts):
            try:
                with self._sem:
                    r = fn()
                self.stats["calls"] += 1
                self.stats["retries"] += retries
                return r, retries, limited
            except RATE as e:
                last, retries, limited = e, retries + 1, True
                self.stats["rate_limits"] += 1
                wait = min(2 ** attempt * 2, 120)
                print(f"  [{self.alias}] rate limit, waiting {wait}s "
                      f"({attempt + 1}/{max_attempts})", flush=True)
                time.sleep(wait)
            except CONN as e:
                # APITimeoutError subclasses this and is caught above
                last, retries = e, retries + 1
                self.stats["connection_errors"] += 1
                wait = min(2 ** attempt * 2, 60)
                print(f"  [{self.alias}] connection error ({type(e).__name__}: {str(e)[:100]}), "
                      f"waiting {wait}s ({attempt + 1}/{max_attempts})", flush=True)
                time.sleep(wait)
            except ANY as e:
                # retry 5xx; raise 4xx immediately
                status = getattr(e, "status_code", None)
                if status and 500 <= status < 600 and attempt < max_attempts - 1:
                    last, retries = e, retries + 1
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise
        raise RuntimeError(
            f"{self.alias}: {max_attempts} attempts exhausted ({type(last).__name__}). Raising "
            f"rather than returning empty text, which an adapter would score as a refusal."
        ) from last

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0,
             respect_cap: bool = False, _retry_ok: bool = True) -> Response:
        """Every provider client implements this exact signature."""
        raise NotImplementedError

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.0) -> Response:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        return self.chat(msgs, max_tokens=max_tokens, temperature=temperature)


class _OpenAICompatible(_Base):
    """OpenAI, DeepSeek, Moonshot, Mistral and local vLLM."""

    def __init__(self, alias, model_id, base_url=None, api_key_env=None,
                 is_pinned=False, extra_body=None, reasoning=None, strict_alternation=False):
        super().__init__(alias, model_id, is_pinned, reasoning)
        self.strict_alternation = bool(strict_alternation)
        from openai import OpenAI
        key = os.environ.get(api_key_env, "EMPTY") if api_key_env else "EMPTY"
        # no SDK retries: _call_with_retry retries and counts them
        kw = dict(api_key=key, timeout=REQUEST_TIMEOUT, max_retries=0)
        self.client = OpenAI(base_url=base_url, **kw) if base_url else OpenAI(**kw)
        self.extra_body = extra_body or {}
        # changed by _adapt() when the provider rejects a parameter
        self._tok_param = "max_tokens"
        self._temp_ok = True

    def _adapt(self, err) -> bool:
        """On a 400 that names a parameter we sent, rename or drop it and return True."""
        msg = str(err).lower()
        if "400" not in msg and "invalid_request" not in msg:
            return False
        changed = False
        if "max_tokens" in msg and self._tok_param == "max_tokens":
            self._tok_param = "max_completion_tokens"
            changed = True
        if "temperature" in msg and self._temp_ok:
            self._temp_ok = False
            print(f"  [WARN] {self.alias}: provider rejected our temperature "
                  f"({str(err)[:90]}...). Dropped; recorded as not honoured.")
            changed = True
        return changed

    @staticmethod
    def _alternate(messages: list[dict]) -> list[dict]:
        """Fold messages into strict user/assistant alternation; later system turns become user."""
        out: list[dict] = []
        for i, m in enumerate(messages):
            role, content = m["role"], m["content"]
            if role == "system" and i > 0:
                role = "user"
            if out and out[-1]["role"] == role:
                out[-1] = {"role": role, "content": out[-1]["content"] + chr(10) * 2 + content}
            else:
                out.append({"role": role, "content": content})
        return out

    def chat(self, messages, max_tokens=512, temperature=0.0, respect_cap=False,
             _retry_ok=True) -> Response:
        budget, raised = self._budget(max_tokens, respect_cap)
        if self.strict_alternation:
            messages = self._alternate(messages)

        def go():
            kw = {self._tok_param: budget}
            if self.extra_body:
                kw["extra_body"] = self.extra_body
            if self._temp_ok and temperature is not None:
                kw["temperature"] = temperature      # None = provider default
            return self.client.chat.completions.create(
                model=self.model_id, messages=messages, **kw)

        # a provider may reject several parameters; adapt one rejection per attempt
        for _ in range(3):
            try:
                r, retries, limited = self._call_with_retry(go)
                break
            except Exception as e:
                if not self._adapt(e):
                    raise
        else:
            r, retries, limited = self._call_with_retry(go)
        served = getattr(r, "model", None)
        self._note_served(served, getattr(r, "system_fingerprint", None))
        text = (r.choices[0].message.content or "")
        resp = Response(text=text, served_model=served or self.model_id,
                        requested_model=self.model_id,
                        prompt_tokens=getattr(r.usage, "prompt_tokens", 0) if r.usage else 0,
                        completion_tokens=getattr(r.usage, "completion_tokens", 0) if r.usage else 0,
                        retries=retries, rate_limited=limited,
                        temperature_sent=temperature if (self._temp_ok and temperature is not None)
                        else None,
                        temperature_honoured=self._temp_ok,
                        token_budget=budget, budget_raised=raised,
                        finish_reason=getattr(r.choices[0], "finish_reason", None)
                        if getattr(r, "choices", None) else None)
        if _retry_ok and self._exhausted(resp):
            bigger = self._escalate(budget)
            if bigger is None:
                print(f"  [WARN] {self.alias}: spent all {resp.token_budget} completion tokens "
                      f"with no visible text, at the {REASONING_TOKEN_CEILING} ceiling. Recorded "
                      f"as a parse failure rather than retried further.")
                return resp
            print(f"  [note] {self.alias}: spent all {resp.token_budget} completion tokens with "
                  f"no visible text. Retrying at {bigger} (up to the ceiling).")
            self.reasoning = "measured"
            # the recursion ends when _escalate returns None at the ceiling
            return self.chat(messages, max_tokens=bigger, temperature=temperature,
                             respect_cap=respect_cap, _retry_ok=True)
        if resp.refused:
            self.stats["refusals"] += 1
        return resp


class _Anthropic(_Base):
    def __init__(self, alias, model_id, is_pinned=False, reasoning=None):
        super().__init__(alias, model_id, is_pinned, reasoning)
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                                          timeout=REQUEST_TIMEOUT, max_retries=0)

    def chat(self, messages, max_tokens=512, temperature=0.0, respect_cap=False,
             _retry_ok=True) -> Response:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        if not turns:
            # the Messages API needs a user turn, so a system-only prompt is sent as one
            turns, system = [{"role": "user", "content": system}], ""
        budget, raised = self._budget(max_tokens, respect_cap)

        def go():
            # no temperature is sent; the Response records it as not honoured
            kw = dict(model=self.model_id, max_tokens=budget, messages=turns)
            if system.strip():
                kw["system"] = system        # omitted when empty; system=None is rejected
            return self.client.messages.create(**kw)
        r, retries, limited = self._call_with_retry(go)
        served = getattr(r, "model", None)
        self._note_served(served)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        resp = Response(text=text, served_model=served or self.model_id,
                        requested_model=self.model_id,
                        prompt_tokens=getattr(r.usage, "input_tokens", 0),
                        completion_tokens=getattr(r.usage, "output_tokens", 0),
                        retries=retries, rate_limited=limited,
                        temperature_sent=None, temperature_honoured=False,
                        token_budget=budget, budget_raised=raised,
                        finish_reason=getattr(r, "stop_reason", None))
        if not text.strip():
            print(f"  [note] {self.alias}: empty reply, stop_reason={resp.finish_reason!r}, "
                  f"blocks={[getattr(b, 'type', '?') for b in r.content]}")
        if _retry_ok and self._exhausted(resp):
            bigger = self._escalate(budget)
            if bigger is None:
                print(f"  [WARN] {self.alias}: spent all {resp.token_budget} completion tokens "
                      f"with no visible text, at the {REASONING_TOKEN_CEILING} ceiling. Recorded "
                      f"as a parse failure rather than retried further.")
                return resp
            print(f"  [note] {self.alias}: spent all {resp.token_budget} completion tokens with "
                  f"no visible text. Retrying at {bigger} (up to the ceiling).")
            self.reasoning = "measured"
            # the recursion ends when _escalate returns None at the ceiling
            return self.chat(messages, max_tokens=bigger, temperature=temperature,
                             respect_cap=respect_cap, _retry_ok=True)
        if resp.refused:
            self.stats["refusals"] += 1
        return resp


_PROVIDERS = {
    "openai":    dict(base_url=None, key="OPENAI_API_KEY"),
    "deepseek":  dict(base_url="https://api.deepseek.com/v1", key="DEEPSEEK_API_KEY"),
    "moonshot":  dict(base_url="https://api.moonshot.ai/v1",  key="MOONSHOT_API_KEY"),
    "mistral":   dict(base_url="https://api.mistral.ai/v1",   key="MISTRAL_API_KEY"),
}

_CACHE: dict[str, _Base] = {}


def route_of(alias: str, cfg: dict) -> tuple[str, dict]:
    """Which arm serves this alias ("local" or "api"), and the entry to serve it from.

    A `local:` entry with `route: api` is served from its nested `api:` block.
    """
    local = cfg["models"].get("local", {})
    # pinned agents are served by API
    api = {**cfg["models"].get("api", {}), **cfg["models"].get("pinned_agents", {})}
    if alias in local:
        m = local[alias]
        if m.get("route", "local") == "api":
            nested = m.get("api") or {}
            if not nested.get("provider"):
                raise ValueError(f"{alias}: route is api but no api.provider is declared")
            return "api", {**m, **nested}
        return "local", m
    if alias in api:
        return "api", api[alias]
    raise ValueError(f"unknown model alias '{alias}'")


def get_client(alias: str, cfg: dict, is_pinned: bool = False) -> _Base:
    """Build the client for a model alias, or return the cached one."""
    if alias in _CACHE:
        return _CACHE[alias]

    arm, m = route_of(alias, cfg)

    if arm == "local":
        port = m.get("port") or os.environ.get("LOCAL_API_PORT", "8000")
        base = os.environ.get("LOCAL_API_BASE_URL", f"http://localhost:{port}/v1")
        # extra_body: vLLM-only request fields from models.yaml, e.g. Qwen3's enable_thinking
        c = _OpenAICompatible(alias, m["id"], base_url=base, is_pinned=is_pinned,
                              reasoning=m.get("reasoning"), extra_body=m.get("extra_body"),
                              strict_alternation=m.get("strict_alternation", False))
    else:
        if m.get("id") in (None, "TBD"):
            raise ValueError(
                f"{alias}: id is TBD in models.yaml. Fill in a dated snapshot id before running "
                f"- a bare alias can be re-pointed mid-collection.")
        prov = m.get("provider")
        if prov == "anthropic":
            c = _Anthropic(alias, m["id"], is_pinned, reasoning=m.get("reasoning"))
        elif prov in _PROVIDERS:
            p = _PROVIDERS[prov]
            c = _OpenAICompatible(alias, m["id"], base_url=p["base_url"],
                                  api_key_env=p["key"], is_pinned=is_pinned,
                                  reasoning=m.get("reasoning"), extra_body=m.get("extra_body"))
        else:
            raise ValueError(f"{alias}: unknown provider '{prov}'")

    _CACHE[alias] = c
    return c


def set_max_concurrent(n: int):
    """Set the shared cap on in-flight calls."""
    _Base._sem = threading.BoundedSemaphore(n)


def reset_cache():
    _CACHE.clear()
