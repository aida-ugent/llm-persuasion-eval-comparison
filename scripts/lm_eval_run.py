"""lm-eval's CLI with patches applied in memory: null chat content becomes "" (scored wrong),
request fields are adjusted per provider by model-id prefix, and Anthropic replies keep only their
text blocks. Same arguments as `lm_eval`. Called by scripts/capability_run.sh.
"""
import os
import sys

import requests
from lm_eval.models import anthropic_llms, openai_completions

DEBUG = bool(os.environ.get("LM_EVAL_DEBUG"))


def _debug(p):
    if DEBUG:
        print("[payload]", {k: v for k, v in p.items() if k != "messages"}, file=sys.stderr)
    return p


# print the provider's error text on a 4xx/5xx
_orig_rfs = requests.Response.raise_for_status


def _rfs(self):
    if self.status_code >= 400:
        print(f"[http {self.status_code}] {self.text[:300]}", file=sys.stderr)
    return _orig_rfs(self)


requests.Response.raise_for_status = _rfs

# null content
_orig_parse = openai_completions.LocalChatCompletion.parse_generations


@staticmethod
def _parse_chat(outputs, **kwargs):
    if not isinstance(outputs, list):
        outputs = [outputs]
    for out in outputs:
        for c in out.get("choices", []):
            if c.get("message", {}).get("content") is None:
                c["message"]["content"] = ""
    return _orig_parse(outputs, **kwargs)


openai_completions.LocalChatCompletion.parse_generations = _parse_chat
openai_completions.OpenAIChatCompletion.parse_generations = _parse_chat


# provider request fields
def _wrap_chat_payload(cls):
    orig = cls._create_payload

    def _payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kw):
        p = orig(self, messages, generate=generate, gen_kwargs=gen_kwargs, seed=seed, eos=eos, **kw)
        if str(self.model).startswith("gpt-5"):
            # max_completion_tokens instead of max_tokens; no temperature, seed or stop
            if "max_tokens" in p:
                p["max_completion_tokens"] = p.pop("max_tokens")
            for k in ("temperature", "seed", "stop"):
                p.pop(k, None)
        if str(self.model).startswith(("mistral", "kimi")):
            p.pop("seed", None)   # not accepted by Mistral or Moonshot
        if str(self.model).startswith("kimi"):
            p.pop("temperature", None)   # Kimi accepts only its default temperature
        if str(self.model).startswith(("kimi", "mistral")):
            # drop empty assistant turns and merge adjacent same-role turns
            folded = []
            for m in p["messages"]:
                if m.get("role") == "assistant" and not (m.get("content") or "").strip():
                    continue
                if folded and folded[-1]["role"] == m["role"]:
                    folded[-1] = {"role": m["role"], "content": folded[-1]["content"] + "\n\n" + m["content"]}
                else:
                    folded.append({"role": m["role"], "content": m["content"]})
            p["messages"] = folded
        return _debug(p)

    cls._create_payload = _payload


_wrap_chat_payload(openai_completions.LocalChatCompletion)
_wrap_chat_payload(openai_completions.OpenAIChatCompletion)

_orig_anthropic = anthropic_llms.AnthropicChat._create_payload


def _payload_anthropic(self, messages, generate=True, gen_kwargs=None, eos="\n\nHuman:", **kw):
    p = _orig_anthropic(self, messages, generate=generate, gen_kwargs=gen_kwargs, eos=eos, **kw)
    p.pop("temperature", None)   # not accepted by claude-fable
    # the Messages API accepts only role and content per message
    p["messages"] = [{"role": m["role"], "content": m["content"]} for m in p["messages"]]
    return _debug(p)


anthropic_llms.AnthropicChat._create_payload = _payload_anthropic


def _parse_anthropic(self, outputs, **kwargs):
    if not isinstance(outputs, list):
        outputs = [outputs]
    return ["".join(c.get("text", "") for c in out.get("content", []) if c.get("type") == "text")
            for out in outputs]


anthropic_llms.AnthropicChat.parse_generations = _parse_anthropic


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate
    sys.exit(cli_evaluate())
