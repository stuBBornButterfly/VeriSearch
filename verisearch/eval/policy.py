"""The real policy behind the episode loop: a vLLM server, over HTTP.

C2. Dhrubo's `env.episode.run_episode` takes a `Policy` - one method,
`generate(prompt, *, stop, max_new_tokens) -> Generation` - and until now only
fakes implemented it. Eval and serving need the real model behind that method.
Training does not use this: it runs its own `VllmTurnPolicy` inside TRL, which
needs token ids and logprobs. Eval and serving need the text.

Three details decide whether an eval measures the trained policy at all:

* **The stop string stays in the text.** vLLM's OpenAI-compatible server strips
  the matched stop string from `text` unless asked not to. The parser needs
  `</search>` in the text to see the action; without it every search turn
  parses as an unclosed tag and every episode ends on turn one. So
  `include_stop_str_in_output` goes on every request, and it is the first thing
  the tests check.
* **Raw text in, through `/v1/completions`.** `render_prompt` produces the text
  training feeds the model, with no chat template. `/v1/chat/completions` would
  wrap it in one, and the eval would measure a different input from the one the
  policy was trained on.
* **A failed request raises.** A timeout or a 5xx scored as an empty generation
  would count as the model failing to answer, and section 9's failure breakdown
  would report infrastructure as policy behaviour. The same rule as
  `HttpRetriever`.

Greedy by default (temperature 0), so an eval is reproducible and two
checkpoints are compared on the same footing. The sampling settings are on the
object, so whatever writes the report can record them.

Owner: Ayesha.
"""

from __future__ import annotations

import json
import urllib.error
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from verisearch.env.episode import Generation, Question
from verisearch.retrieval.client import Transport, UrllibTransport

if TYPE_CHECKING:
    from collections.abc import Sequence

    from verisearch.data.datasets import QAExample
    from verisearch.env.episode import Policy

__all__ = [
    "PolicyError",
    "Sampling",
    "VllmHttpPolicy",
    "conformance_witness",
    "question_from",
]

# One generation is at most `env.max_new_tokens` (512) tokens. On a 1.5B model
# that is seconds, but a server under a GRPO-sized batch queues requests, and a
# timeout that fires on a queued request reads as a dead server.
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRIES = 1

COMPLETIONS_PATH = "/v1/completions"


class PolicyError(RuntimeError):
    """Raised when the policy server cannot produce a generation."""


class Sampling(BaseModel):
    """How the policy samples. Record it with every eval: two runs that differ
    only here are not comparable, and nothing else would say so."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    # vLLM's per-request seed. Irrelevant when greedy; with sampling it is what
    # makes a rerun produce the same text.
    seed: int | None = None


class _Choice(BaseModel):
    # vLLM returns more than this (logprobs, stop_reason, token ids). Ignored,
    # not forbidden: the server's extra fields are not this module's contract.
    model_config = ConfigDict(frozen=True, extra="ignore")

    text: str


class _Completion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    choices: tuple[_Choice, ...]


class VllmHttpPolicy:
    """Implements `env.episode.Policy` against `vllm serve`.

    `model` is the name the server serves: the base model, or a LoRA adapter's
    name when the server was started with `--lora-modules`. Holds no state that
    a call changes, so one instance is safe to share across threads, which
    `serve.server.PolicyRunner` requires.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        sampling: Sampling | None = None,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            # Checked here so the transport's own refusal, which speaks of a
            # retrieval URL, can never be what the caller sees.
            raise PolicyError(f"refusing a non-HTTP policy URL: {base_url!r}")
        if not model.strip():
            raise PolicyError("model must name what the server serves")
        if retries < 0:
            raise PolicyError(f"retries must be >= 0, got {retries}")
        if timeout <= 0:
            raise PolicyError(f"timeout must be positive, got {timeout}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sampling = sampling if sampling is not None else Sampling()
        self._transport: Transport = transport if transport is not None else UrllibTransport()
        self._timeout = timeout
        self._retries = retries

    def request_body(
        self, prompt: str, *, stop: Sequence[str], max_new_tokens: int
    ) -> dict[str, str | int | float | bool | list[str]]:
        """The JSON sent for one turn. Public so a test, or a log, can see it."""
        body: dict[str, str | int | float | bool | list[str]] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "stop": list(stop),
            # The one field that cannot be left to its default. See the module
            # docstring.
            "include_stop_str_in_output": True,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "n": 1,
        }
        if self.sampling.seed is not None:
            body["seed"] = self.sampling.seed
        return body

    def generate(self, prompt: str, *, stop: Sequence[str], max_new_tokens: int) -> Generation:
        """One turn: the prompt so far in, the model's continuation out."""
        if max_new_tokens < 1:
            raise PolicyError(f"max_new_tokens must be positive, got {max_new_tokens}")
        payload = json.dumps(
            self.request_body(prompt, stop=stop, max_new_tokens=max_new_tokens)
        ).encode("utf-8")
        raw = self._post(payload)
        try:
            completion = _Completion.model_validate_json(raw)
        except ValidationError as exc:
            raise PolicyError(
                f"{self._url} returned something that is not a completion: {exc}"
            ) from exc
        if len(completion.choices) != 1:
            raise PolicyError(
                f"{self._url} returned {len(completion.choices)} choices for n=1. Taking the "
                f"first would score a sample nobody chose."
            )
        return Generation(text=completion.choices[0].text)

    @property
    def _url(self) -> str:
        return f"{self.base_url}{COMPLETIONS_PATH}"

    def _post(self, payload: bytes) -> bytes:
        last: Exception | None = None
        for _ in range(self._retries + 1):
            try:
                return self._transport.request(self._url, payload, self._timeout)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                # A 4xx is this request's own fault - an unknown model name, a
                # prompt past the context window - and fails the same way again.
                if exc.code < 500:
                    raise PolicyError(f"{self._url} returned HTTP {exc.code}: {detail}") from exc
                last = PolicyError(f"HTTP {exc.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
        raise PolicyError(
            f"{self._url} failed after {self._retries + 1} attempt(s): {last}. Not returning an "
            f"empty generation - it would be scored as the model failing to answer."
        ) from last


def question_from(example: QAExample) -> Question:
    """The episode loop's view of a dataset example.

    Built here rather than by a `from_example` on `Question`: the conversion is
    three fields, and keeping it on this side needs nothing from `env/`.
    """
    return Question(
        question_id=example.question_id,
        question=example.question,
        gold_answers=example.answers,
    )


def conformance_witness(policy: VllmHttpPolicy) -> Policy:
    """Return the policy typed as `env.episode.Policy`, so mypy checks the fit.

    The same mechanism `verisearch.testing.fakes` uses: a conformance check
    written in `tests/` is never type-checked, because `tests/` is outside
    mypy's `files` list.
    """
    return policy
