"""Typed configuration, loaded from YAML.

PROJECT.md section 3: "Config as data. Pydantic settings from YAML. No magic
numbers in code." Every value in `configs/base.yaml` has a field here, and
every field is validated.

The validators are the point. A config file is the easiest place in this project
to introduce a silent, expensive mistake - an embedding dimension that does not
match the embedder builds an index that returns garbage, and nothing crashes.
Each `@model_validator` below encodes a rule from PROJECT.md that would
otherwise only exist in prose.

Precedence, highest first:

    1. environment variables   VERISEARCH_RETRIEVAL__SERVICE_URL=...
    2. .env file
    3. the YAML files passed to `load_config`, merged left to right

Secrets are deliberately absent. `WANDB_API_KEY` and `HF_TOKEN` are read from
the environment by wandb and huggingface_hub themselves. They must never enter
this object, because this object gets serialised into run metadata.

Owner: Ayesha.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

# Embedders whose output dimension we know. Used to catch a dim/model mismatch,
# which otherwise produces a silently broken index rather than an error.
# An embedder absent from this table is allowed - we just cannot check it.
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


class ConfigError(Exception):
    """Raised when a config cannot be loaded or violates a PROJECT.md rule.

    PROJECT.md section 3: "Fail loud. Typed exceptions."
    """


class Dtype(StrEnum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class IndexType(StrEnum):
    HNSW = "hnsw"
    FLAT = "flat"
    IVF = "ivf"


_SECTION = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(BaseModel):
    """Policy model. PROJECT.md section 5."""

    model_config = _SECTION

    policy: str
    policy_repo: str
    dtype: Dtype = Dtype.BFLOAT16
    load_in_4bit: bool = False

    @model_validator(mode="after")
    def _check_dtype_against_quantisation(self) -> Self:
        # NF4 keeps a bf16 or fp16 compute dtype; fp32 with 4-bit weights is a
        # configuration nobody wants and signals a copy-paste error.
        if self.load_in_4bit and self.dtype is Dtype.FLOAT32:
            raise ValueError("load_in_4bit with dtype float32 is not a meaningful combination")
        return self


class LoraConfig(BaseModel):
    """LoRA adapter. PROJECT.md section 5: r=16, alpha=32, dropout=0.05."""

    model_config = _SECTION

    r: int = Field(ge=1)
    alpha: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] = Field(min_length=1)


class RetrievalConfig(BaseModel):
    """Embedder and index. Ayesha's half."""

    model_config = _SECTION

    embedder: str
    embedding_dim: int = Field(ge=1)
    index_type: IndexType = IndexType.HNSW
    top_k: int = Field(ge=1)
    service_url: str

    @model_validator(mode="after")
    def _check_dim_matches_embedder(self) -> Self:
        expected = KNOWN_EMBEDDING_DIMS.get(self.embedder)
        if expected is not None and self.embedding_dim != expected:
            raise ValueError(
                f"embedder {self.embedder} outputs {expected} dims, "
                f"but embedding_dim is {self.embedding_dim}. A mismatch here builds an "
                f"index that returns nonsense without raising."
            )
        return self


class FaithfulnessConfig(BaseModel):
    """Offline NLI scorer. PROJECT.md section 5: never in the training hot loop."""

    model_config = _SECTION

    nli_model: str
    entailment_threshold: float = Field(ge=0.0, le=1.0)


class EnvConfig(BaseModel):
    """Episode limits. PROJECT.md section 6: SEARCH_BUDGET = 4."""

    model_config = _SECTION

    search_budget: int = Field(ge=1)
    max_turns: int = Field(ge=2)
    max_new_tokens: int = Field(ge=1)
    stop_sequences: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_budget_is_reachable(self) -> Self:
        # The policy needs one generation turn per search plus one to answer. If
        # max_turns is lower, the search budget is unreachable and every
        # trajectory terminates on MAX_TURNS - which looks like the model
        # refusing to answer rather than a config bug.
        required = self.search_budget + 1
        if self.max_turns < required:
            raise ValueError(
                f"max_turns={self.max_turns} cannot accommodate search_budget="
                f"{self.search_budget}: the policy needs {required} turns to spend the "
                f"budget and still answer"
            )
        return self


class PenaltyConfig(BaseModel):
    """Penalties from PROJECT.md section 6. All non-positive."""

    model_config = _SECTION

    duplicate_query: float = Field(le=0.0)
    no_answer: float = Field(le=0.0)
    empty_query: float = Field(le=0.0)


class RewardConfig(BaseModel):
    """Reward weights. PROJECT.md section 6."""

    model_config = _SECTION

    w_correctness: float = Field(ge=0.0)
    w_faithfulness: float = Field(ge=0.0)
    w_format: float = Field(ge=0.0)
    lambda_retrieval_cost: float = Field(ge=0.0)
    penalties: PenaltyConfig

    @model_validator(mode="after")
    def _check_correctness_dominates(self) -> Self:
        # PROJECT.md section 6, design rules: "Correctness must dominate." and
        # "Format reward stays small. Large format rewards produce beautifully
        # formatted wrong answers." Both are stated as prose in the brief; a
        # transposed pair of weights would otherwise pass review and quietly
        # change what the experiment measures.
        if self.w_correctness <= self.w_faithfulness:
            raise ValueError(
                f"w_correctness ({self.w_correctness}) must exceed w_faithfulness "
                f"({self.w_faithfulness}): PROJECT.md section 6 requires correctness to dominate"
            )
        if self.w_format >= self.w_faithfulness:
            raise ValueError(
                f"w_format ({self.w_format}) must stay below w_faithfulness "
                f"({self.w_faithfulness}): large format rewards buy well-formed wrong answers"
            )
        return self


class TrainConfig(BaseModel):
    """GRPO. PROJECT.md section 5."""

    model_config = _SECTION

    group_size: int = Field(ge=2)
    # 0.0 disables the KL term. PROJECT.md section 5 calls it "the main
    # anti-hacking guard", so zero is only ever a deliberate ablation - allowed,
    # but it has to be written down in a config file someone reviews.
    kl_beta: float = Field(ge=0.0)
    learning_rate: float = Field(gt=0.0)
    max_steps: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_group_size(self) -> Self:
        # GRPO's advantage is (r_i - mean(r)) / std(r) over the group. At group
        # size 1 the advantage is undefined; at 2 it is +/-1 and carries no
        # ranking information. The floor of 2 is enforced by the field; this
        # flags the degenerate-but-legal range.
        if self.group_size < 4:
            raise ValueError(
                f"group_size={self.group_size} gives a near-degenerate advantage estimate; "
                f"PROJECT.md section 5 specifies 8"
            )
        return self


class EvalConfig(BaseModel):
    """Bootstrap confidence intervals. PROJECT.md section 9."""

    model_config = _SECTION

    bootstrap_resamples: int = Field(ge=100)
    confidence_level: float = Field(gt=0.0, lt=1.0)


class LoggingConfig(BaseModel):
    """Weights & Biases. No credentials here - see the module docstring."""

    model_config = _SECTION

    wandb_project: str
    wandb_entity: str | None = None


class Config(BaseSettings):
    """The whole experiment, as data.

    Construct with `load_config()`, not directly - the loader merges YAML files
    and raises `ConfigError` instead of leaking Pydantic internals.
    """

    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        env_prefix="VERISEARCH_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    seed: int
    model: ModelConfig
    lora: LoraConfig
    retrieval: RetrievalConfig
    faithfulness: FaithfulnessConfig
    env: EnvConfig
    reward: RewardConfig
    train: TrainConfig
    eval: EvalConfig
    logging: LoggingConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],  # noqa: ARG003 - signature fixed by pydantic-settings
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Put the environment ahead of the YAML passed to `__init__`.

        Pydantic's default order puts init kwargs first, which would make a
        committed YAML file silently beat a machine-specific override. The YAML
        is the shared experiment definition; the environment is the local
        deviation from it, so the environment has to win.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    @property
    def fingerprint(self) -> str:
        """Short stable hash of every value in this config.

        PROJECT.md section 15: no number in any doc that did not come from a
        logged run. Log this next to the run so a result can be traced back to
        the exact configuration that produced it - including values that came
        from the environment and are therefore not in any committed file.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def with_lambda(self, value: float) -> Config:
        """Return a copy with a different retrieval-cost lambda.

        The lambda sweep in PROJECT.md section 6 is four runs differing in this
        one number. Everything else must be identical or the frontier plot
        compares runs that differ in more than the dial being swept.
        """
        if value < 0.0:
            raise ConfigError(f"lambda must be non-negative, got {value}")
        reward = self.reward.model_copy(update={"lambda_retrieval_cost": value})
        return self.model_copy(update={"reward": reward})


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    try:
        loaded: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"{path}: expected a mapping at the top level, got {type(loaded).__name__}"
        )
    return {str(key): value for key, value in loaded.items()}


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    """Merge `override` onto `base`, recursing into nested mappings.

    Shallow merge would mean an override file that sets one reward weight wipes
    the other three. Lists are replaced wholesale, not concatenated - appending
    to `target_modules` by accident is worse than restating it.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(
                {str(k): v for k, v in existing.items()},
                {str(k): v for k, v in value.items()},
            )
        else:
            merged[key] = value
    return merged


def load_config(*paths: str | Path) -> Config:
    """Load and validate a config.

    With no arguments, loads `configs/base.yaml`. With several, merges them left
    to right, so `load_config("configs/base.yaml", "configs/train_3b.yaml")`
    layers an override on the base. Environment variables beat both.

    Raises `ConfigError` on a missing file, malformed YAML, or any validation
    failure - callers get one exception type, not a mix of OSError, YAMLError
    and ValidationError.
    """
    targets = [Path(p) for p in paths] or [DEFAULT_CONFIG_PATH]

    merged: dict[str, object] = {}
    for path in targets:
        merged = _deep_merge(merged, _load_yaml(path))

    try:
        # mypy cannot match a `**dict[str, object]` splat against a typed
        # `__init__`, and it has to be a splat: `Config.model_validate(merged)`
        # bypasses BaseSettings' sources, which would silently drop every
        # environment override. Pydantic validates all of it at runtime - the
        # ignore costs no safety, only a static check that cannot be expressed.
        return Config(**merged)  # type: ignore[arg-type]
    except ValueError as exc:
        # Pydantic's ValidationError subclasses ValueError. Chaining keeps the
        # per-field detail while giving callers one type to catch.
        joined = ", ".join(str(p) for p in targets)
        raise ConfigError(f"invalid config from [{joined}]:\n{exc}") from exc


# ---------------------------------------------------------------------------
# Lambda sweep. PROJECT.md section 6 and the P4 exit criterion.
# ---------------------------------------------------------------------------


class SweepPoint(BaseModel):
    """One run of the sweep: a lambda, a run name, and the config to run."""

    model_config = _SECTION

    lambda_value: float = Field(ge=0.0)
    run_name: str = Field(min_length=1)
    config: Config


class SweepSpec(BaseModel):
    """The lambda sweep, as data.

    This is not a `Config` overlay and must not be loaded as one. A single
    config holds one lambda; the sweep is N runs that differ in exactly that
    number, so the manifest lists the lambdas and the base files they share.

    Typed rather than left as an array in `scripts/04_sweep.sh` because the
    failure it prevents is expensive: a sweep launched with a duplicated or
    missing lambda produces a frontier plot with a point missing or two points
    on top of each other, and you find out after the GPU hours are spent.
    """

    model_config = _SECTION

    name: str = Field(min_length=1)
    # Composed left to right by `load_config`, so later files win.
    base: tuple[Path, ...] = Field(min_length=1)
    lambdas: tuple[float, ...] = Field(min_length=2)
    output_dir: Path

    @model_validator(mode="after")
    def _check_lambdas(self) -> Self:
        negative = [value for value in self.lambdas if value < 0.0]
        if negative:
            # `Field(min_length=2)` constrains the tuple, not its elements, so
            # without this a negative lambda loads cleanly here and only fails
            # later inside `with_lambda` - after the manifest has been reviewed
            # and the sweep launched.
            raise ValueError(
                f"lambdas must be non-negative, got {negative}; a negative retrieval cost "
                f"pays the policy to search, which is the opposite of the experiment"
            )
        if len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError(
                f"lambdas must be distinct, got {list(self.lambdas)}; a repeated value "
                f"spends a training run producing a point that is already on the plot"
            )
        if 0.0 not in self.lambdas:
            # The frontier shows accuracy traded against retrieval count. Every
            # other point is read relative to the un-penalised run, so without
            # it the plot has no anchor and the trade-off cannot be quantified.
            raise ValueError(
                f"lambdas must include 0.0 as the control point, got {list(self.lambdas)}"
            )
        if list(self.lambdas) != sorted(self.lambdas):
            raise ValueError(
                f"lambdas must be ascending, got {list(self.lambdas)}; the frontier is read "
                f"left to right and a shuffled manifest makes run names sort wrongly"
            )
        return self


def load_sweep(path: str | Path) -> SweepSpec:
    """Read a sweep manifest. Same error contract as `load_config`."""
    target = Path(path)
    try:
        return SweepSpec(**_load_yaml(target))  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfigError(f"invalid sweep manifest {target}:\n{exc}") from exc


def sweep_points(spec: SweepSpec, *, root: str | Path | None = None) -> tuple[SweepPoint, ...]:
    """Expand a manifest into one ready-to-run config per lambda.

    `spec.base` is loaded **once** and then varied with `Config.with_lambda`, so
    the N configs are structurally identical apart from that one float. Loading
    the YAML N times would admit a difference between runs if a file changed
    mid-sweep, which is exactly the kind of thing that is invisible afterwards.

    `root` resolves the manifest's relative paths; it defaults to the current
    directory, matching how `load_config` is called from the repo root.
    """
    base_dir = Path(root) if root is not None else Path()
    shared = load_config(*[base_dir / path for path in spec.base])
    return tuple(
        SweepPoint(
            lambda_value=value,
            # `:g` so 0.0 reads as "0" and 0.02 as "0.02" - these names become
            # W&B run names and directory names, and "lam0.000000" is unreadable.
            run_name=f"{spec.name}-lam{value:g}",
            config=shared.with_lambda(value),
        )
        for value in spec.lambdas
    )
