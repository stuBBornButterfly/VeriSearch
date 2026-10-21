"""Tests for `verisearch/config.py`.

Most of these exercise validators rather than loading. That is deliberate: a bad
config does not crash, it produces a run whose numbers are quietly wrong, and
PROJECT.md section 15 forbids reporting a number we cannot trace. Each test
below corresponds to a rule stated as prose somewhere in PROJECT.md.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

from verisearch.config import (
    _YAML_LAYER,
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    Dtype,
    EnvConfig,
    IndexType,
    ModelConfig,
    PenaltyConfig,
    RetrievalConfig,
    RewardConfig,
    ServeConfig,
    TrainConfig,
    load_config,
    load_sweep,
    sweep_points,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    WriteConfig = Callable[[dict[str, object]], Path]


@pytest.fixture
def base_dict() -> dict[str, object]:
    """`configs/base.yaml` as a fresh mutable dict."""
    loaded = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


@pytest.fixture
def write_config(tmp_path: Path) -> WriteConfig:
    """Factory writing a dict to a YAML file and returning its path."""
    counter = {"n": 0}

    def _write(data: dict[str, object]) -> Path:
        counter["n"] += 1
        path = tmp_path / f"cfg{counter['n']}.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


def _penalties() -> PenaltyConfig:
    return PenaltyConfig(duplicate_query=-0.1, no_answer=-0.5, empty_query=-0.05)


def _reward(**overrides: float) -> RewardConfig:
    fields: dict[str, float] = {
        "w_correctness": 1.0,
        "w_faithfulness": 0.5,
        "w_format": 0.2,
        "lambda_retrieval_cost": 0.05,
    }
    fields.update(overrides)
    return RewardConfig(penalties=_penalties(), **fields)


# ---------------------------------------------------------------------------
# The committed config must load and must say what PROJECT.md says
# ---------------------------------------------------------------------------


def test_base_yaml_loads() -> None:
    """If this fails, every script in the repo fails."""
    config = load_config()
    assert isinstance(config, Config)


def test_base_yaml_matches_project_md() -> None:
    """Spot-checks against PROJECT.md sections 5 and 6, not a restatement of the file."""
    config = load_config()
    assert config.seed == 42
    assert config.model.policy_repo == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.model.dtype is Dtype.BFLOAT16
    assert config.lora.r == 16
    assert config.lora.alpha == 32
    assert config.retrieval.embedding_dim == 384
    assert config.retrieval.index_type is IndexType.HNSW
    assert config.env.search_budget == 4
    assert config.reward.w_correctness == 1.0
    assert config.reward.w_faithfulness == 0.5
    assert config.reward.w_format == 0.2
    assert config.train.group_size == 8
    assert config.train.kl_beta == 0.04


def test_config_is_frozen() -> None:
    config = load_config()
    with pytest.raises(ValidationError):
        config.seed = 7  # type: ignore[misc]


def _all_keys(value: object) -> set[str]:
    """Every field name at every depth of a dumped config."""
    if isinstance(value, dict):
        keys: set[str] = set()
        for key, nested in value.items():
            keys.add(str(key))
            keys |= _all_keys(nested)
        return keys
    if isinstance(value, list):
        return {k for item in value for k in _all_keys(item)}
    return set()


def test_no_credential_fields_on_the_config_object() -> None:
    """Secrets must not be reachable from an object that gets logged to W&B.

    Checks field names exactly rather than searching the serialised blob -
    `max_new_tokens` legitimately contains "token".
    """
    forbidden = {
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "credentials",
        "wandb_api_key",
        "hf_token",
    }
    assert not (_all_keys(load_config().model_dump(mode="json")) & forbidden)


def test_credentials_in_the_environment_do_not_leak_into_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The env prefix is VERISEARCH_, so unprefixed secrets cannot be picked up.

    This is the mechanism that keeps keys out of run metadata. If someone adds
    a credential field or widens the prefix, this fails.
    """
    monkeypatch.setenv("WANDB_API_KEY", "sentinel-wandb-key")
    monkeypatch.setenv("HF_TOKEN", "sentinel-hf-token")
    dumped = load_config().model_dump_json()
    assert "sentinel-wandb-key" not in dumped
    assert "sentinel-hf-token" not in dumped


# ---------------------------------------------------------------------------
# Loading, merging, failure modes
# ---------------------------------------------------------------------------


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("seed: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_unreadable_file_raises_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that exists but cannot be read is still a ConfigError, not an OSError.

    Callers get one exception type. PROJECT.md section 3: "Fail loud."
    """
    path = tmp_path / "locked.yaml"
    path.write_text("seed: 42", encoding="utf-8")

    def _boom(*args: object, **kwargs: object) -> str:
        raise PermissionError("locked by another process")

    monkeypatch.setattr(Path, "read_text", _boom)
    with pytest.raises(ConfigError, match="could not read"):
        load_config(path)


def test_non_mapping_top_level_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_config(path)


def test_empty_file_raises_config_error_for_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid config"):
        load_config(path)


def test_unknown_key_is_rejected(base_dict: dict[str, object], write_config: WriteConfig) -> None:
    """extra="forbid" - a typo in a config key must not be silently ignored."""
    base_dict["sed"] = 42
    path = write_config(base_dict)
    with pytest.raises(ConfigError, match="invalid config"):
        load_config(path)


def test_override_merges_deeply(base_dict: dict[str, object], write_config: WriteConfig) -> None:
    """Setting one reward weight must not wipe the other three."""
    base_path = write_config(base_dict)
    override_path = write_config({"reward": {"lambda_retrieval_cost": 0.1}})

    config = load_config(base_path, override_path)

    assert config.reward.lambda_retrieval_cost == 0.1
    assert config.reward.w_correctness == 1.0
    assert config.reward.w_faithfulness == 0.5
    assert config.reward.penalties.no_answer == -0.5


def test_override_replaces_lists_wholesale(
    base_dict: dict[str, object], write_config: WriteConfig
) -> None:
    """Appending to target_modules by accident is worse than restating it."""
    base_path = write_config(base_dict)
    override_path = write_config({"lora": {"target_modules": ["q_proj"]}})

    config = load_config(base_path, override_path)

    assert config.lora.target_modules == ("q_proj",)


def test_environment_beats_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """The YAML is the shared experiment; the environment is the local deviation."""
    monkeypatch.setenv("VERISEARCH_RETRIEVAL__SERVICE_URL", "http://10.0.0.5:9000")
    assert load_config().retrieval.service_url == "http://10.0.0.5:9000"


def test_the_yaml_layer_does_not_outlive_the_load(
    base_dict: dict[str, object], write_config: WriteConfig
) -> None:
    """`load_config` hands the YAML to its settings source through a context
    variable. Left set - after a success or a failure - the next construction
    would quietly read the previous experiment."""
    load_config()
    assert _YAML_LAYER.get() is None
    base_dict["not_a_section"] = 1
    with pytest.raises(ConfigError):
        load_config(write_config(base_dict))
    assert _YAML_LAYER.get() is None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_and_sensitive() -> None:
    """PROJECT.md section 15: a result must be traceable to the exact config."""
    config = load_config()
    assert config.fingerprint == load_config().fingerprint
    assert len(config.fingerprint) == 12
    assert config.fingerprint != config.with_lambda(0.1).fingerprint


def test_with_lambda_leaves_the_original_untouched() -> None:
    config = load_config()
    swept = config.with_lambda(0.1)
    assert swept.reward.lambda_retrieval_cost == 0.1
    assert config.reward.lambda_retrieval_cost == 0.05


def test_with_lambda_changes_nothing_else() -> None:
    """The frontier plot is only meaningful if lambda is the single difference."""
    config = load_config()
    swept = config.with_lambda(0.1)

    before = config.model_dump(mode="json")
    after = swept.model_dump(mode="json")
    before["reward"]["lambda_retrieval_cost"] = None
    after["reward"]["lambda_retrieval_cost"] = None
    assert before == after


@pytest.mark.parametrize("value", [0.0, 0.02, 0.05, 0.1])
def test_with_lambda_accepts_every_sweep_point(value: float) -> None:
    """The four values in PROJECT.md section 6."""
    assert load_config().with_lambda(value).reward.lambda_retrieval_cost == value


def test_with_lambda_rejects_negative() -> None:
    with pytest.raises(ConfigError, match="non-negative"):
        load_config().with_lambda(-0.01)


# ---------------------------------------------------------------------------
# Validators. Each one guards a rule PROJECT.md states only in prose.
# ---------------------------------------------------------------------------


def test_embedding_dim_must_match_a_known_embedder() -> None:
    """A mismatch builds an index that returns nonsense without raising."""
    with pytest.raises(ValidationError, match="384 dims"):
        RetrievalConfig(
            embedder="BAAI/bge-small-en-v1.5",
            embedding_dim=768,
            top_k=3,
            service_url="http://127.0.0.1:8000",
        )


def test_unknown_embedder_is_allowed() -> None:
    """We cannot check a model we do not know, and we must not block it."""
    config = RetrievalConfig(
        embedder="some-org/some-new-embedder",
        embedding_dim=512,
        top_k=3,
        service_url="http://127.0.0.1:8000",
    )
    assert config.embedding_dim == 512


@pytest.mark.parametrize(
    ("search_budget", "max_turns", "ok"),
    [
        pytest.param(4, 6, True, id="base-config"),
        pytest.param(4, 5, True, id="exactly-enough"),
        pytest.param(4, 4, False, id="budget-unreachable"),
        pytest.param(8, 6, False, id="budget-far-too-large"),
    ],
)
def test_max_turns_must_accommodate_the_search_budget(
    search_budget: int, max_turns: int, ok: bool
) -> None:
    """Otherwise every trajectory ends on MAX_TURNS and it looks like the model
    refusing to answer rather than a config bug."""
    kwargs = {
        "search_budget": search_budget,
        "max_turns": max_turns,
        "max_new_tokens": 512,
        "stop_sequences": ("</search>", "</answer>"),
    }
    if ok:
        assert EnvConfig(**kwargs).max_turns == max_turns  # type: ignore[arg-type]
    else:
        with pytest.raises(ValidationError, match="cannot accommodate"):
            EnvConfig(**kwargs)  # type: ignore[arg-type]


def test_stop_sequences_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        EnvConfig(search_budget=4, max_turns=6, max_new_tokens=512, stop_sequences=())


def test_correctness_must_dominate_faithfulness() -> None:
    """PROJECT.md section 6, design rules."""
    with pytest.raises(ValidationError, match="must exceed w_faithfulness"):
        _reward(w_correctness=0.4)


def test_transposed_weights_are_caught() -> None:
    """The realistic failure: someone swaps two numbers while editing YAML."""
    with pytest.raises(ValidationError, match="correctness to dominate"):
        _reward(w_correctness=0.5, w_faithfulness=1.0)


def test_format_weight_must_stay_small() -> None:
    """ "Large format rewards produce beautifully formatted wrong answers.\""""
    with pytest.raises(ValidationError, match="must stay below"):
        _reward(w_format=0.6)


@pytest.mark.parametrize("field", ["duplicate_query", "no_answer", "empty_query"])
def test_penalties_must_be_non_positive(field: str) -> None:
    """A positive "penalty" rewards the behaviour it was meant to discourage."""
    kwargs: dict[str, float] = {"duplicate_query": -0.1, "no_answer": -0.5, "empty_query": -0.05}
    kwargs[field] = 0.1
    with pytest.raises(ValidationError):
        PenaltyConfig(**kwargs)


def test_lambda_must_be_non_negative() -> None:
    """A negative lambda pays the model to search, inverting the experiment."""
    with pytest.raises(ValidationError):
        _reward(lambda_retrieval_cost=-0.1)


@pytest.mark.parametrize(
    ("group_size", "ok"),
    [
        pytest.param(8, True, id="project-md-value"),
        pytest.param(4, True, id="floor"),
        pytest.param(3, False, id="near-degenerate"),
        pytest.param(1, False, id="advantage-undefined"),
    ],
)
def test_group_size_must_support_an_advantage_estimate(group_size: int, ok: bool) -> None:
    """GRPO advantage is (r_i - mean(r)) / std(r) over the group."""
    kwargs = {
        "group_size": group_size,
        "kl_beta": 0.04,
        "learning_rate": 1e-6,
        "max_steps": 1000,
    }
    if ok:
        assert TrainConfig(**kwargs).group_size == group_size  # type: ignore[arg-type]
    else:
        with pytest.raises(ValidationError):
            TrainConfig(**kwargs)  # type: ignore[arg-type]


def test_kl_beta_zero_is_allowed_as_a_deliberate_ablation() -> None:
    """PROJECT.md section 5 calls KL "the main anti-hacking guard", so disabling
    it must be possible - and must live in a config file someone reviews."""
    assert TrainConfig(group_size=8, kl_beta=0.0, learning_rate=1e-6, max_steps=1000).kl_beta == 0.0


def test_learning_rate_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        TrainConfig(group_size=8, kl_beta=0.04, learning_rate=0.0, max_steps=1000)


def test_four_bit_with_float32_is_rejected() -> None:
    """NF4 keeps a bf16 or fp16 compute dtype; fp32 signals a copy-paste error."""
    with pytest.raises(ValidationError, match="not a meaningful combination"):
        ModelConfig(
            policy="Qwen2.5-1.5B-Instruct",
            policy_repo="Qwen/Qwen2.5-1.5B-Instruct",
            dtype=Dtype.FLOAT32,
            load_in_4bit=True,
        )


def test_four_bit_with_bfloat16_is_fine() -> None:
    """The documented fallback for a card that cannot hold the model in bf16."""
    config = ModelConfig(
        policy="Qwen2.5-3B-Instruct",
        policy_repo="Qwen/Qwen2.5-3B-Instruct",
        dtype=Dtype.BFLOAT16,
        load_in_4bit=True,
    )
    assert config.load_in_4bit is True


# ---------------------------------------------------------------------------
# The committed config files
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize("overlay", ["dev_1.5b.yaml", "train_3b.yaml"])
def test_committed_overlays_compose_onto_base(overlay: str) -> None:
    """Every committed overlay must load. A broken one fails at launch, not here.

    PROJECT.md section 3 puts config in data precisely so a bad value is caught
    by validation; that is worth nothing if no test ever loads the files.
    """
    config = load_config(CONFIG_DIR / "base.yaml", CONFIG_DIR / overlay)
    assert config.seed == 42


def test_dev_overlay_is_cheaper_than_base_without_changing_the_experiment() -> None:
    """A dev setting that leaked into a reported number would be unfindable."""
    base = load_config(CONFIG_DIR / "base.yaml")
    dev = load_config(CONFIG_DIR / "base.yaml", CONFIG_DIR / "dev_1.5b.yaml")

    assert dev.env.max_new_tokens < base.env.max_new_tokens
    assert dev.train.max_steps < base.train.max_steps
    assert dev.eval.bootstrap_resamples < base.eval.bootstrap_resamples

    # Untouched: everything the experiment measures.
    assert dev.reward == base.reward
    assert dev.train.group_size == base.train.group_size
    assert dev.train.kl_beta == base.train.kl_beta
    assert dev.train.learning_rate == base.train.learning_rate


def test_final_overlay_selects_the_3b_policy() -> None:
    config = load_config(CONFIG_DIR / "base.yaml", CONFIG_DIR / "train_3b.yaml")
    assert config.model.policy == "Qwen2.5-3B-Instruct"
    assert config.model.policy_repo == "Qwen/Qwen2.5-3B-Instruct"
    assert config.train.group_size == 8


def test_overlays_differ_in_fingerprint() -> None:
    """`fingerprint` is what ties a logged run to its configuration."""
    dev = load_config(CONFIG_DIR / "base.yaml", CONFIG_DIR / "dev_1.5b.yaml")
    final = load_config(CONFIG_DIR / "base.yaml", CONFIG_DIR / "train_3b.yaml")
    assert dev.fingerprint != final.fingerprint


# ---------------------------------------------------------------------------
# Sweep manifest
# ---------------------------------------------------------------------------


def test_committed_sweep_manifest_is_the_four_points_from_project_md() -> None:
    spec = load_sweep(CONFIG_DIR / "sweep_lambda.yaml")
    assert spec.lambdas == (0.0, 0.02, 0.05, 0.1)
    assert spec.name == "lambda_frontier"


def test_sweep_points_vary_only_lambda() -> None:
    """The entire claim of the frontier plot, asserted."""
    spec = load_sweep(CONFIG_DIR / "sweep_lambda.yaml")
    points = sweep_points(spec, root=CONFIG_DIR.parent)

    assert [point.lambda_value for point in points] == [0.0, 0.02, 0.05, 0.1]
    assert [point.run_name for point in points] == [
        "lambda_frontier-lam0",
        "lambda_frontier-lam0.02",
        "lambda_frontier-lam0.05",
        "lambda_frontier-lam0.1",
    ]

    stripped = {
        point.config.model_copy(
            update={"reward": point.config.reward.model_copy(update={"lambda_retrieval_cost": 0.0})}
        ).fingerprint
        for point in points
    }
    assert len(stripped) == 1, "runs differ in something other than lambda"
    assert len({point.config.fingerprint for point in points}) == 4


def test_sweep_points_use_the_manifest_base_files() -> None:
    spec = load_sweep(CONFIG_DIR / "sweep_lambda.yaml")
    points = sweep_points(spec, root=CONFIG_DIR.parent)
    assert all(point.config.model.policy == "Qwen2.5-3B-Instruct" for point in points)


def write_sweep(tmp_path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "name": "s",
        "base": ["configs/base.yaml"],
        "lambdas": [0.0, 0.1],
        "output_dir": "results/raw/s",
    }
    payload.update(overrides)
    target = tmp_path / "sweep.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return target


def test_duplicate_lambdas_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be distinct"):
        load_sweep(write_sweep(tmp_path, lambdas=[0.0, 0.05, 0.05]))


def test_sweep_without_the_zero_control_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"must include 0\.0 as the control point"):
        load_sweep(write_sweep(tmp_path, lambdas=[0.02, 0.05]))


def test_unsorted_lambdas_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be ascending"):
        load_sweep(write_sweep(tmp_path, lambdas=[0.1, 0.0]))


def test_single_lambda_is_not_a_sweep(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="at least 2 items"):
        load_sweep(write_sweep(tmp_path, lambdas=[0.0]))


def test_negative_lambda_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be non-negative"):
        load_sweep(write_sweep(tmp_path, lambdas=[-0.1, 0.0]))


def test_sweep_manifest_rejects_unknown_keys(tmp_path: Path) -> None:
    """`extra="forbid"`: a typo in a manifest must not silently do nothing."""
    with pytest.raises(ConfigError, match="lamdas"):
        load_sweep(write_sweep(tmp_path, lamdas=[0.0, 0.1]))


def test_missing_sweep_manifest_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_sweep(tmp_path / "absent.yaml")


# ---------------------------------------------------------------------------
# serve: - PROJECT.md section 9, serving row
# ---------------------------------------------------------------------------


def _serve(**overrides: object) -> ServeConfig:
    fields: dict[str, object] = {
        "url": "http://127.0.0.1:8100",
        "concurrency_levels": (1, 8, 32),
        "requests_per_level": 500,
        "warmup_requests": 32,
        "request_timeout_s": 120.0,
        "throughput_batches": 10,
    }
    fields.update(overrides)
    return ServeConfig.model_validate(fields)


def test_the_base_config_benchmarks_at_section_9s_three_levels() -> None:
    """Section 9: "at concurrency 1, 8, 32". The model allows other levels on
    purpose, for smoke overlays - so the committed base is held to it here."""
    assert load_config().serve.concurrency_levels == (1, 8, 32)


def test_the_base_config_sends_enough_requests_for_a_real_p99() -> None:
    """Section 9 asks for p99, and the benchmark only reports one with five
    observations above it. The base config must make that possible."""
    from verisearch.serve.benchmark import min_samples_for

    assert load_config().serve.requests_per_level >= min_samples_for(0.99)


def test_the_policy_server_does_not_share_a_port_with_retrieval() -> None:
    config = load_config()
    assert config.serve.url != config.retrieval.service_url


@pytest.mark.parametrize(
    "levels",
    [
        pytest.param((8, 1, 32), id="unsorted"),
        pytest.param((1, 8, 8), id="duplicate"),
    ],
)
def test_concurrency_levels_must_be_strictly_increasing(levels: tuple[int, ...]) -> None:
    """A duplicate is one row measured twice under one label, the second on a
    warmer cache."""
    with pytest.raises(ValidationError, match="strictly increasing"):
        _serve(concurrency_levels=levels)


def test_a_concurrency_level_below_one_is_rejected() -> None:
    with pytest.raises(ValidationError, match=">= 1"):
        _serve(concurrency_levels=(0, 8))


def test_concurrency_levels_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        _serve(concurrency_levels=())


@pytest.mark.parametrize(
    ("warmup", "ok"),
    [
        pytest.param(32, True, id="one-full-wave"),
        pytest.param(64, True, id="two-waves"),
        pytest.param(31, False, id="one-short"),
    ],
)
def test_warmup_must_cover_a_full_wave_at_the_highest_level(warmup: int, ok: bool) -> None:
    """The first wave at concurrency 32 is the coldest moment of the whole run."""
    if ok:
        assert _serve(warmup_requests=warmup).warmup_requests == warmup
    else:
        with pytest.raises(ValidationError, match="shorter than one wave"):
            _serve(warmup_requests=warmup)


def test_a_smoke_overlay_with_small_counts_is_allowed() -> None:
    """Honest at runtime rather than refused here: p99 comes out n/a."""
    smoke = _serve(concurrency_levels=(1, 2), requests_per_level=20, warmup_requests=2)
    assert smoke.requests_per_level == 20
