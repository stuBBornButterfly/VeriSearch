"""Tests for `verisearch/cli.py`.

`main` returns an exit code and never raises `SystemExit`, so every command is
driven directly here with an argv list. That is the reason the logic is in the
package rather than in the `.sh` files: a bash script that decides anything
cannot be tested, and these decide which index a report was measured against and
whether a sweep is complete.

Seven of the eight commands run end to end with no optional extras installed:
`prepare-data` (with the download replaced by fixture rows), `eval-context`,
`eval-report`, `frontier`, `regression-set`, `eval-regression` and
`serve-bench` - the last against `serve.local`'s stdlib server and a fake
policy, over a real socket.
`build-index` is driven to the point just before the embedder loads, which is
the only line in the module that needs torch.

Every assertion on stderr matters: an expected failure must print one line and
return 2, not raise. A traceback reaching the user is a bug in the CLI.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from verisearch.cli import CONTEXT_FILENAME, REPORT_JSON, RESULTS_JSONL, _force_utf8, main
from verisearch.config import DEFAULT_CONFIG_PATH
from verisearch.data.datasets import (
    DatasetSpec,
    Distribution,
    QAExample,
    Split,
    load_jsonl,
    write_jsonl,
)
from verisearch.eval.baselines import BaselineKind, EvalContext, context_for
from verisearch.eval.frontier import REPORT_FILENAME
from verisearch.eval.harness import (
    EvalReport,
    aggregate,
    read_results,
    score_example,
    write_report,
    write_results,
)
from verisearch.eval.metrics import ConfidenceInterval, FailureReason
from verisearch.retrieval.client import RetrievalError
from verisearch.retrieval.embedder import EmbedderSpec, l2_normalize
from verisearch.retrieval.index import BruteForceIndex
from verisearch.serve.benchmark import REPORT_FILENAME as BENCH_FILENAME
from verisearch.serve.benchmark import REQUESTS_FILENAME, TABLE_HEADER, BenchReport
from verisearch.serve.local import FakePolicyRunner, make_policy_server, serving
from verisearch.serve.server import handle_health
from verisearch.types import Passage, Termination, Trajectory, Turn, TurnKind

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SPEC = EmbedderSpec(model_name="cli-test", dim=4, normalize=True)
LAMBDAS = (0.0, 0.02, 0.05, 0.1)


def write_corpus_file(path: Path, n: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"doc_id": f"d{i}", "title": f"Title {i}", "text": f"Body of document {i}."})
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_real_index(directory: Path, n: int = 3) -> Path:
    """A real on-disk index, with no embedder.

    `BruteForceIndex` is a core-dependency class, so the vectors can be written
    by hand. What the CLI reads from here is `meta.json` - specifically the
    embedder fingerprint - and that is genuine.
    """
    vectors = l2_normalize(np.eye(4, dtype=np.float32)[:n])
    index = BruteForceIndex.build(vectors, [f"d{i}" for i in range(n)], SPEC)
    index.save(directory)
    return directory


def write_examples(path: Path, n: int = 3) -> Path:
    write_jsonl(
        [
            QAExample(
                question_id=f"q{i}",
                question=f"Question {i}?",
                answers=("Ada Lovelace",),
                dataset="hotpotqa",
                distribution=Distribution.IN_DISTRIBUTION,
                supporting_titles=("Title 0",),
                split=Split.DEV,
            )
            for i in range(n)
        ],
        path,
    )
    return path


def make_results_file(path: Path, n_rows: int, *, correct: int = 2) -> Path:
    rows = []
    for index in range(n_rows):
        example = QAExample(
            question_id=f"q{index}",
            question=f"Question {index}?",
            answers=("Ada Lovelace",),
            dataset="hotpotqa",
            distribution=Distribution.IN_DISTRIBUTION,
            supporting_titles=("Title 0",),
            split=Split.DEV,
        )
        answer = "Ada Lovelace" if index < correct else "Nobody"
        trajectory = Trajectory(
            question_id=example.question_id,
            question=example.question,
            gold_answers=example.answers,
            turns=(
                Turn(kind=TurnKind.SEARCH, text="ada"),
                Turn(
                    kind=TurnKind.INFO,
                    text="<info/>",
                    passages=(Passage(doc_id="d0", title="Title 0", text="Body."),),
                ),
                Turn(kind=TurnKind.ANSWER, text=answer),
            ),
            answer=answer,
            termination=Termination.ANSWERED,
        )
        rows.append(score_example(example, trajectory))
    write_results(rows, path)
    return path


def ci(point: float, half: float = 0.02) -> ConfidenceInterval:
    return ConfidenceInterval(
        point=point, low=point - half, high=point + half, confidence=0.95, resamples=1000, n=3
    )


def write_sweep_reports(
    root: Path,
    *,
    searches: tuple[float, ...] = (3.4, 2.8, 2.1, 1.3),
    fingerprints: tuple[str, ...] | None = None,
    skip: float | None = None,
) -> None:
    prints = fingerprints if fingerprints is not None else (SPEC.fingerprint,) * 4
    for value, count, fingerprint in zip(LAMBDAS, searches, prints, strict=True):
        if value == skip:
            continue
        run_name = f"lambda_frontier-lam{value:g}"
        report = EvalReport(
            run_name=run_name,
            kind=BaselineKind.TRAINED,
            context=context_for(
                ("q0", "q1", "q2"),
                dataset="hotpotqa",
                split="dev",
                retriever_fingerprint=fingerprint,
                k=3,
            ),
            exact_match=ci(0.5),
            f1=ci(0.62 - 0.03 * LAMBDAS.index(value)),
            searches_per_question=ci(count, 0.1),
            answer_tokens=ci(6.0, 0.5),
            unsupported_rate=ci(0.2),
            precision_at_k=ci(0.5),
            recall_at_k=ci(0.7),
            failure_breakdown=dict.fromkeys((r.value for r in FailureReason), 0.2),
            duplicate_query_episode_rate=0.1,
        )
        target = root / "out" / run_name / REPORT_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def write_manifest(path: Path, *, name: str = "lambda_frontier") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"name: {name}",
                "base:",
                "  - configs/base.yaml",
                "lambdas: [0.0, 0.02, 0.05, 0.1]",
                "output_dir: out",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_no_subcommand_is_a_usage_error() -> None:
    """argparse exits for a usage error rather than returning, which is correct -
    a missing subcommand is not something a caller recovers from."""
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2


def test_an_unknown_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        main(["not-a-command"])


# ---------------------------------------------------------------------------
# build-index
# ---------------------------------------------------------------------------


def test_build_index_plan_prints_the_fingerprint_without_embedding_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The point of --plan on this command: an hour of embedding is a bad way to
    find out the model flag was misspelled."""
    corpus = write_corpus_file(tmp_path / "corpus.jsonl")
    code = main(
        [
            "--plan",
            "build-index",
            "--corpus",
            str(corpus),
            "--out",
            str(tmp_path / "index"),
            "--model",
            "cli-test",
            "--dim",
            "4",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert f"fingerprint={SPEC.fingerprint}" in out
    assert "3 docs" in out
    assert "nothing written" in out
    assert not (tmp_path / "index").exists()


def test_build_index_refuses_to_overwrite_an_existing_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An index overwritten partway through a sweep invalidates every report
    that already cited its fingerprint, and those reports stay on disk looking
    perfectly valid."""
    corpus = write_corpus_file(tmp_path / "corpus.jsonl")
    index_dir = build_real_index(tmp_path / "index")

    code = main(["build-index", "--corpus", str(corpus), "--out", str(index_dir)])
    assert code == 2
    assert "invalidates every report" in capsys.readouterr().err


def test_force_gets_past_the_overwrite_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = write_corpus_file(tmp_path / "corpus.jsonl")
    index_dir = build_real_index(tmp_path / "index")

    # --plan still stops before the embedder, so this reaches the plan output
    # rather than needing torch.
    code = main(
        ["--plan", "build-index", "--corpus", str(corpus), "--out", str(index_dir), "--force"]
    )
    assert code == 0
    assert "nothing written" in capsys.readouterr().out


def test_build_index_reports_a_missing_corpus_as_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["build-index", "--corpus", str(tmp_path / "absent.jsonl"), "--out", str(tmp_path / "i")]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("error: ")
    assert err.count("\n") == 1
    assert "Traceback" not in err


def test_build_index_rejects_a_corpus_with_a_duplicate_doc_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Caught before embedding, not after: a duplicate doc_id makes every hit on
    that id resolve by read order."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"doc_id":"d1","title":"A","text":"Body."}\n{"doc_id":"d1","title":"B","text":"Other."}\n',
        encoding="utf-8",
    )
    code = main(["--plan", "build-index", "--corpus", str(corpus), "--out", str(tmp_path / "i")])
    assert code == 2
    assert "duplicate doc_id" in capsys.readouterr().err


def test_flags_that_change_the_vectors_change_the_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`EmbedderSpec.fingerprint` is what makes a mismatched index loud. A flag
    that changed the vectors without changing it would be silent."""
    corpus = write_corpus_file(tmp_path / "corpus.jsonl")
    base = ["--plan", "build-index", "--corpus", str(corpus), "--out", str(tmp_path / "i")]

    fingerprints: list[str] = []
    for extra in ([], ["--no-normalize"], ["--query-instruction", "Represent: "], ["--dim", "768"]):
        assert main([*base, *extra]) == 0
        line = next(row for row in capsys.readouterr().out.splitlines() if "fingerprint=" in row)
        fingerprints.append(line.split("fingerprint=")[1].strip())

    assert len(set(fingerprints)) == 4


def test_batch_size_does_not_change_the_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It changes throughput, not output, so an index built at batch 32 is still
    valid when queried at batch 128."""
    corpus = write_corpus_file(tmp_path / "corpus.jsonl")
    base = ["--plan", "build-index", "--corpus", str(corpus), "--out", str(tmp_path / "i")]

    prints: list[str] = []
    for size in ("16", "128"):
        assert main([*base, "--batch-size", size]) == 0
        prints.append(
            next(row for row in capsys.readouterr().out.splitlines() if "fingerprint=" in row)
        )
    assert prints[0] == prints[1]


def test_build_index_without_the_retrieval_extra_says_how_to_install_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The path a user actually hits first. Without torch installed this has to
    print the install command, not a traceback about a missing module."""
    corpus = write_corpus_file(tmp_path / "corpus.jsonl")

    if importlib.util.find_spec("sentence_transformers") is not None:  # pragma: no cover
        pytest.skip("the [retrieval] extra is installed, so this path succeeds")

    code = main(
        [
            "build-index",
            "--corpus",
            str(corpus),
            "--out",
            str(tmp_path / "index"),
            "--model",
            "cli-test",
            "--dim",
            "4",
        ]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "retrieval extra" in err
    assert "Traceback" not in err
    # Nothing half-written: a directory with vectors and no meta.json would be
    # loaded by nothing and deleted by nobody.
    assert not (tmp_path / "index" / "meta.json").exists()


# ---------------------------------------------------------------------------
# eval-context - the keystone
# ---------------------------------------------------------------------------


def test_eval_context_takes_the_fingerprint_from_the_index_not_a_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Section 8's "identical retriever" is a promise until the fingerprint is
    read out of the index's own meta.json. There is deliberately no flag to
    override it."""
    index_dir = build_real_index(tmp_path / "index")
    examples = write_examples(tmp_path / "dev.jsonl")
    out = tmp_path / CONTEXT_FILENAME

    code = main(
        [
            "eval-context",
            "--index",
            str(index_dir),
            "--examples",
            str(examples),
            "--dataset",
            "hotpotqa",
            "--split",
            "dev",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "every baseline run must load this file" in capsys.readouterr().out

    context = EvalContext.model_validate_json(out.read_text(encoding="utf-8"))
    assert context.retriever_fingerprint == SPEC.fingerprint
    assert context.n_examples == 3
    assert context.dataset == "hotpotqa"
    assert context.k == 3


def test_eval_context_plan_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = build_real_index(tmp_path / "index")
    examples = write_examples(tmp_path / "dev.jsonl")
    out = tmp_path / CONTEXT_FILENAME

    code = main(
        [
            "--plan",
            "eval-context",
            "--index",
            str(index_dir),
            "--examples",
            str(examples),
            "--dataset",
            "hotpotqa",
            "--split",
            "dev",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "nothing written" in capsys.readouterr().out
    assert not out.exists()


def test_eval_context_on_a_missing_index_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = write_examples(tmp_path / "dev.jsonl")
    code = main(
        [
            "eval-context",
            "--index",
            str(tmp_path / "nope"),
            "--examples",
            str(examples),
            "--dataset",
            "hotpotqa",
            "--split",
            "dev",
        ]
    )
    assert code == 2
    assert "no index metadata" in capsys.readouterr().err


def test_eval_context_on_a_missing_eval_set_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = build_real_index(tmp_path / "index")
    code = main(
        [
            "eval-context",
            "--index",
            str(index_dir),
            "--examples",
            str(tmp_path / "absent.jsonl"),
            "--dataset",
            "hotpotqa",
            "--split",
            "dev",
        ]
    )
    assert code == 2
    assert "dataset file not found" in capsys.readouterr().err


def test_two_indexes_give_two_contexts_that_refuse_to_compare(tmp_path: Path) -> None:
    """End to end: rebuild the index with a different embedder and the contexts
    stop matching, which is what stops a comparison across the two."""
    examples = write_examples(tmp_path / "dev.jsonl")
    first = tmp_path / "ctx_a.json"
    second = tmp_path / "ctx_b.json"

    build_real_index(tmp_path / "index_a")
    other = EmbedderSpec(model_name="a-different-model", dim=4, normalize=True)
    vectors = l2_normalize(np.eye(4, dtype=np.float32)[:3])
    BruteForceIndex.build(vectors, ["d0", "d1", "d2"], other).save(tmp_path / "index_b")

    for index_dir, out in ((tmp_path / "index_a", first), (tmp_path / "index_b", second)):
        assert (
            main(
                [
                    "eval-context",
                    "--index",
                    str(index_dir),
                    "--examples",
                    str(examples),
                    "--dataset",
                    "hotpotqa",
                    "--split",
                    "dev",
                    "--out",
                    str(out),
                ]
            )
            == 0
        )

    from verisearch.eval.baselines import EvalError, check_comparable

    left = EvalContext.model_validate_json(first.read_text(encoding="utf-8"))
    right = EvalContext.model_validate_json(second.read_text(encoding="utf-8"))
    with pytest.raises(EvalError, match="retriever_fingerprint"):
        check_comparable(left, right)


# ---------------------------------------------------------------------------
# eval-report
# ---------------------------------------------------------------------------


def context_on_disk(tmp_path: Path, *, n: int = 3) -> tuple[Path, EvalContext]:
    context = context_for(
        tuple(f"q{i}" for i in range(n)),
        dataset="hotpotqa",
        split="dev",
        retriever_fingerprint=SPEC.fingerprint,
        k=3,
    )
    path = tmp_path / CONTEXT_FILENAME
    path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    return path, context


def test_eval_report_aggregates_rows_into_section_9s_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context_path, context = context_on_disk(tmp_path)
    results = make_results_file(tmp_path / "results.jsonl", context.n_examples)
    out = tmp_path / "report.json"

    code = main(
        [
            "eval-report",
            "--results",
            str(results),
            "--context",
            str(context_path),
            "--run-name",
            "prompted",
            "--kind",
            "prompted_iterative",
            "--resamples",
            "50",
            "--out",
            str(out),
        ]
    )
    printed = capsys.readouterr().out

    assert code == 0
    assert "EM" in printed
    assert "searches/question" in printed

    report = EvalReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.run_name == "prompted"
    assert report.kind is BaselineKind.PROMPTED_ITERATIVE
    assert report.context == context
    assert report.f1.point == pytest.approx(2 / 3)


def test_eval_report_refuses_rows_that_do_not_match_the_context(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The context carries the denominators. Aggregating a different number of
    rows under it would describe a set the run did not measure."""
    context_path, _ = context_on_disk(tmp_path, n=5)
    # Three rows written against a five-example context.
    results = make_results_file(tmp_path / "results.jsonl", 3)

    code = main(
        [
            "eval-report",
            "--results",
            str(results),
            "--context",
            str(context_path),
            "--run-name",
            "x",
            "--kind",
            "trained",
            "--resamples",
            "50",
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    assert code == 2
    assert "denominators would be wrong" in capsys.readouterr().err


def test_eval_report_rejects_a_context_file_that_is_not_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "ctx.json"
    bad.write_text('{"dataset":"hotpotqa"}', encoding="utf-8")
    code = main(
        [
            "eval-report",
            "--results",
            str(tmp_path / "whatever.jsonl"),
            "--context",
            str(bad),
            "--run-name",
            "x",
            "--kind",
            "trained",
        ]
    )
    assert code == 2
    assert "is not an eval context" in capsys.readouterr().err


def test_eval_report_rejects_an_unknown_baseline_kind() -> None:
    """The five kinds come straight off `BaselineKind`, so a typo in a script is
    a usage error rather than a run labelled with something meaningless."""
    with pytest.raises(SystemExit):
        main(
            [
                "eval-report",
                "--results",
                "x.jsonl",
                "--context",
                "c.json",
                "--run-name",
                "x",
                "--kind",
                "rlhf",
            ]
        )


def test_eval_report_plan_prints_the_table_without_writing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Useful on its own: it is how you read section 9's table for a run without
    adding a report file that a later comparison would pick up."""
    context_path, context = context_on_disk(tmp_path)
    results = make_results_file(tmp_path / "results.jsonl", context.n_examples)
    out = tmp_path / "report.json"

    code = main(
        [
            "--plan",
            "eval-report",
            "--results",
            str(results),
            "--context",
            str(context_path),
            "--run-name",
            "x",
            "--kind",
            "trained",
            "--resamples",
            "50",
            "--out",
            str(out),
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "F1" in printed
    assert "nothing written" in printed
    assert not out.exists()


def test_eval_report_is_seeded_so_two_runs_agree(tmp_path: Path) -> None:
    context_path, context = context_on_disk(tmp_path)
    results = make_results_file(tmp_path / "results.jsonl", context.n_examples)
    argv = [
        "eval-report",
        "--results",
        str(results),
        "--context",
        str(context_path),
        "--run-name",
        "x",
        "--kind",
        "trained",
        "--resamples",
        "50",
    ]

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    assert main([*argv, "--out", str(first)]) == 0
    assert main([*argv, "--out", str(second)]) == 0
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# frontier
# ---------------------------------------------------------------------------


def test_frontier_reads_every_run_and_writes_the_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(tmp_path / "configs" / "sweep_lambda.yaml")
    write_sweep_reports(tmp_path)
    out = tmp_path / "frontier.md"

    code = main(
        ["frontier", "--manifest", str(manifest), "--root", str(tmp_path), "--out", str(out)]
    )
    printed = capsys.readouterr().out

    assert code == 0
    assert "trade-off present" in printed
    table = out.read_text(encoding="utf-8")
    for value in LAMBDAS:
        assert f"| {value:g} |" in table


def test_frontier_exits_non_zero_when_the_sweep_measured_no_trade_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sweep where searches did not fall is not a frontier. The script must not
    quietly pass it on to a figure."""
    manifest = write_manifest(tmp_path / "configs" / "sweep_lambda.yaml")
    write_sweep_reports(tmp_path, searches=(2.5, 2.5, 2.5, 2.5))

    code = main(
        [
            "frontier",
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path),
            "--out",
            str(tmp_path / "frontier.md"),
        ]
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "NO TRADE-OFF MEASURED" in captured.out
    assert "not writing this up as a frontier" in captured.err
    # The table is still written: the numbers are the evidence for the verdict.
    assert (tmp_path / "frontier.md").is_file()


def test_frontier_names_the_missing_lambda(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three of four reports draw a curve that looks exactly like a complete
    one, and the caption would still say four-point sweep."""
    manifest = write_manifest(tmp_path / "configs" / "sweep_lambda.yaml")
    write_sweep_reports(tmp_path, skip=0.05)

    code = main(["frontier", "--manifest", str(manifest), "--root", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2
    assert "lambda=0.05" in err


def test_frontier_refuses_a_sweep_whose_points_used_different_indexes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sweep takes hours per point, so an index rebuild partway through is an
    ordinary thing to happen - and the curve comes out smooth."""
    manifest = write_manifest(tmp_path / "configs" / "sweep_lambda.yaml")
    write_sweep_reports(
        tmp_path,
        fingerprints=(SPEC.fingerprint, SPEC.fingerprint, SPEC.fingerprint, "rebuilt-index"),
    )

    code = main(["frontier", "--manifest", str(manifest), "--root", str(tmp_path)])
    assert code == 2
    assert "not measured the same way" in capsys.readouterr().err


def test_frontier_plan_prints_the_table_without_writing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(tmp_path / "configs" / "sweep_lambda.yaml")
    write_sweep_reports(tmp_path)
    out = tmp_path / "frontier.md"

    code = main(
        [
            "--plan",
            "frontier",
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "| λ |" in capsys.readouterr().out
    assert not out.exists()


def test_frontier_rejects_a_manifest_that_is_not_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "sweep.yaml"
    bad.write_text("name: x\nlambdas: [0.1]\n", encoding="utf-8")
    code = main(["frontier", "--manifest", str(bad), "--root", str(tmp_path)])
    assert code == 2
    assert "invalid sweep manifest" in capsys.readouterr().err


def test_asking_for_a_figure_without_matplotlib_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`matplotlib` is in the [plot] extra, so this is the path CI takes. The
    table is already on disk by then - the figure is the optional part."""
    manifest = write_manifest(tmp_path / "configs" / "sweep_lambda.yaml")
    write_sweep_reports(tmp_path)

    code = main(
        [
            "frontier",
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path),
            "--out",
            str(tmp_path / "frontier.md"),
            "--figure",
            str(tmp_path / "frontier.png"),
        ]
    )
    err = capsys.readouterr().err

    if importlib.util.find_spec("matplotlib") is None:
        assert code == 2
        assert "needs the plot extra" in err
        # The table is already on disk. The figure is the optional half, so a
        # missing extra must not cost the numbers.
        assert (tmp_path / "frontier.md").is_file()
    else:  # pragma: no cover - only when the [plot] extra is installed
        assert code == 0
        assert (tmp_path / "frontier.png").is_file()


def test_force_utf8_reconfigures_a_real_text_stream_and_leaves_a_fake_one_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both arms of the isinstance guard, because only one runs under pytest.

    `_force_utf8` exists so the frontier table's lambda column survives a cp1252
    console. It is guarded by an isinstance check rather than a try/except
    because pytest replaces `sys.stdout` with a capture object that has no
    `reconfigure` - which means the normal test run only ever exercises the
    branch that does nothing, and the branch that does the actual work would
    ship unexercised.
    """
    fake = io.StringIO()  # no `reconfigure`, like pytest's capture object
    monkeypatch.setattr(sys, "stdout", fake)
    monkeypatch.setattr(sys, "stderr", fake)
    _force_utf8()
    assert not fake.closed  # the guard skipped it rather than raising

    path = tmp_path / "out.txt"
    with path.open("w", encoding="cp1252") as raw:
        assert isinstance(raw, io.TextIOWrapper)
        monkeypatch.setattr(sys, "stdout", raw)
        monkeypatch.setattr(sys, "stderr", raw)
        _force_utf8()
        assert raw.encoding.lower() in {"utf-8", "utf8"}
        # The point of the reconfigure: this is what used to arrive as λ.
        raw.write("λ=0.05")

    assert path.read_text(encoding="utf-8") == "λ=0.05"


# ---------------------------------------------------------------------------
# serve-bench
# ---------------------------------------------------------------------------

SERVE_HEALTH = handle_health(
    policy="Qwen/Qwen2.5-1.5B-Instruct",
    adapter="ckpt/step-1000",
    hardware="test-cpu",
    config_fingerprint="abc123def456",
)


def write_serve_overlay(
    path: Path,
    *,
    url: str | None = None,
    levels: tuple[int, ...] = (1, 2),
    requests: int = 20,
    warmup: int = 2,
) -> Path:
    """A cheap `serve:` section on top of `configs/base.yaml`."""
    lines = [
        "serve:",
        f"  concurrency_levels: [{', '.join(str(level) for level in levels)}]",
        f"  requests_per_level: {requests}",
        f"  warmup_requests: {warmup}",
        "  request_timeout_s: 10",
        "  throughput_batches: 4",
        "eval:",
        "  bootstrap_resamples: 100",
    ]
    if url is not None:
        lines.insert(1, f"  url: {url}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def serve_bench_argv(tmp_path: Path, overlay: Path, *extra: str) -> list[str]:
    return [
        "serve-bench",
        "--examples",
        str(write_examples(tmp_path / "dev.jsonl", n=3)),
        "--config",
        str(DEFAULT_CONFIG_PATH),
        str(overlay),
        "--out",
        str(tmp_path / "serve"),
        "--table",
        str(tmp_path / "serving.md"),
        *extra,
    ]


class FailsOnOneQuestion:
    """A policy whose retrieval dies for one particular question."""

    def __init__(self, bad: str) -> None:
        self._bad = bad
        self._inner = FakePolicyRunner()

    def run(self, question_id: str, question: str) -> Trajectory:
        if question == self._bad:
            raise RetrievalError("index unavailable")
        return self._inner.run(question_id, question)


@pytest.mark.integration
def test_serve_bench_measures_every_level_and_exits_0_on_a_publishable_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The full path at section 9's request count, so p99 is real and the
    command can say the table is publishable. 500 per level is what it takes.

    The small delay is what makes four requests genuinely overlap on the
    server, so `concurrency_reached` is earned rather than lucky.
    """
    runner = FakePolicyRunner(delay_s=0.002)
    overlay = write_serve_overlay(tmp_path / "serve.yaml", levels=(1, 4), requests=500, warmup=4)
    with serving(make_policy_server(runner, SERVE_HEALTH)) as url:
        code = main(serve_bench_argv(tmp_path, overlay, "--url", url))

    out = capsys.readouterr()
    assert code == 0, out.err
    assert runner.calls == 2 * (500 + 4)
    assert "hardware    test-cpu" in out.out
    assert "adapter ckpt/step-1000" in out.out

    table = (tmp_path / "serving.md").read_text(encoding="utf-8")
    assert table.startswith(TABLE_HEADER)
    assert "| 1 | " in table
    assert "| 4 | " in table
    assert "n/a" not in table
    assert "`test-cpu`" in table

    report = BenchReport.model_validate_json(
        (tmp_path / "serve" / BENCH_FILENAME).read_text(encoding="utf-8")
    )
    assert report.server == SERVE_HEALTH
    assert report.problems() == ()
    rows = (tmp_path / "serve" / REQUESTS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 * (500 + 4)


@pytest.mark.integration
def test_serve_bench_plan_reads_health_and_sends_no_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = FakePolicyRunner()
    overlay = write_serve_overlay(tmp_path / "serve.yaml")
    with serving(make_policy_server(runner, SERVE_HEALTH)) as url:
        code = main(["--plan", *serve_bench_argv(tmp_path, overlay, "--url", url)])

    out = capsys.readouterr().out
    assert code == 0
    assert runner.calls == 0
    assert "no load sent" in out
    assert "44 in all" in out  # 2 levels x (20 + 2)
    assert not (tmp_path / "serve").exists()
    assert not (tmp_path / "serving.md").exists()


@pytest.mark.integration
def test_serve_bench_too_small_for_p99_writes_everything_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A smoke run is useful and is not section 9's table. It says which."""
    overlay = write_serve_overlay(tmp_path / "serve.yaml")
    with serving(make_policy_server(FakePolicyRunner(), SERVE_HEALTH)) as url:
        code = main(serve_bench_argv(tmp_path, overlay, "--url", url))

    err = capsys.readouterr().err
    assert code == 1
    assert "p99 needs 500 successful requests" in err
    assert (tmp_path / "serving.md").is_file()
    assert (tmp_path / "serve" / REQUESTS_FILENAME).is_file()


@pytest.mark.integration
def test_serve_bench_counts_failed_requests_and_refuses_the_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    overlay = write_serve_overlay(tmp_path / "serve.yaml", levels=(1,))
    runner = FailsOnOneQuestion("Question 1?")
    with serving(make_policy_server(runner, SERVE_HEALTH)) as url:
        code = main(serve_bench_argv(tmp_path, overlay, "--url", url))

    err = capsys.readouterr().err
    assert code == 1
    assert "HTTP 503" in err
    table = (tmp_path / "serving.md").read_text(encoding="utf-8")
    # 20 measured requests cycle three questions from index 2, so "Question 1?"
    # is asked at indices 4, 7, ... 19 - six times.
    assert "| 6 / 20 |" in table


@pytest.mark.integration
def test_serve_bench_takes_the_url_from_the_config_when_no_flag_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with serving(make_policy_server(FakePolicyRunner(), SERVE_HEALTH)) as url:
        overlay = write_serve_overlay(tmp_path / "serve.yaml", url=url)
        code = main(["--plan", *serve_bench_argv(tmp_path, overlay)])

    assert code == 0
    assert f"server      {url}" in capsys.readouterr().out


def test_serve_bench_against_nothing_says_so_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    overlay = write_serve_overlay(tmp_path / "serve.yaml")
    code = main(serve_bench_argv(tmp_path, overlay, "--url", "http://127.0.0.1:9"))

    err = capsys.readouterr().err
    assert code == 2
    assert err.count("\n") == 1
    assert not (tmp_path / "serve").exists()


def test_serve_bench_rejects_a_serve_section_that_breaks_its_rules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    overlay = write_serve_overlay(tmp_path / "serve.yaml", levels=(1, 8), warmup=2)
    code = main(serve_bench_argv(tmp_path, overlay, "--url", "http://127.0.0.1:9"))

    assert code == 2
    assert "shorter than one wave" in capsys.readouterr().err


def test_serve_bench_refuses_an_unsendable_question_before_any_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One over-long question in the eval set must stop the run up front, not an
    hour into the second level."""
    examples = tmp_path / "long.jsonl"
    write_jsonl(
        [
            QAExample(
                question_id="q-long",
                question="x" * 5000,
                answers=("Ada Lovelace",),
                dataset="hotpotqa",
                distribution=Distribution.IN_DISTRIBUTION,
            )
        ],
        examples,
    )
    runner = FakePolicyRunner()
    overlay = write_serve_overlay(tmp_path / "serve.yaml")
    argv = serve_bench_argv(tmp_path, overlay)
    argv[argv.index("--examples") + 1] = str(examples)
    with serving(make_policy_server(runner, SERVE_HEALTH)) as url:
        code = main([*argv, "--url", url])

    assert code == 2
    assert "cannot be sent" in capsys.readouterr().err
    assert runner.calls == 0
    assert not (tmp_path / "serve").exists()


# ---------------------------------------------------------------------------
# regression-set and eval-regression
# ---------------------------------------------------------------------------


def test_regression_set_writes_the_set_its_fingerprint_describes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The printed fingerprint must be the one `eval-context` then computes from
    the written file. They are different code paths and the fingerprint is
    order-sensitive, so this is the check that they agree."""
    source = write_examples(tmp_path / "dev.jsonl", n=8)
    out = tmp_path / "regression" / "examples.jsonl"

    code = main(["regression-set", "--examples", str(source), "--n", "5", "--out", str(out)])

    printed = capsys.readouterr().out
    assert code == 0
    assert len(load_jsonl(out)) == 5

    index = build_real_index(tmp_path / "index")
    context_path = tmp_path / "context.json"
    assert (
        main(
            [
                "eval-context",
                "--index",
                str(index),
                "--examples",
                str(out),
                "--dataset",
                "hotpotqa",
                "--split",
                "dev",
                "--out",
                str(context_path),
            ]
        )
        == 0
    )
    context = EvalContext.model_validate_json(context_path.read_text(encoding="utf-8"))
    assert f"fingerprint={context.examples_fingerprint}" in printed


def test_regression_set_refuses_to_replace_the_set_a_reference_was_scored_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_examples(tmp_path / "dev.jsonl", n=8)
    out = tmp_path / "examples.jsonl"
    argv = ["regression-set", "--examples", str(source), "--n", "5", "--out", str(out)]
    assert main(argv) == 0
    capsys.readouterr()

    assert main(argv) == 2
    assert "already exists" in capsys.readouterr().err
    assert main([*argv, "--force"]) == 0


def test_regression_set_plan_writes_nothing(tmp_path: Path) -> None:
    source = write_examples(tmp_path / "dev.jsonl", n=8)
    out = tmp_path / "examples.jsonl"
    argv = ["--plan", "regression-set", "--examples", str(source), "--n", "5", "--out", str(out)]
    assert main(argv) == 0
    assert not out.exists()


def test_regression_set_from_too_small_a_pool_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_examples(tmp_path / "dev.jsonl", n=3)
    code = main(
        ["regression-set", "--examples", str(source), "--n", "5", "--out", str(tmp_path / "x")]
    )
    assert code == 2
    assert "found 3" in capsys.readouterr().err


def make_reference(
    directory: Path, *, n: int = 3, correct: int = 3, spec: EmbedderSpec = SPEC
) -> Path:
    """A promoted reference: report.json and results.jsonl, as 03_eval.sh writes them."""
    directory.mkdir(parents=True, exist_ok=True)
    results = make_results_file(directory / RESULTS_JSONL, n, correct=correct)
    context = context_for(
        [f"q{i}" for i in range(n)],
        dataset="hotpotqa",
        split="dev",
        retriever_fingerprint=spec.fingerprint,
        k=3,
    )
    report = aggregate(
        read_results(results),
        run_name="reference",
        kind=BaselineKind.TRAINED,
        context=context,
        resamples=100,
    )
    write_report(report, directory / REPORT_JSON)
    return directory


def regression_argv(tmp_path: Path, *, candidate_correct: int, n: int = 3) -> list[str]:
    candidate = make_results_file(tmp_path / "candidate.jsonl", 3, correct=candidate_correct)
    return [
        "eval-regression",
        "--examples",
        str(write_examples(tmp_path / "regression.jsonl", n=3)),
        "--index",
        str(build_real_index(tmp_path / "index")),
        "--reference",
        str(tmp_path / "reference"),
        "--candidate",
        str(candidate),
        "--dataset",
        "hotpotqa",
        "--split",
        "dev",
        "--n",
        str(n),
        "--max-f1-drop",
        "2.0",
        "--resamples",
        "100",
    ]


def test_eval_regression_passes_when_f1_held(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_reference(tmp_path / "reference", correct=3)
    code = main(regression_argv(tmp_path, candidate_correct=3))
    assert code == 0
    assert "verdict             held" in capsys.readouterr().out


def test_eval_regression_fails_when_f1_fell_and_the_interval_agrees(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every question that was right is now wrong: the interval is [-1, -1]."""
    make_reference(tmp_path / "reference", correct=3)
    code = main(regression_argv(tmp_path, candidate_correct=0))
    out = capsys.readouterr()
    assert code == 1
    assert "REGRESSED" in out.err
    assert "verdict             regressed" in out.out


def test_eval_regression_refuses_a_set_of_the_wrong_size(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--n is the gate's definition. A different file is a different gate."""
    make_reference(tmp_path / "reference")
    code = main(regression_argv(tmp_path, candidate_correct=3, n=100))
    assert code == 2
    assert "not the 100 this gate is defined on" in capsys.readouterr().err


def test_eval_regression_without_a_reference_cannot_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A comparison that could not be made is never a pass."""
    code = main(regression_argv(tmp_path, candidate_correct=3))
    assert code == 2
    assert "no report at" in capsys.readouterr().err


def test_eval_regression_refuses_a_reference_from_another_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The candidate's context comes from tonight's index, not the reference's
    file - so a reference scored against a different index is caught."""
    other = EmbedderSpec(model_name="another-embedder", dim=4, normalize=True)
    make_reference(tmp_path / "reference", spec=other)
    code = main(regression_argv(tmp_path, candidate_correct=3))
    assert code == 2
    assert "retriever_fingerprint" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# prepare-data
# ---------------------------------------------------------------------------


def fixture_rows(spec: DatasetSpec, limit: int | None = None) -> Iterator[dict[str, object]]:
    """Stands in for `download_rows`: a handful of rows in each dataset's shape."""
    n = 12 if limit is None else min(limit, 12)
    for i in range(n):
        if spec.name == "hotpotqa":
            yield {
                "id": f"h{i}",
                "question": f"Hotpot {i}?",
                "answer": "Ada Lovelace",
                "supporting_facts": {"title": ["Ada"], "sent_id": [0]},
                "context": {"title": ["Ada", f"Other {i}"], "sentences": [["Ada."], ["No."]]},
            }
        elif spec.name == "2wikimultihopqa":
            yield {
                "id": f"w{i}",
                "question": f"Wiki {i}?",
                "answer": "yes",
                "supporting_facts": [["Wiki gold", 0]],
                "context": [["Wiki gold", ["Yes."]]],
            }
        elif spec.name == "musique":
            yield {"id": f"m{i}", "question": f"Musique {i}?", "answer": "x"}
        else:
            yield {"Question": f"Bamboogle {i}?", "Answer": "y"}


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the download with fixture rows, and record what was fetched."""
    fetched: list[str] = []

    def fake(spec: DatasetSpec, limit: int | None = None) -> Iterator[dict[str, object]]:
        fetched.append(spec.name)
        return fixture_rows(spec, limit)

    monkeypatch.setattr("verisearch.cli.download_rows", fake)
    return fetched


def test_prepare_data_writes_what_the_index_and_eval_scripts_read(
    tmp_path: Path, offline: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """The point of the command: the two default paths downstream exist after it."""
    out = tmp_path / "processed"
    code = main(["prepare-data", "--out", str(out)])

    printed = capsys.readouterr().out
    assert code == 0
    assert offline == ["hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
    assert (out / "corpus.jsonl").is_file()  # 01_build_index.sh's CORPUS default
    assert (out / "manifest.json").is_file()
    assert any(out.glob("hotpotqa_*.jsonl"))
    assert (out / "musique_ood.jsonl").is_file()
    assert "coverage UNMEASURED" in printed


def test_a_prepared_dev_file_feeds_eval_context(tmp_path: Path, offline: list[str]) -> None:
    """`03_eval.sh` reads `<dataset>_dev.jsonl`. Chained here so the naming
    convention is checked by the thing that depends on it."""
    out = tmp_path / "processed"
    assert main(["prepare-data", "--out", str(out), "--datasets", "hotpotqa"]) == 0
    dev_files = sorted(out.glob("hotpotqa_*.jsonl"))
    assert dev_files, "no hotpotqa split files"
    code = main(
        [
            "eval-context",
            "--index",
            str(build_real_index(tmp_path / "index")),
            "--examples",
            str(dev_files[0]),
            "--dataset",
            "hotpotqa",
            "--split",
            dev_files[0].stem.split("_", 1)[1],
            "--out",
            str(tmp_path / "context.json"),
        ]
    )
    assert code == 0


def test_prepare_data_plan_downloads_nothing(
    tmp_path: Path, offline: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "processed"
    assert main(["--plan", "prepare-data", "--out", str(out)]) == 0
    assert offline == []
    assert not out.exists()
    assert "hotpotqa/hotpot_qa (distractor)" in capsys.readouterr().out


def test_prepare_data_refuses_to_overwrite_a_finished_build(
    tmp_path: Path, offline: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "processed"
    assert main(["prepare-data", "--out", str(out)]) == 0
    offline.clear()
    capsys.readouterr()

    assert main(["prepare-data", "--out", str(out)]) == 2
    assert "already holds a finished build" in capsys.readouterr().err
    assert offline == []  # refused before downloading anything
    assert main(["prepare-data", "--out", str(out), "--force"]) == 0


def test_prepare_data_passes_the_limit_to_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both halves: the download is asked for 3 rows, and the manifest says the
    build is limited. Either alone would let a smoke build pass as the data."""
    limits: list[int | None] = []

    def recording(spec: DatasetSpec, limit: int | None = None) -> Iterator[dict[str, object]]:
        limits.append(limit)
        return fixture_rows(spec, limit)

    monkeypatch.setattr("verisearch.cli.download_rows", recording)
    code = main(["prepare-data", "--out", str(tmp_path / "p"), "--limit", "3"])
    assert code == 0
    assert limits == [3, 3, 3, 3]
    assert "LIMITED to 3 rows per split" in capsys.readouterr().out


def test_prepare_data_exits_1_when_a_gold_paragraph_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def gold_elsewhere(spec: DatasetSpec, limit: int | None = None) -> Iterator[dict[str, object]]:
        yield {
            "id": "h0",
            "question": "Where is it?",
            "answer": "here",
            "supporting_facts": {"title": ["Missing"], "sent_id": [0]},
            "context": {"title": ["Present"], "sentences": [["Text."]]},
        }

    monkeypatch.setattr("verisearch.cli.download_rows", gold_elsewhere)
    out = tmp_path / "processed"
    code = main(["prepare-data", "--out", str(out), "--datasets", "hotpotqa"])

    assert code == 1
    assert "recall is capped" in capsys.readouterr().err
    assert (out / "manifest.json").is_file()  # written, for diagnosis


def test_prepare_data_reports_a_bad_field_map_in_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first live download is what checks the field maps. When one is
    wrong, the message has to say which keys the row really had."""

    def wrong_schema(spec: DatasetSpec, limit: int | None = None) -> Iterator[dict[str, object]]:
        yield {"qid": "x", "q": "?", "a": "!"}

    monkeypatch.setattr("verisearch.cli.download_rows", wrong_schema)
    code = main(["prepare-data", "--out", str(tmp_path / "p"), "--datasets", "hotpotqa"])

    err = capsys.readouterr().err
    assert code == 2
    assert "Keys present: a, q, qid" in err
    assert err.count("\n") == 1


def test_prepare_data_rejects_an_unknown_dataset() -> None:
    with pytest.raises(SystemExit):
        main(["prepare-data", "--datasets", "squad"])
