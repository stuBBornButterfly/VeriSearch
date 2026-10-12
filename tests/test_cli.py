"""Tests for `verisearch/cli.py`.

`main` returns an exit code and never raises `SystemExit`, so every command is
driven directly here with an argv list. That is the reason the logic is in the
package rather than in the `.sh` files: a bash script that decides anything
cannot be tested, and these decide which index a report was measured against and
whether a sweep is complete.

Three of the four commands run end to end with no optional extras installed:
`eval-context`, `eval-report` and `frontier`. `build-index` is driven to the
point just before the embedder loads, which is the only line in the module that
needs torch.

Every assertion on stderr matters: an expected failure must print one line and
return 2, not raise. A traceback reaching the user is a bug in the CLI.
"""

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from verisearch.cli import CONTEXT_FILENAME, main
from verisearch.data.datasets import Distribution, QAExample, Split, write_jsonl
from verisearch.eval.baselines import BaselineKind, EvalContext, context_for
from verisearch.eval.frontier import REPORT_FILENAME
from verisearch.eval.harness import EvalReport, score_example, write_results
from verisearch.eval.metrics import ConfidenceInterval, FailureReason
from verisearch.retrieval.embedder import EmbedderSpec, l2_normalize
from verisearch.retrieval.index import BruteForceIndex
from verisearch.types import Passage, Termination, Trajectory, Turn, TurnKind

if TYPE_CHECKING:
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
