"""Command line entry points for the scripts in `scripts/`.

**Why this file exists, given that PROJECT.md section 4 does not list it.**

Section 4 names five shell scripts. A shell script that decides anything is
outside every standard in section 3: bash has no types, `mypy` never sees it,
`ruff` barely does, and `pytest` cannot drive it. And these scripts decide
things that matter - which index a report was measured against, whether a sweep
is complete, whether an index is about to be overwritten.

`files = ["verisearch"]` in `pyproject.toml` is the boundary mypy enforces, and
`scripts/` is outside it. Pointing mypy at `scripts/00_setup_check.py` today
reports five errors, so that is not a boundary this PR can simply move.

So the decisions live here, inside the checked package, and each `.sh` file is a
wrapper: `set -euo pipefail`, defaults, and one `python -m verisearch.cli` call.

Every command returns an exit code and never calls `sys.exit`, so a test can
drive it. Expected failures print one line to stderr; a traceback reaching the
user means a bug here, not bad input.

Owner: Ayesha.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from verisearch.config import DEFAULT_CONFIG_PATH, ConfigError, load_config, load_sweep
from verisearch.data.build_corpus import load_corpus, summarize_corpus
from verisearch.data.datasets import (
    DATASET_SPECS,
    DataError,
    Split,
    load_jsonl,
    spec_for,
    write_jsonl,
)
from verisearch.data.prepare import (
    MANIFEST_FILENAME,
    assemble,
    download_rows,
    read_dataset,
    write_prepared,
)
from verisearch.eval.baselines import (
    BaselineKind,
    EvalContext,
    EvalError,
    context_for,
    examples_fingerprint,
)
from verisearch.eval.frontier import (
    Frontier,
    diagnose,
    frontier_table,
    load_sweep_results,
    pareto_frontier,
)
from verisearch.eval.harness import EvalRun, aggregate, load_run, read_results, write_report
from verisearch.eval.regression import RegressionStatus, check_regression, regression_set
from verisearch.retrieval.client import InMemoryDocStore, RetrievalError
from verisearch.retrieval.embedder import DEFAULT_BATCH_SIZE, EmbedderError, EmbedderSpec
from verisearch.retrieval.index import (
    DEFAULT_EF_CONSTRUCTION,
    DEFAULT_EF_SEARCH,
    DEFAULT_HNSW_M,
    BruteForceIndex,
    VectorIndexError,
    load_meta,
)
from verisearch.serve.benchmark import BenchError, HttpPolicyClient, run_benchmark, write_bench
from verisearch.serve.server import ServeError

if TYPE_CHECKING:
    from collections.abc import Sequence

# PROJECT.md section 5. Repeated here as the CLI default rather than imported
# from a config section that does not exist yet - `TODO` in index.py tracks
# moving the retrieval constants into configs/base.yaml.
DEFAULT_EMBEDDER_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDER_DIM = 384

CONTEXT_FILENAME = "context.json"
# What `03_eval.sh` writes for every run, and so what a promoted reference holds.
REPORT_JSON = "report.json"
RESULTS_JSONL = "results.jsonl"

# Every expected failure in this module is one of these. Anything else is a bug
# and should reach the user as a traceback rather than a tidy message.
_EXPECTED = (
    BenchError,
    ConfigError,
    DataError,
    EmbedderError,
    EvalError,
    RetrievalError,
    ServeError,
    VectorIndexError,
    OSError,
)


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _spec_from_args(args: argparse.Namespace) -> EmbedderSpec:
    """Build the embedder spec the index will be fingerprinted with.

    Every field here changes the vectors, which is why `EmbedderSpec` exists and
    why its fingerprint is written into the index. A flag typo produces a
    different fingerprint, which is loud; a flag typo that did *not* change the
    fingerprint would be silent, which is why `batch_size` is excluded from it.
    """
    instruction: str | None = args.query_instruction
    model: str = args.model
    dim: int = args.dim
    batch_size: int = args.batch_size
    normalize: bool = not args.no_normalize
    return EmbedderSpec(
        model_name=model,
        dim=dim,
        normalize=normalize,
        query_instruction=instruction,
        batch_size=batch_size,
    )


def _load_embedder(spec: EmbedderSpec) -> object:  # pragma: no cover - needs [retrieval]
    """Construct the real embedder. The one line in this module that needs torch.

    Isolated so that everything above and below it - argument resolution, the
    overwrite guard, the corpus load, the stats, the index write - is exercised
    by `tests/test_cli.py` without the extra installed.
    """
    from verisearch.retrieval.embedder import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(spec)


def prepare_data(args: argparse.Namespace) -> int:
    """`scripts/00_prepare_data.sh`. Download, split, build the corpus, check coverage.

    The step before `01_build_index.sh`, and the only one that needs the
    network. `--plan` names what would be downloaded and stops before any of
    it, so it needs neither the network nor the `[data]` extra.

    Exits 1, after writing everything, when a supporting title has no document
    in the corpus - the files exist for diagnosis, but recall on them is capped
    and the exit code says so. Same convention as `frontier` and `serve-bench`.
    """
    out_dir = Path(args.out)
    # argparse `choices` has already refused an unknown name.
    specs = [spec_for(name) for name in args.datasets]

    for spec in specs:
        name = f"{spec.hf_path}" + (f" ({spec.hf_name})" if spec.hf_name else "")
        corpus = f"corpus from {spec.context_key!r}" if spec.context_key else "no paragraphs"
        print(f"{spec.name:<16} {name}  splits {', '.join(spec.hf_splits)}  {corpus}")
    print(f"seed        {args.seed}")
    print(f"limit       {args.limit if args.limit is not None else 'none - the full pool'}")
    print(f"out         {out_dir}")

    if args.plan:
        print("\nplan only, nothing downloaded or written.")
        return 0
    if (out_dir / MANIFEST_FILENAME).exists() and not args.force:
        return _fail(
            f"{out_dir} already holds a finished build. Every eval context and index built "
            f"from it fingerprints these files; pass --force to rebuild anyway."
        )

    try:
        parts = [read_dataset(download_rows(spec, args.limit), spec) for spec in specs]
        prepared = assemble(parts, seed=args.seed, limit=args.limit)
    except _EXPECTED as exc:
        return _fail(str(exc))

    print(f"\n{prepared.manifest.summary()}")
    for path in write_prepared(prepared, out_dir):
        print(f"wrote {path}")

    problems = prepared.manifest.problems()
    for problem in problems:
        print(f"not usable as-is - {problem}", file=sys.stderr)
    return 1 if problems else 0


def build_index(args: argparse.Namespace) -> int:
    """`scripts/01_build_index.sh`. Corpus JSONL in, index directory out."""
    corpus_path = Path(args.corpus)
    out_dir = Path(args.out)
    spec = _spec_from_args(args)

    if (out_dir / "meta.json").is_file() and not args.force:
        # The guard worth having. An index overwritten partway through a sweep
        # invalidates every report that already cited its fingerprint, and the
        # reports stay on disk looking perfectly valid.
        return _fail(
            f"{out_dir} already holds an index. Overwriting it invalidates every report "
            f"that cites its fingerprint, and those reports will not notice. Pass --force "
            f"if that is what you mean, or build into a new directory."
        )

    try:
        docs = load_corpus(corpus_path)
        stats = summarize_corpus(docs)
        # Built and discarded: it refuses a duplicate doc_id, which would make
        # every hit on that id resolve by read order. Cheaper to find now than
        # after embedding the whole corpus.
        store = InMemoryDocStore.from_records(docs)
    except _EXPECTED as exc:
        return _fail(str(exc))

    print(f"corpus      {corpus_path} ({len(docs)} docs, {len(store)} unique ids)")
    print(
        f"lengths     min {stats.min_chars}, median {stats.median_chars}, "
        f"p95 {stats.p95_chars}, max {stats.max_chars} chars"
    )
    print(f"embedder    {spec.model_name} dim={spec.dim} fingerprint={spec.fingerprint}")
    print(f"index       {args.index_type} -> {out_dir}")

    if args.plan:
        print("\nplan only, nothing written. Drop --plan to build.")
        return 0

    return _embed_and_save(args, [doc.doc_id for doc in docs], [doc.text for doc in docs], spec)


def _embed_and_save(  # pragma: no cover - needs the [retrieval] extra
    args: argparse.Namespace,
    doc_ids: list[str],
    texts: list[str],
    spec: EmbedderSpec,
) -> int:
    """Embed the corpus and write the index. The only part that needs torch.

    Split out so the pragma covers exactly the lines that cannot run in CI.
    Everything above it in `build_index` - the overwrite guard, the corpus load,
    the duplicate-id check, the stats, the fingerprint - is tested.
    """
    out_dir = Path(args.out)
    try:
        embedder = _load_embedder(spec)
        vectors = embedder.encode_passages(texts)  # type: ignore[attr-defined]
        index = _build(args, vectors, doc_ids, spec)
        index.save(out_dir)
    except _EXPECTED as exc:
        return _fail(str(exc))

    print(f"\nwrote {out_dir}")
    print(f"fingerprint {spec.fingerprint} - every eval run must report this one")
    return 0


def _build(  # pragma: no cover - only reached once vectors exist
    args: argparse.Namespace,
    vectors: object,
    doc_ids: list[str],
    spec: EmbedderSpec,
) -> BruteForceIndex:
    if args.index_type == "hnsw":
        from verisearch.retrieval.index import FaissHnswIndex

        return FaissHnswIndex.build(  # type: ignore[return-value]
            vectors,  # type: ignore[arg-type]
            doc_ids,
            spec,
            m=args.hnsw_m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
    return BruteForceIndex.build(vectors, doc_ids, spec)  # type: ignore[arg-type]


def eval_context(args: argparse.Namespace) -> int:
    """`scripts/03_eval.sh`, first half. Resolve the context every run shares.

    **The keystone of section 8.** "All on identical eval sets with the identical
    retriever" is a promise until the retriever fingerprint is read from the
    index itself rather than typed on a command line. This command reads
    `meta.json`, pairs it with the eval set, and writes one `context.json` that
    every baseline run then loads.

    Written once, shared by all five runs. If it were derived per run, two runs
    could disagree about what they measured and still compare cleanly.
    """
    index_dir = Path(args.index)
    examples_path = Path(args.examples)
    out_path = Path(args.out)

    try:
        meta = load_meta(index_dir)
        examples = load_jsonl(examples_path)
        context = context_for(
            [example.question_id for example in examples],
            dataset=args.dataset,
            split=args.split,
            retriever_fingerprint=meta.embedder_fingerprint,
            k=args.k,
        )
    except _EXPECTED as exc:
        return _fail(str(exc))

    print(f"index       {index_dir} ({meta.index_type}, {meta.n_vectors} vectors)")
    print(f"embedder    {meta.embedder_model} fingerprint={meta.embedder_fingerprint}")
    print(f"eval set    {examples_path} ({context.n_examples} examples)")
    print(f"examples    fingerprint={context.examples_fingerprint}")
    print(f"k           {context.k}")

    if args.plan:
        print("\nplan only, nothing written.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print("every baseline run must load this file, not rebuild it")
    return 0


def eval_report(args: argparse.Namespace) -> int:
    """`scripts/03_eval.sh`, second half. Scored rows in, section 9 table out."""
    results_path = Path(args.results)
    context_path = Path(args.context)
    out_path = Path(args.out)

    try:
        context = EvalContext.model_validate_json(Path(context_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _fail(f"{context_path} is not an eval context: {exc}")

    try:
        results = read_results(results_path)
        report = aggregate(
            results,
            run_name=args.run_name,
            kind=BaselineKind(args.kind),
            context=context,
            resamples=args.resamples,
            seed=args.seed,
        )
    except _EXPECTED as exc:
        return _fail(str(exc))

    print(report.summary())
    if args.plan:
        print("\nplan only, nothing written.")
        return 0

    write_report(report, out_path)
    print(f"\nwrote {out_path}")
    return 0


def frontier(args: argparse.Namespace) -> int:
    """`scripts/04_sweep.sh`, final step. Manifest in, frontier table out.

    Refuses an incomplete sweep and a sweep whose points were measured
    differently, both inside `load_sweep_results`. Exit code 1 rather than 0
    when the diagnostics say there is no trade-off: a sweep that did not measure
    one is a result the script should not quietly pass on to a figure.
    """
    manifest_path = Path(args.manifest)

    try:
        spec = load_sweep(manifest_path)
        points, _ = load_sweep_results(spec, root=args.root)
    except _EXPECTED as exc:
        return _fail(str(exc))

    curve = pareto_frontier(points)
    diagnostics = diagnose(points)
    table = frontier_table(curve, diagnostics)
    print(table)

    if not args.plan:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table + "\n", encoding="utf-8")
        print(f"\nwrote {out_path}")

        if args.figure is not None:
            # `no branch`: the false arm needs matplotlib, so CI can only ever
            # take the failure path. The test covers both arms conditionally on
            # whether the [plot] extra happens to be installed.
            figure_code = _write_figure(curve, args.figure, spec.name)
            if figure_code != 0:  # pragma: no branch
                return figure_code

    if not diagnostics.is_a_frontier:
        print(
            "\nnot writing this up as a frontier: the sweep did not measure a trade-off",
            file=sys.stderr,
        )
        return 1
    return 0


def _write_figure(curve: Frontier, path: str, title: str) -> int:
    """Draw the PNG. Needs the [plot] extra; the table is already on disk.

    A missing extra must not cost the numbers, so this is the last thing the
    command does and its failure is reported without discarding the table.
    """
    try:
        from verisearch.eval.frontier import save_frontier

        written = save_frontier(curve, path, title=title)
    except _EXPECTED as exc:
        return _fail(str(exc))
    print(f"wrote {written}")  # pragma: no cover - needs the [plot] extra
    return 0  # pragma: no cover - needs the [plot] extra


def regression_set_command(args: argparse.Namespace) -> int:
    """Write the fixed set `eval-regression` measures. Run once, then keep the file.

    The file is the contract, the way `context.json` is for an experiment: the
    reference and every nightly run are scored on exactly these questions. So it
    refuses to overwrite one that exists. Rewriting it - even deterministically,
    from a dev file that has since changed - makes the reference incomparable,
    and `eval-regression` would then refuse every night until a new reference
    is promoted.
    """
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        return _fail(
            f"{out_path} already exists. It is the set the reference was scored on; "
            f"pass --force only alongside promoting a new reference."
        )
    try:
        examples = load_jsonl(Path(args.examples))
        chosen = regression_set(examples, args.n, seed=args.seed, split=Split(args.split))
    except _EXPECTED as exc:
        return _fail(str(exc))

    fingerprint = examples_fingerprint(example.question_id for example in chosen)
    print(f"source      {args.examples}")
    print(f"chosen      {len(chosen)} {args.split} examples, seed {args.seed}")
    print(f"examples    fingerprint={fingerprint}")
    if args.plan:
        print("\nplan only, nothing written.")
        return 0
    write_jsonl(chosen, out_path)
    print(f"\nwrote {out_path}")
    return 0


def eval_regression(args: argparse.Namespace) -> int:
    """`scripts/03_eval.sh --regression`. Section 10's nightly `eval-regression`.

    Exit 0 when F1 held or the drop is within noise, 1 when it regressed, 2 when
    the comparison could not be made at all. A comparison that cannot be made is
    never a pass: a missing reference, a reference from another set or another
    index, and candidate rows for the wrong questions all exit 2.

    The candidate's context is rebuilt here from the index's own `meta.json` and
    the regression set, rather than copied from the reference - otherwise
    `check_comparable` would be comparing the reference with itself.
    """
    reference_dir = Path(args.reference)
    try:
        examples = load_jsonl(Path(args.examples))
        if len(examples) != args.n:
            return _fail(
                f"{args.examples} holds {len(examples)} examples, not the {args.n} this gate "
                f"is defined on. Build it with `regression-set --n {args.n}`."
            )
        meta = load_meta(Path(args.index))
        context = context_for(
            [example.question_id for example in examples],
            dataset=args.dataset,
            split=args.split,
            retriever_fingerprint=meta.embedder_fingerprint,
            k=args.k,
        )
        reference = load_run(reference_dir / REPORT_JSON, reference_dir / RESULTS_JSONL)
        rows = read_results(Path(args.candidate))
        report = aggregate(
            rows,
            run_name=args.run_name,
            kind=reference.report.kind,
            context=context,
            resamples=args.resamples,
            seed=args.seed,
        )
        verdict = check_regression(
            reference,
            EvalRun(report=report, results=rows),
            max_drop=args.max_f1_drop / 100.0,
            resamples=args.resamples,
            seed=args.seed,
        )
    except _EXPECTED as exc:
        return _fail(str(exc))

    print(verdict.summary())
    if verdict.status is RegressionStatus.REGRESSED:
        print(
            f"\nREGRESSED: F1 fell {-verdict.f1_delta.point * 100:.1f} pts and the interval "
            f"excludes zero",
            file=sys.stderr,
        )
        return 1
    return 0


def serve_bench(args: argparse.Namespace) -> int:
    """`scripts/05_serve_bench.sh`. Section 9's serving row, at every configured level.

    Reads what it is measuring from the server's `/health` - policy, adapter,
    hardware - so the footer under the table cannot describe a machine other
    than the one that answered. `--plan` goes as far as that read and stops
    before any load is sent.

    Exits 1, after writing everything, when `BenchReport.problems` finds a
    reason the table is not section 9's: a failed request, a level that never
    reached its concurrency, or a p99 with too few requests behind it. Same
    convention as `frontier`: the numbers are kept, the exit code refuses them.
    """
    examples_path = Path(args.examples)
    try:
        config = load_config(*args.config)
        serve = config.serve
        url = args.url if args.url is not None else serve.url
        examples = load_jsonl(examples_path)
        client, server = HttpPolicyClient.connect(url, timeout=serve.request_timeout_s)
    except _EXPECTED as exc:
        return _fail(str(exc))

    adapter = server.adapter if server.adapter is not None else "none (base model)"
    per_level = serve.warmup_requests + serve.requests_per_level
    print(f"server      {url} (protocol {server.protocol_version})")
    print(f"policy      {server.policy}, adapter {adapter}")
    print(f"hardware    {server.hardware}")
    print(f"config      server {server.config_fingerprint}, this run {config.fingerprint}")
    print(f"levels      {', '.join(str(level) for level in serve.concurrency_levels)}")
    print(
        f"requests    {serve.requests_per_level} per level after {serve.warmup_requests} "
        f"warmup, {per_level * len(serve.concurrency_levels)} in all"
    )
    print(f"questions   {len(examples)} from {examples_path}")

    if args.plan:
        print("\nplan only, no load sent, nothing written.")
        return 0

    try:
        run = run_benchmark(
            client,
            server,
            [example.question for example in examples],
            levels=serve.concurrency_levels,
            requests_per_level=serve.requests_per_level,
            warmup=serve.warmup_requests,
            request_timeout_s=serve.request_timeout_s,
            batches=serve.throughput_batches,
            resamples=config.eval.bootstrap_resamples,
            confidence=config.eval.confidence_level,
            seed=config.seed,
        )
    except _EXPECTED as exc:
        return _fail(str(exc))

    table = run.report.table()
    print(f"\n{table}")
    report_path, requests_path = write_bench(run, args.out)
    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nwrote {report_path}\nwrote {requests_path}\nwrote {table_path}")

    problems = run.report.problems()
    for problem in problems:
        print(f"not section 9's table - {problem}", file=sys.stderr)
    return 1 if problems else 0


def _add_prepare_data(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "prepare-data", help="download the datasets, split them, and build the corpus"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SPECS),
        default=list(DATASET_SPECS),
        help="PROJECT.md section 7's four by default, in this order - the order decides "
        "which text wins when two datasets share a title",
    )
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seed", type=int, default=42, help="split seed; configs/base.yaml seed")
    parser.add_argument(
        "--limit", type=int, default=None, help="rows per HF split, for a smoke build only"
    )
    parser.add_argument("--force", action="store_true", help="overwrite a finished build")
    parser.set_defaults(handler=prepare_data)


def _add_build_index(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("build-index", help="embed a corpus and build a vector index")
    parser.add_argument("--corpus", required=True, help="corpus JSONL from build_corpus")
    parser.add_argument("--out", required=True, help="directory to write the index into")
    parser.add_argument("--model", default=DEFAULT_EMBEDDER_MODEL)
    parser.add_argument("--dim", type=int, default=DEFAULT_EMBEDDER_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--query-instruction",
        default=None,
        help="BGE query prefix. Default off - BAAI report v1.5 works without it, "
        "and it is untested here. Measure both before trusting either.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="store raw vectors. Inner product then stops equalling cosine.",
    )
    parser.add_argument("--index-type", choices=("brute", "hnsw"), default="hnsw")
    parser.add_argument("--hnsw-m", type=int, default=DEFAULT_HNSW_M)
    parser.add_argument("--ef-construction", type=int, default=DEFAULT_EF_CONSTRUCTION)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument(
        "--force", action="store_true", help="overwrite an index that is already there"
    )
    parser.set_defaults(handler=build_index)


def _add_eval_context(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "eval-context", help="resolve the shared context every baseline run must use"
    )
    parser.add_argument("--index", required=True, help="index directory, read for its fingerprint")
    parser.add_argument("--examples", required=True, help="eval set JSONL")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default=CONTEXT_FILENAME)
    parser.set_defaults(handler=eval_context)


def _add_eval_report(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval-report", help="aggregate scored rows into a report")
    parser.add_argument("--results", required=True, help="results JSONL from the harness")
    parser.add_argument("--context", required=True, help="context.json from eval-context")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--kind", choices=tuple(kind.value for kind in BaselineKind), required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="report.json")
    parser.set_defaults(handler=eval_report)


def _add_frontier(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("frontier", help="build the accuracy vs retrieval frontier")
    parser.add_argument("--manifest", required=True, help="sweep manifest YAML")
    parser.add_argument("--root", default=None, help="resolves the manifest's relative paths")
    parser.add_argument("--out", default="results/frontier.md")
    parser.add_argument("--figure", default=None, help="also write a PNG; needs the [plot] extra")
    parser.set_defaults(handler=frontier)


def _add_regression_set(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "regression-set", help="write the fixed example set the nightly regression gate uses"
    )
    parser.add_argument("--examples", required=True, help="processed examples JSONL, splits set")
    parser.add_argument("--n", type=int, default=100, help="section 10: 100")
    parser.add_argument("--split", choices=tuple(split.value for split in Split), default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/regression/examples.jsonl")
    parser.add_argument("--force", action="store_true", help="overwrite an existing set")
    parser.set_defaults(handler=regression_set_command)


def _add_eval_regression(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "eval-regression", help="compare tonight's run with the reference on the fixed set"
    )
    parser.add_argument("--examples", required=True, help="the set from regression-set")
    parser.add_argument("--index", required=True, help="index directory, read for its fingerprint")
    parser.add_argument(
        "--reference", required=True, help="directory holding report.json and results.jsonl"
    )
    parser.add_argument("--candidate", required=True, help="tonight's results JSONL on the set")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=100, help="the size the gate is defined on")
    parser.add_argument(
        "--max-f1-drop", type=float, default=2.0, help="in F1 points; section 10 says 2"
    )
    parser.add_argument("--run-name", default="nightly")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(handler=eval_regression)


def _add_serve_bench(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "serve-bench", help="load-benchmark the policy server at every configured concurrency"
    )
    parser.add_argument(
        "--examples", required=True, help="eval set JSONL; its questions are the load"
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=[str(DEFAULT_CONFIG_PATH)],
        help="config YAML files, merged left to right; levels and counts come from serve:",
    )
    parser.add_argument("--url", default=None, help="policy server; default serve.url")
    parser.add_argument(
        "--out", default="results/raw/serve", help="directory for bench.json and requests.jsonl"
    )
    parser.add_argument("--table", default="results/tables/serving.md")
    parser.set_defaults(handler=serve_bench)


_DESCRIPTION = """Entry points for the scripts in scripts/. The decisions live here rather
than in bash because mypy, ruff and pytest reach this package and do not reach
a shell script. Pass --plan before the subcommand to resolve and print what
would happen without writing anything."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m verisearch.cli", description=_DESCRIPTION)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="resolve and print what would happen, write nothing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare_data(subparsers)
    _add_build_index(subparsers)
    _add_eval_context(subparsers)
    _add_eval_report(subparsers)
    _add_frontier(subparsers)
    _add_regression_set(subparsers)
    _add_eval_regression(subparsers)
    _add_serve_bench(subparsers)
    return parser


def _force_utf8() -> None:
    """Print UTF-8 regardless of the console's codepage.

    A Windows console defaults to cp1252, and Python then writes stderr with
    `backslashreplace` - so the frontier table's lambda column arrives as a
    literal `\u03bb`. The numbers are still right and the output is unreadable.

    Guarded by an isinstance check rather than a try/except: under pytest,
    `sys.stdout` is a capture object with no `reconfigure`.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch. Returns an exit code and never raises `SystemExit` itself.

    `--plan` is global and comes before the subcommand, so every command can be
    driven to the point just before it writes anything. That is what makes the
    shell scripts auditable: `01_build_index.sh --plan` prints the fingerprint
    the index will carry without spending an hour embedding a corpus.
    """
    _force_utf8()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = args.handler
    result: int = handler(args)
    return result


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
