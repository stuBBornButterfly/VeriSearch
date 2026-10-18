"""Tests for corpus construction.

The corpus is the one artefact every retrieval number depends on. A corpus that
is silently missing gold paragraphs caps accuracy at a number that looks exactly
like a model which failed to learn, so the checks here are about catching that
rather than about paragraph plumbing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verisearch.data.build_corpus import (
    CorpusDoc,
    build_corpus,
    build_corpus_from_hf,
    doc_id_for,
    find_missing_titles,
    load_corpus,
    normalize_title,
    paragraphs_from_context,
    summarize_corpus,
    write_corpus,
)
from verisearch.data.datasets import (
    DATASET_SPECS,
    DataError,
    Distribution,
    QAExample,
    spec_for,
)

SPEC = DATASET_SPECS["hotpotqa"]

# The two published shapes of the same two paragraphs.
MAPPING_CONTEXT: dict[str, object] = {
    "title": ["Ada Lovelace", "Analytical Engine"],
    "sentences": [
        ["Ada Lovelace was an English mathematician.", " She wrote notes on the engine."],
        ["The Analytical Engine was designed by Charles Babbage."],
    ],
}
PAIRS_CONTEXT: list[object] = [
    [
        "Ada Lovelace",
        ["Ada Lovelace was an English mathematician.", " She wrote notes on the engine."],
    ],
    ["Analytical Engine", ["The Analytical Engine was designed by Charles Babbage."]],
]


def example(question_id: str, *titles: str) -> QAExample:
    return QAExample(
        question_id=question_id,
        question="who wrote notes on the engine?",
        answers=("Ada Lovelace",),
        dataset="hotpotqa",
        distribution=Distribution.IN_DISTRIBUTION,
        supporting_titles=titles,
    )


# ---------------------------------------------------------------------------
# Title normalisation and ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ada Lovelace", "ada lovelace"),
        ("Ada_Lovelace", "ada lovelace"),
        ("  Ada   Lovelace  ", "ada lovelace"),
        ("ADA LOVELACE", "ada lovelace"),
        ("Ada__Lovelace", "ada lovelace"),
        ("Ada\tLovelace", "ada lovelace"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_title_normalisation_matches_the_eval_rule(raw: str, expected: str) -> None:
    """Pins the rule `eval.metrics.normalize_title` also implements.

    The two are deliberate duplicates - `data/` must not import `eval/` - so this
    table is the contract between them. If someone changes one, this test still
    says what the agreed behaviour was. Punctuation is kept: "Ada (mathematician)"
    and "Ada mathematician" are different Wikipedia pages.
    """
    assert normalize_title(raw) == expected


def test_punctuation_is_not_stripped_from_titles() -> None:
    assert normalize_title("Ada (mathematician)") == "ada (mathematician)"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Ada Lovelace", "Ada_Lovelace"),
        ("Ada Lovelace", "  ada   lovelace "),
        ("ANALYTICAL ENGINE", "Analytical Engine"),
    ],
)
def test_equivalent_titles_with_the_same_text_share_a_doc_id(left: str, right: str) -> None:
    assert doc_id_for(left, "Same text.") == doc_id_for(right, "Same text.")


def test_different_titles_get_different_ids() -> None:
    assert doc_id_for("Ada Lovelace", "Text.") != doc_id_for("Charles Babbage", "Text.")


def test_two_versions_of_one_page_are_two_documents() -> None:
    """The first full build kept one version per title and lost the answering
    sentence for 6.6% of 2WikiMultihopQA's facts. A version is a document."""
    assert doc_id_for("Ada Lovelace", "One version.") != doc_id_for(
        "Ada Lovelace", "Another version."
    )


def test_a_case_collision_between_two_articles_stays_two_documents() -> None:
    """ "Smash Cut" is a film and "Smash cut" an editing technique. They normalise
    to one title, and keyed on the title alone they became one document."""
    film = doc_id_for("Smash Cut", "Smash Cut is a 2009 Canadian horror film.")
    technique = doc_id_for("Smash cut", "A smash cut is a technique in film.")
    assert film != technique


def test_doc_id_shape() -> None:
    doc_id = doc_id_for("Ada Lovelace", "Text.")
    assert len(doc_id) == 16
    assert set(doc_id) <= set("0123456789abcdef")


@pytest.mark.parametrize("title", ["", "   ", "___"])
def test_blank_title_cannot_have_an_id(title: str) -> None:
    with pytest.raises(DataError, match="blank title"):
        doc_id_for(title, "Text.")


def test_doc_id_is_stable_across_processes() -> None:
    """The reason this is `hashlib`, not `hash()`.

    Python randomises string hashing per process unless `PYTHONHASHSEED` is set,
    and the Jenkins pipeline leaves it unset on purpose. An id built from
    `hash()` would therefore change between the run that builds the index and
    the run that queries it - and the symptom is an index that returns nothing,
    with no error anywhere.
    """
    code = (
        "from verisearch.data.build_corpus import doc_id_for; "
        "print(doc_id_for('Ada Lovelace', 'Text.'))"
    )
    outputs = set()
    for seed in ("0", "1", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        outputs.add(result.stdout.strip())
    assert outputs == {doc_id_for("Ada Lovelace", "Text.")}


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------


def test_both_published_context_shapes_agree() -> None:
    """A mirror that ships one shape must not produce a different corpus."""
    from_mapping = paragraphs_from_context(MAPPING_CONTEXT, "context", SPEC)
    from_pairs = paragraphs_from_context(PAIRS_CONTEXT, "context", SPEC)
    assert from_mapping == from_pairs


def test_sentences_are_joined_with_single_spaces() -> None:
    pairs = paragraphs_from_context(MAPPING_CONTEXT, "context", SPEC)
    assert pairs[0] == (
        "Ada Lovelace",
        "Ada Lovelace was an English mathematician. She wrote notes on the engine.",
    )


@pytest.mark.parametrize(
    "sentences",
    [
        ["a.", "b."],
        ["a. ", " b."],
        ["  a.  ", "  b.  "],
        ["a.", "", "b."],
    ],
)
def test_inconsistent_sentence_spacing_normalises_to_one_text(sentences: list[str]) -> None:
    """The published data is not consistent about trailing spaces.

    Joining raw gives "a.b." in some rows and "a.  b." in others, which embeds as
    different text for paragraphs that are supposed to be identical.
    """
    context = {"title": ["T"], "sentences": [sentences]}
    assert paragraphs_from_context(context, "context", SPEC) == (("T", "a. b."),)


def test_bare_string_sentence_group_is_accepted() -> None:
    context = {"title": ["T"], "sentences": ["one sentence"]}
    assert paragraphs_from_context(context, "context", SPEC) == (("T", "one sentence"),)


def test_unusable_sentence_group_yields_empty_text() -> None:
    context = {"title": ["T"], "sentences": [12345]}
    assert paragraphs_from_context(context, "context", SPEC) == (("T", ""),)


def test_non_string_titles_are_skipped() -> None:
    context = {"title": ["T", 7], "sentences": [["a."], ["b."]]}
    assert paragraphs_from_context(context, "context", SPEC) == (("T", "a."),)


def test_title_and_sentence_counts_must_match() -> None:
    context = {"title": ["a", "b"], "sentences": [["x."]]}
    with pytest.raises(DataError, match="2 titles for 1 sentence groups"):
        paragraphs_from_context(context, "context", SPEC)


def test_mapping_without_the_expected_lists_is_rejected() -> None:
    with pytest.raises(DataError, match="must hold 'title' and 'sentences' lists"):
        paragraphs_from_context({"title": "not a list"}, "context", SPEC)


@pytest.mark.parametrize("value", [42, None, 3.5])
def test_context_of_the_wrong_type_is_rejected(value: object) -> None:
    with pytest.raises(DataError, match="expected a mapping or sequence"):
        paragraphs_from_context(value, "context", SPEC)


def test_short_pairs_are_ignored() -> None:
    assert paragraphs_from_context([["lonely-title"]], "context", SPEC) == ()


# ---------------------------------------------------------------------------
# build_corpus
# ---------------------------------------------------------------------------


def test_build_deduplicates_across_rows() -> None:
    build = build_corpus([{"context": MAPPING_CONTEXT}, {"context": MAPPING_CONTEXT}], SPEC)
    assert build.n_rows == 2
    assert build.n_paragraphs == 4
    assert len(build.docs) == 2
    assert build.n_duplicate_paragraphs == 2
    assert build.n_extra_versions == 0


def test_build_keeps_every_version_of_a_paragraph() -> None:
    """The same title with different text is two documents, and counted, because
    several versions of one page can fill a top-k between them."""
    first = {"title": ["T"], "sentences": [["one version."]]}
    second = {"title": ["T"], "sentences": [["a different version."]]}
    build = build_corpus([{"context": first}, {"context": second}], SPEC)
    assert sorted(doc.text for doc in build.docs) == ["a different version.", "one version."]
    assert build.n_extra_versions == 1
    assert build.n_duplicate_paragraphs == 0


def test_the_corpus_does_not_depend_on_row_order() -> None:
    """Keeping the first version seen made the text of a document a function of
    row order. Now every version is kept and, for identical text under two
    spellings of a title, the smaller spelling wins - in any order."""
    rows = [
        {"context": {"title": ["Ada_Lovelace"], "sentences": [["Same."]]}},
        {"context": {"title": ["Ada Lovelace"], "sentences": [["Same."]]}},
        {"context": {"title": ["Ada Lovelace"], "sentences": [["Different."]]}},
    ]
    forward = build_corpus(rows, SPEC)
    backward = build_corpus(list(reversed(rows)), SPEC)
    assert forward.docs == backward.docs
    assert {doc.title for doc in forward.docs} == {"Ada Lovelace"}


def test_build_skips_empty_paragraphs() -> None:
    context = {"title": ["Real", "Blank"], "sentences": [["text."], ["   "]]}
    build = build_corpus([{"context": context}], SPEC)
    assert [doc.title for doc in build.docs] == ["Real"]
    assert build.n_empty_skipped == 1


def test_build_skips_paragraphs_with_blank_titles() -> None:
    context = {"title": ["  ", "Real"], "sentences": [["text."], ["more."]]}
    build = build_corpus([{"context": context}], SPEC)
    assert [doc.title for doc in build.docs] == ["Real"]
    assert build.n_empty_skipped == 1


def test_build_is_order_independent() -> None:
    """Same input, any row order, byte-identical corpus."""
    a = {"context": {"title": ["A"], "sentences": [["alpha."]]}}
    b = {"context": {"title": ["B"], "sentences": [["beta."]]}}
    forward = build_corpus([a, b], SPEC).docs
    backward = build_corpus([b, a], SPEC).docs
    assert forward == backward


def test_docs_are_sorted_by_doc_id() -> None:
    build = build_corpus([{"context": MAPPING_CONTEXT}], SPEC)
    assert [doc.doc_id for doc in build.docs] == sorted(doc.doc_id for doc in build.docs)


def test_missing_context_key_names_the_keys_present() -> None:
    with pytest.raises(DataError, match=r"row has no key 'context'\. Keys present: question"):
        build_corpus([{"question": "q"}], SPEC)


def test_custom_context_key() -> None:
    build = build_corpus([{"paras": MAPPING_CONTEXT}], SPEC, context_key="paras")
    assert len(build.docs) == 2


def test_empty_input_builds_an_empty_corpus() -> None:
    build = build_corpus([], SPEC)
    assert build.docs == ()
    assert build.n_rows == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_summarize_empty_corpus() -> None:
    stats = summarize_corpus([])
    assert stats.n_docs == 0
    assert stats.max_chars == 0


def test_summarize_uses_nearest_rank_percentiles() -> None:
    docs = tuple(
        CorpusDoc(doc_id=f"{i:016x}", title="t", text="x" * length)
        for i, length in enumerate(range(1, 21))
    )
    stats = summarize_corpus(docs)
    # "t " adds 2 characters to every document.
    assert stats.n_docs == 20
    assert stats.min_chars == 3
    assert stats.max_chars == 22
    assert stats.median_chars == 13
    assert stats.p95_chars == 21


def test_summarize_measures_title_plus_text() -> None:
    stats = summarize_corpus([CorpusDoc(doc_id="a", title="ab", text="cde")])
    assert stats.n_chars == len("ab cde")


# ---------------------------------------------------------------------------
# Coverage against the gold paragraphs
# ---------------------------------------------------------------------------


def test_missing_supporting_title_is_reported() -> None:
    build = build_corpus([{"context": MAPPING_CONTEXT}], SPEC)
    missing = find_missing_titles([example("q1", "Ada Lovelace", "Charles Babbage")], build.docs)
    assert missing == ("charles babbage",)


def test_nothing_missing_when_the_corpus_covers_the_gold() -> None:
    build = build_corpus([{"context": MAPPING_CONTEXT}], SPEC)
    assert find_missing_titles([example("q1", "Ada Lovelace")], build.docs) == ()


def test_coverage_check_normalises_both_sides() -> None:
    """An underscore in one dataset and a space in the other is not a miss."""
    build = build_corpus([{"context": {"title": ["Ada_Lovelace"], "sentences": [["x."]]}}], SPEC)
    assert find_missing_titles([example("q1", "Ada Lovelace")], build.docs) == ()


def test_blank_supporting_titles_are_not_counted_as_missing() -> None:
    build = build_corpus([{"context": MAPPING_CONTEXT}], SPEC)
    assert find_missing_titles([example("q1", "   ")], build.docs) == ()


def test_missing_titles_are_sorted_and_deduplicated() -> None:
    build = build_corpus([{"context": MAPPING_CONTEXT}], SPEC)
    missing = find_missing_titles(
        [example("q1", "Zeta", "Alpha"), example("q2", "Alpha")], build.docs
    )
    assert missing == ("alpha", "zeta")


# ---------------------------------------------------------------------------
# JSONL round trip
# ---------------------------------------------------------------------------


def test_write_then_load_round_trips(tmp_path: Path) -> None:
    build = build_corpus([{"context": MAPPING_CONTEXT}], SPEC)
    target = tmp_path / "corpus" / "hotpotqa.jsonl"
    assert write_corpus(build.docs, target) == 2
    assert load_corpus(target) == build.docs


def test_written_file_is_byte_stable(tmp_path: Path) -> None:
    build = build_corpus([{"context": MAPPING_CONTEXT}], SPEC)
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    write_corpus(build.docs, first)
    write_corpus(reversed(build.docs), second)
    assert first.read_bytes() == second.read_bytes()


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    target = tmp_path / "c.jsonl"
    payload = json.dumps({"doc_id": "a", "title": "t", "text": "x"})
    target.write_text(f"\n{payload}\n\n", encoding="utf-8")
    assert len(load_corpus(target)) == 1


def test_missing_file_is_a_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="corpus file not found"):
        load_corpus(tmp_path / "nope.jsonl")


def test_bad_json_reports_the_line_number(tmp_path: Path) -> None:
    target = tmp_path / "d.jsonl"
    good = json.dumps({"doc_id": "a", "title": "t", "text": "x"})
    target.write_text(f"{good}\nnot json\n", encoding="utf-8")
    with pytest.raises(DataError, match=r"d\.jsonl:2: not valid JSON"):
        load_corpus(target)


def test_invalid_record_reports_the_line_number(tmp_path: Path) -> None:
    target = tmp_path / "e.jsonl"
    target.write_text(json.dumps({"doc_id": "a", "title": "t"}) + "\n", encoding="utf-8")
    with pytest.raises(DataError, match=r"e\.jsonl:1:"):
        load_corpus(target)


def test_empty_text_is_rejected_on_load(tmp_path: Path) -> None:
    """`min_length=1` here is the same guard `build_corpus` applies in memory."""
    target = tmp_path / "f.jsonl"
    target.write_text(
        json.dumps({"doc_id": "a", "title": "t", "text": ""}) + "\n", encoding="utf-8"
    )
    with pytest.raises(DataError, match=r"f\.jsonl:1:"):
        load_corpus(target)


# ---------------------------------------------------------------------------
# Hugging Face entry point
# ---------------------------------------------------------------------------


def test_unknown_dataset_lists_the_known_ones() -> None:
    with pytest.raises(DataError, match=r"unknown dataset 'nope'\. Known: 2wikimultihopqa"):
        spec_for("nope")


def test_spec_for_returns_the_named_spec() -> None:
    assert spec_for("hotpotqa").hf_name == "distractor"


def test_the_download_entry_point_validates_before_importing_datasets() -> None:
    """`datasets` is not installed in CI, so a bad name must fail first."""
    with pytest.raises(DataError, match=r"unknown dataset 'nope'"):
        build_corpus_from_hf("nope")
