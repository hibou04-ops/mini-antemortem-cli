"""CI gate for the analytical-preflight false-positive corpus.

Every case in ``benchmarks/false_positive/benign_cases.json`` claims the named
trap should remain silent (label GHOST). A regression that flips one to REAL /
NEW / UNRESOLVED is a false positive — this test surfaces the offending case
ids so the diff is easy to read in CI logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_false_positive_audit import MANIFEST, POSITIVE_LABELS, run_audit
from scripts.repo_facts import project_facts


def _report() -> dict:
    return run_audit()


def test_no_unexpected_false_positives_on_benign_corpus():
    """Fail when a benign case starts firing that isn't acknowledged in the manifest."""
    report = _report()
    unexpected = report["unexpected_false_positive_case_ids"]
    assert unexpected == [], (
        "Benign cases unexpectedly fired (not in acknowledged_false_positives): "
        + ", ".join(unexpected)
    )


def test_no_stale_acknowledgements():
    """If an acknowledged FP no longer fires, the entry should be removed."""
    report = _report()
    stale = report["stale_acknowledgement_case_ids"]
    assert stale == [], (
        "Acknowledged false positives that no longer fire (remove from manifest): "
        + ", ".join(stale)
    )


def test_benign_corpus_covers_all_traps():
    report = _report()
    covered = set(report["per_trap"].keys())
    assert covered == set(project_facts().trap_ids), (
        "Benign corpus is missing coverage for: "
        f"{sorted(set(project_facts().trap_ids) - covered)}"
    )


def test_benign_corpus_minimum_three_cases_per_trap():
    """A trap with fewer than 3 benign cases gives a weak FP signal."""
    report = _report()
    thin = {
        trap_id: stats["total_cases"]
        for trap_id, stats in report["per_trap"].items()
        if stats["total_cases"] < 3
    }
    assert thin == {}, f"Traps with under-seeded benign coverage: {thin}"


def test_manifest_positive_labels_constant_matches_doc():
    """The manifest documents which labels count as FPs; keep code + doc aligned."""
    body = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))
    declared_negative = set(body["expected_negative_labels"])
    # The two sets are complements over the known label space.
    assert declared_negative.isdisjoint(POSITIVE_LABELS)
    assert "GHOST" in declared_negative
