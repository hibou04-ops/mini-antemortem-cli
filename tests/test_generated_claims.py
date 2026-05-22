from pathlib import Path

from scripts.generate_readme_claims import build_claim_docs
from scripts.repo_facts import ROOT, project_facts


def test_generated_claim_docs_are_current():
    docs = build_claim_docs()
    for rel, expected in docs.items():
        assert (ROOT / rel).read_text(encoding="utf-8") == expected


def test_generated_claims_include_actual_trap_ids():
    facts = project_facts()
    body = "\n".join(build_claim_docs(facts).values())
    assert facts.trap_count == 9
    for trap_id in facts.trap_ids:
        assert trap_id in body


def test_generated_claims_do_not_hard_code_test_counts():
    for content in build_claim_docs().values():
        assert "passing" not in content.lower()
        assert "tests-" not in content.lower()

