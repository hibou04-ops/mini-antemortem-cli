from scripts.run_golden_cases import load_expected, run_checks
from scripts.repo_facts import project_facts


def test_golden_cases_pass():
    assert run_checks() == []


def test_golden_cases_cover_all_traps():
    expected = load_expected()
    covered = {case["trap_id"] for case in expected["cases"] if case.get("trap_id")}
    assert covered == set(project_facts().trap_ids)

