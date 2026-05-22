from scripts.check_repo_consistency import run_checks


def test_repository_consistency_contract():
    assert run_checks() == []

