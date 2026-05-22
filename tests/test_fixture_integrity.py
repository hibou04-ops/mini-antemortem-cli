from pathlib import Path

from scripts.verify_fixture_integrity import sha256_file, verify_manifest


def test_fixture_integrity_manifest_is_current():
    assert verify_manifest() == []


def test_fixture_integrity_detects_tampering(tmp_path: Path):
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("actual", encoding="utf-8")
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text("0" * 64 + "  benchmarks/golden_cases/expected_cases.json\n", encoding="utf-8")

    errors = verify_manifest(root=Path.cwd(), manifest=manifest)
    assert errors


def test_sha256_file_changes_with_content(tmp_path: Path):
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("a", encoding="utf-8")
    first = sha256_file(fixture)
    fixture.write_text("b", encoding="utf-8")
    assert sha256_file(fixture) != first

