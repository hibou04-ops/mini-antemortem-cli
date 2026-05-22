"""Verify deterministic fixture SHA-256 digests."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "golden_cases" / "manifest.sha256"
FIXTURE_FILES = (
    Path("benchmarks/golden_cases/expected_cases.json"),
    Path("examples/_demo_output.txt"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_manifest_lines(root: Path = ROOT) -> list[str]:
    return [f"{sha256_file(root / rel)}  {rel.as_posix()}" for rel in FIXTURE_FILES]


def load_manifest(manifest: Path = MANIFEST) -> dict[Path, str]:
    entries: dict[Path, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        entries[Path(rel.strip())] = digest
    return entries


def verify_manifest(root: Path = ROOT, manifest: Path = MANIFEST) -> list[str]:
    if not manifest.exists():
        return [f"missing manifest: {manifest}"]
    entries = load_manifest(manifest)
    errors: list[str] = []
    for rel in FIXTURE_FILES:
        path = root / rel
        if rel not in entries:
            errors.append(f"manifest missing fixture: {rel.as_posix()}")
            continue
        if not path.exists():
            errors.append(f"fixture missing: {rel.as_posix()}")
            continue
        actual = sha256_file(path)
        if actual != entries[rel]:
            errors.append(
                f"digest mismatch for {rel.as_posix()}: expected {entries[rel]}, got {actual}"
            )
    extra = sorted(set(entries) - set(FIXTURE_FILES))
    for rel in extra:
        errors.append(f"manifest contains unknown fixture: {rel.as_posix()}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Rewrite manifest with current digests.")
    args = parser.parse_args(argv)

    if args.update:
        MANIFEST.write_text("\n".join(expected_manifest_lines()) + "\n", encoding="utf-8")
        print(f"updated {MANIFEST.relative_to(ROOT).as_posix()}")
        return 0

    errors = verify_manifest()
    if errors:
        for error in errors:
            print(f"FIXTURE_INTEGRITY_FAIL: {error}", file=sys.stderr)
        return 1
    print("FIXTURE_INTEGRITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

