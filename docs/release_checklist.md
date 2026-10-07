# Release Checklist

This checklist is no-network unless explicitly stated. It does not publish to
PyPI, push tags, or create a GitHub release.

## Trusted Publishing Setup

PyPI publishing uses GitHub OIDC Trusted Publishing through
`.github/workflows/publish.yml`. PyPI API tokens, passwords, and `__token__`
secrets are not required and must not be added.

External setup that cannot be verified from this repository:

| PyPI Trusted Publisher field | Required value |
|---|---|
| Owner | `hibou04-ops` |
| Repository | `mini-antemortem-cli` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

Use the existing GitHub Environment named `pypi` and preserve its configured
protection rules. If required reviewers are configured, publication waits for
their approval. The workflow grants `id-token: write` only to the publish job;
normal CI never publishes. Do not bypass environment protection.

## Local Preflight

Run these commands before creating a release tag:

1. Verify generated claims:

   ```bash
   python scripts/generate_readme_claims.py --check
   ```

2. Verify repository consistency:

   ```bash
   python scripts/check_repo_consistency.py
   ```

3. Run tests and deterministic artifacts:

   ```bash
   python -m pytest -q
   python examples/demo_replay.py
   python scripts/run_golden_cases.py --check
   python scripts/verify_fixture_integrity.py
   ```

4. Run release audit:

   ```bash
   python scripts/release_audit.py --no-network
   ```

5. Build and smoke install:

   ```bash
   python -m build
   python scripts/wheel_smoke_install.py dist/*.whl
   ```

6. Run publish readiness:

   ```bash
   python scripts/publish_readiness.py --no-network
   ```

If a required executable or package is missing, scripts must report
`TOOLING_MISSING`. If the local environment cannot execute required commands,
report `ENVIRONMENT_BLOCKED`.

## GitHub Publish Sequence

1. Confirm local preflight is green.
2. Commit the release-ready tree.
3. Create an annotated version tag using the `vMAJOR.MINOR.PATCH` format:

   ```bash
   git tag -a v0.10.1 -m "mini-antemortem-cli v0.10.1"
   ```

4. Push the tag after required checks and release authorization:

   ```bash
   git push origin v0.10.1
   ```

5. Tag push alone does not publish. Dispatch the existing workflow from main,
   passing the exact tag; it checks out that tag for verification and build:

   ```bash
   gh workflow run publish.yml --ref main -f release_tag=v0.10.1
   ```
6. The `verify-build` job runs deterministic tests, generated claims,
   repository consistency, demo replay, golden cases, fixture integrity,
   release audit, `python -m build`, wheel smoke install, and
   `publish_readiness.py --no-network`.
7. The `publish` job obeys the `pypi` environment rules and then uses
   `pypa/gh-action-pypi-publish` with Trusted Publishing. No password or token
   arguments are supplied.

Manual dispatch is the only publish trigger, with an existing tag input.
The workflow rejects refs that do not match `v*.*.*`.

## Rollback / Failure Notes

- If verification fails before publishing, fix the repository and create a new
  tag. Do not reuse a failed release tag unless no artifact was published and
  project policy explicitly permits moving that tag.
- If PyPI publishing fails before upload, fix the Trusted Publisher setup or
  GitHub Environment approval and rerun the workflow for the same tag.
- If an artifact is published to PyPI, it cannot be overwritten. Fix forward by
  incrementing the version in `pyproject.toml`, regenerating claims, and
  creating a new tag.
- Do not create a GitHub release until post-release verification confirms the
  intended PyPI artifact is available.
