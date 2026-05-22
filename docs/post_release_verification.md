# Post-Release Verification

Default mode is no-network and only verifies local artifacts. It must not claim
that PyPI or GitHub release state is correct unless a network check is actually
performed.

## No-Network Default

```bash
python scripts/post_release_verify.py
```

This checks local generated docs, repository consistency, and fixture integrity.
It reports that remote PyPI/GitHub checks were skipped.

## Optional Network Path

```bash
python scripts/post_release_verify.py --network
```

The network path may query remote package metadata if network access is
available. Failure to reach the network is not release approval; it is an
environment/tooling issue.

## Trusted Publishing Reminder

Post-release verification does not configure PyPI Trusted Publishing. Before
the first publish, PyPI must have a Trusted Publisher entry for owner
`hibou04-ops`, repository `mini-antemortem-cli`, workflow `publish.yml`, and
environment `pypi`. No PyPI token secret is required.
