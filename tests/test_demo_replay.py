from examples.demo_replay import EXPECTED_PATH, build_demo_output


def test_demo_replay_fixture_is_current():
    assert build_demo_output() == EXPECTED_PATH.read_text(encoding="utf-8")


def test_demo_replay_contains_real_and_ghost_findings():
    output = build_demo_output()
    assert '"label": "REAL"' in output
    assert '"label": "GHOST"' in output
    assert "mini-antemortem-cli check" in output

