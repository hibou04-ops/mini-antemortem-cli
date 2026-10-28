"""Development regressions for typed exposure, explicit policy, and paired scope."""
import json
from unittest.mock import patch

import pytest
from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, JudgeRubric
from omegaprompt.domain.params import PromptVariants

from mini_antemortem_cli.traps import (
    TrapPolicy, _max_pairwise_jaccard, _normalize_example_field,
    _tokenize_for_jaccard, analytical_preflight,
)


def run(*, train=None, test=None, shots=(), **kwargs):
    return {f.trap_id: f for f in analytical_preflight(
        target_provider="openai", target_model="a", judge_provider="anthropic", judge_model="b",
        train_dataset=train or Dataset(items=[DatasetItem(id="train", input="unrelated train")]),
        test_dataset=test, rubric=JudgeRubric(dimensions=[Dimension(name="quality", description="coherent", weight=1)]),
        variants=PromptVariants(system_prompts=["Answer precisely", "Explain the reasoning carefully"], few_shot_examples=list(shots)),
        **kwargs,
    )}


@pytest.mark.parametrize("label", ["yes", "positive", "0", "42"])
def test_shared_output_label_does_not_expose_unrelated_input(label):
    ds=Dataset(items=[DatasetItem(id="eval", input="different question", reference=label)])
    result=run(test=ds, shots=[{"input": "training question", "output": label}])
    assert result["few_shot_leakage_into_test"].label == "GHOST"


def test_input_and_answer_exposure_is_detected_but_unused_pool_is_distinguished():
    ds=Dataset(items=[DatasetItem(id="eval", input="same question", reference="yes")])
    shots=[{"input": "same question", "output": "yes"}, {"input":"other question","output":"no"}]
    potential=run(test=ds, shots=shots)["few_shot_leakage_into_test"]
    actual=run(test=ds, shots=shots, selected_few_shot_indices=[0])["few_shot_leakage_into_test"]
    unused=run(test=ds, shots=shots, selected_few_shot_indices=[1])["few_shot_leakage_into_test"]
    assert potential.label == actual.label == "REAL"
    assert "potential exposure" in potential.note
    assert "actual input exposure" in actual.note
    assert unused.label == "GHOST"


def test_normalization_preserves_types_case_and_structural_equality():
    norm=_normalize_example_field
    assert norm({"a":1,"b":["é"]}) == norm({"b":["e\u0301"],"a":1})
    assert norm(1) != norm("1")
    assert norm(True) != norm(1)
    assert norm("Code_A") != norm("code_a")
    assert norm({"a":1}) != norm('{"a": 1}')
    assert norm(None) == norm("") == ""
    assert norm("  alpha\t beta ") == norm("alpha beta")
    with pytest.raises(ValueError): norm(float("nan"))


@pytest.mark.parametrize("kwargs", [
    {"min_test_items_high": True}, {"min_total_items_medium":-1},
    {"near_duplicate_jaccard":float("nan")}, {"near_duplicate_jaccard":1.01},
    {"rubric_concentration_threshold":"0.8"}, {"fail_on_missing_test":"false"},
])
def test_invalid_policy_values_fail_loudly(kwargs):
    with pytest.raises(ValueError): TrapPolicy(**kwargs)


def test_policy_json_unknown_key_is_not_silently_discarded(tmp_path):
    path=tmp_path/"policy.json"
    path.write_text(json.dumps({"fail_on_mising_test":False}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown policy field"): TrapPolicy.from_json_file(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"): TrapPolicy.from_json_file(path)


def test_missing_test_policy_has_actual_effect():
    assert run()["no_held_out_slice"].label == "REAL"
    finding=run(policy=TrapPolicy(fail_on_missing_test=False))["no_held_out_slice"]
    assert finding.label == "GHOST"
    assert "unmeasured" in finding.note


def test_paired_mode_checks_content_not_only_mode_or_ids():
    train=Dataset(items=[DatasetItem(id="a",input="first"), DatasetItem(id="b",input="second")])
    same=Dataset(items=list(reversed(train.items)))
    paired=run(train=train,test=same,validation_mode="paired")["train_test_id_overlap"]
    assert paired.label == "UNRESOLVED" and paired.severity.value == "medium"
    for mode in ("auto","disjoint"):
        assert run(train=train,test=same,validation_mode=mode)["train_test_id_overlap"].severity.value == "blocker"
    changed=Dataset(items=[DatasetItem(id="a",input="changed"),train.items[1]])
    partial=Dataset(items=[train.items[0]])
    for ds in (changed,partial):
        assert run(train=train,test=ds,validation_mode="paired")["train_test_id_overlap"].severity.value == "blocker"


@pytest.mark.parametrize("mode", ["auto","paired","disjoint"])
def test_content_overlap_survives_id_renaming(mode):
    train=Dataset(items=[DatasetItem(id="train",input="seen question",reference="yes")])
    test=Dataset(items=[DatasetItem(id="renamed",input="seen question",reference="no")])
    finding=run(train=train,test=test,validation_mode=mode)["train_test_id_overlap"]
    assert finding.severity.value == "blocker"
    assert "content-overlap" in finding.cite


def test_within_slice_duplicate_ids_without_cross_content_remains_medium():
    train=Dataset(items=[DatasetItem(id="dup",input="training only")]*2)
    test=Dataset(items=[DatasetItem(id="eval",input="independent question")])
    assert run(train=train,test=test)["train_test_id_overlap"].severity.value == "medium"


def test_exact_jaccard_avoids_quadratic_duplicate_work():
    with patch("mini_antemortem_cli.traps._tokenize_for_jaccard", wraps=_tokenize_for_jaccard) as tokenize:
        assert _max_pairwise_jaccard(["same words"]*10000) == 1.0
        assert tokenize.call_count <= 2
    assert _max_pairwise_jaccard(["", "", "a"]) == 0
    assert _max_pairwise_jaccard(["a b", "b c", "unrelated"]) == pytest.approx(1/3)


def test_mcp_passes_paired_mode_and_selected_pool_through_real_classifier(monkeypatch):
    from mini_antemortem_cli.mcp import server
    ds=Dataset(items=[DatasetItem(id="same",input="question")])
    monkeypatch.setattr(server,"_resolve_dataset",lambda value:ds)
    findings=server.analytical_preflight(target_provider="openai",judge_provider="anthropic",
        train_dataset="train.jsonl",test_dataset="test.jsonl",
        rubric={"dimensions":[{"name":"q","description":"q","weight":1}]},
        variants={"system_prompts":["a","b"],"few_shot_examples":[{"input":"question","output":"yes"}]},
        validation_mode="paired",selected_few_shot_indices=[])
    by_id={f["trap_id"]:f for f in findings}
    assert by_id["train_test_id_overlap"]["label"] == "UNRESOLVED"
    assert by_id["few_shot_leakage_into_test"]["label"] == "GHOST"


def test_cli_invalid_policy_is_exit_two_and_no_findings(tmp_path,capsys):
    from mini_antemortem_cli.cli import main
    path=tmp_path/"policy.json"
    path.write_text('{"fail_on_missing_test":"false"}',encoding="utf-8")
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    code=main(["check","--target-provider","openai","--judge-provider","anthropic",
        "--train",str(root/"examples/demo_config/train.jsonl"),
        "--rubric",str(root/"examples/demo_config/rubric.json"),
        "--variants",str(root/"examples/demo_config/variants.json"),"--policy",str(path)])
    output=capsys.readouterr()
    assert code == 2
    assert "boolean" in output.err
    assert output.out == ""
