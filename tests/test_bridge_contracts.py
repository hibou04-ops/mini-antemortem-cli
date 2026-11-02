"""Optional integration: compare the leaf contracts with the real parent models.

Only an absent parent skips this file. A present but broken/incompatible
installation must fail, rather than hiding schema drift behind an optional skip.
"""

from __future__ import annotations

import importlib
import importlib.util
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from mini_antemortem_cli import _vendor as local
from mini_antemortem_cli.bridge import to_omegaprompt_findings


if importlib.util.find_spec("omegaprompt") is None:
    pytest.skip("optional [bridge] integration requires omegaprompt", allow_module_level=True)

parent_dataset = importlib.import_module("omegaprompt.domain.dataset")
parent_judge = importlib.import_module("omegaprompt.domain.judge")
parent_params = importlib.import_module("omegaprompt.domain.params")
parent_enums = importlib.import_module("omegaprompt.domain.enums")
parent_contracts = importlib.import_module("omegaprompt.preflight.contracts")
pytestmark = pytest.mark.bridge

ITEM = {"id": "probe", "input": "검증 😀", "reference": None,
        "metadata": {"nested": [1, None, {"trusted": False}]}, "external_tag": "retained"}
DIMENSION = {"name": "quality", "description": "정확성", "weight": 1.0, "scale": [0, 4]}
GATE = {"name": "format", "description": "valid format", "evaluator": "post"}
FINDING = {"trap_id": "train_test_id_overlap", "label": "UNRESOLVED", "hypothesis": "check inputs",
           "severity": "high", "note": "보류 — 😀", "remediation": "supply a holdout", "cite": None}

MODEL_CASES = [
    (local.DatasetItem, parent_dataset.DatasetItem, ITEM),
    (local.Dataset, parent_dataset.Dataset, {"items": [ITEM]}),
    (local.Dimension, parent_judge.Dimension, DIMENSION),
    (local.HardGate, parent_judge.HardGate, GATE),
    (local.JudgeRubric, parent_judge.JudgeRubric, {"dimensions": [DIMENSION], "hard_gates": [GATE]}),
    (local.PromptVariants, parent_params.PromptVariants,
     {"system_prompts": ["Answer precisely", "근거 확인"], "few_shot_examples": [{"input": "1+1", "output": "2"}]}),
    (local.AnalyticalFinding, parent_contracts.AnalyticalFinding, FINDING),
]


def _semantic_schema(value):
    """Remove schema prose only; field names, defaults and enum data are intact."""
    if not isinstance(value, dict):
        return value
    result = {key: item for key, item in value.items() if key != "description"}
    for keyword in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas"):
        if isinstance(result.get(keyword), dict):
            result[keyword] = {name: _semantic_schema(schema) for name, schema in result[keyword].items()}
    for keyword in ("items", "additionalProperties", "unevaluatedProperties", "contains",
                    "propertyNames", "not", "if", "then", "else", "contentSchema"):
        if keyword in result:
            result[keyword] = _semantic_schema(result[keyword])
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if isinstance(result.get(keyword), list):
            result[keyword] = [_semantic_schema(schema) for schema in result[keyword]]
    return result


def test_schema_comparison_retains_description_fields_and_arbitrary_default_data():
    schema = {
        "description": "documentation only",
        "properties": {"description": {"type": "string", "default": "kept", "minLength": 1,
                                       "description": "field documentation"}},
        "required": ["description"],
        "default": {"description": "actual data", "properties": {"description": "also data"}},
    }
    normalized = _semantic_schema(schema)
    assert normalized["properties"]["description"] == {"type": "string", "default": "kept", "minLength": 1}
    assert normalized["required"] == ["description"]
    assert normalized["default"] == schema["default"]


@pytest.mark.parametrize("local_type,parent_type,payload", MODEL_CASES,
                         ids=[case[0].__name__ for case in MODEL_CASES])
def test_models_preserve_fields_schema_and_json_in_both_directions(local_type, parent_type, payload):
    assert local_type is not parent_type
    assert set(local_type.model_fields) == set(parent_type.model_fields)
    schema = _semantic_schema(local_type.model_json_schema())
    assert set(schema["properties"]) == set(local_type.model_fields)
    assert schema == _semantic_schema(parent_type.model_json_schema())
    leaf = local_type.model_validate(payload)
    parent = parent_type.model_validate_json(leaf.model_dump_json())
    returned_leaf = local_type.model_validate_json(parent.model_dump_json())
    assert type(parent) is parent_type
    assert type(returned_leaf) is local_type
    assert returned_leaf.model_dump(mode="json") == leaf.model_dump(mode="json") == parent.model_dump(mode="json")
    parent_first = parent_type.model_validate(payload)
    leaf_second = local_type.model_validate_json(parent_first.model_dump_json())
    returned_parent = parent_type.model_validate_json(leaf_second.model_dump_json())
    assert returned_parent.model_dump(mode="json") == parent_first.model_dump(mode="json")


@pytest.mark.parametrize("leaf,parent", [
    (local.OutputBudgetBucket, parent_enums.OutputBudgetBucket),
    (local.PreflightSeverity, parent_contracts.PreflightSeverity),
])
def test_enum_members_and_json_values_roundtrip(leaf, parent):
    assert leaf is not parent
    assert [(name, item.value) for name, item in leaf.__members__.items()] == [
        (name, item.value) for name, item in parent.__members__.items()]
    for member in leaf:
        foreign = TypeAdapter(parent).validate_json(TypeAdapter(leaf).dump_json(member))
        back = TypeAdapter(leaf).validate_json(TypeAdapter(parent).dump_json(foreign))
        assert type(foreign) is parent and type(back) is leaf
        assert back.value == member.value


@pytest.mark.parametrize("leaf,parent,payload,error_type,error_location", [
    (local.DatasetItem, parent_dataset.DatasetItem, {"id": "a", "input": ""}, "string_too_short", ("input",)),
    (local.Dimension, parent_judge.Dimension, {**DIMENSION, "weight": -1}, "greater_than_equal", ("weight",)),
    (local.Dimension, parent_judge.Dimension, {**DIMENSION, "scale": [4, 4]}, "value_error", ("scale",)),
    (local.HardGate, parent_judge.HardGate, {**GATE, "evaluator": "unknown"}, "literal_error", ("evaluator",)),
    (local.JudgeRubric, parent_judge.JudgeRubric, {"dimensions": [DIMENSION, DIMENSION]}, "value_error", ("dimensions",)),
    (local.PromptVariants, parent_params.PromptVariants, {"system_prompts": []}, "value_error", ("system_prompts",)),
    (local.PromptVariants, parent_params.PromptVariants,
     {"system_prompts": ["valid"], "few_shot_examples": [{"input": "x"}]}, "value_error", ("few_shot_examples",)),
    (local.AnalyticalFinding, parent_contracts.AnalyticalFinding, {**FINDING, "severity": "urgent"}, "enum", ("severity",)),
    (local.AnalyticalFinding, parent_contracts.AnalyticalFinding, {**FINDING, "unexpected": True}, "extra_forbidden", ("unexpected",)),
])
def test_validation_error_types_and_locations_match(leaf, parent, payload, error_type, error_location):
    errors = []
    for model in (leaf, parent):
        with pytest.raises(ValidationError) as caught:
            model.model_validate(payload)
        errors.append([(item["type"], item["loc"]) for item in caught.value.errors()])
    assert errors[0] == errors[1]
    assert (error_type, error_location) in errors[0]


def test_dataset_loader_exceptions_are_shared_builtins_not_copied_error_models(tmp_path):
    missing = tmp_path / "missing.jsonl"
    for dataset in (local.Dataset, parent_dataset.Dataset):
        with pytest.raises(FileNotFoundError):
            dataset.from_jsonl(missing)
        with pytest.raises(ValueError, match="duplicate ids"):
            dataset.from_items([{"id": "x", "input": "one"}, {"id": "x", "input": "two"}])
    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("{not-json}\n", encoding="utf-8")
    for dataset in (local.Dataset, parent_dataset.Dataset):
        with pytest.raises(ValueError, match="not valid JSON"):
            dataset.from_jsonl(malformed)


def test_explicit_bridge_builds_parent_models_and_roundtrips_enum_fields():
    findings = [local.AnalyticalFinding.model_validate({**FINDING, "severity": severity.value})
                for severity in local.PreflightSeverity]
    expected = [item.model_dump(mode="json") for item in findings]
    converted = to_omegaprompt_findings(findings)
    assert all(type(item) is parent_contracts.AnalyticalFinding for item in converted)
    assert all(type(item.severity) is parent_contracts.PreflightSeverity for item in converted)
    assert [item.model_dump(mode="json") for item in converted] == expected
    returned = [local.AnalyticalFinding.model_validate_json(item.model_dump_json()) for item in converted]
    assert [item.model_dump(mode="json") for item in returned] == expected
    assert json.loads(json.dumps(expected, ensure_ascii=True)) == expected
