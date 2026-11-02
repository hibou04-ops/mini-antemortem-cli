# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak
# Vendored from omegaprompt 27e23788990a1ea951525a401a1cb6a880e83948.
# Retained local contracts only; see NOTICE for source and modifications.
"""Provider-neutral prompt parameter space and variants."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

class PromptVariants(BaseModel):
    """User-supplied discrete pools of prompt components.

    The searcher picks an index into ``system_prompts`` and a count into
    ``few_shot_examples``. Both axes are first-class in every calibration
    run and cannot be fully disabled (set ``len(system_prompts) == 1`` to
    effectively freeze the system prompt).
    """

    model_config = ConfigDict(extra="forbid")

    system_prompts: list[str] = Field(
        default_factory=list,
        description="Ordered list of candidate system prompt strings.",
    )
    few_shot_examples: list[dict[str, str]] = Field(
        default_factory=list,
        description="Pool of {input, output} dicts used as few-shot demos.",
    )

    @field_validator("system_prompts")
    @classmethod
    def _non_empty_variants(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("PromptVariants.system_prompts must contain at least one prompt.")
        if any(not isinstance(s, str) or not s.strip() for s in v):
            raise ValueError("Every system prompt must be a non-empty string.")
        return v

    @field_validator("few_shot_examples")
    @classmethod
    def _few_shot_shape(cls, v: list[dict[str, str]]) -> list[dict[str, str]]:
        """Each few-shot must have ``input`` and ``output`` non-empty strings.

        Reviewer P1 #10: provider ``_messages()`` indexes ``shot["input"]``
        and ``shot["output"]`` directly. A malformed shot (missing either
        key, or an empty value) would either KeyError mid-eval or silently
        send an empty user/assistant turn that confuses the target.
        Catching it at construction time keeps the failure on the user's
        config, not the eval pipeline.
        """
        for i, shot in enumerate(v):
            if not isinstance(shot, dict):
                raise ValueError(
                    f"few_shot_examples[{i}] must be a dict with 'input' "
                    f"and 'output' keys, got {type(shot).__name__}."
                )
            for key in ("input", "output"):
                val = shot.get(key)
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(
                        f"few_shot_examples[{i}].{key} must be a non-empty "
                        f"string, got {val!r}."
                    )
        return v

