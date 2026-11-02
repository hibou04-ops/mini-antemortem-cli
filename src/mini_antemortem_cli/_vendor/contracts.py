# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak
# Vendored from omegaprompt 27e23788990a1ea951525a401a1cb6a880e83948.
# Retained local contracts only; see NOTICE for source and modifications.
"""Shared Pydantic contracts for the preflight layer.

These types are the stable interface between the two preflight sub-units
(analytical `mini_antemortem`, empirical `mini_omega_lock`) and the main
`omegaprompt` calibration pipeline. External repositories implementing the
full sub-units must emit records conforming to these shapes.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict


class PreflightSeverity(str, Enum):
    """Severity of a preflight finding.

    BLOCKER findings abort the run under guarded execution profile.
    HIGH findings force an adaptation override (e.g., raise `min_kc4`).
    MEDIUM findings are recorded; adaptation is advisory.
    LOW findings are noted only; no parameter change.
    """

    BLOCKER = "blocker"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnalyticalFinding(BaseModel):
    """One classification produced by the analytical preflight sub-unit.

    Mirrors the shape of :class:`omegaprompt.domain.judge.JudgeResult`'s
    sibling in antemortem-cli: each trap carries a label, a severity, a
    free-form note, and an optional remediation hint the adaptation
    layer consumes.
    """

    model_config = ConfigDict(extra="forbid")

    trap_id: str
    label: str  # REAL / GHOST / NEW / UNRESOLVED
    hypothesis: str
    severity: PreflightSeverity = PreflightSeverity.MEDIUM
    note: str = ""
    remediation: str = ""
    cite: str | None = None  # e.g. "variants.json:3-9"

