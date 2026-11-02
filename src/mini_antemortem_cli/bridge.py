# SPDX-License-Identifier: Apache-2.0
"""Explicit JSON conversion into the optional calibration engine's types.

Importing this module does not import omegaprompt. The engine is loaded only
when the caller requests conversion; the standalone classifier never needs it.
"""

from __future__ import annotations

from collections.abc import Iterable
import importlib
from typing import Any

from mini_antemortem_cli._vendor.contracts import AnalyticalFinding


class BridgeUnavailableError(RuntimeError):
    """The explicitly requested omegaprompt bridge is not installed."""


def to_omegaprompt_findings(
    findings: Iterable[AnalyticalFinding | dict[str, Any]],
) -> list[Any]:
    """Validate local findings as the installed engine's own finding models.

    Pydantic model classes in separate packages have distinct identities.
    Converting via JSON fields preserves the wire contract without making
    the calibration engine a required dependency of this classifier.
    """
    try:
        ParentFinding = importlib.import_module(
            "omegaprompt.preflight.contracts"
        ).AnalyticalFinding
    except ModuleNotFoundError as exc:
        if exc.name != "omegaprompt":
            raise
        raise BridgeUnavailableError(
            'Install "mini-antemortem-cli[bridge]" to convert findings for omegaprompt.'
        ) from exc
    return [
        ParentFinding.model_validate(
            finding if isinstance(finding, dict) else finding.model_dump(mode="json")
        )
        for finding in findings
    ]
