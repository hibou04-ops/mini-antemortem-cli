# SPDX-License-Identifier: Apache-2.0
"""mini-antemortem-cli - analytical preflight for omegaprompt calibration.

Reads the run configuration and classifies eleven calibration-specific
trap patterns against deterministic rules. No API calls, no network;
reasoning is deterministic given the inputs. Emits
local ``AnalyticalFinding`` records. Their JSON representation can feed
``omegaprompt.preflight.derive_adaptation_plan`` through the optional bridge.

Covered traps::

    self_agreement_bias
    small_sample_kc4_power
    variants_homogeneous
    rubric_weight_concentration
    judge_budget_too_small
    empty_reference_with_strict_rubric
    no_held_out_slice
    train_test_id_overlap
    routed_provider_opaque_family
    few_shot_leakage_into_test
    rubric_dead_weight_dimension

Public API::

    from mini_antemortem_cli import analytical_preflight, analytical_traps

Standalone installation requires Pydantic but no calibration engine::

    pip install mini-antemortem-cli

For integration with the optional calibration engine::

    pip install "mini-antemortem-cli[bridge]"

Companion to ``mini-omega-lock`` (empirical preflight). Either can be
used alone; both compose into the same ``PreflightReport``.
"""

from mini_antemortem_cli.traps import (
    CALIBRATION_TRAPS,
    TrapPattern,
    TrapPolicy,
    analytical_preflight,
    analytical_traps,
    summarize_findings,
)

__version__ = "0.11.2"

__all__ = [
    "CALIBRATION_TRAPS",
    "TrapPattern",
    "TrapPolicy",
    "analytical_preflight",
    "analytical_traps",
    "summarize_findings",
    "__version__",
]
