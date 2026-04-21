"""mini-antemortem-cli - analytical preflight for omegaprompt calibration.

Reads the run configuration and classifies seven calibration-specific
trap patterns against deterministic rules. No API calls, no network;
reasoning is deterministic given the inputs. Emits
:class:`omegaprompt.preflight.AnalyticalFinding` records that feed
:func:`omegaprompt.preflight.derive_adaptation_plan`.

Covered traps::

    self_agreement_bias
    small_sample_kc4_power
    variants_homogeneous
    rubric_weight_concentration
    judge_budget_too_small
    empty_reference_with_strict_rubric
    no_held_out_slice

Public API::

    from mini_antemortem_cli import analytical_preflight, analytical_traps

Separate package; not part of the main ``omegaprompt`` install. Install
alongside omegaprompt when you want analytical preflight before
calibration::

    pip install omegaprompt mini-antemortem-cli

Companion to ``mini-omega-lock`` (empirical preflight). Either can be
used alone; both compose into the same ``PreflightReport``.
"""

from mini_antemortem_cli.traps import (
    CALIBRATION_TRAPS,
    TrapPattern,
    analytical_preflight,
    analytical_traps,
)

__version__ = "0.1.0"

__all__ = [
    "CALIBRATION_TRAPS",
    "TrapPattern",
    "analytical_preflight",
    "analytical_traps",
    "__version__",
]
