# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak
# Vendored from omegaprompt 27e23788990a1ea951525a401a1cb6a880e83948.
# Retained local contracts only; see NOTICE for source and modifications.
"""Output-budget wire values used by analytical preflight callers."""

from enum import Enum


class OutputBudgetBucket(str, Enum):
    """Discretised max-output-tokens bucket.

    The calibration searcher picks a bucket; adapters resolve it to a
    concrete ``max_tokens`` int appropriate for their vendor.
    """

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

