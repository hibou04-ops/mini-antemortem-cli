# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""MCP server exposing mini-antemortem-cli's analytical preflight classifier.

Run with:

    python -m mini_antemortem_cli.mcp           # stdio transport
    python -m mini_antemortem_cli.mcp --http    # streamable-http transport

Two tools:

* ``analytical_preflight`` — classify a calibration config against the seven
  trap patterns (self-agreement bias, small-sample power, variant homogeneity,
  rubric concentration, judge budget, empty reference, missing held-out).
* ``list_traps``           — introspection: return the seven trap patterns
  with their ids and hypotheses.

Deterministic; zero LLM calls.
"""

from __future__ import annotations

from mini_antemortem_cli.mcp.server import mcp_app

__all__ = ["mcp_app"]
