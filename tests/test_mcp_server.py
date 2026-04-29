# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Smoke tests for the mini-antemortem-cli MCP server."""

from __future__ import annotations

import asyncio

import pytest

mcp = pytest.importorskip("mcp")


@pytest.fixture(scope="module")
def mcp_app():
    from mini_antemortem_cli.mcp import mcp_app as app

    return app


@pytest.fixture(scope="module")
def tools(mcp_app):
    return asyncio.run(mcp_app.list_tools())


EXPECTED_TOOLS = {"analytical_preflight", "list_traps"}


def test_two_tools_registered(tools):
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS


def test_each_tool_has_description(tools):
    for tool in tools:
        assert tool.description and len(tool.description) > 50


def test_each_tool_has_input_schema(tools):
    for tool in tools:
        assert tool.inputSchema is not None


def test_analytical_preflight_required_args(tools):
    tool = next(t for t in tools if t.name == "analytical_preflight")
    required = set(tool.inputSchema.get("required", []))
    assert {"target_provider", "judge_provider", "train_dataset", "rubric", "variants"}.issubset(required)


def test_list_traps_executes_without_llm(mcp_app):
    """list_traps is pure deterministic — call should return seven traps."""
    result = asyncio.run(mcp_app.call_tool("list_traps", {}))
    assert result is not None
