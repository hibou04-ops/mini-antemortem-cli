# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Entry point for ``python -m mini_antemortem_cli.mcp``."""

from __future__ import annotations

import argparse
import sys

from mini_antemortem_cli.mcp.server import mcp_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mini-antemortem-cli-mcp",
        description="MCP server exposing mini-antemortem-cli's analytical preflight classifier.",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run streamable-http transport instead of the default stdio.",
    )
    args = parser.parse_args(argv)

    if args.http:
        mcp_app.run(transport="streamable-http")
    else:
        mcp_app.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
