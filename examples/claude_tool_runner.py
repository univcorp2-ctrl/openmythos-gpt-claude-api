"""Minimal helper for executing a Claude tool call against this API.

This file does not call Claude directly. In a real Claude API app, when Claude
returns a tool call named `generate_with_openmythos`, pass its input to
`call_openmythos_tool` and send the result back as a tool_result.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


OPENMYTHOS_API_URL = os.getenv("OPENMYTHOS_API_URL", "http://localhost:8000")
OPENMYTHOS_API_KEY = os.getenv("OPENMYTHOS_API_KEY")


def call_openmythos_tool(tool_input: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if OPENMYTHOS_API_KEY:
        headers["Authorization"] = f"Bearer {OPENMYTHOS_API_KEY}"

    response = httpx.post(
        f"{OPENMYTHOS_API_URL.rstrip('/')}/generate",
        headers=headers,
        json=tool_input,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print(
        call_openmythos_tool(
            {
                "prompt": "Hello from Claude tool use",
                "max_new_tokens": 32,
                "n_loops": 4,
            }
        )
    )
