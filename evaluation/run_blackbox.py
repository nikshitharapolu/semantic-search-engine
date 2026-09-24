from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_URL = os.getenv(
    "EVALUATION_API_URL",
    "http://api:8000",
).rstrip("/")

CASES_PATH = Path(
    os.getenv(
        "EVALUATION_CASES",
        "/evaluation/cases.json",
    )
)

CITATION_PATTERN = re.compile(r"\[\d+\]")


def call_api(
    endpoint: str,
    payload: dict[str, Any],
) -> tuple[int, dict[str, Any], float]:

    request = urllib.request.Request(
        f"{API_URL}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request,
            timeout=35,
        ) as response:
            status = response.status
            body = response.read()

    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read()

    elapsed = time.perf_counter() - started

    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = {
            "raw_body": body.decode(
                "utf-8",
                errors="replace",
            )
        }

    return status, parsed, elapsed


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []

    try:
        status, response, latency = call_api(
            case["endpoint"],
            case["payload"],
        )
    except Exception as error:
        return {
            "id": case["id"],
            "passed": False,
            "errors": [
                f"request failed: {error}"
            ],
        }

    if status != case["expected_status"]:
        errors.append(
            f"expected HTTP {case['expected_status']}, "
            f"received {status}"
        )

    answer = str(response.get("answer", ""))
    sources = response.get("sources", [])

    if (
        case.get("require_citation")
        and not CITATION_PATTERN.search(answer)
    ):
        errors.append(
            "answer did not contain a numbered citation"
        )

    minimum_sources = case.get("min_sources", 0)

    if len(sources) < minimum_sources:
        errors.append(
            f"expected at least {minimum_sources} sources, "
            f"received {len(sources)}"
        )

    for forbidden in case.get(
        "forbidden_substrings",
        [],
    ):
        if forbidden.casefold() in answer.casefold():
            errors.append(
                f"answer contained forbidden text: {forbidden}"
            )

    maximum_latency = case.get(
        "max_latency_seconds",
        float("inf"),
    )

    if latency > maximum_latency:
        errors.append(
            f"latency {latency:.3f}s exceeded limit"
        )

    return {
        "id": case["id"],
        "passed": not errors,
        "status": status,
        "latency_seconds": round(latency, 3),
        "source_count": len(sources),
        "errors": errors,
    }


def probe_sandbox() -> list[dict[str, Any]]:
    probes = []

    try:
        Path("/sandbox-write-probe").write_text(
            "unexpected",
            encoding="utf-8",
        )
        probes.append(
            {
                "id": "read_only_root",
                "passed": False,
                "detail": "write unexpectedly succeeded",
            }
        )
    except OSError as error:
        probes.append(
            {
                "id": "read_only_root",
                "passed": True,
                "detail": type(error).__name__,
            }
        )

    try:
        with socket.create_connection(
            ("1.1.1.1", 53),
            timeout=1,
        ):
            pass

        probes.append(
            {
                "id": "external_network_blocked",
                "passed": False,
                "detail": "connection unexpectedly succeeded",
            }
        )
    except OSError as error:
        probes.append(
            {
                "id": "external_network_blocked",
                "passed": True,
                "detail": type(error).__name__,
            }
        )

    return probes


def main() -> int:
    cases = json.loads(
        CASES_PATH.read_text(encoding="utf-8")
    )

    results = [
        run_case(case)
        for case in cases
    ]

    sandbox_results = probe_sandbox()

    passed = all(
        result["passed"]
        for result in [
            *results,
            *sandbox_results,
        ]
    )

    report = {
        "suite": "sandboxed-api-blackbox",
        "passed": passed,
        "api_url": API_URL,
        "tests": results,
        "sandbox_controls": sandbox_results,
    }

    print(json.dumps(report, indent=2))

    if (
        os.getenv("STRICT_MODE") == "1"
        and not passed
    ):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())