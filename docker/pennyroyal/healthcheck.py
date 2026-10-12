#!/usr/bin/env python3
"""Container health probe: ask the engine, on the port the engine chose.

The server port is a launch-time flag, not a container constant: the canonical
recipe serves --port 8001, while the Pennyroyal-3 mamba-only launch serves
--port 30000 and publishes 8001:30000. A probe that hardcodes a port therefore
reports `unhealthy` for a container that is serving correctly -- and a health
report that lies is worse than no health report, because `restart: always`,
`--wait condition=healthy`, and any orchestrator will act on it.

So resolve the port instead of assuming it:

1. PENNYROYAL_HEALTH_PORT, if the operator set it explicitly (highest trust).
2. The --port value of the running engine process, read from /proc.
3. 8001, the canonical default.

Exit 0 only if /health answers 2xx/3xx. Failure stays failure: if the engine is
gone, or the port cannot be resolved, the probe exits non-zero so the container
goes unhealthy rather than reporting a false green.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

DEFAULT_PORT = "8001"
ENGINE_MARKERS = ("sglang", "vllm", "pennyroyal")


def _port_from_env() -> str | None:
    value = os.environ.get("PENNYROYAL_HEALTH_PORT", "").strip()
    return value or None


def _port_from_proc() -> str | None:
    """First --port value belonging to an engine-looking process, lowest pid."""
    pids = [p for p in os.listdir("/proc") if p.isdigit()]
    for pid in sorted(pids, key=int):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")
        except OSError:
            continue  # exited between listing and reading, or not ours to read
        args = [a.decode("utf-8", "replace") for a in argv if a]
        if not args:
            continue
        joined = " ".join(args)
        if not any(marker in joined for marker in ENGINE_MARKERS):
            continue
        for i, arg in enumerate(args):
            if arg in ("--port", "--http-port") and i + 1 < len(args):
                candidate = args[i + 1].strip()
                if candidate.isdigit():
                    return candidate
            for flag in ("--port=", "--http-port="):
                if arg.startswith(flag):
                    candidate = arg[len(flag) :].strip()
                    if candidate.isdigit():
                        return candidate
    return None


def resolve_port() -> str:
    return _port_from_env() or _port_from_proc() or DEFAULT_PORT


def main() -> int:
    port = resolve_port()
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            resp.read(1)
            code = getattr(resp, "status", None) or resp.getcode()
        if 200 <= code < 400:
            return 0
        print(f"health probe: {url} -> HTTP {code}", file=sys.stderr)
        return 1
    except urllib.error.HTTPError as exc:
        print(f"health probe: {url} -> HTTP {exc.code}", file=sys.stderr)
        return 1
    except Exception as exc:  # refused, timed out, no such port
        print(f"health probe: {url} -> {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
