from __future__ import annotations

import json
import os
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from graspbench.perception import ModelServiceError


def get_json(url: str, *, headers: dict[str, str] | None = None) -> dict:
    try:
        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request, timeout=10) as response:
            value = json.load(response)
    except Exception as exc:
        raise ModelServiceError(f"Cannot reach {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelServiceError(f"Service {url} returned a non-object response")
    return value


def replace_path(url: str, path: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def main() -> None:
    endpoint = os.getenv("GRASPBENCH_YOLO_URL", "http://127.0.0.1:8765/infer")
    health_url = replace_path(endpoint, "/healthz")
    health = get_json(health_url)
    if health.get("ok") is not True:
        raise ModelServiceError(f"YOLO health check failed: {health}")
    print(
        "YOLO ready: "
        f"endpoint={endpoint} weights={health.get('weights')} "
        f"device={health.get('device')}"
    )


if __name__ == "__main__":
    main()
