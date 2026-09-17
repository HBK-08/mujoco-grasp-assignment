from __future__ import annotations

import json
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from graspbench.perception import FoundationModelPerception, ModelServiceError
from graspbench.vlm import VLMConfig


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
    vlm = VLMConfig.from_env()
    models_url = f"{vlm.base_url}/models"
    models = get_json(models_url, headers={"Authorization": f"Bearer {vlm.api_key}"})
    model_ids = {
        str(item.get("id") or item.get("name"))
        for item in models.get("data", models.get("models", []))
        if isinstance(item, dict)
    }
    if model_ids and vlm.model not in model_ids:
        raise ModelServiceError(
            f"Configured VLM {vlm.model!r} is unavailable; endpoint advertised={sorted(model_ids)}"
        )

    client = FoundationModelPerception()
    sam_health_url = replace_path(client.sam3_endpoint, "/healthz")
    sam_health = get_json(sam_health_url)
    if sam_health.get("ok") is not True:
        raise ModelServiceError(f"SAM3 health check failed: {sam_health}")

    print(f"VLM ready: model={vlm.model} endpoint={vlm.chat_endpoint}")
    print(f"SAM3 ready: endpoint={client.sam3_endpoint}")


if __name__ == "__main__":
    main()
