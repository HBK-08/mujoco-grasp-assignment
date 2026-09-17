from __future__ import annotations

import io
import json
from unittest.mock import patch

import numpy as np

from graspbench.perception import SAM3DetectionDebug
from graspbench.types import CameraObservation, DetectedObject
from graspbench.vlm import OpenAICompatibleVLM, VLMConfig


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._buffer = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self._buffer

    def __exit__(self, *_args) -> None:
        self._buffer.close()


def _camera() -> CameraObservation:
    return CameraObservation(
        name="overhead",
        rgb=np.full((16, 16, 3), 127, dtype=np.uint8),
        depth=np.ones((16, 16), dtype=np.float32),
        position=np.zeros(3),
        rotation=np.eye(3),
        fovy_degrees=45.0,
    )


def _detected(name: str) -> DetectedObject:
    return DetectedObject(
        name=name,
        color="red",
        shape="cube",
        position=np.array([0.5, 0.0, 0.426]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def _evidence(name: str) -> SAM3DetectionDebug:
    return SAM3DetectionDebug(
        target_id=name,
        prompt="small red object",
        score=0.9,
        box_xyxy=(3.0, 4.0, 9.0, 10.0),
        mask_pixels=42,
        geometry_pixels=40,
        latency_s=0.1,
        endpoint="http://sam.test/infer",
    )


def test_openai_compatible_vlm_sends_live_image_and_validates_candidate() -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _FakeResponse(
            {"choices": [{"message": {"content": '{"target_id":"red_cube","reason":"red cube"}'}}]}
        )

    client = OpenAICompatibleVLM(
        VLMConfig("https://vlm.example/v1", "test-key", "vision-model", timeout_s=12)
    )
    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.ground_target(
            "pick the red cube",
            _camera(),
            {"red_cube": _detected("red_cube")},
            {"red_cube": _evidence("red_cube")},
        )

    assert result.target_id == "red_cube"
    assert result.service == "openai_compatible_vlm"
    assert len(requests) == 1
    request, timeout = requests[0]
    payload = json.loads(request.data)
    image = payload["messages"][1]["content"][1]["image_url"]["url"]
    assert request.full_url == "https://vlm.example/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert timeout == 12
    assert image.startswith("data:image/jpeg;base64,")


def test_openai_compatible_vlm_caches_identical_visual_grounding() -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return _FakeResponse(
            {"choices": [{"message": {"content": '{"target_id":"red_cube","reason":"red cube"}'}}]}
        )

    client = OpenAICompatibleVLM(VLMConfig("https://vlm.example/v1", "test-key", "vision-model"))
    with patch("urllib.request.urlopen", fake_urlopen):
        first = client.ground_target(
            "pick the red cube",
            _camera(),
            {"red_cube": _detected("red_cube")},
            {"red_cube": _evidence("red_cube")},
        )
        second = client.ground_target(
            "pick the red cube",
            _camera(),
            {"red_cube": _detected("red_cube")},
            {"red_cube": _evidence("red_cube")},
        )

    assert calls == 1
    assert not first.cached
    assert second.cached


def test_openai_compatible_vlm_validates_multistep_pick_and_place_plan() -> None:
    def fake_urlopen(request, timeout):
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"actions":[{"pick_id":"red_cube","place_id":"square_tray"}],'
                                '"reason":"The image shows the requested cube and square tray."}'
                            )
                        }
                    }
                ]
            }
        )

    client = OpenAICompatibleVLM(VLMConfig("https://vlm.example/v1", "test-key", "vision-model"))
    detections = {"red_cube": _detected("red_cube"), "square_tray": _detected("square_tray")}
    evidence = {"red_cube": _evidence("red_cube"), "square_tray": _evidence("square_tray")}
    with patch("urllib.request.urlopen", fake_urlopen):
        plan = client.plan_task("put the red cube in the square tray", _camera(), detections, evidence)

    assert [action.as_dict() for action in plan.actions] == [
        {"pick_id": "red_cube", "place_id": "square_tray"}
    ]
