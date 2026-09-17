from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import imageio.v3 as iio
import numpy as np

from .config import CONTAINER_SPEC_BY_NAME, OBJECT_SPEC_BY_NAME
from .perception import GroundingResult, ModelServiceError, SAM3DetectionDebug
from .types import CameraObservation, DetectedObject


@dataclass(frozen=True)
class VLMConfig:
    """Configuration for an OpenAI-compatible, image-capable chat endpoint.

    The base URL must end at the API version root (for example ``.../v1``),
    not at ``chat/completions``. Secrets are deliberately read only from the
    process environment and never copied into logs or PolicyDecision.debug.
    """

    base_url: str
    api_key: str
    model: str
    timeout_s: float = 45.0

    @classmethod
    def from_env(cls) -> VLMConfig:
        base_url = os.getenv("GRASPBENCH_VLM_BASE_URL", "").rstrip("/")
        api_key = os.getenv("GRASPBENCH_VLM_API_KEY", "")
        model = os.getenv("GRASPBENCH_VLM_MODEL", "qwen3-vl-flash")
        missing = [
            name
            for name, value in (
                ("GRASPBENCH_VLM_BASE_URL", base_url),
                ("GRASPBENCH_VLM_API_KEY", api_key),
            )
            if not value
        ]
        if missing:
            raise ModelServiceError(
                "OpenAI-compatible VLM is not configured; set " + ", ".join(missing)
            )
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=float(os.getenv("GRASPBENCH_VLM_TIMEOUT_S", "45")),
        )

    @property
    def chat_endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"


@dataclass(frozen=True)
class PlannedAction:
    pick_id: str
    place_id: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {"pick_id": self.pick_id, "place_id": self.place_id}


@dataclass(frozen=True)
class GroundingPlan:
    actions: tuple[PlannedAction, ...]
    reason: str
    model: str
    latency_s: float
    raw_response: str
    endpoint: str
    cached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "service": "openai_compatible_vlm",
            "model": self.model,
            "actions": [action.as_dict() for action in self.actions],
            "reason": self.reason,
            "latency_s": self.latency_s,
            "raw_response": self.raw_response,
            "endpoint": self.endpoint,
            "cached": self.cached,
        }


class OpenAICompatibleVLM:
    """Ground a language instruction in a live RGB frame via ``/chat/completions``.

    This is intentionally a narrow adapter: any model that accepts the standard
    image_url content part and returns text/JSON can replace the VLM by changing
    environment variables. SAM remains responsible for metric mask-depth
    geometry; the VLM selects only among detections produced from the current
    camera frame.
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig.from_env()
        self._cache: dict[str, GroundingResult] = {}
        self._plan_cache: dict[str, GroundingPlan] = {}

    def ground_target(
        self,
        instruction: str,
        camera: CameraObservation,
        detections: dict[str, DetectedObject],
        evidence: dict[str, SAM3DetectionDebug],
    ) -> GroundingResult:
        if not detections:
            raise ModelServiceError("Cannot ground an instruction without live visual detections")

        encoded = iio.imwrite(
            "<bytes>", np.asarray(camera.rgb, dtype=np.uint8), extension=".jpg", quality=92
        )
        image_b64 = base64.b64encode(encoded).decode("ascii")
        candidates = [
            {
                "target_id": target_id,
                "sam_prompt": item.prompt,
                "sam_score": round(item.score, 6),
                "box_xyxy": [round(value, 3) for value in item.box_xyxy],
            }
            for target_id, item in evidence.items()
            if target_id in detections
        ]
        if not candidates:
            raise ModelServiceError("SAM returned detections without usable visual evidence")

        cache_key = hashlib.sha256(
            (instruction + "\0" + image_b64 + "\0" + json.dumps(candidates, sort_keys=True)).encode(
                "utf-8"
            )
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return GroundingResult(
                target_id=cached.target_id,
                reason=cached.reason,
                model=cached.model,
                latency_s=0.0,
                raw_response=cached.raw_response,
                service="openai_compatible_vlm",
                endpoint=self.config.chat_endpoint,
                cached=True,
            )

        prompt = (
            "You are the visual grounding module of a tabletop grasping robot. "
            "Inspect the supplied current overhead camera image and the SAM detections. "
            "Select exactly one detected candidate that satisfies the instruction. "
            "Return a JSON object with exactly two fields: target_id and reason. "
            "target_id must be copied exactly from detected_candidates; never invent an object.\n"
            f"instruction={json.dumps(instruction, ensure_ascii=False)}\n"
            f"detected_candidates={json.dumps(candidates, ensure_ascii=False)}"
        )
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the requested JSON object. Use the image as evidence.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }
        if os.getenv("GRASPBENCH_VLM_RESPONSE_FORMAT", "json_object") != "none":
            payload["response_format"] = {"type": "json_object"}
        response, latency = self._post_json(payload)
        try:
            raw_response = str(response["choices"][0]["message"]["content"])
            parsed = self._parse_json_object(raw_response)
            target_id = str(parsed["target_id"])
            reason = str(parsed["reason"]).strip()
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ModelServiceError(
                "Invalid OpenAI-compatible VLM grounding response (expected choices[0].message.content JSON)"
            ) from exc
        if target_id not in detections:
            raise ModelServiceError(
                f"VLM selected {target_id!r}, but it is absent from the current SAM detections"
            )
        if not reason:
            raise ModelServiceError("VLM returned an empty grounding reason")
        result = GroundingResult(
            target_id=target_id,
            reason=reason,
            model=self.config.model,
            latency_s=latency,
            raw_response=raw_response,
            service="openai_compatible_vlm",
            endpoint=self.config.chat_endpoint,
        )
        self._cache[cache_key] = result
        return result

    def plan_task(
        self,
        instruction: str,
        camera: CameraObservation,
        detections: dict[str, DetectedObject],
        evidence: dict[str, SAM3DetectionDebug],
    ) -> GroundingPlan:
        """Turn one visual instruction into one or more validated pick/place actions.

        The VLM only chooses semantic IDs from live SAM candidates.  Metric pick and
        placement coordinates remain entirely in the RGB-D controller.
        """
        if not detections:
            raise ModelServiceError("Cannot plan a task without live visual detections")
        encoded = iio.imwrite(
            "<bytes>", np.asarray(camera.rgb, dtype=np.uint8), extension=".jpg", quality=92
        )
        image_b64 = base64.b64encode(encoded).decode("ascii")
        candidates = [
            {
                "candidate_id": target_id,
                "kind": "object" if target_id in OBJECT_SPEC_BY_NAME else "container",
                "sam_prompt": item.prompt,
                "sam_score": round(item.score, 6),
                "box_xyxy": [round(value, 3) for value in item.box_xyxy],
            }
            for target_id, item in evidence.items()
            if target_id in detections
        ]
        if not candidates:
            raise ModelServiceError("SAM returned detections without usable visual evidence")
        cache_key = hashlib.sha256(
            (instruction + "\0" + image_b64 + "\0" + json.dumps(candidates, sort_keys=True)).encode(
                "utf-8"
            )
        ).hexdigest()
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            return GroundingPlan(
                actions=cached.actions,
                reason=cached.reason,
                model=cached.model,
                latency_s=0.0,
                raw_response=cached.raw_response,
                endpoint=cached.endpoint,
                cached=True,
            )

        prompt = (
            "You are the visual task-planning module of a tabletop robot. Inspect the "
            "current overhead image and the SAM candidates. Convert the instruction into "
            "an ordered list of pick-and-place actions. For a lift-only instruction, use "
            "place_id=null. For a sorting instruction, include every object that must move. "
            "Return a JSON object with exactly two fields: actions and reason. actions is a "
            "non-empty list of objects with exactly pick_id and place_id fields. pick_id must "
            "be an object candidate; a non-null place_id must be a container candidate. Copy "
            "IDs exactly and never invent candidates.\n"
            f"instruction={json.dumps(instruction, ensure_ascii=False)}\n"
            f"detected_candidates={json.dumps(candidates, ensure_ascii=False)}"
        )
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the requested JSON object. Use the image as evidence.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                },
            ],
        }
        if os.getenv("GRASPBENCH_VLM_RESPONSE_FORMAT", "json_object") != "none":
            payload["response_format"] = {"type": "json_object"}
        response, latency = self._post_json(payload)
        try:
            raw_response = str(response["choices"][0]["message"]["content"])
            parsed = self._parse_json_object(raw_response)
            raw_actions = parsed["actions"]
            reason = str(parsed["reason"]).strip()
            if not isinstance(raw_actions, list) or not raw_actions or not reason:
                raise ValueError("actions must be a non-empty list and reason must be non-empty")
            actions = tuple(
                PlannedAction(
                    pick_id=str(item["pick_id"]),
                    place_id=None if item.get("place_id") is None else str(item["place_id"]),
                )
                for item in raw_actions
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ModelServiceError(
                "Invalid VLM task-plan response (expected actions and reason JSON)"
            ) from exc
        picks = [action.pick_id for action in actions]
        if len(set(picks)) != len(picks):
            raise ModelServiceError("VLM task plan selects the same object more than once")
        for action in actions:
            if action.pick_id not in detections or action.pick_id not in OBJECT_SPEC_BY_NAME:
                raise ModelServiceError(f"VLM selected invalid pick candidate: {action.pick_id!r}")
            if action.place_id is not None and (
                action.place_id not in detections or action.place_id not in CONTAINER_SPEC_BY_NAME
            ):
                raise ModelServiceError(
                    f"VLM selected invalid destination candidate: {action.place_id!r}"
                )
        result = GroundingPlan(
            actions=actions,
            reason=reason,
            model=self.config.model,
            latency_s=latency,
            raw_response=raw_response,
            endpoint=self.config.chat_endpoint,
        )
        self._plan_cache[cache_key] = result
        return result

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response does not contain a JSON object")
        value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise TypeError("grounding response must be a JSON object")
        return value

    def _post_json(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        request = urllib.request.Request(
            self.config.chat_endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                value = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2_000).decode("utf-8", errors="replace")
            raise ModelServiceError(
                f"VLM endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelServiceError(f"VLM endpoint failed: {exc}") from exc
        if not isinstance(value, dict):
            raise ModelServiceError("VLM endpoint returned a non-object JSON response")
        if "error" in value:
            raise ModelServiceError(f"VLM endpoint returned: {value['error']}")
        return value, time.perf_counter() - started
