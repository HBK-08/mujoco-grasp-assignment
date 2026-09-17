"""YOLO detections converted to auditable RGB-D world coordinates."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

import imageio.v3 as iio
import numpy as np

from graspbench.config import (
    CONTAINER_SPECS,
    OBJECT_SPECS,
    TABLE_TOP_Z,
    ContainerSpec,
    ObjectSpec,
)
from graspbench.perception import ModelServiceError
from graspbench.types import CameraObservation, DetectedObject


@dataclass(frozen=True)
class YOLODetectionEvidence:
    """Compact evidence saved in the evaluator JSONL log."""

    target_id: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    geometry_pixels: int
    latency_s: float
    endpoint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "service": "yolo",
            "model": "yolov8n_finetuned",
            "target_id": self.target_id,
            "score": self.score,
            "box_xyxy": list(self.box_xyxy),
            "geometry_pixels": self.geometry_pixels,
            "latency_s": self.latency_s,
            "endpoint": self.endpoint,
        }


def _split_two_spatial_clusters(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split two spatially separated tray surfaces without assigning a class.

    The split is deliberately invariant to world-frame left/right placement.
    PCA only supplies deterministic initial centres for a small two-means fit;
    semantic tray labels are assigned later from each cluster's shape.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 80:
        raise ValueError("insufficient tray point-cloud geometry")

    centered = points - np.median(points, axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, -1]
    projection = centered @ principal_axis
    initial_projection = np.percentile(projection, [20.0, 80.0])
    centres = np.vstack(
        [
            np.median(points[projection <= initial_projection[0]], axis=0),
            np.median(points[projection >= initial_projection[1]], axis=0),
        ]
    )

    labels = np.zeros(len(points), dtype=np.int8)
    for _ in range(20):
        distance = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        new_labels = np.argmin(distance, axis=1).astype(np.int8)
        if np.all(new_labels == new_labels[0]):
            raise ValueError("tray point cloud did not separate into two clusters")
        new_centres = np.vstack(
            [np.median(points[new_labels == index], axis=0) for index in range(2)]
        )
        converged = np.array_equal(labels, new_labels) and np.allclose(
            centres, new_centres, atol=1e-6
        )
        labels = new_labels
        centres = new_centres
        if converged:
            break

    clusters = (points[labels == 0], points[labels == 1])
    if min(len(cluster) for cluster in clusters) < 40:
        raise ValueError("one tray cluster has insufficient geometry")
    return clusters


def _tray_cluster_geometry(cluster: np.ndarray) -> dict[str, Any]:
    """Return a robust centre and a square-vs-round shape score."""
    lower, upper = np.percentile(cluster, [1.0, 99.0], axis=0)
    span = upper - lower
    if np.any(span < 0.12) or np.any(span > 0.34):
        raise ValueError("tray cluster has an implausible footprint")
    centre = 0.5 * (lower + upper)
    half_span = np.maximum(0.5 * span, 1e-6)
    normalized = np.abs((cluster - centre) / half_span)

    # A filled square retains surface samples near all four bounding-box
    # corners; a disk does not. This discriminant is independent of which side
    # of the table the tray occupies and of rotations of the circular tray.
    corner_fraction = float(
        np.mean((normalized[:, 0] >= 0.70) & (normalized[:, 1] >= 0.70))
    )
    return {
        "position": np.array([centre[0], centre[1], TABLE_TOP_Z]),
        "points": len(cluster),
        "corner_fraction": corner_fraction,
        "footprint_span": span,
    }


def _classify_container_point_cloud(points_xy: np.ndarray) -> dict[str, dict[str, Any]]:
    """Classify two tray surfaces by geometry, never by absolute position."""
    clusters = _split_two_spatial_clusters(points_xy)
    geometry = [_tray_cluster_geometry(cluster) for cluster in clusters]
    score_gap = abs(
        geometry[0]["corner_fraction"] - geometry[1]["corner_fraction"]
    )
    if score_gap < 0.015:
        raise ValueError("tray shapes are not geometrically distinguishable")
    square_index = int(
        geometry[1]["corner_fraction"] > geometry[0]["corner_fraction"]
    )
    round_index = 1 - square_index
    return {
        "square_tray": geometry[square_index],
        "round_tray": geometry[round_index],
    }


def refine_container_positions_from_depth(
    detections: dict[str, DetectedObject], camera: CameraObservation
) -> dict[str, dict[str, Any]]:
    """Refine detected tray centres from live RGB-D shape evidence.

    Spatial clustering discovers two anonymous surfaces. Their class labels
    are then assigned from footprint shape, so swapping the trays' positions
    does not change the result. If geometry is incomplete or ambiguous, the
    original YOLO projections remain untouched.
    """
    depth = np.asarray(camera.depth, dtype=np.float64)
    finite = np.isfinite(depth) & (depth > 0)
    if int(finite.sum()) < 80:
        return {}
    rows, cols = np.nonzero(finite)
    points = YOLOPerception._unproject_many(camera, cols, rows)
    raised_workspace = (
        (points[:, 0] >= 0.15)
        & (points[:, 0] <= 0.50)
        & (np.abs(points[:, 1]) <= 0.34)
        & (points[:, 2] >= TABLE_TOP_Z + 0.005)
        & (points[:, 2] <= TABLE_TOP_Z + 0.070)
    )
    try:
        classified = _classify_container_point_cloud(
            points[raised_workspace, :2]
        )
    except ValueError:
        return {}

    debug: dict[str, dict[str, Any]] = {}
    for container_id, item in classified.items():
        if container_id not in detections:
            continue
        position = np.asarray(item["position"], dtype=np.float64)
        detections[container_id] = replace(
            detections[container_id], position=position
        )
        debug[container_id] = {
            "method": "depth_cluster_shape",
            "position": position.tolist(),
            "points": item["points"],
            "corner_fraction": item["corner_fraction"],
            "footprint_span": np.asarray(item["footprint_span"]).tolist(),
        }
    return debug


class YOLOPerception:
    """Call the local detector only at planning and recovery events."""

    TALL_OBJECT_HALF_HEIGHT = 0.06
    VALIDATION_TOLERANCE_M = 0.003

    def __init__(self, *, endpoint: str | None = None, timeout_s: float = 30.0) -> None:
        self.endpoint = endpoint or os.getenv(
            "GRASPBENCH_YOLO_URL", "http://127.0.0.1:8765/infer"
        )
        self.timeout_s = float(timeout_s)
        self.spec_by_name: dict[str, ObjectSpec | ContainerSpec] = {
            **{spec.name: spec for spec in OBJECT_SPECS},
            **{spec.name: spec for spec in CONTAINER_SPECS},
        }

    def detect_scene(
        self,
        camera: CameraObservation,
        candidate_ids: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, DetectedObject], dict[str, YOLODetectionEvidence]]:
        target_ids = candidate_ids or tuple(self.spec_by_name)
        unknown = set(target_ids) - set(self.spec_by_name)
        if unknown:
            raise ModelServiceError(f"unknown YOLO candidate ids: {sorted(unknown)}")

        response, latency = self._call_yolo(camera.rgb, target_ids)
        by_target = self._group_detections(response, target_ids)
        detections: dict[str, DetectedObject] = {}
        evidence: dict[str, YOLODetectionEvidence] = {}

        for target_id in target_ids:
            spec = self.spec_by_name[target_id]
            try:
                detected, debug = self._decode_detection(
                    camera,
                    spec,
                    by_target.get(target_id, []),
                    latency_s=latency,
                    prior_xy=None,
                    prior_radius=None,
                    allow_flat=False,
                    surface_percentile=(
                        spec.surface_percentile if isinstance(spec, ObjectSpec) else 70.0
                    ),
                )
            except (TypeError, ValueError):
                if not (
                    isinstance(spec, ObjectSpec)
                    and spec.half_height >= self.TALL_OBJECT_HALF_HEIGHT
                ):
                    continue
                relaxed_spec = replace(
                    spec, half_height=spec.half_height - self.VALIDATION_TOLERANCE_M
                )
                try:
                    detected, debug = self._decode_detection(
                        camera,
                        relaxed_spec,
                        by_target.get(target_id, []),
                        latency_s=latency,
                        prior_xy=None,
                        prior_radius=None,
                        allow_flat=False,
                        surface_percentile=relaxed_spec.surface_percentile,
                    )
                except (TypeError, ValueError):
                    continue
                position = detected.position.copy()
                position[2] = TABLE_TOP_Z + spec.half_height
                detected = DetectedObject(
                    name=spec.name,
                    color=spec.color,
                    shape=spec.shape,
                    position=position,
                    quaternion=detected.quaternion,
                )
            detections[target_id] = detected
            evidence[target_id] = debug

        if not detections:
            raise ModelServiceError("YOLO did not detect any requested tabletop candidate")
        return detections, evidence

    def detect_target(
        self,
        camera: CameraObservation,
        target_id: str,
        *,
        prior_xy: np.ndarray,
        surface_percentile: float = 98.0,
    ) -> tuple[DetectedObject, YOLODetectionEvidence]:
        if target_id not in self.spec_by_name:
            raise ModelServiceError(f"unsupported YOLO target: {target_id}")
        response, latency = self._call_yolo(camera.rgb, (target_id,))
        by_target = self._group_detections(response, (target_id,))
        try:
            return self._decode_detection(
                camera,
                self.spec_by_name[target_id],
                by_target.get(target_id, []),
                latency_s=latency,
                prior_xy=np.asarray(prior_xy, dtype=np.float64),
                prior_radius=None,
                allow_flat=False,
                surface_percentile=surface_percentile,
            )
        except (TypeError, ValueError) as exc:
            raise ModelServiceError(str(exc)) from exc

    def detect_recovery_target(
        self,
        camera: CameraObservation,
        target_id: str,
    ) -> tuple[DetectedObject, YOLODetectionEvidence]:
        if target_id not in self.spec_by_name:
            raise ModelServiceError(f"unknown recovery target: {target_id}")
        spec = self.spec_by_name[target_id]
        if not isinstance(spec, ObjectSpec):
            raise ModelServiceError(f"recovery target is not an object: {target_id}")
        response, latency = self._call_yolo(camera.rgb, (target_id,))
        by_target = self._group_detections(response, (target_id,))
        relaxed_spec = replace(spec, half_height=min(spec.half_height, 0.015))
        try:
            return self._decode_detection(
                camera,
                relaxed_spec,
                by_target.get(target_id, []),
                latency_s=latency,
                prior_xy=None,
                prior_radius=None,
                allow_flat=True,
                surface_percentile=98.0,
            )
        except (TypeError, ValueError) as exc:
            raise ModelServiceError(str(exc)) from exc

    def _call_yolo(
        self, rgb: np.ndarray, candidate_ids: tuple[str, ...]
    ) -> tuple[dict[str, Any], float]:
        encoded = iio.imwrite(
            "<bytes>", np.asarray(rgb, dtype=np.uint8), extension=".jpg", quality=92
        )
        payload = {
            "image_jpeg_b64": base64.b64encode(encoded).decode("ascii"),
            "candidate_ids": list(candidate_ids),
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                value = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2_000).decode("utf-8", errors="replace")
            raise ModelServiceError(
                f"YOLO service returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelServiceError(f"YOLO service failed: {exc}") from exc
        if not isinstance(value, dict):
            raise ModelServiceError("YOLO service returned a non-object response")
        return value, time.perf_counter() - started

    @staticmethod
    def _group_detections(
        response: dict[str, Any], candidate_ids: tuple[str, ...]
    ) -> dict[str, list[dict[str, Any]]]:
        raw = response.get("detections")
        if not isinstance(raw, list):
            raise ModelServiceError("YOLO response has no detections list")
        allowed = set(candidate_ids)
        grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in candidate_ids}
        for item in raw:
            if not isinstance(item, dict):
                raise ModelServiceError("YOLO detection is not an object")
            target_id = item.get("target_id")
            if target_id in allowed:
                grouped[target_id].append(item)
        return grouped

    def _decode_detection(
        self,
        camera: CameraObservation,
        spec: ObjectSpec | ContainerSpec,
        candidates: list[dict[str, Any]],
        *,
        latency_s: float,
        prior_xy: np.ndarray | None,
        prior_radius: float | None,
        allow_flat: bool,
        surface_percentile: float = 70.0,
    ) -> tuple[DetectedObject, YOLODetectionEvidence]:
        if not candidates:
            raise ValueError(f"YOLO found no instance for {spec.name}")
        best = max(candidates, key=lambda item: float(item.get("score", -1.0)))
        score = float(best.get("score", -1.0))
        box_values = best.get("box_xyxy")
        if score < 0.0 or not isinstance(box_values, list) or len(box_values) != 4:
            raise TypeError(f"invalid YOLO detection for {spec.name}")
        box = tuple(float(value) for value in box_values)
        height, width = camera.depth.shape
        x1, y1, x2, y2 = box
        if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
            raise ValueError(f"YOLO box for {spec.name} is outside the image")
        u_c = 0.5 * (x1 + x2)
        v_c = 0.5 * (y1 + y2)

        ix1 = int(np.clip(np.floor(x1), 0, width - 1))
        ix2 = int(np.clip(np.ceil(x2), 0, width - 1))
        iy1 = int(np.clip(np.floor(y1), 0, height - 1))
        iy2 = int(np.clip(np.ceil(y2), 0, height - 1))
        sub = camera.depth[iy1 : iy2 + 1, ix1 : ix2 + 1]
        finite = np.isfinite(sub) & (sub > 0)
        geometry_pixels = int(np.count_nonzero(finite))
        if geometry_pixels < 10:
            raise ValueError(f"YOLO box for {spec.name} has insufficient depth")
        if isinstance(spec, ObjectSpec):
            # A detector box is intentionally looser than an object mask. A
            # single projected box-centre pixel can therefore land in a
            # banana's empty crescent or on background next to an irregular
            # object. Recover the object centre from all current RGB-D surface
            # points in the box, matching the benchmark's mask geometry rule.
            local_rows, local_cols = np.nonzero(finite)
            rows = local_rows + iy1
            cols = local_cols + ix1
            points = self._unproject_many(camera, cols, rows)
            keep = (
                (points[:, 0] >= 0.34)
                & (points[:, 0] <= 0.72)
                & (np.abs(points[:, 1]) <= 0.29)
                & (points[:, 2] >= 0.405)
                & (points[:, 2] <= 0.76)
            )
            maximum_prior_distance = 0.085 if prior_radius is None else prior_radius
            if prior_xy is not None:
                keep &= (
                    np.linalg.norm(points[:, :2] - prior_xy[None, :], axis=1)
                    <= maximum_prior_distance
                )
            points = points[keep]
            geometry_pixels = len(points)
            if geometry_pixels < 10:
                raise ValueError(
                    f"YOLO box for {spec.name} has no valid RGB-D geometry"
                )
            xy = np.median(points[:, :2], axis=0)
            visible_surface_z = float(
                np.percentile(points[:, 2], surface_percentile)
            )
            surface = np.array([xy[0], xy[1], visible_surface_z])
        else:
            # Tray boxes cover a large part of the workspace. Their centre
            # pixel is sometimes occupied by the robot or a tall object, so a
            # single centre-depth validity check can discard a high-confidence
            # tray detection. Use the far-depth part of the current box only
            # as a provisional projection. The controller immediately refines
            # both tray centres from the current RGB-D point cloud before any
            # target waypoint is created.
            depth = float(np.percentile(sub[finite], 85.0))
            surface = self._unproject(camera, u_c, v_c, depth)

        maximum_prior_distance = 0.085 if prior_radius is None else prior_radius
        if prior_xy is not None and np.linalg.norm(surface[:2] - prior_xy) > maximum_prior_distance:
            raise ValueError(f"YOLO box centre for {spec.name} is far from the prior")
        if isinstance(spec, ObjectSpec) and not (
            TABLE_TOP_Z - 0.02 <= surface[2] <= 0.70
        ):
            raise ValueError(f"YOLO depth for {spec.name} is outside the tabletop workspace")

        if isinstance(spec, ObjectSpec):
            center_z = float(surface[2] - spec.half_height)
            minimum_center_z = TABLE_TOP_Z + spec.half_height - 0.012
            if center_z < minimum_center_z and not allow_flat:
                raise ValueError(f"RGB-D geometry is inconsistent with {spec.name}")
            if prior_xy is None and center_z > TABLE_TOP_Z + spec.half_height + 0.020 and not allow_flat:
                raise ValueError(f"RGB-D geometry is inconsistent with resting {spec.name}")
            color = spec.color
        else:
            center_z = TABLE_TOP_Z
            color = "container"

        detected = DetectedObject(
            name=spec.name,
            color=color,
            shape=spec.shape,
            position=np.array([surface[0], surface[1], center_z], dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        evidence = YOLODetectionEvidence(
            target_id=spec.name,
            score=score,
            box_xyxy=box,
            geometry_pixels=geometry_pixels,
            latency_s=latency_s,
            endpoint=self.endpoint,
        )
        return detected, evidence

    @staticmethod
    def _unproject(
        camera: CameraObservation, u: float, v: float, depth: float
    ) -> np.ndarray:
        height, width = camera.depth.shape
        focal = 0.5 * height / np.tan(np.deg2rad(camera.fovy_degrees) * 0.5)
        point_camera = np.array(
            [
                (u - (width - 1) * 0.5) * depth / focal,
                ((height - 1) * 0.5 - v) * depth / focal,
                -depth,
            ]
        )
        return camera.position + camera.rotation @ point_camera

    @staticmethod
    def _unproject_many(
        camera: CameraObservation, cols: np.ndarray, rows: np.ndarray
    ) -> np.ndarray:
        height, width = camera.depth.shape
        depth = camera.depth[rows, cols].astype(np.float64)
        focal = 0.5 * height / np.tan(np.deg2rad(camera.fovy_degrees) * 0.5)
        points_camera = np.column_stack(
            [
                (cols - (width - 1) * 0.5) * depth / focal,
                ((height - 1) * 0.5 - rows) * depth / focal,
                -depth,
            ]
        )
        return camera.position + points_camera @ camera.rotation.T


class Task3YOLOPerception(YOLOPerception):
    """YOLO box decoder retaining the geometry tuned for multi-object sorting."""

    def _decode_detection(
        self,
        camera: CameraObservation,
        spec: ObjectSpec | ContainerSpec,
        candidates: list[dict[str, Any]],
        *,
        latency_s: float,
        prior_xy: np.ndarray | None,
        prior_radius: float | None,
        allow_flat: bool,
        surface_percentile: float = 70.0,
    ) -> tuple[DetectedObject, YOLODetectionEvidence]:
        if not candidates:
            raise ValueError(f"YOLO found no instance for {spec.name}")

        best = max(candidates, key=lambda item: float(item.get("score", -1.0)))
        score = float(best.get("score", -1.0))
        box_values = best.get("box_xyxy")
        if score < 0.0 or not isinstance(box_values, list) or len(box_values) != 4:
            raise TypeError(f"invalid YOLO detection for {spec.name}")

        box = tuple(float(value) for value in box_values)
        height, width = camera.depth.shape
        x1, y1, x2, y2 = box
        if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
            raise ValueError(f"YOLO box for {spec.name} is outside the image")

        u_center = 0.5 * (x1 + x2)
        v_center = 0.5 * (y1 + y2)
        ix1 = int(np.clip(np.floor(x1), 0, width - 1))
        ix2 = int(np.clip(np.ceil(x2), 0, width - 1))
        iy1 = int(np.clip(np.floor(y1), 0, height - 1))
        iy2 = int(np.clip(np.ceil(y2), 0, height - 1))
        depth_region = camera.depth[iy1 : iy2 + 1, ix1 : ix2 + 1]
        finite = np.isfinite(depth_region) & (depth_region > 0)
        geometry_pixels = int(np.count_nonzero(finite))
        if geometry_pixels == 0:
            raise ValueError(f"YOLO box for {spec.name} has no valid depth")

        if isinstance(spec, ObjectSpec) and allow_flat:
            # A side-lying package spans a wide detector box whose centre ray
            # can hit background, a fingertip, or the tray rim.  Project all
            # live RGB-D samples in the model box and use their robust spatial
            # centre, just as the generic decoder does for irregular objects.
            # This branch remains fully model-grounded: no simulator pose or
            # seed-specific coordinate participates in the estimate.
            local_rows, local_cols = np.nonzero(finite)
            rows = local_rows + iy1
            cols = local_cols + ix1
            points = self._unproject_many(camera, cols, rows)
            keep = (
                (points[:, 0] > 0.10)
                & (points[:, 0] < 0.85)
                & (np.abs(points[:, 1]) < 0.40)
                & (points[:, 2] > TABLE_TOP_Z + 0.008)
                & (points[:, 2] < 0.70)
            )
            maximum_prior_distance = 0.085 if prior_radius is None else prior_radius
            if prior_xy is not None:
                keep &= (
                    np.linalg.norm(points[:, :2] - prior_xy[None, :], axis=1)
                    <= maximum_prior_distance
                )
            points = points[keep]
            geometry_pixels = len(points)
            if geometry_pixels < 10:
                raise ValueError(
                    f"YOLO box for {spec.name} has no valid RGB-D geometry"
                )
            xy = np.median(points[:, :2], axis=0)
            surface = np.array(
                [
                    xy[0],
                    xy[1],
                    float(np.percentile(points[:, 2], surface_percentile)),
                ]
            )
        elif isinstance(spec, ObjectSpec):
            surface_depth = float(np.percentile(depth_region[finite], 5.0))
        else:
            center_col = int(np.clip(round(u_center), 0, width - 1))
            center_row = int(np.clip(round(v_center), 0, height - 1))
            surface_depth = float(camera.depth[center_row, center_col])
            if not np.isfinite(surface_depth) or surface_depth <= 0:
                raise ValueError(f"YOLO box centre for {spec.name} has invalid depth")
        if not (isinstance(spec, ObjectSpec) and allow_flat):
            surface = self._unproject(camera, u_center, v_center, surface_depth)

        maximum_prior_distance = 0.085 if prior_radius is None else prior_radius
        if (
            prior_xy is not None
            and np.linalg.norm(surface[:2] - prior_xy) > maximum_prior_distance
        ):
            raise ValueError(f"YOLO box centre for {spec.name} is far from the prior")
        if not (TABLE_TOP_Z - 0.02 <= surface[2] <= 0.70):
            raise ValueError(f"YOLO box centre for {spec.name} is not on the tabletop")

        if isinstance(spec, ObjectSpec):
            center_z = float(surface[2] - spec.half_height)
            minimum_center_z = TABLE_TOP_Z + spec.half_height - 0.012
            if center_z < minimum_center_z and not allow_flat:
                raise ValueError(f"RGB-D geometry is inconsistent with {spec.name}")
            if (
                prior_xy is None
                and center_z > TABLE_TOP_Z + spec.half_height + 0.020
                and not allow_flat
            ):
                raise ValueError(f"RGB-D geometry is inconsistent with resting {spec.name}")
            color = spec.color
        else:
            center_z = TABLE_TOP_Z
            color = "container"

        detected = DetectedObject(
            name=spec.name,
            color=color,
            shape=spec.shape,
            position=np.array([surface[0], surface[1], center_z], dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        evidence = YOLODetectionEvidence(
            target_id=spec.name,
            score=score,
            box_xyxy=box,
            geometry_pixels=geometry_pixels,
            latency_s=latency_s,
            endpoint=self.endpoint,
        )
        return detected, evidence
