"""Student-side post-action verification from a fresh RGB-D observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graspbench.config import OBJECT_SPEC_BY_NAME, TABLE_TOP_Z
from graspbench.types import CameraObservation

_VERIFY_RADIUS_M = {
    "banana": 0.070,
    "mustard_bottle": 0.060,
    "apple": 0.048,
    "orange": 0.048,
}

# A side-lying mustard bottle is often partially occluded by the square-tray
# rim in the overhead RGB-D view.  Four coherent elevated samples are still
# sufficient for that tall object; all other targets retain the stricter count.
_VERIFY_MIN_POINTS = {"mustard_bottle": 4}


@dataclass(frozen=True)
class PlacementVerification:
    """Auditable result of a fresh depth-geometry occupancy check."""

    target_id: str
    expected_xy: tuple[float, float]
    radius_m: float
    elevated_point_count: int
    median_surface_z: float | None
    verified: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "fresh_rgbd_expected_slot_occupancy",
            "target_id": self.target_id,
            "expected_xy": list(self.expected_xy),
            "radius_m": self.radius_m,
            "elevated_point_count": self.elevated_point_count,
            "median_surface_z": self.median_surface_z,
            "verified": self.verified,
        }


def verify_placement(
    camera: CameraObservation,
    *,
    target_id: str,
    expected_xy: np.ndarray,
) -> PlacementVerification:
    """Check for object-height geometry at the model-grounded placement slot.

    The check reads only the new overhead depth frame and calibration.  The
    target identity and expected slot originate from the earlier model plan;
    no simulator bodies, task answers, seed, or evaluator predicate are used.
    A failed check is deliberately inconclusive and must trigger model
    re-perception rather than being treated as proof of task failure.
    """
    height, width = camera.depth.shape
    depth = camera.depth.astype(np.float64)
    focal = 0.5 * height / np.tan(np.deg2rad(camera.fovy_degrees) * 0.5)
    rows, cols = np.mgrid[0:height, 0:width]
    points_camera = np.stack(
        [
            (cols - (width - 1) * 0.5) * depth / focal,
            ((height - 1) * 0.5 - rows) * depth / focal,
            -depth,
        ],
        axis=-1,
    )
    world = camera.position + points_camera @ camera.rotation.T

    xy = np.asarray(expected_xy, dtype=np.float64)
    radius = _VERIFY_RADIUS_M.get(target_id, 0.045)
    object_spec = OBJECT_SPEC_BY_NAME.get(target_id)
    max_surface_z = (
        TABLE_TOP_Z + 2.0 * object_spec.half_height + 0.045
        if object_spec is not None
        else TABLE_TOP_Z + 0.19
    )
    distance_sq = (world[..., 0] - xy[0]) ** 2 + (world[..., 1] - xy[1]) ** 2
    finite = np.isfinite(world).all(axis=-1)
    elevated = (
        finite
        & (distance_sq <= radius * radius)
        & (world[..., 2] >= TABLE_TOP_Z + 0.018)
        & (world[..., 2] <= max_surface_z)
    )
    surfaces = world[..., 2][elevated]
    count = int(surfaces.size)
    median_z = float(np.median(surfaces)) if count else None
    return PlacementVerification(
        target_id=target_id,
        expected_xy=(float(xy[0]), float(xy[1])),
        radius_m=radius,
        elevated_point_count=count,
        median_surface_z=median_z,
        verified=count >= _VERIFY_MIN_POINTS.get(target_id, 8),
    )

