from __future__ import annotations

import numpy as np

from .types import CameraObservation


def unproject_pixel(
    camera: CameraObservation,
    pixel_xy: tuple[float, float],
    *,
    depth: float | None = None,
) -> np.ndarray:
    """Unproject an RGB-D pixel to world coordinates.

    MuJoCo cameras look along local -Z, local +Y is image-up, and rendered
    depth is metric optical-axis depth. `pixel_xy` uses the common (u, v)
    convention with v increasing down the image.
    """
    u, v = map(float, pixel_xy)
    height, width = camera.depth.shape
    if depth is None:
        row = int(np.clip(round(v), 0, height - 1))
        col = int(np.clip(round(u), 0, width - 1))
        depth = float(camera.depth[row, col])
    focal = 0.5 * height / np.tan(np.deg2rad(camera.fovy_degrees) * 0.5)
    point_camera = np.array(
        [
            (u - (width - 1) * 0.5) * depth / focal,
            ((height - 1) * 0.5 - v) * depth / focal,
            -depth,
        ]
    )
    return camera.position + camera.rotation @ point_camera


def camera_by_name(observation, name: str) -> CameraObservation:
    try:
        return next(camera for camera in observation.cameras if camera.name == name)
    except StopIteration as exc:
        raise ValueError(f"Camera {name!r} is not present in the observation") from exc
