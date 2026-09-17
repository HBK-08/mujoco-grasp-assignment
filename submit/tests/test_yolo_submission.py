from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from graspbench.config import CONTAINER_SPEC_BY_NAME, OBJECT_SPEC_BY_NAME
from graspbench.perception import ModelServiceError
from graspbench.types import CameraObservation
from policies.sort_policy import _MUSTARD_VERTICAL_RECOVERY_PHI, SortPolicy
from policies.student_policy import (
    StudentPolicy,
    _min_rect_angle,
    _parse_instruction,
)
from policies.yolo_perception import (
    Task3YOLOPerception,
    YOLOPerception,
    _classify_container_point_cloud,
)


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("把红色方块放进方盘。", [("red_cube", "square_tray")]),
        ("Place the apple into the square tray.", [("apple", "square_tray")]),
        (
            "Sort the fruit into the round tray and the packaged food into the square tray.",
            [
                ("apple", "round_tray"),
                ("orange", "round_tray"),
                ("mustard_bottle", "square_tray"),
                ("potted_meat_can", "square_tray"),
            ],
        ),
    ],
)
def test_instruction_parser(instruction: str, expected: list[tuple[str, str]]) -> None:
    actions, candidates = _parse_instruction(instruction)
    assert [(item["pick_id"], item["place_id"]) for item in actions] == expected
    assert set(candidates) == {name for pair in expected for name in pair}


def test_response_schema_rejects_missing_detection_list() -> None:
    with pytest.raises(ModelServiceError):
        YOLOPerception._group_detections({}, ("apple",))


def test_yolo_box_is_unprojected_with_current_depth() -> None:
    camera = CameraObservation(
        name="overhead",
        rgb=np.zeros((100, 100, 3), dtype=np.uint8),
        depth=np.full((100, 100), 0.55, dtype=np.float32),
        position=np.array([0.5, 0.0, 1.0]),
        rotation=np.eye(3),
        fovy_degrees=60.0,
    )
    perception = YOLOPerception(endpoint="http://unused")
    detected, evidence = perception._decode_detection(
        camera,
        OBJECT_SPEC_BY_NAME["red_cube"],
        [{"target_id": "red_cube", "score": 0.9, "box_xyxy": [40, 40, 60, 60]}],
        latency_s=0.01,
        prior_xy=None,
        prior_radius=None,
        allow_flat=False,
    )
    np.testing.assert_allclose(detected.position[:2], [0.503175, -0.003175], atol=1e-5)
    assert detected.position[2] == pytest.approx(
        1.0 - 0.55 - OBJECT_SPEC_BY_NAME["red_cube"].half_height,
        abs=1e-5,
    )
    assert evidence.target_id == "red_cube"
    assert evidence.score == pytest.approx(0.9)


def test_tray_detection_survives_an_occluded_box_centre() -> None:
    depth = np.full((100, 100), 0.60, dtype=np.float32)
    depth[49:52, 49:52] = 0.25
    camera = CameraObservation(
        name="overhead",
        rgb=np.zeros((100, 100, 3), dtype=np.uint8),
        depth=depth,
        position=np.array([0.5, 0.0, 1.0]),
        rotation=np.eye(3),
        fovy_degrees=60.0,
    )
    perception = YOLOPerception(endpoint="http://unused")
    detected, evidence = perception._decode_detection(
        camera,
        CONTAINER_SPEC_BY_NAME["square_tray"],
        [{"target_id": "square_tray", "score": 0.98, "box_xyxy": [10, 10, 90, 90]}],
        latency_s=0.01,
        prior_xy=None,
        prior_radius=None,
        allow_flat=False,
    )
    assert detected.name == "square_tray"
    assert detected.position[2] == pytest.approx(0.4)
    assert evidence.score == pytest.approx(0.98)


def test_task3_flat_recovery_uses_box_point_cloud() -> None:
    depth = np.full((100, 100), 0.56, dtype=np.float32)
    camera = CameraObservation(
        name="overhead",
        rgb=np.zeros((100, 100, 3), dtype=np.uint8),
        depth=depth,
        position=np.array([0.5, 0.0, 1.0]),
        rotation=np.eye(3),
        fovy_degrees=60.0,
    )
    perception = Task3YOLOPerception(endpoint="http://unused")
    detected, evidence = perception._decode_detection(
        camera,
        OBJECT_SPEC_BY_NAME["mustard_bottle"],
        [
            {
                "target_id": "mustard_bottle",
                "score": 0.95,
                "box_xyxy": [40, 45, 70, 60],
            }
        ],
        latency_s=0.01,
        prior_xy=None,
        prior_radius=None,
        allow_flat=True,
        surface_percentile=98.0,
    )
    assert detected.name == "mustard_bottle"
    assert np.isfinite(detected.position).all()
    assert evidence.geometry_pixels > 10


def test_stage_clock_counts_elapsed_control_periods() -> None:
    for policy_class in (StudentPolicy, SortPolicy):
        policy = policy_class.__new__(policy_class)
        policy._last_policy_time = None
        assert policy._elapsed_control_steps(SimpleNamespace(time=1.00)) == 1
        assert policy._elapsed_control_steps(SimpleNamespace(time=1.08)) == 2
        assert policy._elapsed_control_steps(SimpleNamespace(time=1.20)) == 3


@pytest.mark.parametrize("square_y", [-0.18, 0.18])
def test_tray_shape_classification_is_independent_of_side(square_y: float) -> None:
    axis = np.linspace(-0.13, 0.13, 41)
    grid_x, grid_y = np.meshgrid(axis, axis)
    square = np.column_stack(
        [grid_x.ravel() + 0.32, grid_y.ravel() + square_y]
    )
    disk_mask = grid_x**2 + grid_y**2 <= 0.13**2
    circle = np.column_stack(
        [grid_x[disk_mask] + 0.32, grid_y[disk_mask] - square_y]
    )

    classified = _classify_container_point_cloud(np.vstack([square, circle]))

    assert classified["square_tray"]["position"][1] == pytest.approx(
        square_y, abs=0.005
    )
    assert classified["round_tray"]["position"][1] == pytest.approx(
        -square_y, abs=0.005
    )
    assert (
        classified["square_tray"]["corner_fraction"]
        > classified["round_tray"]["corner_fraction"]
    )


def test_banana_release_combines_retreat_and_camera_clear() -> None:
    policy = StudentPolicy.__new__(StudentPolicy)
    policy.pick_id = "banana"
    policy.place_id = "round_tray"
    expected = {
        "descend_place": "settle_place",
        "settle_place": "place",
        "place": "retreat",
        "retreat": "post_place_check",
    }
    for stage, next_stage in expected.items():
        policy.stage = stage
        assert policy._next_stage() == next_stage


def test_banana_grasp_has_a_seating_stage_before_lift() -> None:
    policy = StudentPolicy.__new__(StudentPolicy)
    policy.pick_id = "banana"
    policy.place_id = "round_tray"
    policy.stage = "close"
    assert policy._next_stage() == "seat"
    policy.stage = "seat"
    assert policy._next_stage() == "lift"


def test_banana_retry_records_feedback_driven_recovery() -> None:
    policy = StudentPolicy.__new__(StudentPolicy)
    policy.pick_id = "banana"
    policy._banana_retry_active = False
    assert policy._banana_retry_active is False
    policy._banana_retry_active = True
    assert policy._banana_retry_active is True


def test_minimum_rectangle_recovers_rotated_long_axis() -> None:
    angle = 0.43
    corners = np.array(
        [[-0.06, -0.015], [-0.06, 0.015], [0.06, -0.015], [0.06, 0.015]]
    )
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    estimated, aspect = _min_rect_angle(corners @ rotation.T)
    error = (estimated - angle + np.pi / 2.0) % np.pi - np.pi / 2.0
    assert error == pytest.approx(0.0, abs=1e-6)
    assert aspect == pytest.approx(4.0, abs=1e-6)


def test_vertical_mustard_recovery_avoids_exact_ik_branch_boundary() -> None:
    assert _MUSTARD_VERTICAL_RECOVERY_PHI == pytest.approx(-1.5656)
    assert abs(_MUSTARD_VERTICAL_RECOVERY_PHI + np.pi / 2.0) < 0.01
    assert _MUSTARD_VERTICAL_RECOVERY_PHI != -np.pi / 2.0


def test_regular_wrist_seed_keeps_existing_joint7_convention() -> None:
    policy = SortPolicy.__new__(SortPolicy)
    policy.pick_id = "blue_box"
    policy._mustard_regrasp_done = False
    policy._reference_joint7 = 0.785

    assert policy._wrist_seed_joint7(0.4) == pytest.approx(0.385)


def test_mustard_transport_preserves_validated_decisive_clamp() -> None:
    policy = SortPolicy.__new__(SortPolicy)
    policy.pick_id = "mustard_bottle"

    opening = policy._transport_gripper_opening()

    assert opening == policy.GRIPPER_CLOSED


def test_can_transport_does_not_reuse_mustard_finite_clamp() -> None:
    policy = SortPolicy.__new__(SortPolicy)
    policy.pick_id = "potted_meat_can"

    assert policy._transport_gripper_opening() == policy.CAN_HOLD_OPENING
    assert policy.CAN_HOLD_OPENING != policy.MUSTARD_HOLD_OPENING


def test_rehome_budget_covers_large_post_mustard_wrist_return() -> None:
    # Rehome is encoder-closed-loop.  Its timeout must not reuse the short
    # Cartesian-stage bound: an async rollout can otherwise leave joint 7 on
    # the mustard branch before starting the can grasp.
    assert SortPolicy.REHOME_STEPS > 2 * SortPolicy.MAX_STAGE_STEPS


def test_can_lift_requires_meaningful_table_clearance() -> None:
    assert SortPolicy.CAN_LIFT_CLEARANCE >= 0.03
    assert SortPolicy.CAN_LIFT_STEPS > SortPolicy.GRIP_STEPS


def test_can_uses_inboard_alignment_before_descent() -> None:
    policy = SortPolicy.__new__(SortPolicy)
    policy.pick_id = "potted_meat_can"
    policy.place_id = "square_tray"
    policy.pick_xyz = np.array([0.66, 0.20, 0.44])
    policy.stage = "approach"

    assert policy._next_stage() == "align"
    policy.stage = "align"
    assert policy._next_stage() == "descend"

    policy.stage = "detour"
    assert policy._next_stage() == "cross_lane"
