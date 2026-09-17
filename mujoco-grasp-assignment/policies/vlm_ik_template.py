from __future__ import annotations

import mujoco

from graspbench.camera import camera_by_name
from graspbench.config import HOME_Q
from graspbench.ik import DampedLeastSquaresIK
from graspbench.perception import FoundationModelPerception, ModelServiceError
from graspbench.types import JointPositionCommand, Observation, PolicyDecision
from graspbench.vlm import OpenAICompatibleVLM


class VLMAndIKTemplate:
    """VLM + SAM3 starter; ``act`` runs on the framework's single async worker."""

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        self.task = task
        self.ik = DampedLeastSquaresIK(model)
        self.perception = FoundationModelPerception()
        self.vlm = OpenAICompatibleVLM()
        self.target_id = None
        self.target_world = None
        self.model_evidence = {}

    def act(self, observation: Observation) -> PolicyDecision:
        # ``observation`` is a stable snapshot. Do not retain env.data or
        # launch another per-step thread: the evaluator keeps physics live
        # while this event-triggered remote request is pending.
        if self.target_world is None:
            camera = camera_by_name(observation, "overhead")
            try:
                detections, sam_evidence = self.perception.detect_scene(camera)
                grounding = self.vlm.ground_target(
                    observation.instruction, camera, detections, sam_evidence
                )
            except ModelServiceError as exc:
                return PolicyDecision(
                    command=JointPositionCommand(observation.joint_position, 1.0),
                    stage="model_error",
                    rationale=f"Model service failed; hold safely: {exc}",
                    done=True,
                )
            self.target_id = grounding.target_id
            self.target_world = detections[self.target_id].position.copy()
            self.model_evidence = {
                "grounding": grounding.as_dict(),
                "sam3": sam_evidence[self.target_id].as_dict(),
            }

        # TODO: implement PREGRASP -> DESCEND -> CLOSE -> LIFT -> VERIFY and
        # call self.ik.solve(...) for each Cartesian waypoint. Re-run SAM3 after
        # action events or failures instead of calling the language model at 50 Hz.
        return PolicyDecision(
            command=JointPositionCommand(HOME_Q, 1.0),
            stage="todo_model_ik",
            rationale="The VLM selected a live SAM3 detection; motion state machine remains TODO.",
            target_id=self.target_id,
            debug={
                "target_world": self.target_world.tolist(),
                "model_evidence": self.model_evidence,
            },
        )
