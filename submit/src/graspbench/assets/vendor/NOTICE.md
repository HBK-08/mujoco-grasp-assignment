# Vendored robot asset

`franka_emika_panda/` is copied from Google DeepMind's MuJoCo Menagerie at
commit `71f066ad0be9cd271f7ed58c030243ef157af9f4` (2026-07-04).

- Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
- License: Apache-2.0; the upstream `LICENSE` is retained in the directory.
- Local changes: `panda.xml` points `meshdir` at its vendored location, adds a
  small `grasp_site` between the finger pads for task-space Jacobians, and
  removes the upstream keyframe because this course scene adds free joints for
  tabletop objects.

No upstream mesh was modified.
