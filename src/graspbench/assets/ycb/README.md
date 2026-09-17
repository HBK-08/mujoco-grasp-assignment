# YCB asset provenance

This directory vendors selected meshes and texture maps from the Yale-CMU-Berkeley
Object and Model Set (YCB): `011_banana`, `013_apple`, `017_orange`,
`006_mustard_bottle`, `010_potted_meat_can`, `037_scissors`, and
`040_large_marker`.

Files were obtained from the official YCB public object archives at
<https://ycb-benchmarks.s3.amazonaws.com/index.html>, using the corresponding
`google_16k` models. They are included only to make the course scene runnable
without a separate asset deployment step. The course code is MIT licensed;
vendored YCB assets retain their upstream terms and attribution.

The MuJoCo scene uses each visual mesh for rendering. Most objects use a simple
collision proxy for robust, real-time classroom simulation. The banana is the
exception: its curved silhouette is represented by six convex macro-hulls
generated offline with `scripts/generate_banana_vhacd.py` and pyVHACD. This
keeps the gripper collision close to the rendered fruit without introducing a
runtime mesh-decomposition dependency or exposing privileged poses to student
policies.
