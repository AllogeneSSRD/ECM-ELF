# Ely pet artwork

`rig.rs` and `LICENSE` are copied from ELYTerminal commit
`fe07813fdfa9fd600274066d9fcd88d5f5a3bcc0`, `src/pet/rig.rs` (MIT).
The source SHA256 and generated atlas geometry are recorded in
`assets/gui_v2/pet_ely.json`. The generator preserves attribution and converts
the layered vector artwork to seven moods, eight frames each. No Rust/GPUI
runtime is linked. Native positioning and behavior are adapted in the v2 pet module.

Regenerate SVGs with `tools/gen/generate_gui_v2_pet.py`, then rasterize offline
with `tools/gen/render_gui_v2_pet.cjs`. Browser tooling is build-time only.
