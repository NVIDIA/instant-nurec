<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# internal/scripts

One-off maintenance scripts. Not part of the public surface; not shipped
in the wheel.

## export_kelvin_jit.py

Builds the shipped `kelvin_jit.pt` TorchScript artifact from the legacy
pickled `kelvin_full.pt`. Run once per release; the wheel only depends
on the resulting `kelvin_jit.pt`.

How it works:

1. Pre-registers `sys.modules` aliases + class-attribute aliases so the
   legacy `kelvin_full.pt` (which baked in qualnames like
   `instant_nurec.config_schema.models.KelvinModelConfig` and
   `instant_nurec.model.kelvin.KelvinInstantNuRec`) can be unpickled
   against the post-relocation classes under
   `instant_nurec_internal.*`.
2. Loads the pickled `GaussiansInstantNuRecSystem`, builds a fresh
   `KelvinStaticCore` from its config (with `checkpointing` disabled —
   the activation-checkpoint wrapper inserts a `_NoopSaveInputs`
   autograd Function that `torch.jit.save` cannot serialize), and
   copies weights.
3. Registers JIT-baked scalars + input-shape constraints as persistent
   buffers on `KelvinStaticCore` (`scene_rescale_buffer`, `expected_b`,
   `expected_v`, `expected_h`, `expected_w`) so the runtime adapter can
   read them back without any Pydantic config plumbing.
4. Loads one representative ncorev4 batch via the existing dataloader,
   extracts the trace-input tensors, and runs `torch.jit.trace` on
   `TraceableStaticCore` (a thin wrapper exposing
   `KelvinStaticCore.forward_tensors` as the module's `forward`).
5. Saves the traced module to the requested output path and round-trips
   it (`torch.jit.save` → `torch.jit.load` → re-run on the same input)
   to confirm shape, dtype, and finiteness of the load output.

End-to-end PLY parity against the eager pickle path is the actual
user-facing gate and runs at inference time via
`internal/benchmark/validate_parity.py`, not inside this script.

Usage:

```bash
INSTANT_NUREC_FULL_PT=/path/to/kelvin_full.pt \
python internal/scripts/export_kelvin_jit.py \
    --ncore-path /path/to/clips/<uuid>/pai_<uuid>.json \
    --output /path/to/kelvin_jit.pt
```

After running, set `INSTANT_NUREC_FULL_PT=/path/to/kelvin_jit.pt` (the
runtime auto-detects the file format) or upload the artifact to the
Hugging Face repo so the auto-download path picks it up.

## migrate_kelvin_full_pt.py

Re-pickles a legacy `kelvin_full.pt` artifact under the current class
qualnames. Needed when a class rename inside `instant_nurec.*` makes
old pickles unloadable (the unpickler fails because
`old.qualname.OldClass` no longer exists).

Largely retired now that the runtime has retired the pickle path
entirely (`make()` only consumes JIT artifacts). Kept for the
`export_kelvin_jit.py` workflow above, where loading the legacy pickle
is a prerequisite for tracing the JIT artifact.

How it works:

1. Installs runtime aliases that map the legacy qualnames
   (e.g. `instant_nurec.config_schema.nrm.NRMConfig`) onto the
   current classes.
2. Calls `torch.load(old_pt, map_location="cpu", weights_only=False)`,
   which uses those aliases to materialize the pickled instance.
3. Calls `torch.save(system, new_pt)`, which writes the same tensor
   data under the current qualnames. The result loads cleanly with no
   aliasing on subsequent runs.

Usage:

```bash
python internal/scripts/migrate_kelvin_full_pt.py /path/to/old/kelvin_full.pt /path/to/new/kelvin_full.pt
```
