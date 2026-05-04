<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# internal/scripts

One-off maintenance scripts. Not part of the public surface.

## migrate_kelvin_full_pt.py

Re-pickles a `kelvin_full.pt` artifact under the current class
qualnames. Needed when a class rename inside `instant_nurec.*` makes
old pickles unloadable (the unpickler fails because
`old.qualname.OldClass` no longer exists).

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

After running once, replace your `kelvin_full.pt` cache (or set
`INSTANT_NUREC_FULL_PT`) to point at the migrated file. Future-rename
maintainers should update the alias table in the script.
