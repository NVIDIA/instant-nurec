# Phase 2 Step 7 — `nvdiffrast.dr.texture` → `torch.nn.functional.grid_sample` research

Author: claude (kelvin-standalone, 2026-05-02 overnight)
Status: pre-implementation research; no code changes yet.

## Single nvdiffrast call site

After Phase 1 step 4.3 strip iteration, the only remaining `nvdiffrast` call
in the predict graph is:

```python
# nre/nrm/utils/cubemap.py:112
sky_color = dr.texture(
    cubemap[None],          # (1, 6, H, H, 3)
    opengl_rays_d[None, None],  # (1, 1, N, 3)
    filter_mode="linear",
    boundary_mode="cube",
)
```

invoked from `rotate_sky_cubemap(cubemap, rotation)`. Callers:

- `nre/nrm/primitives/kelvin_primitive.py:322` — `KelvinNRMPrimitive.rigid_transform()`,
  which is invoked from `nre/nrm/predict/export_ply.py:106` and
  `nre/nrm/predict/primitive_merge.py:193`.

So the rotation is `T_new[:3, :3]` of whatever per-chunk-to-world rigid
transform the export / merge passes apply.

## Convention notes

`cubemap_ray_directions(size)` returns `(6, H, H, 3)` ordered as
`[right, left, top, bottom, front, back]`. Decoding the per-face direction
constructions:

| index | NRE name | dominant axis (pre-Y-flip) |
|---|---|---|
| 0 | right_dirs   = (1, vv, -uu) | +X |
| 1 | left_dirs    = (-1, vv, uu) | -X |
| 2 | top_dirs     = (uu, -1, vv) | -Y |
| 3 | bottom_dirs  = (uu, 1, -vv) | +Y |
| 4 | front_dirs   = (uu, vv, 1)  | +Z |
| 5 | back_dirs    = (-uu, vv, -1) | -Z |

Note this is **not** the OpenGL cubemap face order (`+X, -X, +Y, -Y, +Z, -Z`):
indices 2 and 3 are swapped (NRE puts -Y before +Y).

Before sampling, the code negates Y on the query rays:

```python
opengl_rays_d = torch.stack([query_rays[:, 0], -query_rays[:, 1], query_rays[:, 2]], dim=-1)
```

This Y-flip compensates for the layout mismatch — when nvdiffrast (OpenGL
convention) sees a +Y-dominant ray, its "+Y" face index 2 in OpenGL maps to
NRE's slot 2 which actually holds the -Y face content. The Y-flip swaps the
+Y/-Y slots so the lookup lands on the right content.

## Pure-torch replacement sketch

```python
def _rotate_sky_cubemap_torch(cubemap: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """grid_sample-based replacement for the dr.texture cube-boundary path."""
    H = cubemap.shape[1]
    query_rays = cubemap_ray_directions(H, device=cubemap.device) @ rotation.float()
    query_rays = query_rays.reshape(-1, 3)  # (6*H*H, 3)

    # 1. determine which face each ray hits
    abs_xyz = query_rays.abs()
    dominant_axis = abs_xyz.argmax(dim=-1)  # 0=x, 1=y, 2=z
    sign = query_rays.gather(-1, dominant_axis.unsqueeze(-1)).squeeze(-1) > 0

    # 2. NRE face index from (axis, sign)
    face_idx = torch.where(
        dominant_axis == 0, torch.where(sign, 0, 1),  # +X=0, -X=1
        torch.where(
            dominant_axis == 1, torch.where(sign, 3, 2),  # +Y=3, -Y=2
            torch.where(sign, 4, 5),                       # +Z=4, -Z=5
        ),
    )

    # 3. project ray onto face plane to get (u, v)
    # For the +X face: parameterized as (1, vv, -uu) → u = -z/x, v = y/x.
    # Each face has its own (u, v) formula. Since absolute-value of dominant
    # axis is the projection denom, divide non-dominant components by it.
    # Sign of u/v depends on which face — see table.
    # ... (this is the part that needs careful per-face implementation)

    # 4. for each pixel, gather the right face slice and grid_sample with bilinear
    # (One option: split by face_idx and process each face's pixels in a loop;
    # another: build a (N, 3) grid_sample input over a single 2-D slab made by
    # tiling the 6 faces side by side, with a face-offset added to u.)

    # 5. reshape back to (6, H, H, 3)
    return result.reshape(6, H, H, 3)
```

## Per-face (u, v) formulas

For a ray `r = (x, y, z)` hitting each face, the projected (u, v) is the
inverse of the parameterization above. Letting `a = abs(dominant_component)`:

| face | dominant | u | v |
|---|---|---|---|
| 0 (+X) | +x | -z/x | y/x |
| 1 (-X) | -x | -z/x = z/|x| ⋅ (-sign) — recompute | … |
| 2 (-Y) | -y | x/(-y) | z/(-y) |
| 3 (+Y) | +y | x/y | -z/y |
| 4 (+Z) | +z | x/z | y/z |
| 5 (-Z) | -z | -x/z | y/z |

All formulas need to be re-derived carefully and verified against
`cubemap_ray_directions` outputs (i.e., feed an axis-aligned ray, check the
formula recovers (uu=0, vv=0) ↔ pixel center on the right face).

## Validation strategy

1. **Identity-rotation test** — rotating any cubemap by the identity rotation
   should reproduce the input. Both `dr.texture` and the torch replacement
   must satisfy this.
2. **End-to-end parity** — toggle between the two implementations via env var
   (e.g. `INSTANT_NUREC_TORCH_CUBEMAP=1`), run both `--merge none` and
   `--merge frustum-ownership`, validate that the PLY outputs match within
   `tests/tolerance.json` (Phase 2 step 7.4 allows ratcheting that file
   upwards).

## What blocks autonomous completion

- nvdiffrast is not installable in the cpu-only `.venv`, so reference-equivalence
  tests (the cleaner TDD form) require a bazel-runtime test or a CUDA-enabled
  test environment.
- Per-face (u, v) sign conventions must be verified empirically (each face has
  its own swizzle / negation pattern). Getting this wrong silently degrades sky
  rendering quality without crashing.
- The end-to-end parity tolerance may need to ratchet upward (cross-face seams
  no longer get nvdiffrast's smooth interpolation); user approval per
  CLAUDE.md §4.1.2.5.

## Recommended next session shape

1. Implement `_rotate_sky_cubemap_torch` per the table above.
2. Add identity-rotation invariant test (no nvdiffrast dep).
3. Toggle the call site behind `INSTANT_NUREC_TORCH_CUBEMAP` env var.
4. Run e2e parity for both modes; compare PLY against baseline.
5. If outside tolerance: bump tolerance.json with a commit body documenting
   each property's old vs new bound.
6. Once accepted: drop the env-var toggle, remove `import nvdiffrast.torch as dr`,
   drop `nvdiffrast` from BUILD.bazel deps.
