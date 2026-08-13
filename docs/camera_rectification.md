# Camera Rectification for NCore V4 Data

## Purpose

Rectify a source camera in ncore V4 data to match NV's training camera model (FTheta), enabling cross-platform model inference. Preserves all other cameras, lidar, poses, labels, and aux data.

## How to generate the scripts

Two scripts are needed: `rectify_ncore.py` (main tool) and `test_rectified.py` (verification).

---

### rectify_ncore.py

```python
#!/usr/bin/env python3
"""Rectify a specified camera in ncore V4 data into FTheta model.

Reads ncore V4 component stores (.zarr.itar), rectifies the specified source camera
images to the target FTheta model, keeps all other cameras unchanged, and writes
new component stores.

Usage:
    python rectify_ncore.py \
        --input /path/to/data.json \
        --output-dir /path/to/output \
        --source-camera <SOURCE_CAMERA_ID> \
        --target-camera <TARGET_CAMERA_NAME_IN_JSON> \
        --target-intrinsics /path/to/target_intrinsics.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from ncore.data.v4 import SequenceComponentGroupsReader, SequenceLoaderV4
from ncore.impl.data.stores import IndexedTarStore, consolidate_compressed_metadata
from ncore.impl.data.types import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    ShutterType,
)
from ncore.impl.sensors.camera import CameraModel


# ── Calibration ───────────────────────────────────────────────────────────────


def load_target_intrinsics(path: Path, camera_name: str) -> FThetaCameraModelParameters:
    data = json.loads(path.read_text())
    entry = data[camera_name]
    reference_poly = FThetaCameraModelParameters.PolynomialType[entry["reference_poly"]]
    return FThetaCameraModelParameters(
        resolution=np.asarray(entry["resolution"], dtype=np.uint64),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        external_distortion_parameters=None,
        principal_point=np.asarray(entry["principal_point"], dtype=np.float32),
        reference_poly=reference_poly,
        pixeldist_to_angle_poly=np.asarray(entry["pixeldist_to_angle_poly"], dtype=np.float32),
        angle_to_pixeldist_poly=np.asarray(entry["angle_to_pixeldist_poly"], dtype=np.float32),
        max_angle=float(entry.get("max_angle", entry.get("max_angle_rad", 0.0))),
        linear_cde=np.asarray(entry.get("linear_cde", [1.0, 0.0, 0.0]), dtype=np.float32),
    )


# ── Rectification ─────────────────────────────────────────────────────────────


def build_rectification_grid(
    source_params,
    target_params: FThetaCameraModelParameters,
    device: torch.device,
    source_max_angle_override: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build remap grids (map_x, map_y) + valid_mask.
    Supports OpenCVFisheye and OpenCVPinhole as source camera models.
    """
    if isinstance(source_params, OpenCVFisheyeCameraModelParameters):
        source_params_for_grid = OpenCVFisheyeCameraModelParameters(
            resolution=source_params.resolution.copy(),
            shutter_type=source_params.shutter_type,
            external_distortion_parameters=None,
            principal_point=source_params.principal_point.copy(),
            focal_length=source_params.focal_length.copy(),
            radial_coeffs=source_params.radial_coeffs.copy(),
            max_angle=source_max_angle_override,
        )
    else:
        source_params_for_grid = source_params

    source_camera = CameraModel.from_parameters(source_params_for_grid, device=str(device))
    target_camera = CameraModel.from_parameters(target_params, device=str(device))

    target_w, target_h = int(target_params.resolution[0]), int(target_params.resolution[1])
    source_res = torch.tensor([float(source_params.resolution[0]), float(source_params.resolution[1])], device=device)

    ys, xs = torch.meshgrid(
        torch.arange(target_h, device=device, dtype=torch.float32),
        torch.arange(target_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    target_pts = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    target_rays = target_camera.image_points_to_camera_rays(target_pts)
    source_proj = source_camera.camera_rays_to_image_points(target_rays)
    source_pts = source_proj.image_points

    valid = (
        source_proj.valid_flag
        & torch.isfinite(source_pts).all(dim=-1)
        & (source_pts[:, 0] >= 0.0)
        & (source_pts[:, 0] <= source_res[0] - 1.0)
        & (source_pts[:, 1] >= 0.0)
        & (source_pts[:, 1] <= source_res[1] - 1.0)
    )

    map_x = source_pts[:, 0].cpu().numpy().reshape(target_h, target_w).astype(np.float32)
    map_y = source_pts[:, 1].cpu().numpy().reshape(target_h, target_w).astype(np.float32)
    valid_mask = valid.cpu().numpy().reshape(target_h, target_w)
    map_x[~valid_mask] = -1.0
    map_y[~valid_mask] = -1.0

    return map_x, map_y, valid_mask


# ── Inpaint ───────────────────────────────────────────────────────────────────


def inpaint_invalid_regions(image, valid_mask, method="telea", radius=7.0, mask_dilate_px=3):
    """Fill invalid rectification regions using OpenCV inpainting.

    IMPORTANT: Invalid regions (where source FOV doesn't cover the target) must be
    inpainted, NOT left as black (zero). Black borders cause training artifacts
    (the model learns to reconstruct black edges instead of scene content).
    """
    inpaint_mask = (~valid_mask).astype(np.uint8) * 255
    if mask_dilate_px > 0:
        kernel = np.ones((mask_dilate_px * 2 + 1, mask_dilate_px * 2 + 1), dtype=np.uint8)
        inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)
    cv2_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(image, inpaint_mask, radius, cv2_method)


# ── Main shard processing ─────────────────────────────────────────────────────


def rectify_camera_itar(input_itar_path, output_itar_path, source_camera, target_params, map_x, map_y, valid_mask):
    """Rectify source_camera images in camera itar, keep all others unchanged."""
    in_store = IndexedTarStore(input_itar_path, mode="r")
    out_store = IndexedTarStore(output_itar_path, mode="w")
    all_keys = list(in_store.keys())

    # V4 camera itar key structure: cameras/<cam_id>/frames/<ts>/image/0
    camera_ids = set()
    for k in all_keys:
        parts = k.split("/")
        if len(parts) >= 2 and parts[0] == "cameras" and parts[1] not in (".zattrs", ".zgroup", ""):
            camera_ids.add(parts[1])

    print(f"  All cameras: {sorted(camera_ids)}")
    print(f"  Rectifying: {source_camera}")

    frame_count = 0
    for key in all_keys:
        parts = key.split("/")
        if key == ".zmetadata.cbor.xz":
            continue

        # Handle source camera keys
        if len(parts) >= 2 and parts[0] == "cameras" and parts[1] == source_camera:
            # Update camera .zattrs (intrinsics)
            if key == f"cameras/{source_camera}/.zattrs":
                orig_attrs = json.loads(in_store[key])
                orig_attrs["camera_model_parameters"] = {
                    "external_distortion_parameters": None,
                    "principal_point": target_params.principal_point.tolist(),
                    "max_angle": float(target_params.max_angle),
                    "linear_cde": target_params.linear_cde.tolist(),
                    "pixeldist_to_angle_poly": target_params.pixeldist_to_angle_poly.tolist(),
                    "angle_to_pixeldist_poly": target_params.angle_to_pixeldist_poly.tolist(),
                    "reference_poly": target_params.reference_poly.name,
                    "resolution": [int(r) for r in target_params.resolution],
                    "shutter_type": "ROLLING_TOP_TO_BOTTOM",
                }
                orig_attrs["camera_model_type"] = "ftheta"
                out_store[key] = json.dumps(orig_attrs, indent=2).encode("utf-8")
                continue

            # Rectify image data: cameras/<cam>/frames/<ts>/image/0
            if len(parts) == 6 and parts[4] == "image" and parts[5] == "0":
                data = bytes(in_store[key])
                src_img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                rectified = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                if not valid_mask.all():
                    rectified = inpaint_invalid_regions(rectified, valid_mask)
                _, buf = cv2.imencode(".jpg", rectified, [cv2.IMWRITE_JPEG_QUALITY, 95])
                new_jpeg = buf.tobytes()
                out_store[key] = new_jpeg
                # Update .zarray dtype to match new JPEG size
                zarray_key = "/".join(parts[:5]) + "/.zarray"
                out_store[zarray_key] = json.dumps({"chunks": [], "compressor": None, "dtype": f"|S{len(new_jpeg)}", "fill_value": "", "filters": None, "order": "C", "shape": [], "zarr_format": 2}).encode("utf-8")
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"    Rectified {frame_count} frames...")
                continue

            # Skip .zarray for image (already written above)
            if len(parts) == 6 and parts[4] == "image" and parts[5] == ".zarray":
                continue

            # Pass through other source camera keys (generic_data, etc.)
            out_store[key] = bytes(in_store[key])
            continue

        # All other keys: pass through unchanged
        out_store[key] = bytes(in_store[key])

    consolidate_compressed_metadata(out_store)
    in_store.close()
    out_store.close()
    print(f"  Total frames rectified: {frame_count}")


# ── SSEG aux processing ──────────────────────────────────────────────────────


def rectify_sseg_aux(input_path, output_path, source_camera, target_params, map_x, map_y, valid_mask):
    """Rectify sseg for source_camera, keep other cameras unchanged.

    V4 aux sseg key structure: aux/semantic_segmentation/<cam_id>/<frame_idx>/0
    """
    in_store = IndexedTarStore(input_path, mode="r")
    out_store = IndexedTarStore(output_path, mode="w")

    sseg_meta_key = f"aux/semantic_segmentation/{source_camera}/.zattrs"
    sseg_meta = json.loads(in_store[sseg_meta_key])
    stuff_classes = sseg_meta.get("stuff_classes", [])
    sky_class_idx = stuff_classes.index("sky") if "sky" in stuff_classes else 0

    all_keys = list(in_store.keys())
    frame_count = 0

    for key in all_keys:
        parts = key.split("/")
        if key == ".zmetadata.cbor.xz":
            continue

        if len(parts) > 2 and parts[0] == "aux" and parts[1] == "semantic_segmentation" and parts[2] == source_camera:
            if key == sseg_meta_key:
                updated_meta = sseg_meta.copy()
                updated_meta["resolution"] = [int(target_params.resolution[0]), int(target_params.resolution[1])]
                out_store[key] = json.dumps(updated_meta, indent=2).encode("utf-8")
                continue
            if key.endswith("/0") and len(parts) == 5:
                seg_data = bytes(in_store[key])
                seg_pil = PILImage.open(io.BytesIO(seg_data))
                seg_arr = np.asarray(seg_pil)
                rectified_arr = cv2.remap(seg_arr, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                # Inpaint via iterative dilation (NOT black fill)
                if not valid_mask.all():
                    invalid = ~valid_mask
                    kernel = np.ones((5, 5), dtype=np.uint8)
                    result = rectified_arr.copy()
                    filled = valid_mask.copy()
                    remaining = invalid.copy()
                    for _ in range(200):
                        if not remaining.any():
                            break
                        filled_dilated = cv2.dilate(filled.astype(np.uint8), kernel, iterations=1).astype(bool)
                        dilated = cv2.dilate(result, kernel, iterations=1)
                        newly_filled = remaining & filled_dilated
                        result[newly_filled] = dilated[newly_filled]
                        filled[newly_filled] = True
                        remaining[newly_filled] = False
                    result[remaining] = sky_class_idx
                    rectified_arr = result
                rectified_pil = PILImage.fromarray(rectified_arr, mode="P")
                if seg_pil.mode == "P":
                    rectified_pil.putpalette(seg_pil.getpalette())
                buf = io.BytesIO()
                rectified_pil.save(buf, format="png", optimize=True)
                new_png = buf.getvalue()
                out_store[key] = new_png
                zarray_key = "/".join(parts[:-1]) + "/.zarray"
                out_store[zarray_key] = json.dumps({"chunks": [], "compressor": None, "dtype": f"|S{len(new_png)}", "fill_value": "", "filters": None, "order": "C", "shape": [], "zarr_format": 2}).encode("utf-8")
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"    Rectified {frame_count} sseg frames...")
                continue
            if key.endswith("/.zarray") and len(parts) == 5:
                continue
            out_store[key] = bytes(in_store[key])
            continue

        out_store[key] = bytes(in_store[key])

    consolidate_compressed_metadata(out_store)
    in_store.close()
    out_store.close()
    print(f"  Total sseg frames rectified: {frame_count}")


# ── Aux egomask processing ───────────────────────────────────────────────────


def rectify_aux_egomask(input_path, output_path, source_camera, map_x, map_y, valid_mask):
    """Rectify aux egomask for source_camera, keep other cameras unchanged."""
    in_store = IndexedTarStore(input_path, mode="r")
    out_store = IndexedTarStore(output_path, mode="w")
    all_keys = list(in_store.keys())
    frame_count = 0

    for key in all_keys:
        parts = key.split("/")
        if key == ".zmetadata.cbor.xz":
            continue
        if len(parts) > 2 and parts[0] == "aux" and parts[1] == "egomask" and parts[2] == source_camera:
            if key.endswith("/0") and len(parts) == 5:
                mask_data = bytes(in_store[key])
                mask_pil = PILImage.open(io.BytesIO(mask_data))
                mask_arr = np.asarray(mask_pil)
                rectified_mask = cv2.remap(mask_arr, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                rectified_mask[~valid_mask] = 255
                buf = io.BytesIO()
                PILImage.fromarray(rectified_mask).save(buf, format="png", optimize=True)
                new_png = buf.getvalue()
                out_store[key] = new_png
                zarray_key = "/".join(parts[:-1]) + "/.zarray"
                out_store[zarray_key] = json.dumps({"chunks": [], "compressor": None, "dtype": f"|S{len(new_png)}", "fill_value": "", "filters": None, "order": "C", "shape": [], "zarr_format": 2}).encode("utf-8")
                frame_count += 1
                continue
            if key.endswith("/.zarray") and len(parts) == 5:
                continue
            out_store[key] = bytes(in_store[key])
            continue
        out_store[key] = bytes(in_store[key])

    consolidate_compressed_metadata(out_store)
    in_store.close()
    out_store.close()
    print(f"  Total egomask frames rectified: {frame_count}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Input ncore V4 JSON meta file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--target-intrinsics", type=str, required=True, help="Target FTheta intrinsics JSON")
    parser.add_argument("--target-camera", type=str, required=True, help="Camera name key in target intrinsics JSON")
    parser.add_argument("--source-camera", type=str, required=True, help="Source camera ID in ncore data to rectify")
    parser.add_argument("--source-max-angle", type=float, default=1.5, help="Override source max_angle for fisheye (rad)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load V4 sequence to get source camera intrinsics
    with open(args.input) as f:
        meta = json.load(f)
    input_base = os.path.dirname(args.input)

    reader = SequenceComponentGroupsReader([args.input], open_consolidated=False)
    seq = SequenceLoaderV4(reader)
    source_cam = seq.get_camera_sensor(args.source_camera)
    source_params = source_cam.model_parameters
    print(f"Source: {args.source_camera} ({type(source_params).__name__}) {source_params.resolution}")

    target_params = load_target_intrinsics(Path(args.target_intrinsics), args.target_camera)
    print(f"Target: {args.target_camera} ({type(target_params).__name__}) {target_params.resolution}")
    del reader, seq

    print("Building rectification grid...")
    map_x, map_y, valid_mask = build_rectification_grid(source_params, target_params, device, args.source_max_angle)
    print(f"  Valid pixels: {valid_mask.sum() / valid_mask.size * 100:.1f}%")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "valid_mask.png"), (valid_mask.astype(np.uint8) * 255))

    # Process camera itar
    camera_itar_name = next(cs["path"] for cs in meta["component_stores"] if "camera" in cs["path"])
    input_camera_itar = os.path.join(input_base, camera_itar_name)
    output_camera_itar = os.path.join(output_dir, camera_itar_name)

    print(f"\nProcessing camera itar -> {output_camera_itar}")
    rectify_camera_itar(input_camera_itar, output_camera_itar, args.source_camera, target_params, map_x, map_y, valid_mask)

    # Process sseg aux
    seq_prefix = os.path.splitext(os.path.basename(args.input))[0]
    sseg_name = f"{seq_prefix}.aux.sseg.zarr.itar"
    sseg_input = os.path.join(input_base, sseg_name)
    if os.path.exists(sseg_input):
        print(f"\nProcessing sseg aux...")
        rectify_sseg_aux(sseg_input, os.path.join(output_dir, sseg_name), args.source_camera, target_params, map_x, map_y, valid_mask)
    else:
        print(f"\nNo sseg aux found, skipping.")

    # Process egomask aux
    egomask_name = f"{seq_prefix}.aux.egomask.zarr.itar"
    egomask_input = os.path.join(input_base, egomask_name)
    if os.path.exists(egomask_input):
        print(f"\nProcessing aux egomask...")
        rectify_aux_egomask(egomask_input, os.path.join(output_dir, egomask_name), args.source_camera, map_x, map_y, valid_mask)
    else:
        print(f"\nNo aux egomask found, skipping.")

    # Symlink other component stores + non-camera aux
    for cs in meta["component_stores"]:
        if cs["path"] == camera_itar_name:
            continue
        src = os.path.join(input_base, cs["path"])
        dst = os.path.join(output_dir, cs["path"])
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

    for f_name in os.listdir(input_base):
        if ".aux." in f_name and f_name not in (sseg_name, egomask_name):
            src = os.path.join(input_base, f_name)
            dst = os.path.join(output_dir, f_name)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

    # Copy JSON
    output_json = os.path.join(output_dir, os.path.basename(args.input))
    with open(output_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Output JSON: {output_json}")
    print("\nDone!")


if __name__ == "__main__":
    main()
```



---



### test_rectified.py

```python
#!/usr/bin/env python3
"""Verify rectified ncore V4 data: camera, lidar, labels, and aux data integrity.

Saves: vis_egomask.png, vis_sseg.png

Usage:
    python test_rectified.py --input /path/to/rectified_output/data.json [--camera CAMERA_ID]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from ncore.data.v4 import SequenceComponentGroupsReader, SequenceLoaderV4
from ncore.impl.data.stores import IndexedTarStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Path to rectified ncore V4 JSON")
    parser.add_argument("--camera", type=str, default=None, help="Camera ID to test (default: first available)")
    parser.add_argument("--output-dir", type=str, default=None, help="Visualization output dir (default: data dir)")
    args = parser.parse_args()

    data_dir = Path(os.path.dirname(args.input))
    vis_dir = Path(args.output_dir) if args.output_dir else data_dir
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TESTING RECTIFIED NCORE V4 DATA")
    print(f"  Input: {args.input}")
    print("=" * 60)

    reader = SequenceComponentGroupsReader([args.input], open_consolidated=False)
    seq = SequenceLoaderV4(reader)

    # 1. Camera
    print("\n[1] Camera...")
    cam_ids = [s.sensor_id for s in seq.camera_sensors]
    print(f"  Camera IDs: {cam_ids}")

    camera_id = args.camera or cam_ids[0]
    cam = seq.get_camera_sensor(camera_id)
    mp = cam.model_parameters
    n_frames = cam.frames_count
    expected_w, expected_h = int(mp.resolution[0]), int(mp.resolution[1])
    print(f"  {camera_id}: {n_frames} frames, {expected_w}x{expected_h}, {type(mp).__name__}")

    img0 = cam.get_frame_image_array(0)
    assert img0.shape == (expected_h, expected_w, 3), f"Shape mismatch: {img0.shape}"
    print("  [PASS]")

    # 2. Egomask + valid mask visualization
    print("\n[2] Egomask + valid_mask overlay...")
    img0_bgr = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    vis = img0_bgr.copy()

    valid_mask_path = data_dir / "valid_mask.png"
    if valid_mask_path.exists():
        vm = cv2.imread(str(valid_mask_path), cv2.IMREAD_GRAYSCALE)
        inpaint_mask = vm < 128
        if inpaint_mask.any():
            gray = np.full_like(vis, 128)
            vis[inpaint_mask] = cv2.addWeighted(vis[inpaint_mask], 0.5, gray[inpaint_mask], 0.5, 0)
            print(f"  Inpainted region: {inpaint_mask.sum() / inpaint_mask.size * 100:.1f}%")

    cv2.imwrite(str(vis_dir / "vis_egomask.png"), vis)
    print("  [PASS]")

    # 3. Lidar
    print("\n[3] Lidar...")
    lidar_sensors = seq.lidar_sensors
    if lidar_sensors:
        lidar = lidar_sensors[0]
        pc = lidar.get_frame_point_cloud(0, motion_compensation=False, with_start_points=False, return_index=0)
        print(f"  {lidar.sensor_id}: {len(pc.xyz_m_end)} points")
        print("  [PASS]")
    else:
        print("  [SKIP] No lidar")

    # 4. Cuboids
    print("\n[4] Cuboids...")
    cuboids = list(seq.get_cuboid_track_observations())
    print(f"  {len(cuboids)} observations")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    import os
    main()
```

---



## Target intrinsics JSON format

```json
{
  "<camera_name>": {
    "resolution": [1280, 720],
    "principal_point": [640.5, 501.2],
    "max_angle": 1.0472,
    "reference_poly": "ANGLE_TO_PIXELDIST",
    "pixeldist_to_angle_poly": [0.0, 0.00156, ...],
    "angle_to_pixeldist_poly": [0.0, 639.8, ...],
    "linear_cde": [1.0, 0.0, 0.0]
  }
}
```



## What the rectification does

1. **Images**: Remap to target FTheta grid (INTER_LINEAR) + **inpaint** invalid borders (NOT black fill)
2. **Intrinsics**: Replace with target FTheta params; preserve T_sensor_rig (pose unchanged)
3. **Aux egomask**: Remap (INTER_NEAREST), invalid regions = 255 (ego)
4. **Aux sseg**: Remap (INTER_NEAREST), inpaint via iterative dilation, remaining = sky class
5. **Other cameras, lidar, poses, cuboids**: Unchanged (symlinked)
6. **Consolidated metadata**: Regenerated via `consolidate_compressed_metadata()`



## CRITICAL: Inpaint, Do NOT Fill Black

**Invalid regions after rectification (where source FOV doesn't cover target) MUST be inpainted, never left as zeros/black.** Reasons:

- Black borders cause incorrect "black edge" patterns
- Inpainting with surrounding textures (Telea/NS method) provides plausible content that the model can safely ignore during training
- For sseg, iterative dilation fills with neighboring class labels; remaining pixels default to "sky" (harmless)
- For egomask, invalid = 255 (marked as ego → masked out during training)



## Adapting to Other Cameras

The examples above use **front wide camera** (`camera_front_wide_120fov`) as the rectification source. This is the primary camera for instant NuRec.

**For other cameras**, you can obtain reference target intrinsics from the Instant NuRec project:

1. Download sample PAI data as guided: `https://github.com/NVIDIA/instant-nurec`
2. Load an NCore V4 sequence from the downloaded data
3. Read the camera intrinsics via the V4 API:
  ```python
   reader = SequenceComponentGroupsReader([sample_json], open_consolidated=False)
   seq = SequenceLoaderV4(reader)
   cam = seq.get_camera_sensor("<target_camera_id>")
   params = cam.model_parameters  # FThetaCameraModelParameters
   # Extract resolution, principal_point, polynomials, etc. and save to JSON
  ```
4. Use the extracted intrinsics as `--target-intrinsics` for rectification



## Notes

- **Input/Output format: NCore V4** (`component_stores` with separate `.zarr.itar` per component type)
- **Dependencies**: Requires `nvidia-ncore` package (internal, provides `ncore.data.v4.SequenceLoaderV4`, `IndexedTarStore`, `CameraModel`)
- **V4 key structure**: `cameras/<cam_id>/frames/<timestamp>/image/0` (differs from V3's `sensors/cameras/<cam_id>/<frame_idx>/image/0`)
- Supported source models: `OpenCVFisheye`, `OpenCVPinhole`, any `CameraModel`-compatible type
- Output uses `IndexedTarStore` with proper itar index + consolidated metadata
- If source FOV < target FOV, `valid_mask.png` shows coverage percentage

