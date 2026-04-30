# Use Asset Harvester Output in Reconstructions

Use the steps in this guide to remove, insert, or replace 3D assets created from the Asset Harvester (AH) tool and modify reconstructed scenes (.usdz).

**Before you begin,** you must have already run the Asset Harvester following the steps in [3D Asset Harvesting Inference Pipeline](asset-harvester) and have access to the output from Asset Harvester (`/path/to/AH/output`).

## Add Asset Harvester Output to a Reconstruction

To run asset editing operations on a target reconstruction (USDZ file), the USDZ file must be repackaged with the harvested assets from the [3D Asset Harvesting Inference Pipeline](../nurec/asset-harvester).

Run the following command to repackage the harvested assets as a USDZ file:

```bash
docker run --shm-size=64g -it --rm --gpus all \
--net=host \
--privileged \
--volume /path/to/output/folder:/workdir/output \
nvcr.io/nvidia/nre/nre:latest \
export-external-assets \
--artifact-path /path/to/target.usdz \
--external-assets-dir /path/to/AH/output
--output-edit-file /path/to/output/edit-assets.json \
--output-artifact-path /path/to/output/target-external-assets.usdz
```

Edit the following parameters in the command with the corresponding paths on your system:

- `--artifact-path`: Path to the target USDZ that should be repackaged with the harvested assets.
- `--external-assets-dir`: Path to the Asset Harvester output folder (where the harvested assets exist).
- `--output-edit-file`: Path where the JSON file output should be placed, to be used in the next step.
- `--output-artifact-path`: Path where the repackaged USDZ file should be placed.

## Edit Actors in a USDZ dataset

Use the gRPC API to specify a JSON file that explicitly describes the changes you want to make to the USDZ scene. Use the `--edit-assets` parameter to point to the JSON file path.

The basic structure of the JSON file is as follows. When you run the script to add harvested assets to an existing USDZ file, it outputs this template file to the path specified in the `--output-edit-file` parameter.

```json

  "metadata": {
    "output_artifact_path": "/path/to/output/last.usdz",
    "external_assets_metadata": []
  },
  "replace": [],
  "remove": [],
  "insert": {
    "asset_ids": [],
    "data": {}
  }


```

Each section contains the following information:

### `external_assets_metadata`

This is populated automatically by the external asset import process.

### `remove`

This is a list of `track_ids` you want to remove. These IDs must exist in the `sequence_tracks.json` file. Tracks are filtered out during render request creation.

For example:

```json

 // Removes track ids 8 and 13 from rendering
 "remove": ["8", "13"],

```

### `replace`

This is a list of objects mapping `original_id` (track from artifact) to `replacement_id` (asset in `external_assets`).

Note the following caveats:

- `original_id` must exist in the artifact's `sequence_tracks`, and `replacement_id` must exist in USDZ file's `external_assets`.
- Each replacement action has an `object_size` field: a list of 3 floats \[size_x, size_y, size_z\] representing AABB dimensions. If `object_size` is missing or empty, `render_grpc` will fall back to `cuboid_dims` from the metadata for that `replacement_id`.

For example:

```json

"replace": [
    // replace track_id '8' with asset '13' using specified dimensions
    {
        "original_id": "8",
        "replacement_id": "13",
        "object_size": [4.5, 2.0, 1.8]
    },
    // replace track_id '18' with asset '22', use 22's cuboid_dims
    // replace track_id '6' with asset '7', use 7's cuboid_dims
    {
       "original_id": "18",
       "replacement_id": "22",
       "object_size": []
    },
    {
       "original_id": "6",
       "replacement_id": "7"
    }
]

```

### `insert`

To insert a new asset, populate the `insert` data structure with the asset IDs you want to insert from the external assets, and the Cuboid Track data as shown below. This is the same structure as the `sequence_tracks.json` file stored in a USDZ reconstruction. See [the Load Trajectory Data section in Render the Physical AI Dataset with NuRec](../nurec/physical-ai-data) for a python example working with this data structure.

The `tracks_id` can be any string that doesn't conflict with existing IDs in the USDZ file's `sequence_tracks.json` file.  
The asset_ids must exist in the USDZ's external_assets folder, must be same length as tracks_id, and correspond 1:1 in order.

For example (replace the placeholder variables with your information):

```json

"insert": {
   "asset_ids": ["18"],
   "data": {
     "tracks_data": {
         "tracks_id": ["car_18"],
	       "tracks_poses": ["YOUR_INPUT_HERE"]
         "tracks_timestamps_us": ["YOUR_INPUT_HERE"]
         "tracks_label_class": ["YOUR_INPUT_HERE"],
      },
      "cuboidtracks_data": {
         "cuboids_dims": ["YOUR_INPUT_HERE"]
      }
}

```

Once you have the `edit-assets.json` file, create the rendered outputs with the gRPC API as follows:

1. Start the GRPC server:

```bash
docker run --shm-size=64g -it --rm --gpus all \
        --net=host \
        --privileged \
        --volume /path/to/output/folder:/workdir/output \
        nvcr.io/nvidia/nre/nre:latest \
        serve-grpc \
        --artifact-glob /workdir/output/<RUN-ID>/usd-out/last.usdz --no-enable-nrend --test-scenes-are-valid --enable-editing-actors
```

2. Run the `render-grpc` command:

```bash
   docker run --shm-size=64g -it --rm --gpus all \
        --network host \
        --volume /path/to/output/folder:/workdir/output \
        nvcr.io/nvidia/nre/nre:latest \
        render-grpc \
        --artifact-path /workdir/output/<RUN-ID>/usd-out/last.usdz \
        --edit-assets <edit assets json>

```
