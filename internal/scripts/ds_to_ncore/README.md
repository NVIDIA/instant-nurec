# DS2NCore

This document describes how to generate an ncore dataset from a drivesim scenario. It is a two step process which can be run locally or on maglev: Run the drivesim scenario, and convert the output to the ncore format.

### Requirements

- A drivesim scenario (usd/usda) file with a rig and cameras that move along a trajectory.
- Access to maglev, swiftstack, drivesim, and nre.

### Creating drivesim data

Run drivesim's headless internal with the `NCoreDSWriter` to capture the raw data required for ncore. To capture also auxiliary ground truth data (depth, semantics, ect) pass `aux_data=True`. The following command will save the data to `/tmp/ncore/<EXPERIMENT_NAME>`

```
./generate_SDG_headless_internal.sh <SCENARIO_PATH> <NUM_FRAMES> --writer="NCoreDSWriter:aux_data=True;run_id=<EXPERIMENT_NAME>" --merge-config=assets/scenarios/nre/recording.toml
```

### Copy lidar json data

The lidar json file contains important details about the angles of each ray, which are needed for the lidar model.
We can get it from [Hesai_P128_V4P5_HR10.json](https://gitlab-master.nvidia.com/omniverse/sensors/sensors-abandoned/-/blob/master/source/extensions/omni.sensors.nv.common/data/lidar/Hesai_P128_V4P5_HR10.json).

```
cp Hesai_P128_V4P5_HR10.json /tmp/ncore/<EXPERIMENT_NAME>/<LIDAR_ID>/.
```

### Converting the data to NCore

Run `ds_to_ncore.py` to convert the data to NCore format and generate a `.zarr.itar` file. This can only be run from bazel because of the dependency on ncore's nvidia_utils.

```
bazel run //internal/scripts/ds_to_ncore:ds_to_ncore -- --input-dir <EXPERIMENT_PATH> --run-id <EXPERIMENT_NAME> --output-dir <OUTPUT_DIR> --camera-ids <CAMERA_IDS> --lidar-ids <LIDAR_IDS>
```

### Generate Metadata JSON

You will also need to use the [ncore repository](https://gitlab-master.nvidia.com/nrs/ncore) to generate a JSON metadata file from the `.zarr.itar` data, which is then used to train NRE.

The command to generate the JSON metadata is:

```
bazel run //internal/scripts:ncore_sequence_meta -- --shard-file-pattern=<OUTPUT_DIR>/<RUN_ID>.zarr.itar --output-dir=<METADATA_OUTPUT_DIR>
```

# Maglev Batch Workflow

The above steps can be applied to a batch of scenarios and all run on maglev.

### Brief Summary of Steps

1. Create one or more `.sh` files describing the scenario(s), number of frames, and sensors, and place them in a new folder in swiftstack.
2. Modify the workflow `ds2ncore.yaml` file to point to your swiftstack file and output folder.
3. Run the maglev workflow

### Setup

Setup your `omni-asset-auth` maglev secret (search on slack for instructions). This is required to run the drivesim image.

Replace both instances of `ds2ncore-tasks/task_1` in the workflow file with the path to your own swift-stack folder containing the `.sh` files. Also replace the swift url with your own:

`swift://pbss.s8k.io/team-ct-omni-syntheticdata/<username>`

If you haven't already, setup your swiftstack maglev secrets. This can be named whatever you want, but must match the value in the maglev workflow: `storageSecret: my-custom-swift-creds`

```
maglev storage-secrets set my-custom-swift-creds \
   --access-key-id <account> \
   --secret-access-key <s3apikey>
```

### Detailed Steps

The maglev batch workflow takes as input a directory in swiftstack containing `.sh` files. These `.sh` files describe the scenarios and sensors for each Drivesim run. A separate maglev process will run for each `.sh` file in the directory, and the results will be named the same as the input file (without the extension).

In our example we will use a single `.sh` file, but the process scales for multiple files in a single maglev workflow. The following `task.sh` file will be placed in an empty folder in swiftstack called `/ds2ncore-tasks/task_1`

### Example `task_1.sh` file

```
export SCENARIO_PATH=omniverse://drivesim2-dev.ov.nvidia.com/Projects/ds2_scenarios/ncore/Rivermark/rivermark_loop_3cam.usda
export NUM_FRAMES=500
export CAMERA_IDS="camera_front_wide_120fov,camera_cross_left_120fov,camera_cross_right_120fov"
export LIDAR_IDS="lidar_gt_top_p128_v4p5"
export AUX_DATA=False
```

### Maglev workflow inputs and outputs

The outputs are named after the task file provided and are saved to swiftstack. They can also be downloaded from maglev directly.

```
swift::.../ds2ncore-tasks/task_folder/my_task.sh ->

    gen-headless-ds-for-ncore
        input: my_task.sh
        output: my_task.tar.zip (drivesim writer output)
    ncore-data-conversion:
        input: my_task.sh, output of gen-headless-ds-for-ncore
        output: my_task.zarr.itar (also saved to swift stack)
    nerf-aux-data:
        input: my_task.sh, my_task.zarr.itar
        output:
            my_task.aux.iseg.zarr.itar
            my_task.aux.lidar-camvis.zarr.itar
            my_task.aux.lidar-sseg.zarr.itar
            my_task.aux.oflow.zarr.itar
            my_task.aux.sflow.zarr.itar
            my_task.aux.sseg.zarr.itar
    ds2ncore-meta:
        input: my_task.sh, my_task.zarr.itar
        output: my_task.json
```
