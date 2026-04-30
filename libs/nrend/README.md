# NREnd : Neural Reconstruction Engine Renderer

NRend is a cuda rendering library for NRE models.
It is used both for fast evaluation and rendering from within NRE and as an exported library for integrating NRE into other applications.
It is also used as a fast differentiable renderer for optimizing some specific models (e.g. Gaussian particles cloud representations).

NRE models are serialized into a packed json dictionnary (msgpack), containing both the description of the model computationnal graph and its parameters.
The serialized model files are versionned for managing the backward compatibility.

NRend deserializes the models and composes a corresponding fully-fused, JIT compiled cuda kernel.
It also manages the cuda resources required to render on every requested cuda device.

NRend expose a rendering C++ interface `NRenderer`. The different implementations of this interface are specific to the serialized model version.

## NRE Renderers

Differenet renderers implementation (instances of `NRenderer` is the object exposing the rendering functionnality for the models version identified by the json _model_ entry `nre`.

The `NRERenderer` owns the graph root model `NREModel` and all the cuda resources `CudaKernelResource` (device buffers and run-time compiled kernels) for each queried device.

Note that the `NRERenderer` has a corresponding cuda kernel (`nreRenderer.cuh`) which exposes the code shared by all graphs (e.g. preprocessing of the rays).

## NRE Computationnal Graph

NRE models defined a computationnal tree (NRE does not support generic computationnal DAG).
This tree is described in the NRE configuration dictionnary by nested relationship of _models_ (aka _modules_ in torch taxonomy).
Note that every models in the configuration are identified by a unique name.
NRend use the name of the model to identify the class to be instanciated for the corresponding node.

In NRend the models are implementations of the `NREModel` interface.
The model implementations expose a run-time generated cuda code and the associated parameter host buffers.
They are created automatically from the json dictionnary thanks to a static register of callback instantiator `NREModel::registerInstantiator`, based on the name of the model.

The models are further categorized as _root_ or _node template instances_.

### Root models

A root model is the root of the computationnal tree.
It is called directly by the renderer cuda code through the `eval` cuda function.
An example of a root model is the `NRENeRFModel`.

### Node templates

Node templates are models conforming to an **implicit** interface.
Every instances of the template must expose this interface in their cuda code.
An example of a node template model is the _Appearance Embedding_ template.
The _Appearance Embedding_ template is implemented by both the `NRESkipAppearanceEmbedding` and the `NREGloAppearanceEmbedding`.
Implementations must expose an `eval` and a `fetch` cuda methods (see `nreAppearanceEmbedding.cuh`).
This interface is defined by the different callers : `NRENeRFModel` is calling `eval` while the `NRESkyMLPBackground` is calling `fetch`.

## Versions

The NRE models are identified by the `nre` model version string.
The version number is the NRE version.

## version @ 25.12.148 [2026-02-04]

- **Optimization** :

  - gutRenderer : Revert of cache configuration for both `renderBackward` kernels and only keep `__launch_bounds__` for lidar.

## version @ 25.12.147 [2026-01-25]

- **Optimization** :

  - gutRenderer : specify "max threads per block" and "min number of block per SM" hints via `__launch_bounds__` for `renderBackward`
  - gutRenderer : set `renderBackward` Cuda functions cache config to CU_FUNC_CACHE_PREFER_SHARED

## version @ 25.12.146 [2025-12-20]

- **Optimization** :

  - gutRenderer : split the `renderBackward` kernel into 2 distinct versions for lidar and camera.

## version @ 25.12.144 [2025-12-04]

- **API modifications** :

  - Adding more rendererHint [rendererFastQuality and rendererQualityFast]
  - DeviceMemoryAllocator : remove inline default callback

- Fixes :

  - GrtOptixRenderer : Avoid rebuilding the AS for static scenes
  - Add support for null ray direction
  - Remove sRGB to linear conversion when postprocessing is disabled

## version @ 25.10.23 [2025-9-09]

- **API modifications** :

  - Rendering parameters flags
    - Adding flags to disable features, sensor features, normals and ray gradient
    - Change the flags to enable extended features as a disabling flag
  - Rendering feature layout :
    - Change signature to include the sensor type

- **Extended Features Support** :
  - Add extended features support for camera and lidar sensors
  - Refactor `NREShGaussianParticles` to support extended features
  - Add sensor-specific extended features parameters
  - Make rendered radiance, normals and ray gradients optional

### version @ 25.9.38 [2025-8-19]

- Fixes :
  - Sensor projection : fix principal point wrt viewport resolution with different aspect ratio
  - Fix safe_normalize backward compilation error

### version @ 25.9.14 [2025-8-14]

- Fixes :
  - Replace slang's built-in normalize with safe_normalize

### version @ 25.8.118 [2025-8-11]

- **PPISP post processing** :
  - Adds PPISP post processing

### version @ 25.8.81 [2025-8-01]

- **API modifications** :
  - Output rendered normal for _3dgrut-nrend_, _3dgrt-optix-nrend_ and _3dgrt-rejection-optix-nrend_

### version @ 25.8.76 [2025-07-31]

- Fixes :

  - `NRESHGaussianModel` :
    - Fix min response for gaussian particles canonical scale
  - `NREGaussianCompositeModel` :
    - Do not clamp extended features (add a `saturate_radiance` option)
  - Compositing : do not write invalid instance id in the instance id buffer

- **API modifications** :
  - Add new render parameters flags :
    - `DisableBackground`
    - `DisablePostProcessing`

### version @ 25.8.35 [2025-07-23]

- Fixes :
  - `GUTRenderer` :
    - Features accumulation vector size (NRE-1556)

### version @ 0.2.715 [2025-06-23]

- **API modifications** :

  - Add RenderingParameters render hints for renderer implementation selection

- Added renderers :
  - _3dgrut-nrend_ : `GRUTRenderer` : an abstract class for `GUTRenderer` and `GRTOptixRenderer`
  - _3dgs-nrend_ : `GSRenderer` : default parametrization of 3DGUT for rendering 3DGS
  - _3dgrt-optix-nrend_ : `GRTOptixRenderer` : 3DGRT implementation using OptiX
  - _3dgrt-rejection-optix-nrend_ : `GRTOptixRejectionRenderer` : 3DGRT rejection sampling parametrization using OptiX

### version @ 0.2.668 [2025-06-07]

- **API modifications** :
  - Added : expose model version in the API for better handling of backward compatibility

### version @ 0.2.648 [2025-05-30]

- Fixes :
  - `GUTRenderer` :
    - Prevent atomic sum when no thread in the warp hit the particle for the `GUTKBufferRenderer` when K=0
    - Re-enable warp atomic sum optimization for the `GUTKBufferRenderer` when K=0 when sensor rays are not coherent

### version @ 0.2.637 [2025-05-27]

- Fixes :
  - `GUTRenderer` :
    - Add option to disable warp atomic sum optimization for the `GUTKBufferRenderer` when K=0
    - Disable warp atomic sum optimization for the `GUTKBufferRenderer` when K=0 when sensor rays are not coherent

### version @ 0.2.601 [2025-05-16]

- **API modifications** :
  - **Updated `RowOffsetStructuredSpinningLidarProjectionParameters` Data Structure:**
    - Renamed `fovMin` and `fovMax` to `fovStart` and `fovSpan`, respectively.
    - This change broadens support for various LiDAR models with non-standard azimuth starting angles.
    - The contents of `tilesToElementsMap` and `tilesPackInfo` have been updated; tiling information is now computed using **relative** angles (with respect to the starting angle) instead of absolute angles.
    - Similarly, the data format for `angleToColumnMap` has been modified to utilize **relative** angles (with respect to the starting angle).

### version @ 0.2.596 [2025-05-14]

- Fixes :
  - `GUTRenderer` :
    - Add specific path for the render backward kernel for the `GUTKBufferRenderer` when K=0

### version @ 0.2.583 [2025-05-11]

- Fixes
  - _gaussians-composite_ : change the particle deformations to be applied post-activation in `NREDynamicShGaussianModel`

### version @ 0.2.582 [2025-05-10]

- Modified _3dgut_ renderer:
  - Invalid Particle Identification: Enhanced the identification of invalid particles by checking if the tile count is zero, which is equivalent to verifying that minTile equals maxTile.
  - Tile Extents Storage: Updated the storage of tile extents to use integers, improving numerical stability throughout the rendering process.
  - Lidar Point Projection: For lidar data, points are now projected onto a virtual pixel space instead of sensor angle space. In this context, a "pixel" refers to the smallest distinguishable resolution unit for lidar sensors.

### version @ 0.2.558 [2025-05-03]

- **API modifications** :

  - added frameTileResolution and frameTileOffset to the renderParameters

- Fixes
  - MGPU JIT layout conversion for:
    - _background_ : `NRESkyMLPBackground`
    - _feature-volume_ : `NREHashGridFeatureVolume`
    - _texture_ : `NREFullyFusedTexture`
  - _gaussians-composite_ : Handling of missing track poses in `NREGaussianCompositeModel`
  - Cuda 11.8 compilations and synchronization issues (cudaMemcpy)
  - _3dgut_ renderer :
    - error handling when updateBindedMemory fails
    - cache the kernel source code table to prevent regeneration in MGPU
    - add missing files in CMakelists
  - _background_ : clamp after interpolation rather than before in `NREEnvMapBackground`

### version @ 0.2.555 [2025-05-02]

- Modified root models:
  - _gaussians-composite_ : `NREGaussianCompositeModel`
    - Fixed support for extended features

### version @ 0.2.548 [2025-04-30]

- **API modifications** :

  - Ray direction length encode the ray spread

- Modified renderer :
  - _3dgut-nrend_ : add anti-aliasing as described in https://arxiv.org/abs/2504.12811

### version @ 0.2.521 [2025-04-16]

- Added root models :
  - _gaussians-composite_ : `NREGaussianCompositeModel`
- Added node template models:
  - Gaussians primitives :
    - _rigid-gaussians_ : `NRERigidSHGaussianModel`
    - _deformable-gaussians_ : `NREDeformableSHGaussianModel`
- Added Cuda RTC kernel with multiple entry points

### version @ 0.2.522 [2025-04-17]

- Modified sensor models:
  - `RowOffsetStructuredSpinningLidarModelParameters` :
    - enable custom resolution (NRows x NColumns)

### version @ 0.2.474 [2025-04-08]

- Added node template models:
  - Post-processings : a list of post-processing operators to be applied to the rendering result in order
    - _post-processing_ : `NREPostProcessings`
  - PPISP [NOOP]:
    - _ppisp-post-processing_ : `NREPPISPPostProcessing`

### version @ 0.2.473 [2025-04-07]

- Fixes :
  - `GUTRenderer` :
    - fix wrong matrix size in sensor projection

### version @ 0.2.460 [2025-04-04]

- Fixes :
  - `GUTRenderer` :
    - Fix missing sigma points clipping for pinhole models

### version @ 0.2.440 [2025-03-30]

- **API modifications** :

  - Remove near/far distance to all camera models (for culling)
  - Add nominal resolution to sensor projection models for RTX renderer integration
  - Add new FThetaCameraModelParameters::PolynomialType::PIXELDIST_TO_ANGLE_RF for RTX integration renderer

- Fixes :
  - `GUTRenderer` :
    - Fix world-to-object support for RTX integration renderer

### version @ 0.2.439 [2025-03-29]

- Fixes :
  - `GUTRenderer` :
    - fix opencv fisheye distortion polynomial evaluation

### version @ 0.2.436 [2025-03-26]

- **API modifications** :

  - Add near/far distance to all camera models (for culling)

- Fixes :
  - `GUTRenderer` :
    - fix sensor culling
    - fix Jacobian
    - fix cuda kernel compilation on cuda 11.8

### version @ 0.2.399 [2025-03-16]

- Fixes :
  - `GUTRenderer` :
    - fix hit distance for the splat rendering
    - fix hit buffer size 0 for the k-buffer rendering
  - `CudaRtcKernel` :
    - fix PTX caching and add obfuscation

### version @ 0.2.387 [2025-03-12]

- Fixes :
  - `GUTRenderer` :
    - fix Jacobian and UT sigma points computation

### version @ 0.2.379 [2025-03-11]

- **API modifications** :

  - Add extended features rendering

- Modified model :
  - _sh-gaussians_ :
    - add extended features computation from extra-signals parameters

### version @ 0.2.353 [2025-03-04]

- Added sensor models:
  - `RowOffsetStructuredSpinningLidarModelParameters` :
    - row-offset structured spinning lidar model parameters

### version @ 0.2.336 [2025-02-27]

- Fixes :
  - `NREDefaultRenderer` :
    - fix sky-mlp background evaluation

### version @ 0.2.334 [2025-02-27]

- Fixes :
  - `GUTRenderer` :
    - splatting mode : compute pixel uv by projecting the ray
    - cleanup slang rtc engine
  - `NREGeneralComposite` :
    - fix sky-mlp background evaluation
  - `NRENeRF` :
    - fix sky-mlp background evaluation
  - `NREGaussiansPrimitive` :
    - fix sky-mlp background evaluation

### version @ 0.2.323 [2025-02-26]

- Modified renderer :
  - _3dgut-nrend_ : adding ray origin and direction gradients to the backward pass

### version @ 0.2.316 [2025-02-24]

- Fixes :
  - `GUTRenderer` :
    - project sensor origin to object space
    - k-buffer : cull rays based on the current composited depth
    - fix inverse depth support through near/far

### version @ 0.2.315 [2025-02-22]

- Add fallback to use the model renderer configuration at creation time
- Fix INRenderer symbol visibility
- Modified renderer :
  - _3dgut-nrend_ :
    - add object-to-world transform to the sensor projections
    - update Slang version to `slang-2025.4`
    - fix Slang filesystem (maintain a copy of unencoded files)
    - prevent compiling backward kernel when not initialized in differentiable mode

### version @ 0.2.297 [2025-02-16]

- Modified renderer :
  - _3dgut-nrend_ : adding profiling callback

### version @ 0.2.291 [2025-02-12]

- Fixes :
  - `GUTRenderer` : adjust culling to better feat the original splat algorithm

### version @ 0.2.278 [2025-02-11]

- Fixes :
  - `NREShGaussianModel` : add missing evaluation function which sets the hit status

### version @ 0.2.263 [2025-02-10]

- **API modifications** :
  - Add INRenderer interface
  - Add ITypes interface, ghosting tcnn types

### version @ 0.2.248 [2025-02-05]

- **API modifications** :
  - Remove all std dependencies from the API

### version @ 0.2.246 [2025-02-04]

- Modified renderer :
  - _3dgut-nrend_ :
    - Splat render mode compatible with [3DGS codebase](https://github.com/graphdeco-inria/gaussian-splatting)
    - Cuda 11.8 compatibility

### version @ 0.2.241 [2025-01-31]

- Modified renderer :
  - _3dgut-nrend_ :
    - UT projection for orthographic, perspective, ocv-pinhole, ocv-fisheye, ftheta models
    - Culling based on max projected Gaussian response on the tile (aka Tile Based Culling)
- Modified root model :
  - _gaussians-primitive_ :
    - Added background evaluation
    - Added radiance clamping

### version @ 0.2.234 [2025-01-28]

- **API modifications** :
  - Add renderer configuration json in the factory to permits specifying different renderer for the same model
  - Change `render` sensor profile information from per ray to global
  - Add to `render` global sensor parameters (camera models following the ncore/vren representation)
  - Add to `render` global start/end sensor poses
  - Add `renderForward` call with an opaque forward context argument
  - Add `renderBackward` call (supported only by specific renderers)
  - Add `updateModelParameters` call
- Added root models :

  - _sh-gaussians_ : `NRESHGaussianModel`
  - _gaussians-primitive_ : `NREGaussiansPrimitiveModel` (support limited to a single _sh-gaussians_ node)

- Added renderer :
  - _3dgut-nrend_ : `GUTRenderer` : a splatting based differentiable renderer for the _gaussians-primitive_ and _sh-gaussians_ models

### version @ 0.2.86 [2024-10-28]

- **API modifications** :

  - Add a per ray sensor profile `int` buffer containing the _unique sensor idx_ and the _unique frame idx_ used for the `per-element-latent` appearance embeddings.

- Modified node template models :
  - Appearance Embedding :
    - _glo-embedding_ : adding support for `per-element-latent` (`per-camera-latent` and `per-frame-camera-latent`)

### version @ 0.2.65 [2024-10-16]

- Fixes :
  - `NREGeneralComposite` : do not reset the current hit data because it is needed on skipTest
  - `NREDenseObjectAccStructure` : align sampling to the NRE implementation (add half delta and force last sample)

### version @ 0.2.40 [2024-09-26]

- Fixes :
  - `NREGloAppearanceEmbedding` : wrong logic when creating the composite embedding
  - `NREFullyFusedTexture` : wrong dimension with appearance embedding
  - `NRESkyMLPBackground` : issue with input dimensions (padding) due to the use of a NetworkWithInputEncoding with a dummy encoding

### version @ 0.2.34 [2024-09-24]

Refactor embeddings :

- Removed node template models :
  - Embedding :
    - _embedding_ : `NREEmbedding`
    - _individual_remap-embedding_ : `NREIndividualRemapEmbedding`
- Added node template models :
  - Embedding :
    - _weighted-instance-input-embedding_ : `NREWeightedInstanceInputEmbedding`
    - _individual-remap-time-input-embedding_ : `NREIndividualRemapTimeInputEmbedding`
- Modified node template models :
  - Appearance Embedding :
    - _glo-embedding_ : adding support for `first-latent` and `mean-latent` evaluation mode

### version @ 0.2.18 [2024-09-18]

Integration of DNSG :

- Added root models:
  - _general-composite_ : `NREGeneralComposite`
- Added node template models:
  - Acceleration structure :
    - _dense-object-acc-structure_ : `NREDenseObjectAccStructure`
  - Feature Volume :
    - _hash-grid-object_ : `NREHashGridObjectFeatureVolume`
  - Background :
    - _skip-background_ : `NRESkipBackground`
  - Embedding :
    - _embedding_ : `NREEmbedding`
    - _individual_remap-embedding_ : `NREIndividualRemapEmbedding`

### version @ 0.1.297 [2024-08-15]

Initial version implementing (partially) :

- Root models
  - _nerf_ : `NRENeRFModel`
- Node template models
  - Acceleration structure :
    - _nerfacc-acc-structure_ : `NRENeRFAccelerationStructure`
  - Geometry :
    - _skip-density_ : `NRESkipGeometry`
  - Feature Volume :
    - _hash-grid_ : `NREHashGridFeatureVolume`
  - Texture :
    - _fully-fused-texture_ : `NREFullyFusedTexture`
  - Appearance Embedding :
    - _skip-appearance_ : `NRESkipAppearanceEmbedding`
    - _glo-embedding_ : `NREGloAppearanceEmbedding`
  - Background :
    - _background-color_ : `NREColorBackground`
    - _sky-mlp_ : `NRESkyMLPBackground`

## How-To

### Implementing new root node model

Create a new class inheriting from `NREModel` and its cuda code snippet.
(The class implementation should ideally be located in a `.h` file located in `include/nrend/models` and its kernel source in either a `.cuh` file located in `include/nrend/kernels/cuda/models` or in a `.slang` file located in `include/nrend/kernels/slang/models`.)

This class must implement the `eval` function with the signature used in the renderer code (see `nreRenderer.cuh`).

Set the name of the class to the name of the correspoding NRE model in the configuration dictionnary.

Add model registration by making sure `REGISTER_NREMODEL_IMPLEMENTATION` is called in `src/nreModel.cu` for the newly created class.

As an example refers to `NRENeRFModel`.

### Implementing new node template model

Create a new class inheriting from `NREModel` and its cuda code snippet.
(The class implementation should ideally be located in a `.h` file located in `include/nrend/models` and its kernel source in either a `.cuh` file located in `include/nrend/kernels/cuda/models` or in a `.slang` file located in `include/nrend/kernels/slang/models`.)

The class must implement the **implicit** node template interface.
This interface is defined by how parent models are calling it in their cuda code.

Set the name of the class to the name of the correspoding NRE model in the configuration dictionnary.

Add model registration by making sure `REGISTER_NREMODEL_IMPLEMENTATION` is called in `src/nreModel.cu` for the newly created class.

As an example refers to `NRESkyMLPBackground` which implements the _Background_ node template.

### Adding a new test

Testing NRend against regression is currently done by comparing a rendering with a reference golden image.

You need to first create the configuration file defining the test in `configs/tests` (e.g. `configs/tests/nrend_test_new_supported_model.yaml`).

Then train a model using this configuration :

```
bazel run //:run --  --config-name=configs/tests/nrend_test_new_supported_model.yaml out_dir=/tmp/nre/outdir mode=train dataset.path=/dataset_path/dataset.json logger=dummy system.test.nrend.enabled=true system.test.nrend.log_level=4
```

This will create a new training experiment directory in `/tmp/nre/outdir` containing the model.

Once the model is trained, run a validation with `system.test.nrend.create_test_case=true` :

```
bazel run //:run -- --config-name=/tmp/nre/outdir/<NEW-TRAINING_EXPERIMENT-DIR>/config/parsed.yaml  resume=last mode=val dataset.n_val_image_subsample=8  dataset.val_camera_frame_step=111 system.test.nrend.create_test_case=true
```

This will create a new validation experiment directory in `/tmp/nre/outdir` containing the model.

Get the URL of the `nrend_test_assets` entry in `MODULE.bazel` and upload the archive into your tmp directory :

```
mkdir /tmp/nrend_test_assets;
cd /tmp/nrend_test_assets;
wget https://gitlab-master.nvidia.com/api/v4/projects/85874/packages/generic/nrend_test_assets/<CURRENT-NREND-VERSION>/nrend_test_assets.tar.gz;
tar -zxvf nrend_test_assets.tar.gz;
rm -f nrend_test_assets.tar.gz;
```

Copy one of the test case in the directory :

```
cp /tmp/nre/outdir/<NEW-VALIDATION_EXPERIMENT-DIR>/val/nrend_test_cases/<CAM-ID>/000001.<NEW-NREND-VERSION>.msgpack
```

Edit the `BUILD.bazel` to add the new test case filename in the `filegroup:srcs`, create the updated archive and reupload the archive :

```
tar zcvf nrend_test_assets.tar.gz *;
curl --header "PRIVATE-TOKEN: glpat-<YOUR-GITLAB-TOKEN>" \
                               --upload-file nrend_test_assets.tar.gz \
                               "https://gitlab-master.nvidia.com/api/v4/projects/85874/packages/generic/nrend_test_assets/<NEW-NREND-VERSION>/nrend_test_assets.tar.gz?select=package_file"
```

In the file `MODULE.bazel`, update the `sha256` and `urls` of the `nrend_test_assets` entry.

Finally validate the test is passing :

```
bazel test //libs/nrend:nrend_tests
```

## Legacy NRE / NGP Renderer

### Versions

The legacy models are identified by no or `ngp` model version string. The version number, when present goes from `0.0.0` to `3.0.0`.
