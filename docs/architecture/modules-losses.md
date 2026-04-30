# Losses Module

## Overview

The Losses module provides a hierarchical architecture for computing loss functions during neural rendering training. The module follows a layered design pattern that separates concerns across kernel execution, autograd integration, neural network module wrapping, and high-level aggregation. The design emphasizes performance through a fused Slang-based kernel implementation that computes multiple loss types simultaneously, while maintaining flexibility through configuration-driven loss selection.

## Architecture

The Losses module follows a four-layer architecture pattern:

1. **Layer 0: Slang Losses Kernel** - Fused CUDA kernel implemented in Slang that computes all loss types
2. **Layer 1: SlangLossesFunction** - PyTorch autograd Function that bridges the kernel with PyTorch's autograd system
3. **Layer 2: Loss Modules** - PyTorch nn.Module(s) that manage loss lifecycle and data preparation. This layer represents a set of modules (one per loss type) that can be implemented as either:
   - **Slang Implementation:** A single fused class (`SlangLosses`) that handles multiple losses together for performance
   - **PyTorch Implementation:** Multiple separate classes (e.g., `RGBLoss`, `LidarLoss`, `BackgroundLoss`) that handle losses individually
4. **Layer 3: LossAggregator** - Configuration-driven orchestrator that aggregates and coordinates all losses

Each layer only accesses the immediate layer below it: Layer 3 → Layer 2 → Layer 1 → Layer 0. This is not a hard requirement of the design, just a nice-to-have for the losses module. The layered design ensures clear separation of concerns and maintainability, and the only requirement is that any given layer only depends on itself or layers below it, i.e. layer N depends on layers M where M ≤ N.

---

### Layer 0: Slang Losses Kernel (`slang_losses_kernel`)

**Current location:** `libs/slang_gaussians/losses/losses.slang`

**Description:** The foundational CUDA kernel implemented in Slang that performs fused computation of multiple loss types. This kernel is differentiable and handles both forward and backward passes automatically through Slang's automatic differentiation capabilities.

**Key Characteristics:**

- **Fused Execution:** All loss types are computed in a single kernel invocation, improving memory bandwidth utilization and reducing kernel launch overhead
- **Differentiable:** Uses Slang's `[Differentiable]` attribute to enable automatic gradient computation
- **Conditional Execution:** Loss types are conditionally computed based on factor values (negative factors indicate disabled losses)
- **Multi-Modal Support:** Handles RGB camera losses, LiDAR losses, background losses, and bilateral grid regularization losses

**Kernel API Design:**

The kernel signature is designed to support multiple loss types while avoiding duplication of shared data. The API follows this pattern:

```slang
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void slang_losses_kernel(
    // Dimension parameters (no_diff)
    no_diff uint B_loss1, H_loss1, W_loss1, ...,
    no_diff uint B_lossN, H_lossN, W_lossN,

    // Loss scaling factors (no_diff, negative values indicate disabled)
    // Each factor represents 1/N_valid for mean reduction, or 0 if disabled
    no_diff float loss1_factor,
    ...,
    no_diff float lossN_factor,

    // Shared input tensors (no_diff)
    // Flags and ground-truth tensors that may be reused across multiple losses
    no_diff TensorView<int32_t> shared_flags_tensor,      // Reused by multiple losses
    no_diff TensorView<float> shared_gt_tensor,          // Reused by multiple losses

    // Loss-specific input tensors (no_diff)
    no_diff TensorView<int32_t> loss1_flags,             // [B_loss1, H_loss1, W_loss1, 1]
    no_diff TensorView<float> loss1_gt,                  // [B_loss1, H_loss1, W_loss1, C]
    ...,
    no_diff TensorView<int32_t> lossN_flags,
    no_diff TensorView<float> lossN_gt,

    // Differentiable input tensors (DiffTensorView)
    // Predictions and model parameters that participate in gradient flow
    DiffTensorView<float> loss1_pred,                    // [B_loss1, H_loss1, W_loss1, C]
    ...,
    DiffTensorView<float> lossN_pred,
    DiffTensorView<float> model_params,                  // Model parameters (e.g., bilateral grids)

    // Output tensors (DiffTensorView for gradient flow)
    DiffTensorView<float> loss1_output,                  // [B_loss1, H_loss1, W_loss1] or [N_rays]
    ...,
    DiffTensorView<float> lossN_output
)
```

**Data Shape Specification:**

Each loss type requires:

- **Dimension parameters:** Batch size (B), height (H), width (W), and other dimensions (D, channels, etc.) as needed
- **Scaling factors:** Normalization factors (1/N_valid) for mean reduction, or negative values to indicate disabled losses
- **Input tensor shapes:** Explicit shape specifications for flags, predictions, and ground-truth tensors
- **Output tensor shapes:** Shape of per-element loss values (typically matching spatial dimensions or flattened ray dimensions)

**Data Reuse Pattern:**

Some losses reuse data from other losses to avoid duplication:

- **Example:** Background losses reuse RGB and LiDAR flags tensors rather than requiring separate flag tensors
- **Design Principle:** When adding new losses, identify shared data (flags, ground-truth) and reuse existing tensors rather than duplicating them in the kernel signature
- **Memory Efficiency:** Data reuse reduces memory footprint and kernel parameter count, improving launch overhead

**Tensor Categories:**

1. **Non-differentiable inputs (no_diff):**

   - **Flags tensors:** Ray flags (`TensorView<int32_t>`) containing masking information (e.g., `RGB_LABEL`, `INVALID`, `SKY_SEMANTIC`)
   - **Ground-truth tensors:** Target values (`TensorView<float>`) that do not participate in gradient computation
   - **Dimension parameters:** Scalar values specifying tensor shapes

2. **Differentiable inputs (DiffTensorView):**

   - **Prediction tensors:** Model outputs that require gradients (e.g., RGB predictions, LiDAR distance predictions, opacity predictions)
   - **Model parameters:** Learnable parameters that require gradients (e.g., bilateral grid tensors)

3. **Output tensors (DiffTensorView):**
   - **Loss tensors:** Per-element loss values that must be differentiable for gradient flow through the loss computation

**Example Losses:**

Examples of losses that can be integrated into the fused kernel:

- **RGB L1 Loss:** Computes L1 distance between predicted and ground-truth RGB values, masked by flags
- **LiDAR L1 Loss:** Computes L1 distance for LiDAR distance predictions, masked by flags
- **Background MSE Loss:** Computes mean squared error for opacity predictions against foreground mask (derived from flags)
- **Bilateral Grid Drift Loss:** Regularization loss penalizing deviation from identity transform in bilateral grids
- **Distance Loss:** Computes distance prediction error with range constraints
- **Semantic Loss:** Cross-entropy loss for semantic segmentation predictions

**Compile-Time Constants:**

Losses may require compile-time constants that are defined in Python and injected into the Slang kernel via `-D` flags during Bazel compilation. Examples include:

- **Ray Flags:** Bit values for ray flags (e.g., `RGB_LABEL`, `DROPPED`, `INVALID`, `SKY_SEMANTIC`) defined in `RayFlags` enum
- **Block Dimensions:** CUDA block size (e.g., `BLOCK_THREADS=256`)
- **Grid Dimensions:** Bilateral grid affine matrix dimensions (e.g., `GRID_NUM_ROWS`, `GRID_NUM_COLS`)
- **Loss-Specific Constants:** Thresholds, ranges, or other constants specific to loss computations

When extending the kernel with new losses, any compile-time constants must be:

1. Imported in `slang_losses_constants.py` if they already exist in Python code, or defined in `slang_losses_constants.py` if they don't exist
2. Extracted and injected via `-D` flags in the Bazel build configuration
3. Used consistently in both Python data preparation and Slang kernel computation

**Tensor Requirements:**

Slang requires all input and output tensors to be contiguous in memory and non-null. This requirement is enforced by Layer 1 before kernel launch.

---

### Layer 1: SlangLossesFunction (Autograd Function)

**Current location:** `nre/losses/slang_losses.py`

**Description:** A stateless PyTorch `torch.autograd.Function` that bridges the Slang kernel with PyTorch's autograd system. This layer handles tensor preparation, kernel invocation, and gradient computation coordination.

**Key Responsibilities:**

- **Tensor Validation:** Validates input tensor shapes, dtypes, and dimensional relationships
- **Memory Layout:** Enforces tensor contiguity as required by Slang (all tensors must be contiguous and non-null)
- **Kernel Launch:** Configures CUDA grid and block dimensions based on tensor sizes
- **Gradient Handling:** Saves tensors for backward pass and computes gradients via backward kernel
- **Forward/Backward Consistency:** Ensures forward and backward static methods are consistent: input of forward matches output of backward, output of forward matches input of backward, objects saved in forward are loaded in backward

**Forward API Design:**

```python
@staticmethod
def forward(
    ctx,
    # Shared tensors (no_diff)
    shared_flags: torch.Tensor | None,
    shared_gt: torch.Tensor | None,

    # Loss-specific inputs
    loss1_flags: torch.Tensor,
    loss1_pred: torch.Tensor,
    loss1_gt: torch.Tensor,
    loss1_factor: float,
    ...,
    lossN_flags: torch.Tensor,
    lossN_pred: torch.Tensor,
    lossN_gt: torch.Tensor,
    lossN_factor: float,

    # Model parameters (differentiable)
    model_params: torch.Tensor,

    # Dimension parameters
    B_loss1: int, H_loss1: int, W_loss1: int,
    ...,
    B_lossN: int, H_lossN: int, W_lossN: int,
) -> tuple[torch.Tensor, ...]:
    """
    Forward pass API design:

    Inputs:
    - ctx: Autograd context for saving tensors
    - shared_*: Tensors reused across multiple losses (can be None)
    - loss*_flags: Flag tensors for each loss (no_diff, int32)
    - loss*_pred: Prediction tensors for each loss (differentiable, float32)
    - loss*_gt: Ground-truth tensors for each loss (no_diff, float32)
    - loss*_factor: Scaling factors (1/N_valid for mean reduction, or negative if disabled)
    - model_params: Model parameters requiring gradients
    - B_*, H_*, W_*: Dimension parameters for each loss

    Returns:
    - tuple[torch.Tensor, ...]: Loss output tensors, one per loss type
    - Order must match backward grad_input order

    Responsibilities:
    1. Validate tensor shapes and dtypes
    2. Ensure tensor contiguity (required by Slang)
    3. Allocate output tensors with correct shapes
    4. Compute CUDA grid/block dimensions
    5. Launch Slang kernel
    6. Save tensors needed for backward pass in ctx
    7. Return loss tensors in consistent order
    """
```

**Backward API Design:**

```python
@staticmethod
def backward(
    ctx,
    grad_loss1: torch.Tensor,
    ...,
    grad_lossN: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    """
    Backward pass API design:

    Inputs:
    - ctx: Autograd context containing saved tensors from forward
    - grad_loss*: Gradient tensors for each loss output
      - Must match order and shape of forward outputs
      - Can be None or zero tensor for disabled losses

    Returns:
    - tuple[torch.Tensor | None, ...]: Gradients for each forward input
      - Order must match forward input order
      - None for non-differentiable inputs (flags, ground-truth, factors, dimensions)
      - Tensor for differentiable inputs (predictions, model_params)

    Responsibilities:
    1. Restore saved tensors from ctx
    2. Handle disabled losses (factor < 0) with zero gradients
    3. Allocate gradient tensors for differentiable inputs
    4. Ensure gradient tensors are contiguous
    5. Launch backward kernel
    6. Return gradients matching forward input order
    """
```

**Forward/Backward Consistency Requirements:**

1. **Input/Output Matching:**

   - Forward inputs must match backward outputs in order and count
   - Forward outputs must match backward inputs in order and count
   - Non-differentiable forward inputs return `None` in backward

2. **Context Saving:**

   - All tensors needed for backward computation must be saved in `ctx` during forward
   - Saved tensors must include: input tensors, output tensors, factors, and any intermediate values
   - Restored tensors in backward must match saved tensors exactly

3. **Gradient Flow:**
   - Only differentiable inputs (predictions, model parameters) receive gradients
   - Flags, ground-truth, factors, and dimensions return `None` gradients
   - Disabled losses (factor < 0) produce zero gradients for corresponding outputs

---

### Layer 2: Loss Modules (Neural Network Modules)

**Conceptual Description:** Layer 2 represents a set of `torch.nn.Module` instances, conceptually one per loss type (RGB loss, LiDAR loss, background loss, etc.). Each module manages the lifecycle of its loss computation, including data preparation, loss selection, and result packaging.

**Implementation Strategies:**

This layer can be implemented in two ways:

1. **Slang Fused Implementation (Current Location: `nre/losses/slang_losses.py`):**

   - A single stateful `torch.nn.Module` class called `SlangLosses` that fuses multiple loss computations
   - Manages all Slang-based losses together for performance optimization
   - Invokes the fused Slang kernel (Layer 0) via the autograd function (Layer 1)
   - Handles data preparation for all losses simultaneously

2. **PyTorch Separate Implementation (Current Location: `nre/losses/losses.py`, `nre/losses/primitive_losses.py`):**
   - Multiple separate `torch.nn.Module` classes, one per loss type (e.g., `RGBLoss`, `LidarLoss`, `BackgroundLoss`)
   - Each class inherits from `BaseRenderLoss` or `BasePrimitiveLoss`
   - Each loss is computed independently with its own kernel invocation
   - Provides flexibility but with more kernel launch overhead

The Slang fused implementation is preferred for performance, while the PyTorch separate implementation is used for losses not yet migrated to Slang or requiring specialized computation.

---

#### Layer 2A: SlangLosses (Fused Slang Implementation)

**Current location:** `nre/losses/slang_losses.py`

**Description:** A stateful `torch.nn.Module` that manages the lifecycle of multiple Slang-based losses simultaneously. This implementation fuses data preparation and kernel invocation for all configured Slang losses.

**Key Responsibilities:**

- **Loss Registration:** Maintains a list of available Slang-based loss types
- **Configuration Management:** Tracks which losses are enabled and their lambda weights
- **Data Preparation:** Extracts and reshapes tensors from model outputs and target data
- **Loss Selection:** Determines which losses to compute based on step, configuration, and data availability
- **Result Packaging:** Wraps computed losses into `LossReturn` objects with lambda weights and reduction functions

**Initialization API Design:**

```python
def __init__(self) -> None:
    """
    Initialization API design:

    Responsibilities:
    - Initialize available loss types list
    - Create sum reduction function for Slang losses
    - Pre-allocate dummy tensors for unused loss types
      - Dummy tensors are used when a loss is disabled due to lack of null tensor support in Slang
      - Shape: minimum required dimensions (typically [1, 1, 1, ...])
      - Dtype: matches actual tensor types (int32 for flags, float32 for others)

    Loss Instances:
    - self.losses contains SlangBaseLoss instances
    - SlangBaseLoss is a concrete class inheriting from BaseLoss
    - Provides configuration functionalities such as step-based execution control
    """
    super().__init__()
    self.available: list[str] = [...]  # Example: ["rgb_l1_mean", "lidar_l1_mean", ...]
    self.losses: list[SlangBaseLoss] = []
    self.sum_reduce_fn: SumReduceFn = SumReduceFn(...)
    # Dummy tensors for each loss type category
    self.dummy_flags: torch.Tensor = ...
    self.dummy_pred: torch.Tensor = ...
    self.dummy_gt: torch.Tensor = ...
```

**Forward API Design:**

```python
def forward(
    self,
    step: int,
    model: BaseModel,
    results: GaussiansCompositeReturn,
    target: DataAndRenderingBatch,
) -> dict[str, LossReturn]:
    """
    Forward pass API design:

    Inputs:
    - step: Current training step (for step-based loss scheduling)
    - model: Model instance (for accessing model parameters, post-processing modules)
    - results: Rendered outputs containing predictions
    - target: Ground-truth data and labels

    Returns:
    - dict[str, LossReturn]: Dictionary mapping loss names to LossReturn objects

    Responsibilities:
    1. Determine which losses should run:
       - Check loss.should_run_fn(step) for step-based execution control
       - Check is_enabled() for data availability
    2. Extract and prepare tensors:
       - Load shared data (flags, ground-truth) once
       - Extract loss-specific predictions from results
       - Extract model parameters (e.g., bilateral grids)
       - Reshape tensors to match kernel expectations
       - Compute normalization factors (1/N_valid)
    3. Use dummy tensors for disabled losses:
       - Replace disabled loss tensors with dummy tensors (due to lack of null tensor support in Slang)
       - Set corresponding factors to negative values
    4. Invoke SlangLossesFunction.apply() with prepared tensors (Layer 1)
    5. Package results:
       - Map kernel outputs to loss names
       - Create LossReturn objects with lambda weights
       - Apply reduction functions
    """
```

**Loss Selection and Data Availability:**

**Step-Based Execution Control:**

- Each loss configuration can specify a `start_step` parameter
- Losses only execute when `current_step >= start_step`
- This allows staged training where losses are introduced gradually
- Lambda schedulers can further control loss weighting over time

**Data Availability Checks (`is_enabled`):**

- Checks if required data is available in the input dataset
- **Example:** A LiDAR-based loss configured in the config may be disabled if the input dataset does not contain LiDAR data
- **Example:** A background loss may be disabled if camera has any difixed label
- Prevents runtime errors from missing data
- Allows configuration to include losses that may not be supported by all datasets

**Data Reuse in Layer 2:**

- When multiple losses share data (e.g., flags), the data is loaded once and reused
- **Example:** Background losses reuse RGB and LiDAR flags tensors rather than loading them separately
- Care must be taken to avoid duplicate loading of shared data
- Shared data extraction happens once per forward pass, then passed to multiple losses

**Example Loss Types:**

Examples of loss types that can be registered in `SlangLosses`:

- `rgb_l1_mean`: RGB L1 loss with mean reduction
- `lidar_l1_mean`: LiDAR L1 loss with mean reduction
- `background_mse_mean`: Background MSE loss with mean reduction
- `background_lidar_mse_mean`: Background LiDAR MSE loss with mean reduction
- `bilateral_grid_drift_identity_distance_mean`: Bilateral grid drift loss with mean reduction
- `distance_l1_mean`: Distance prediction L1 loss with mean reduction
- `semantic_cross_entropy_mean`: Semantic segmentation cross-entropy loss with mean reduction

---

#### Layer 2B: PyTorch Loss Modules (Separate PyTorch Implementation)

**Current location:** `nre/losses/losses.py`, `nre/losses/primitive_losses.py`

**Description:** Multiple separate `torch.nn.Module` classes that implement individual loss computations in pure PyTorch. Each loss type has its own class that inherits from either `BaseRenderLoss` or `BasePrimitiveLoss`.

**Key Characteristics:**

- **Separate Classes:** Each loss type is implemented as an independent class (e.g., `RGBLoss`, `LidarLoss`, `BackgroundLoss`, `DistanceLoss`, `SemanticLoss`)
- **Individual Execution:** Each loss executes independently with its own kernel invocations
- **Flexibility:** Easier to implement specialized or experimental losses without modifying the fused kernel
- **Legacy Status:** These implementations are gradually being migrated to the Slang fused kernel for better performance

**Base Class Hierarchy:**

```python
BaseLoss (abstract)
├── BaseRenderLoss (abstract) - Losses applied to rendered results
│   ├── RGBLoss
│   ├── LidarLoss
│   ├── BackgroundLoss
│   ├── DistanceLoss
│   ├── SemanticLoss
│   ├── LPIPSLoss
│   ├── SSIMLoss
│   └── ... (other render losses)
│
└── BasePrimitiveLoss (abstract) - Losses applied to NRM primitives
    ├── PrimitiveDistanceLoss
    ├── PrimitiveSkyDistanceLoss
    ├── PrimitiveSkyMaskLoss
    └── ... (other primitive losses)
```

**Example Loss Classes:**

- `RGBLoss`: Computes loss between rendered RGB and ground-truth RGB
- `LidarLoss`: Computes loss between rendered LiDAR distance and ground-truth distance
- `BackgroundLoss`: Regularization loss for opacity predictions
- `BilateralGridDriftLoss`: Regularization for bilateral grid parameters
- `SemanticLoss`: Cross-entropy loss for semantic segmentation
- `LPIPSLoss`: Perceptual loss using LPIPS network
- `SSIMLoss`: Structural similarity loss
- `PrimitiveDistanceLoss`: Loss for NRM primitive distance supervision

**Migration Strategy:**

As losses mature and stabilize, they are migrated from PyTorch separate implementations to the Slang fused implementation for improved performance. The migration process involves:

1. Implementing the loss computation in the Slang kernel (Layer 0)
2. Extending the autograd function to handle the new loss (Layer 1)
3. Registering the loss in `SlangLosses.available` (Layer 2A)
4. The aggregator (Layer 3) automatically routes to Slang implementation when available

---

### Layer 3: LossAggregator (Orchestration)

**Current location:** `nre/losses/base.py`

**Description:** The top-level orchestrator that manages all loss functions (both Slang-fused and PyTorch-separate implementations) according to external configuration. This layer provides a unified interface for loss computation and handles the coordination between different Layer 2 implementations (Slang fused vs. PyTorch separate).

**Key Responsibilities:**

- **Loss Registration:** Parses OmegaConf configuration to instantiate loss functions
- **Loss Routing:** Routes losses to Slang fused implementation (Layer 2A) when available, otherwise uses PyTorch separate implementation (Layer 2B)
- **Unified Execution:** Provides a single `__call__` interface for all loss computations regardless of implementation
- **Result Aggregation:** Combines results from both Layer 2A (Slang fused) and Layer 2B (PyTorch separate) implementations into a unified return structure

**Initialization API Design:**

```python
def __init__(
    self,
    config: omegaconf.DictConfig,
    trainer_config: TrainerConfig,
    **kwargs
) -> None:
    """
    Initialization API design:

    Inputs:
    - config: OmegaConf configuration containing loss definitions
    - trainer_config: Trainer configuration for step-based scheduling
    - kwargs: Optional overrides (e.g., force_disable_slang for testing)

    Responsibilities:
    1. Parse configuration:
       - Iterate over loss definitions in config
       - Extract loss name, function type, reduction type
       - Construct full_name: "{loss_name}_{loss_fn}_{reduce_name}"
    2. Route losses:
       - Check if full_name is in SlangLosses.available (Layer 2A)
       - Route to Slang fused implementation if available, otherwise PyTorch separate implementation
    3. Instantiate losses:
       - Create SlangBaseLoss instances for Slang fused losses (Layer 2A)
       - Create BaseRenderLoss or BasePrimitiveLoss instances for PyTorch separate losses (Layer 2B)
    """
```

**Execution API Design:**

```python
def __call__(
    self,
    *,
    step: int,
    model: BaseModel,
    results: GaussiansCompositeReturn | None = None,
    target: DataAndRenderingBatch | None = None,
    primitive: BaseNRMPrimitive | None = None,
    supervision_pack: BaseNRMSupervisionPack | None = None,
    context: DataAndRenderingBatch | None = None,
) -> LossAggregatorReturn:
    """
    Execution API design:

    Inputs:
    - step: Current training step
    - model: Model instance
    - results: Rendered outputs (for BaseRenderLoss)
    - target: Ground-truth data (for BaseRenderLoss)
    - primitive: NRM primitive (for BasePrimitiveLoss)
    - supervision_pack: NRM supervision pack (for BasePrimitiveLoss)
    - context: Context batch (for BasePrimitiveLoss)

    Returns:
    - LossAggregatorReturn: Aggregated loss results

    Responsibilities:
    1. Execute Slang fused losses (Layer 2A - single fused kernel):
       - Call SlangLosses.forward() once for all configured Slang losses
       - Returns dictionary of LossReturn objects
    2. Execute PyTorch separate losses (Layer 2B - individual kernels):
       - Iterate over PyTorch loss modules
       - Check loss.should_run_fn(step) for each
       - Call loss.forward() individually
       - Collect LossReturn objects
    3. Aggregate results:
       - Combine results from Layer 2A (Slang fused) and Layer 2B (PyTorch separate)
       - Return LossAggregatorReturn with unified interface
    """
```

**Configuration-Driven Design:**

Loss selection and weighting are controlled entirely through OmegaConf configuration files. The aggregator automatically routes losses to the appropriate implementation based on availability and naming convention (`{loss_name}_{loss_fn}_{reduce_fn}`).

**Loss Types Supported:**

- **BaseRenderLoss:** Losses applied to rendered results (RGB, LiDAR, background, etc.)
- **BasePrimitiveLoss:** Losses applied to NRM primitives (requires primitive, context, supervision_pack)

---

## Design Principles

### 1. Fused Kernel Execution

The Slang implementation uses a single kernel to compute all loss types, improving performance through:

- Reduced kernel launch overhead
- Better memory bandwidth utilization
- Shared data loading across loss types

### 2. Automatic Differentiation

Slang's automatic differentiation capabilities eliminate the need for manual gradient computation, reducing code complexity and potential errors.

### 3. Conditional Execution

Loss types are conditionally executed based on:

- Training step (via `start_step` and lambda schedulers)
- Data availability (camera, LiDAR, bilateral grids)
- Configuration (loss enabled/disabled)

### 4. Unified Interface

The `LossAggregator` provides a single entry point for all loss computations, abstracting away the differences between Slang and Python implementations.

### 5. Configuration-Driven

Loss selection, weighting, and scheduling are controlled through external configuration files, enabling flexible experimentation without code changes.

### 6. Layered Access Pattern

Each layer only accesses the immediate layer below it: Layer 3 → Layer 2 → Layer 1 → Layer 0. This ensures clear separation of concerns and maintainability.

---

## Layer 2 Implementation Notes

**Note:** The losses module supports two implementation strategies at Layer 2:

- **Layer 2A (Slang Fused):** Preferred implementation for performance-critical losses. Multiple losses are computed in a single fused kernel invocation.
- **Layer 2B (PyTorch Separate):** Used for losses not yet migrated to Slang, or losses requiring specialized computation (e.g., LPIPS, SSIM). Each loss has its own kernel invocations.

The SDK design focuses on the Slang fused architecture (Layer 2A) as the primary implementation path. As losses mature and stabilize, they are migrated from Layer 2B to Layer 2A for improved performance.

---

## Folder Structure

The losses module is organized into a layered folder structure that mirrors the architectural layers:

```
nre/libs/losses/
├── kernel/                          # Layer 0: Slang Kernel
│   ├── __init__.py
│   ├── BUILD.bazel
│   ├── losses.slang                 # Fused Slang CUDA kernel
│   ├── constants.py                 # Constants for kernel compilation (slang_losses_constants.py)
│   └── kernel_test.py               # Tests for kernel layer
│
├── functional/                      # Layer 1: Autograd Function
│   ├── __init__.py
│   ├── BUILD.bazel
│   ├── slang_losses_function.py    # SlangLossesFunction (autograd.Function)
│   └── functional_test.py          # Tests for autograd function layer
│
├── models/                          # Layer 2: Loss Modules
│   ├── __init__.py
│   ├── BUILD.bazel
│   │
│   ├── base_losses.py              # Base classes (BaseLoss, BaseRenderLoss, BasePrimitiveLoss, etc.)
│   ├── registry.py                 # Loss class registration system
│   │
│   ├── slang_losses_module.py      # Layer 2A: SlangLosses (fused Slang implementation)
│   ├── render_losses.py            # Layer 2B: PyTorch render losses (RGBLoss, LidarLoss, etc.)
│   ├── primitive_losses.py         # Layer 2B: PyTorch primitive losses
│   │
│   ├── utils.py                     # Utilities used by Layer 2 (_get_bilateral_grids, SSIM helpers, etc.)
│   ├── loss_fns.py                  # Loss function definitions and registry
│   │
│   ├── lambda_schedulers/           # Lambda scheduling submodule
│   │   ├── __init__.py
│   │   ├── BUILD.bazel
│   │   ├── lambda_schedulers.py
│   │   └── registry.py
│   │
│   ├── reduce_functions/            # Reduction functions submodule
│   │   ├── __init__.py
│   │   ├── BUILD.bazel
│   │   ├── reduce_functions.py
│   │   └── registry.py
│   │
│   └── models_test.py               # Tests for loss modules layer
│
├── orchestration/                   # Layer 3: Orchestration
│   ├── __init__.py
│   ├── BUILD.bazel
│   ├── loss_aggregator.py          # LossAggregator (orchestrator)
│   └── orchestration_test.py       # Tests for orchestration layer
│
├── __init__.py                      # Package initialization
└── BUILD.bazel                      # Root build configuration
```

### Folder Organization Principles

1. **Layer Isolation:** Each layer has its own folder with clear boundaries
2. **Distributed Utilities:** Utilities live in the layer that uses them (higher layers can access lower layer utilities)
3. **Distributed Tests:** Test files live in the layer they test
4. **Submodules in Layer 2:** Lambda schedulers and reduce functions are used primarily by Layer 2 modules, so they reside there
5. **Upward Dependencies Only:** Higher layers can depend on lower layers (Layer 3 → Layer 2 → Layer 1 → Layer 0), but not vice versa

### File Migration Map

| Current Location                                   | New Location                                          | Component                  |
| -------------------------------------------------- | ----------------------------------------------------- | -------------------------- |
| `libs/slang_gaussians/losses/losses.slang`         | `nre/libs/losses/kernel/losses.slang`                 | Layer 0: Slang kernel      |
| `nre/losses/slang_losses_constants.py`             | `nre/libs/losses/kernel/constants.py`                 | Layer 0: Constants         |
| `nre/losses/slang_losses.py` (SlangLossesFunction) | `nre/libs/losses/functional/slang_losses_function.py` | Layer 1: Autograd Function |
| `nre/losses/slang_losses.py` (SlangLosses)         | `nre/libs/losses/models/slang_losses_module.py`       | Layer 2A: Slang module     |
| `nre/losses/losses.py`                             | `nre/libs/losses/models/render_losses.py`             | Layer 2B: Render losses    |
| `nre/losses/primitive_losses.py`                   | `nre/libs/losses/models/primitive_losses.py`          | Layer 2B: Primitive losses |
| `nre/losses/base.py` (Base classes)                | `nre/libs/losses/models/base_losses.py`               | Layer 2: Base classes      |
| `nre/losses/registry.py`                           | `nre/libs/losses/models/registry.py`                  | Layer 2: Loss registration |
| `nre/losses/utils.py`                              | `nre/libs/losses/models/utils.py`                     | Layer 2: Utilities         |
| `nre/losses/loss_fns.py`                           | `nre/libs/losses/models/loss_fns.py`                  | Layer 2: Loss functions    |
| `nre/losses/lambda_schedulers/`                    | `nre/libs/losses/models/lambda_schedulers/`           | Layer 2: Lambda schedulers |
| `nre/losses/reduce_functions/`                     | `nre/libs/losses/models/reduce_functions/`            | Layer 2: Reduce functions  |
| `nre/losses/base.py` (LossAggregator)              | `nre/libs/losses/orchestration/loss_aggregator.py`    | Layer 3: Orchestrator      |

### Current Location Reference

**Before reorganization:** `nre/losses/`

**After reorganization:** `nre/libs/losses/`

This migration moves the losses module from `nre/losses/` to `nre/libs/losses/` to reflect its status as a core library component with clear architectural layering.

---

## Integration Points

### With Rendering Pipeline

- Receives `GaussiansCompositeReturn` containing rendered RGB, LiDAR, and other signals
- Extracts opacity and distance predictions from rendering results
- Accesses bilateral grids from model post-processing modules

### With Data Pipeline

- Receives `DataAndRenderingBatch` containing ground-truth labels
- Extracts RGB labels, LiDAR labels, and semantic flags
- Uses ray flags (`RayFlags`) for masking and conditional computation

### With Training Pipeline

- Integrates with `TrainerConfig` for step-based scheduling
- Supports lambda schedulers for dynamic loss weighting
- Returns `LossAggregatorReturn` containing aggregated loss values

---

## Performance Considerations

1. **Memory Efficiency:** Dummy tensors are pre-allocated for unused loss types to avoid dynamic allocation overhead
2. **Contiguous Memory:** All tensors are ensured to be contiguous as required by Slang before kernel launch (enforced in Layer 1)
3. **Factor-Based Execution:** Negative factors indicate disabled losses, allowing the kernel to skip unnecessary computation
4. **Fused Computation:** Multiple loss types computed in a single kernel reduce synchronization overhead
5. **Data Reuse:** Shared data (flags, ground-truth) is loaded once and reused across multiple losses, avoiding duplicate loading
6. **Dummy Tensors:** Used when losses are disabled due to lack of null tensor support in Slang, avoiding kernel parameter changes

---

## Extending the Losses Module

### Adding New Loss Types to Layer 0 (Kernel)

When adding a new loss type to the fused kernel:

1. **Extend Kernel Signature:**

   - Add dimension parameters for the new loss
   - Add scaling factor parameter
   - Add loss-specific input tensors (flags, predictions, ground-truth)
   - Add output tensor for loss values
   - Reuse shared tensors if applicable (e.g., flags reused by background losses)

2. **Implement Loss Computation:**

   - Add conditional logic based on factor value
   - Map thread indices to tensor coordinates
   - Implement loss computation with proper masking
   - Handle edge cases (disabled losses, empty tensors)

3. **Define Compile-Time Constants:**
   - If needed, define constants in Python constants module
   - Add `-D` flag injection in Bazel build configuration
   - Use constants consistently in kernel code

### Adding New Loss Types to Layer 1 (Autograd Function)

When extending the autograd function:

1. **Update Forward Signature:**

   - Add new loss inputs (flags, pred, gt, factor, dimensions)
   - Maintain consistent order with backward outputs
   - Save all necessary tensors in context

2. **Update Backward Signature:**

   - Add gradient input for new loss output
   - Maintain consistent order with forward outputs
   - Return gradients matching forward input order

3. **Ensure Consistency:**
   - Forward inputs match backward outputs in order and count
   - Forward outputs match backward inputs in order and count
   - Context saving/restoration matches exactly

### Adding New Loss Types to Layer 2A (Slang Fused Module)

When registering a new loss type in the Slang fused implementation:

1. **Register Loss Name:**

   - Add loss name to `SlangLosses.available` list
   - Follow naming convention: `{loss_name}_{loss_fn}_{reduce_fn}`

2. **Add Data Extraction:**

   - Implement data extraction logic for the new loss
   - Reuse shared data extraction when possible
   - Compute normalization factors (1/N_valid)

3. **Update is_enabled:**

   - Add data availability check for the new loss
   - Return False if required data is not available

4. **Add Dummy Tensors:**
   - Pre-allocate dummy tensors for the new loss type
   - Use dummy tensors when loss is disabled

### Adding New Loss Types to Layer 2B (PyTorch Separate Modules)

When implementing a new loss type in PyTorch:

1. **Create Loss Class:**

   - Inherit from `BaseRenderLoss` or `BasePrimitiveLoss`
   - Implement `forward()` method with loss computation
   - Use `@register_loss(name)` decorator for automatic registration

2. **Implement Forward Logic:**

   - Extract required data from inputs (results, target, model)
   - Apply masking and filtering as needed
   - Call `self.apply_loss_fn()` with processed data
   - Return `LossReturn` object

3. **Consider Migration:**
   - If the loss becomes performance-critical, plan migration to Layer 2A (Slang fused)

### Adding New Loss Types to Layer 3 (Orchestrator)

The orchestrator automatically handles new loss types if:

- Loss name follows naming convention: `{loss_name}_{loss_fn}_{reduce_fn}`
- Loss is registered in `SlangLosses.available` (for Layer 2A) or via `@register_loss` decorator (for Layer 2B)
- Configuration specifies the loss with proper structure

No changes to Layer 3 are typically needed when adding new loss types.
