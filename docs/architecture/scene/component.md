# Component Module

## Overview

The Component module defines the fundamental building blocks for scene representation. Components are the primary data containers that hold geometric and appearance parameters directly, along with the component specific TransformContext which contains component specific data used in downstream transformation functions.

**Core Principle**: A Component is a **Data Container** with **Transform Context**. It holds tensor parameters (as views into Scene buffers) and carries the transformation contexts needed for downstream processing.

**Design Pattern**: Interface-based design where `IComponent` provides the base contract and concrete implementations like `GaussianComponent` hold type-specific parameters.

---

## Key Characteristics

- **Identifiable**: Each component has a unique `_id` for scene registration and lookup
- **Data Container**: Directly holds tensor parameters (positions, rotations, scales, etc.) as `nn.Parameter`
- **View Semantics**: Parameters are views that share memory with Scene buffers (zero-copy)
- **Transform Context Carrier**: Stores transformation contexts (`ITransformContext`) applicable to this component
- **Capacity-based**: Pre-allocated capacity (`_max_gaussians`) for efficient memory management
- **Interface-based**: `IComponent` base class with typed implementations (`GaussianComponent`)

---

## IComponent (Base Class)

**Description:** The base interface for all scene components. Provides identity and transformation context management.

**API Design:**

```python
class IComponent:
    """
    Base interface for all scene components.

    Attributes:
        _id: Unique identifier for this component
        _transformation_contexts: List of transform contexts applicable to this component
        _metadata: Optional dictionary for storing auxiliary data specific to this component
    """
    _id: str
    _transformation_contexts: List[ITransformContext]
    _metadata: Optional[dict[str, Any]]

    @property
    def id(self) -> str:
        """Get component identifier."""
        return self._id

    @property
    def transformation_contexts(self) -> List[ITransformContext]:
        """Get read-only access to transformation contexts."""
        return self._transformation_contexts

    def add_transformation_context(self, context: ITransformContext) -> None:
        """
        Add a transformation context to this component.

        Args:
            context: Transform context to add
        """
        self._transformation_contexts.append(context)

    def update_transformation_context(self, context: ITransformContext) -> None:
        """
        Update an existing transformation context by name, or add if not found.

        Searches for a context with matching name and replaces it. If no matching
        context is found, the new context is appended.

        Args:
            context: Transform context to update or add
        """
        for i, existing in enumerate(self._transformation_contexts):
            if existing.name == context.name:
                self._transformation_contexts[i] = context
                return
        self._transformation_contexts.append(context)

    @property
    def metadata(self) -> Optional[dict[str, Any]]:
        """Get read-only access to metadata dictionary (may be None)."""
        return self._metadata

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata value. Initializes metadata dict if None.

        Args:
            key: Metadata key
            value: Metadata value
        """
        if self._metadata is None:
            self._metadata = {}
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default (returns default if metadata is None)
        """
        if self._metadata is None:
            return default
        return self._metadata.get(key, default)
```

---

## GaussianComponent

**Description:** Component for 3D Gaussian primitives. Directly holds gaussian parameter tensors as views into Scene buffers.

**API Design:**

```python
class GaussianComponent(IComponent):
    """
    Component for 3D Gaussian primitives.

    Directly holds gaussian parameter tensors (views into Scene buffers).

    Note:
        The `_metadata` dict can store auxiliary data such as:
        - "cuboid_tracks": CuboidTrack
        - "gaussian_cuboid_ids": nn.Buffer # Mapping from gaussians to their parent cuboid IDs
        - "voxels": Tensor
    """

    # === Identity (inherited) ===
    _id: str                                     # Component identifier

    # === Capacity ===
    _max_gaussians: int                          # Pre-allocated capacity

    # === Core Parameters (nn.Parameter, views) ===
    _positions: nn.Parameter                     # [n_gaussians, 3]
    _rotations: nn.Parameter                     # [n_gaussians, 4] quaternions (wxyz)
    _scales: nn.Parameter                        # [n_gaussians, 3]
    _densities: nn.Parameter                     # [n_gaussians, 1]

    # === Sensor-Specific Signals ===
    _signal: dict[str, nn.Parameter]             # {'extra', 'camera', 'lidar'}: [n_gaussians, dim]

    # === Radiance (Optional) ===
    _radiance: Optional[dict[str, nn.Parameter]] # {'albedo', 'specular'}

    # === Transform Contexts (inherited) ===
    _transformation_contexts: List[ITransformContext]

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Get device where component data resides."""
        return self._positions.device

    @property
    def num_gaussians(self) -> int:
        """Get current number of gaussians."""
        return self._positions.shape[0]

    @property
    def max_gaussians(self) -> int:
        """Get pre-allocated capacity."""
        return self._max_gaussians

    @property
    def positions(self) -> Tensor:
        """Get positions [num_gaussians, 3]."""
        return self._positions

    @property
    def rotations(self) -> Tensor:
        """Get rotations [num_gaussians, 4] in wxyz format."""
        return self._rotations

    @property
    def scales(self) -> Tensor:
        """Get scales [num_gaussians, 3]."""
        return self._scales

    @property
    def densities(self) -> Tensor:
        """Get densities [num_gaussians, 1]."""
        return self._densities

    @property
    def covariance(self) -> Tensor:
        """
        Compute covariance matrices from rotations and scales.

        Returns:
            Tensor of shape [num_gaussians, 3, 3]

        Note:
            Computed as: Σ = R @ diag(scales²) @ R^T
        """
        ...

    # =========================================================================
    # Methods
    # =========================================================================

    def get_rotations(self, quaternion_format: Literal["xyzw", "wxyz"]) -> Tensor:
        """
        Get rotations in specified quaternion format.

        Args:
            quaternion_format: Return format ("xyzw" or "wxyz")

        Returns:
            Tensor of shape [num_gaussians, 4]
        """
        ...

    def get_signal(self, key: str) -> Tensor:
        """
        Get sensor-specific signal tensor.

        Args:
            key: Signal type ('extra', 'camera', or 'lidar')

        Returns:
            Tensor of shape [num_gaussians, signal_dim]

        Raises:
            KeyError: If key not found in signal dictionary
        """
        return self._signal[key]

    def get_parameters(self, quaternion_format: Literal["xyzw", "wxyz"] = "xyzw") -> dict[str, Tensor]:
        """
        Get all gaussian parameters as a dictionary.

        Args:
            quaternion_format: Format for quaternion output

        Returns:
            Dictionary containing:
            {
                'positions': [num_gaussians, 3],
                'rotations': [num_gaussians, 4],
                'scales': [num_gaussians, 3],
                'densities': [num_gaussians, 1],
                'signal': dict[str, Tensor],
                'radiance': Optional[dict[str, Tensor]],
            }
        """
        return {
            'positions': self._positions,
            'rotations': self.get_rotations(quaternion_format),
            'scales': self._scales,
            'densities': self._densities,
            'signal': self._signal,
            'radiance': self._radiance,
        }
```

---

## View Semantics

Components maintain zero-copy view semantics with Scene buffers:

```python
# Component holds views - no data copying
vehicle = scene.get_component("vehicle_1")

# Modifications affect the underlying Scene buffers
vehicle.positions[:, 2] += 1.0  # Lifts gaussians by 1 unit in Z

# Scene buffers are modified (shares memory)
offset = scene._id_to_offset["vehicle_1"]
size = scene._id_to_size["vehicle_1"]
assert torch.equal(
    scene._positions[offset:offset + size],
    vehicle.positions
)
```

**Key Point:** The tensor parameters in GaussianComponent are `nn.Parameter` views that share memory with the Scene's unified buffers. This enables zero-copy access and ensures modifications propagate back to the Scene.

---

## Transform Context Integration

Components carry their transformation contexts, enabling downstream processing:

```python
# Get component from scene
vehicle = scene.get_component("vehicle_1")

# Metadata is already restored from scene storage:
# - vehicle.get_metadata("cuboid_tracks") -> CuboidTracks
# - vehicle.get_metadata("gaussian_cuboid_ids") -> [n_gaussians]

# Create transformation context with calibration and ray data
rigid_context = RigidGaussianTransformContext(
    tracks_calib=tracks_calib,
    rays=rendering_data.rays,
    rays_timestamps_us=rendering_data.rays_timestamps_us,
)
vehicle.add_transformation_context(rigid_context)

# Pass to TransformationStack for processing
stack = TransformationStack(name="vehicles", transforms=[RigidBodyTransform()])
stack.add_component(vehicle)
transformed = stack.apply_transformation_and_split(timestamps)[0]

# Persist back to scene
scene.write_component(transformed)
```
