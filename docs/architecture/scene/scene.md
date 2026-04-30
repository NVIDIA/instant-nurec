# Scene Module

## Overview

The Scene module provides a unified scene representation that combines primitive parameter buffers with component-based semantic partitioning. Scene holds unified buffers directly and uses simple dictionary mappings for component tracking. The design maintains the performance benefits of contiguous global buffers while providing intuitive component-based operations for rendering, optimization, and scene manipulation.

**Core Principle**: Scene owns the unified parameter buffers and provides semantic component access through offset/size mappings. Components are views into these buffers.

**Capacity-based Design**: Scene is initialized with a fixed capacity. Adding components that exceed capacity throws an error. Deletion compacts remaining components.

**Generic Design**: `IScene` base interface enables support for different rendering techniques (Gaussians, etc.) through concrete implementations.

---

## Design Criteria

### Unified Buffer Management

Scene maintains all component data in unified parameter buffers:

**Benefits:**

- **Contiguous Memory:** All primitives in continuous memory for cache efficiency
- **Rendering Performance:** GPU can access all data without fragmentation
- **Simple Allocation:** One allocation for entire scene, not per-component
- **Direct Access:** No intermediate Primitive abstraction layer

**Trade-off:** Capacity is fixed at construction; deletion requires buffer compaction.

### Component Tracking

```python
_id_to_offset: dict[str, int]           # component_id → starting index
_id_to_size: dict[str, int]             # component_id → num primitives
_id_to_metadata: dict[str, dict[str, Any]]  # component_id → metadata dict
```

**Rationale:**

- **Direct Access:** O(1) lookup for offset, size, and metadata
- **Encapsulation:** Mappings are internal implementation detail
- **Metadata Preservation:** Component metadata (cuboid_tracks, gaussian_cuboid_ids, etc.) persists through scene lifecycle

### Performance Considerations

1. **O(1) Component Access:** `get_component()` performs constant-time dict lookup and slice creation
2. **Zero-Copy Views:** Component creation has no memory overhead (tensor views)
3. **Contiguous Rendering:** All primitives in continuous memory for efficient GPU access
4. **Compaction Cost:** Deletion requires O(N) buffer shift and offset updates

---

## IScene (Base Interface)

**Description:** Base interface for all scene implementations. Defines the contract for component management with unified buffer storage.

**API Design:**

```python
class IScene:
    """
    Base interface for all scenes.

    Manages a collection of components with unified buffer storage.
    """

    # === Component Management ===

    def add_component(self, offset: int, component: IComponent) -> None:
        """
        Add a component to the scene at the specified offset.

        Args:
            offset: Starting index in unified buffers
            component: Component with data to copy into scene

        Raises:
            ValueError: If offset + component.num_primitives exceeds capacity

        Algorithm:
            1. Validate offset + size <= capacity
            2. Copy component tensors into scene buffers at offset
            3. Register component.id → (offset, size)
            4. Store component metadata by reference
        """
        ...

    def delete_component(self, component_id: str) -> None:
        """
        Delete a component from the scene.

        Args:
            component_id: ID of component to remove

        Algorithm:
            1. Get (offset, size) for component_id
            2. Shift all buffers after offset left by size
            3. Remove component_id from mappings (offset, size, metadata)
            4. Update offsets for all components after deleted one
        """
        ...

    def write_component(self, component: IComponent) -> None:
        """
        Write external component data to scene buffers.

        Uses component.id to look up registered offset.

        Args:
            component: Component with data to write

        Raises:
            KeyError: If component.id not registered
            ValueError: If component size differs from registered size

        Algorithm:
            1. Look up offset from component.id
            2. Validate component size matches registered size
            3. Copy component tensors into scene buffers at offset

        Note:
            For view components (from get_component), this is a no-op
            since modifications already propagate. Use for external
            components (e.g., from TransformationStack).
        """
        ...

    def get_component(self, component_id: str) -> IComponent:
        """
        Get a component view by ID.

        Args:
            component_id: ID of component to retrieve

        Returns:
            Component with tensor views into scene buffers.
            Metadata restored from scene storage.
            Transformation contexts list is empty.

        Raises:
            KeyError: If component_id not found

        Algorithm:
            1. Look up (offset, size) from component_id
            2. Create component with tensor slices [offset:offset+size]
            3. Restore metadata from scene storage
            4. Return component (views, not copies)
        """
        ...

    # === Query Methods ===

    def get_component_ids(self) -> List[str]:
        """Get list of all component IDs."""
        ...

    @property
    def total_num_primitives(self) -> int:
        """Total number of primitives across all components."""
        ...

    @property
    def capacity(self) -> int:
        """Total pre-allocated capacity."""
        ...

    @property
    def device(self) -> torch.device:
        """Device where scene data resides."""
        ...
```

---

## GaussianScene

**Description:** Scene implementation for Gaussian primitives. Holds unified buffers for all gaussian parameters and tracks component locations via offset/size mappings.

**API Design:**

```python
class GaussianScene(IScene):
    """
    Scene containing Gaussian primitives.

    Holds unified buffers for all gaussian parameters and tracks
    component locations via offset/size mappings.
    """

    # === Core Buffers (always present) ===
    _positions: nn.Parameter                     # [capacity, 3]
    _rotations: nn.Parameter                     # [capacity, 4] quaternions (wxyz)
    _scales: nn.Parameter                        # [capacity, 3]
    _densities: nn.Parameter                     # [capacity, 1]

    # === Optional Buffers (configured separately) ===
    _signal: Optional[dict[str, nn.Parameter]]   # {'extra', 'camera', 'lidar'}: [capacity, dim]
    _radiance: Optional[dict[str, nn.Parameter]] # {'albedo', 'specular'}

    # === Component Tracking ===
    _id_to_offset: dict[str, int]                # component_id → starting index
    _id_to_size: dict[str, int]                  # component_id → num gaussians
    _id_to_metadata: dict[str, dict[str, Any]]   # component_id → metadata dict

    # === Capacity ===
    _capacity: int
    _device: torch.device

    # =========================================================================
    # Constructor
    # =========================================================================

    def __init__(
        self,
        capacity: int,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """
        Initialize GaussianScene with pre-allocated core buffers.

        Args:
            capacity: Total capacity for all components
            device: Device for tensor allocation
        """
        ...

    # =========================================================================
    # Buffer Configuration
    # =========================================================================

    def configure_signal(self, signal_dims: dict[str, int]) -> None:
        """
        Configure signal buffers.

        Args:
            signal_dims: Mapping of signal name to dimension
                         e.g., {'camera': 48, 'lidar': 16}
        """
        ...

    def configure_radiance(self, radiance_dims: dict[str, int]) -> None:
        """
        Configure radiance buffers.

        Args:
            radiance_dims: Mapping of radiance name to dimension
                           e.g., {'albedo': 3, 'specular': 1}
        """
        ...

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def total_num_gaussians(self) -> int:
        """Total number of gaussians across all components."""
        ...

    @property
    def total_num_primitives(self) -> int:
        """Alias for total_num_gaussians (IScene interface)."""
        ...

    @property
    def capacity(self) -> int:
        """Total pre-allocated capacity."""
        ...

    @property
    def device(self) -> torch.device:
        """Device where scene data resides."""
        ...

    # =========================================================================
    # Parameter Export
    # =========================================================================

    def get_gaussian_parameters(self) -> dict[str, torch.Tensor]:
        """
        Export scene buffers as a parameter dictionary.

        Returns views (not copies) of scene buffers sliced to total_num_gaussians.

        Returns:
            Dict with the following keys:
            - "positions": [N, 3] gaussian centers
            - "rotations": [N, 4] quaternions (wxyz)
            - "scales": [N, 3] per-axis scales
            - "densities": [N, 1] opacity values
            - "features": [N, radiance_dim] radiance/SH coefficients
            - "extra_signal": [N, extra_dim] general signals
            - "camera_extra_signal": [N, camera_dim] camera-specific signals
            - "lidar_extra_signal": [N, lidar_dim] lidar-specific signals

        Note:
            Signal keys are only present if corresponding buffers were configured
            via configure_signal(). Returns empty tensors [N, 0] for unconfigured signals.
        """
        ...
```

---

## Component View Semantics

`get_component()` returns views that share memory with Scene:

```python
# Get component (zero-copy view)
vehicle = scene.get_component("vehicle_1")

# Modifications affect scene buffers directly
vehicle.positions[:, 2] += 1.0  # Lift by 1 unit

# Scene buffers are modified (shares memory)
offset = scene._id_to_offset["vehicle_1"]
size = scene._id_to_size["vehicle_1"]
assert torch.equal(
    scene._positions[offset:offset + size],
    vehicle.positions
)
```

**Implication:** Direct Component modifications immediately affect rendering without explicit "commit" step.

---

## write_component() for External Data

When working with TransformationStack (which creates copies, not views), use `write_component()` to persist transformed data:

```python
# Get view (metadata is restored automatically)
vehicle = scene.get_component("vehicle_1")

# Create context with rendering data, obtained from the sensor library
rigid_context = RigidGaussianTransformContext(
    tracks_calib=tracks_calib,
    rays=rendering_data.rays,
    rays_timestamps_us=rendering_data.rays_timestamps_us,
)
vehicle.add_transformation_context(rigid_context)

# Transform via stack (creates copies)
stack = TransformationStack(name="vehicles", transforms=[RigidBodyTransform()])
stack.add_component(vehicle)
transformed = stack.apply_transformation_and_split(timestamps)[0]

# transformed is a NEW component with copied tensors, not a view
# Persist back to scene using write_component
scene.write_component(transformed)
```

**Key Point:** `write_component()` uses `component.id` to look up the registered offset. The component must have been previously registered via `add_component()`.

---

## Usage Example

```python
# Initialize scene with capacity
scene = GaussianScene(capacity=100000, device=torch.device("cuda"))

# Configure optional buffers
scene.configure_signal({'camera': 48, 'lidar': 16})
scene.configure_radiance({'albedo': 3})

# Create external components with data and metadata
background = GaussianComponent(id="background", ...)    # 50000 gaussians

vehicle = GaussianComponent(id="vehicle_1", ...)        # 5000 gaussians
vehicle.set_metadata("cuboid_tracks", vehicle_tracks)
vehicle.set_metadata("gaussian_cuboid_ids", vehicle_cuboid_ids)

pedestrian = GaussianComponent(id="pedestrian_1", ...)  # 3000 gaussians
pedestrian.set_metadata("cuboid_tracks", pedestrian_tracks)
pedestrian.set_metadata("gaussian_cuboid_ids", pedestrian_cuboid_ids)

# Add at specific offsets (metadata is preserved)
scene.add_component(offset=0, component=background)
scene.add_component(offset=50000, component=vehicle)
scene.add_component(offset=55000, component=pedestrian)

# Get view - metadata is restored automatically
vehicle_view = scene.get_component("vehicle_1")
assert vehicle_view.get_metadata("cuboid_tracks") is vehicle_tracks  # Same reference

# Add transform context for rendering
rigid_context = RigidGaussianTransformContext(
    tracks_calib=tracks_calib,
    rays=rendering_data.rays,
    rays_timestamps_us=rendering_data.rays_timestamps_us,
)
vehicle_view.add_transformation_context(rigid_context)

# Query scene
ids = scene.get_component_ids()  # ["background", "vehicle_1", "pedestrian_1"]
total = scene.total_num_primitives  # 58000

# Delete (compacts buffers, updates offsets, removes metadata)
scene.delete_component("pedestrian_1")
# vehicle_1 offset unchanged (50000)
# total_num_primitives now 55000
```

---

## References

- **IComponent / GaussianComponent:** See `component.md` for component data structure and view API
- **TransformContext:** See `transforms.md` for transformation context hierarchy
- **TransformationStack:** See `transforms.md` for transformation pipeline
