# Transform Module

## Overview

The Transform module provides infrastructure for applying composable transformation pipelines to scene components. It consists of three main parts:

1. **TransformContext** - Typed data carriers holding transform-specific data (cuboid_tracks, timestamps, neural network). These are stored on the Component itself.
2. **TransformFunction** - Callable transforms that operate on components, extracting the required context from the component's `transformation_contexts` list
3. **TransformationStack** - Pipeline executor that merges Components, applies transforms, and splits results

**Core Principle**: Merge components for efficient batched transformation, then split back to individual components. For example, multiple dynamic rigid components may be merged into a single component for transformation purposes. See [component.md](./component.md) for more details on the Component class. Each transform function declares its required context type and retrieves it directly from the component.

**Planned Transformation Stacks**: We currently plan to support three types of transformation stacks:

1. **SHFeatureTransformation Stack** - For spherical harmonics feature interpolation based on time
2. **RigidBodyTransformation Stack** - For rigid body motion of dynamic objects (vehicles, pedestrians with box tracks)
3. **DeformableBodyTransformation Stack** - For non-rigid deformations (articulated objects, soft bodies)

---

## Requirements

1. **Component-Owned Contexts**: Components own and manage their transformation contexts. Each component carries the specific `ITransformContext` instances and any state required for transformation.

2. **Stateless Transforms**: Transform functions are stateless. They receive only `timestamps` and `component` as inputs, reading all required data from the component (parameters, metadata, and `transformation_contexts`) at call time.

3. **Parallel Context-Transform Mapping**: The component's `transformation_contexts` list must be parallel with the stack's `_transforms` list. At each index `i`, `type(component.transformation_contexts[i])` must match `_transforms[i].context_type`.

4. **Batched Execution**: Components merged for efficient GPU utilization.

5. **Extensibility**: New context and transform types without breaking existing code.

---

## Design Criteria

### Context vs Stack Responsibilities

- **Context**: Data carrier for transform-specific data (timestamps, networks)
- **Stack**: Pipeline executor (owns components, merges/splits, orchestrates transform execution)

### Why Merge Components?

- **Batched GPU Operations**: Single kernel launch for all gaussians
- **Efficient Memory Access**: Contiguous buffers for all components
- **Simplified Transform Logic**: Transforms operate on single merged component

### Why Flat Hierarchy for Contexts?

- **Type Safety**: Each context type declares exactly what it needs
- **Simple Dispatch**: Transform functions matched to contexts by type
- **Extensibility**: New context types extend `ITransformContext` directly

### Context Type Validation

- Components in the same stack must have identical context type lists
- Enables 1:1 mapping for context merging
- Validated at `add_component()` time

---

## TransformContext Hierarchy

### ITransformContext

**Description:** The base interface for all transform contexts. Provides a name identifier for debugging and dispatch.

**API Design:**

```python
@dataclass
class ITransformContext:
    """
    Base interface for all transform contexts.

    Attributes:
        __name__: Context type identifier
    """
    __name__: str

    @classmethod
    def merge(cls, contexts: List["ITransformContext"]) -> "ITransformContext":
        """
        Merge a list of contexts of this type into a single context.

        Args:
            contexts: List of contexts to merge (all same type)

        Returns:
            Single merged context

        Note:
            Subclasses must implement this method to define type-specific
            merge behavior (e.g., concatenating tensor fields).
        """
        raise NotImplementedError("Subclasses must implement merge()")
```

---

### SHGaussianTransformContext

**Description:** Context for Spherical Harmonics transforms with time-dependent Fourier features. Handles interpolation of `features_albedo` across the Fourier dimension based on frame timestamp.

**API Design:**

```python
@dataclass
class CuboidTrackContext(ITransformContext):
    cuboid_tracks: CuboidTrack
    gaussian_cuboid_ids: torch.Tensor
    tracks_calib: BaseTracksCalib
```

Transform Function : calibrate_tracks()

```python
@dataclass
class SHGaussianTransformContext(ITransformContext):
    """
    Context for SH Gaussian transforms with time-dependent Fourier features.

    Note:
        For individual embedding, the per-gaussian track assignment
        (instance_idx) is retrieved from component.get_metadata("gaussian_cuboid_ids")
    """
    __name__: str = "sh_gaussian"

    time_embedding: BaseInputEmbedding
    fourier_features_dim: int

    @classmethod
    def merge(cls, contexts: List["SHGaussianTransformContext"]) -> "SHGaussianTransformContext":
        """
        Merge contexts from multiple components.

        All fields are shared configuration, so merge validates equality.

        Args:
            contexts: List of SHGaussianTransformContext to merge

        Returns:
            Single merged context (first context if all are equal)

        Raises:
            ValueError: If any fields differ between contexts
        """
        ...
```

---

### RigidGaussianTransformContext

**Description:** Context for rigid body transforms of dynamic Gaussians. Handles track pose calibration, pose interpolation, and per-track appearance/scale modifiers.

**API Design:**

```python
@dataclass
class RigidGaussianTransformContext(ITransformContext):
    """
    Context for rigid body transforms of dynamic Gaussians.

    Note:
        Core track data is retrieved from component metadata:
        - component.get_metadata("cuboid_tracks") -> CuboidTracks
        - component.get_metadata("gaussian_cuboid_ids") -> [n_gaussians] tensor
        - component.get_metadata("track_albedos") -> Optional[Tensor] [n_tracks, 3, 4]
        - component.get_metadata("track_scales") -> Optional[Tensor] [n_tracks, 3]
    """
    __name__: str = "rigid_gaussian"

    # Track calibration module
    tracks_calib: BaseTracksCalib

    # Ray data for timestamp estimation
    rays: torch.Tensor                    # from RenderingData
    rays_timestamps_us: torch.Tensor      # from RenderingData

    @classmethod
    def merge(cls, contexts: List["RigidGaussianTransformContext"]) -> "RigidGaussianTransformContext":
        """
        Merge contexts from multiple components.

        Validates that shared configuration fields (tracks_calib) are equal.
        rays, rays_timestamps_us remain the same

        Args:
            contexts: List of RigidGaussianTransformContext to merge

        Returns:
            Single merged context with concatenated ray data

        Raises:
            ValueError: If configuration fields differ between contexts
        """
        ...
```

---

## TransformFunction Protocol

**Description:** Protocol defining the interface for transform functions. Each transform declares its required context type and retrieves it from the component's `transformation_contexts` list.

**API Design:**

```python
from typing import Protocol, Type

class TransformFunction(Protocol):
    """
    Protocol for transform functions.

    Each transform declares the context type it operates on and retrieves
    it from the component. Multiple transforms can share the same context type.
    """

    @property
    def context_type(self) -> Type[ITransformContext]:
        """Return the context type this transform requires."""
        ...

    def __call__(
        self,
        timestamps_startend_us: torch.Tensor,
        component: IComponent,
    ) -> None:
        """
        Apply transformation.

        The transform retrieves its required context from component.transformation_contexts.

        Args:
            timestamps_startend_us: Timestamp tensor
            component: Component to transform (parameters modified in-place)
        """
        ...
```

### SHFeatureTransform

```python
class SHFeatureTransform:
    """
    Transform that interpolates Fourier features based on frame timestamp.

    Inputs:
        - ctx: SHGaussianTransformContext (time_embedding, fourier_features_dim)
        - component._radiance["albedo"]: [n_gaussians, fourier_features_dim, 3] or [n_gaussians, 3]
        - component._radiance["specular"]: [n_gaussians, specular_sh_dim]
        - component.get_metadata("gaussian_cuboid_ids"): [n_gaussians] (only for IndividualRemapTimeInputEmbedding)

    Output:
        - Component modified in-place:
            - features = concat(transformed(albedo), specular)
    """

    @property
    def context_type(self) -> Type[ITransformContext]:
        return SHGaussianTransformContext

    def __call__(
        self,
        timestamps_startend_us: torch.Tensor,
        component: IComponent,
    ) -> None:
        ctx = self._get_context(component)
        ...

    def _get_context(self, component: IComponent) -> SHGaussianTransformContext:
        for ctx in component.transformation_contexts:
            if isinstance(ctx, SHGaussianTransformContext):
                return ctx
        raise ValueError("No SHGaussianTransformContext found")
```

---

### RigidBodyTransform

```python
class RigidBodyTransform:
    """
    Transform that applies rigid body motion to gaussians.
    1. Apply track pose calibration (delta rotations/translations)
    2. Apply per-track albedo and scale modifiers
    3. Interpolate poses at the render timestamp
    4. Transform gaussian positions/rotations to world frame

    Inputs:
        - ctx: RigidGaussianTransformContext (tracks_calib, rays, rays_timestamps_us)
        - component.get_metadata("cuboid_tracks"): CuboidTracks
        - component.get_metadata("gaussian_cuboid_ids"): [n_gaussians]
        - component.get_metadata("track_albedos"): Optional[Tensor]
        - component.get_metadata("track_scales"): Optional[Tensor]

    Output:
        - Component modified in-place:
            - positions/rotations transformed to world frame
            - scales scaled by track_scales (if present)
            - features[..., :3] transformed by track_albedos (if present)
            - densities zeroed for invalid tracks
    """

    @property
    def context_type(self) -> Type[ITransformContext]:
        return RigidGaussianTransformContext

    def __call__(
        self,
        timestamps_startend_us: torch.Tensor,
        component: IComponent,
    ) -> None:
        ctx = self._get_context(component)
        ...

    def _get_context(self, component: IComponent) -> RigidGaussianTransformContext:
        for ctx in component.transformation_contexts:
            if isinstance(ctx, RigidGaussianTransformContext):
                return ctx
        raise ValueError("No RigidGaussianTransformContext found")
```

---

## TransformationStack

**Description:** Pipeline executor that manages components, merges them for batched transformation, applies transform functions in DAG order, and splits results back to individual components.

**API Design:**

```python
class TransformationStack:
    """
    Transformation pipeline for a collection of components.

    Merges components for efficient batched transformation,
    then splits results back to individual components.
    """

    # === Identity ===
    _name: str                                   # Stack identifier

    # === Transform Pipeline ===
    _transforms: List[TransformFunction]         # Transform functions (DAG order)

    # === Components ===
    _components: List[IComponent]                # Components added to stack
    _merged_component: Optional[IComponent]      # Result of merge_components()

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, name: str, transforms: List[TransformFunction] = None) -> None:
        """
        Initialize TransformationStack.

        Args:
            name: Stack identifier
            transforms: Optional list of transform functions (DAG order)
        """
        self._name = name
        self._transforms = transforms or []
        self._components = []
        self._merged_component = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def name(self) -> str:
        """Get stack identifier."""
        return self._name

    @property
    def num_components(self) -> int:
        """Get number of components in stack."""
        return len(self._components)

    @property
    def components(self) -> List[IComponent]:
        """Get read-only access to components."""
        return self._components

    @property
    def merged_component(self) -> Optional[IComponent]:
        """Get merged component (None if not yet merged)."""
        return self._merged_component

    # =========================================================================
    # Component Management
    # =========================================================================

    def add_component(self, component: IComponent) -> None:
        """
        Add a component to the stack.

        Args:
            component: Component to add

        Raises:
            ValueError: If component's context types don't match stack's expected types
        """
        self._validate_context_types(component)
        self._components.append(component)
        self._merged_component = None  # Invalidate cached merged component

    def add_components(self, components: List[IComponent]) -> None:
        """
        Add multiple components to the stack.

        Args:
            components: List of components to add
        """
        for c in components:
            self.add_component(c)

    def _validate_context_types(self, component: IComponent) -> None:
        """
        Validate that component's context types match the stack's transform requirements.

        The component's transformation_contexts list must be parallel with _transforms:
        - _transforms[i].context_type == type(component.transformation_contexts[i])

        Args:
            component: Component to validate

        Raises:
            ValueError: If context types don't match transform requirements at any index
        """
        # for i, transform_fn in enumerate(self._transforms):
        #     expected = transform_fn.context_type
        #     actual = type(component.transformation_contexts[i])
        #     if expected != actual:
        #         raise ValueError(...)
        ...

    # =========================================================================
    # Merge / Split Operations
    # =========================================================================

    def merge_components(self) -> IComponent:
        """
        Merge all components into a single merged component.

        Concatenates:
            - Tensor parameters (positions, rotations, scales, etc.)
            - Context fields (cuboid_tracks, gaussian_cuboid_ids, etc.)

        Returns:
            Merged component with concatenated parameters and contexts

        Note:
            Component boundaries deduced from _components order + sizes
        """
        if len(self._components) == 0:
            raise ValueError("No components to merge")

        # Concatenate parameters
        merged_positions = torch.cat([c.positions for c in self._components], dim=0)
        merged_rotations = torch.cat([c.rotations for c in self._components], dim=0)
        merged_scales = torch.cat([c.scales for c in self._components], dim=0)
        merged_densities = torch.cat([c.densities for c in self._components], dim=0)

        # Merge signal dicts
        merged_signal = self._merge_signal_dicts()

        # Merge radiance dicts (if present)
        merged_radiance = self._merge_radiance_dicts()

        # Merge contexts (1:1 mapping)
        merged_contexts = self._merge_transformation_contexts()

        # Create merged component
        self._merged_component = GaussianComponent(
            positions=merged_positions,
            rotations=merged_rotations,
            scales=merged_scales,
            densities=merged_densities,
            signal=merged_signal,
            radiance=merged_radiance,
            max_gaussians=sum(c.max_gaussians for c in self._components),
        )
        self._merged_component._transformation_contexts = merged_contexts

        return self._merged_component

    def _merge_transformation_contexts(self) -> List[ITransformContext]:
        """
        Merge transformation contexts from all components.

        For each context position i, collects contexts from all components
        and delegates to the context type's merge classmethod.

        Returns:
            List of merged contexts (same length as individual context lists)
        """
        if len(self._components) == 0:
            return []

        num_contexts = len(self._components[0].transformation_contexts)
        merged_contexts = []

        for i in range(num_contexts):
            contexts_at_i = [c.transformation_contexts[i] for c in self._components]
            # Delegate to the context type's own merge logic
            merged_ctx = type(contexts_at_i[0]).merge(contexts_at_i)
            merged_contexts.append(merged_ctx)

        return merged_contexts

    def split_merged_component(self, merged: IComponent) -> List[IComponent]:
        """
        Split merged component back into individual components.

        Uses _components order and sizes to determine boundaries.

        Args:
            merged: Merged component to split

        Returns:
            List of individual components with sliced parameters
        """

    def _slice_component(self, merged: IComponent, offset: int, size: int) -> IComponent:
        """
        Slice a component from the merged component.

        Args:
            merged: Merged component to slice from
            offset: Starting index
            size: Number of gaussians to slice

        Returns:
            New component with sliced parameters
        """

    # =========================================================================
    # Transformation
    # =========================================================================

    def apply_transformation(self, timestamps_startend_us: torch.Tensor) -> IComponent:
        """
        Apply all transforms to merged component in DAG order.

        Each transform retrieves its required context from the component.
        Multiple transforms can share the same context type.

        Args:
            timestamps_startend_us: Timestamp tensor for transformation

        Returns:
            Transformed merged component
        """
        if self._merged_component is None:
            self.merge_components()

        # Apply transforms in DAG order
        for transform_fn in self._transforms:
            transform_fn(timestamps_startend_us, self._merged_component)

        return self._merged_component

    def apply_transformation_and_split(self, timestamps_startend_us: torch.Tensor) -> List[IComponent]:
        """
        Apply transforms and split back to individual components.

        Args:
            timestamps_startend_us: Timestamp tensor for transformation

        Returns:
            List of transformed individual components
        """
        transformed_merged_component = self.apply_transformation(timestamps_startend_us)
        return self.split_merged_component(transformed_merged_component)
```

---

## Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ TransformationStack                                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  _transforms = [Transform_0, Transform_1, Transform_2]                       │
│                                                                              │
│  add_component(comp_1)                                                       │
│    • validates: type(comp_1.contexts[i]) == _transforms[i].context_type      │
│  add_component(comp_2)                                                       │
│    • validates: type(comp_2.contexts[i]) == _transforms[i].context_type      │
│                                                                              │
│  _components = [comp_1, comp_2]                                              │
│                          ↓                                                   │
│                                                                              │
│  merge_components()                                                          │
│    • Concat parameters: positions, rotations, scales, ...                    │
│    • Merge contexts[i]: delegates to contexts[i].merge()                     │
│                          ↓                                                   │
│                                                                              │
│  _merged_component = GaussianComponent(                                      │
│      positions=[n1+n2, 3],                                                   │
│      ...                                                                     │
│      contexts=[MergedCtx_0, MergedCtx_1, MergedCtx_2]                        │
│  )                                                                           │
│                          ↓                                                   │
│                                                                              │
│  apply_transformation(timestamps)                                            │
│    │                                                                         │
│    │  Parallel mapping (1:1 by index):                                       │
│    │    _transforms[0] ←→ contexts[0]                                        │
│    │    _transforms[1] ←→ contexts[1]                                        │
│    │    _transforms[2] ←→ contexts[2]                                        │
│    │                                                                         │
│    └──► for i, transform_fn in enumerate(_transforms):                       │
│            ctx = component.transformation_contexts[i]                        │
│            transform_fn(timestamps, component)                               │
│              • reads ctx + component params/metadata                         │
│              • modifies component in-place                                   │
│                          ↓                                                   │
│                                                                              │
│  split_merged_component()                                                    │
│    • slice [0:n1]      ──► comp_1'                                           │
│    • slice [n1:n1+n2]  ──► comp_2'                                           │
│                          ↓                                                   │
│                                                                              │
│  Return [comp_1', comp_2']                                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```
