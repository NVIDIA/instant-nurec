# Scene Architecture

## Overview

The Scene Architecture provides a unified representation for Gaussian splatting scenes with component-based semantic partitioning. It combines the performance benefits of contiguous GPU buffers with intuitive component-level operations for rendering, optimization, and scene manipulation.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                GaussianScene                                    │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                Unified Parameter Buffers [capacity]                       │  │
│  │   positions | rotations | scales | densities | signal | radiance          │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  Component Tracking:                                                            │
│    _id_to_offset:   {"background": 0, "vehicle_1": 50000, ...}                  │
│    _id_to_size:     {"background": 50000, "vehicle_1": 5000, ...}               │
│    _id_to_metadata: {"vehicle_1": {cuboid_tracks, gaussian_cuboid_ids}}         │
└─────────────────────────────────────────────────────────────────────────────────┘
          │                              │
          │ get_component()              │ write_component()
          ▼                              ▲
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             GaussianComponent                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐   │
│  │ Tensor Views     │  │ Metadata         │  │ Transform Contexts           │   │
│  │ (into Scene)     │  │ (from Scene)     │  │ (added for rendering)        │   │
│  │                  │  │                  │  │                              │   │
│  │ positions[...]   │  │ cuboid_tracks    │  │ SHGaussianTransformContext   │   │
│  │ rotations[...]   │  │ gaussian_cuboid_ │  │ RigidGaussianTransform...    │   │
│  │ scales[...]      │  │   ids            │  │                              │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ add to stack
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TransformationStack                                  │
│                                                                                 │
│   1. merge_components()  ──►  Single merged component                           │
│   2. apply_transforms()  ──►  SHFeatureTransform, RigidBodyTransform, ...       │
│   3. split_components()  ──►  Individual transformed components                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Design Principles

### Unified Buffers

Scene owns contiguous GPU memory for all primitives. Components are zero-copy views (tensor slices) into these buffers. This ensures cache-efficient rendering without memory fragmentation.

### Component Ownership

Components own their metadata (`cuboid_tracks`, `gaussian_cuboid_ids`, etc.) and carry transformation contexts. Scene persists metadata by reference and restores it when components are retrieved.

### Stateless Transforms

Transform functions are stateless. They receive `timestamps` and `component` as inputs, reading all required data from the component's parameters, metadata, and `transformation_contexts` at call time.

### Batched Execution

TransformationStack merges multiple components into a single component for efficient GPU kernel execution, then splits results back to individual components.

---

## Module Summary

| Module         | Responsibility                                                             | Key Types                                                       |
| -------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Component**  | Data container holding tensor parameters, metadata, and transform contexts | `IComponent`, `GaussianComponent`                               |
| **Scene**      | Unified buffer management with component tracking and lifecycle            | `IScene`, `GaussianScene`                                       |
| **Transforms** | Transformation contexts, functions, and batched pipeline execution         | `ITransformContext`, `TransformFunction`, `TransformationStack` |

---

## Data Flow Example

### Typical Render Loop

```python
# 1. Scene stores components with metadata
vehicle = GaussianComponent(id="vehicle_1", ...)
vehicle.set_metadata("cuboid_tracks", tracks)
vehicle.set_metadata("gaussian_cuboid_ids", cuboid_ids)
scene.add_component(offset=50000, component=vehicle)

# 2. get_component() returns views with restored metadata
vehicle = scene.get_component("vehicle_1")
# vehicle.get_metadata("cuboid_tracks") -> tracks (same reference)

# 3. Add transformation context with rendering data
rigid_context = RigidGaussianTransformContext(
    tracks_calib=tracks_calib,
    rays=rendering_data.rays,
    rays_timestamps_us=rendering_data.rays_timestamps_us,
)
vehicle.add_transformation_context(rigid_context)

# 4. TransformationStack processes
stack = TransformationStack(name="dynamics", transforms=[RigidBodyTransform()])
stack.add_component(vehicle)
transformed = stack.apply_transformation_and_split(timestamps)[0]

# 5. write_component() persists results back to scene
scene.write_component(transformed)
```

---

## Key Interfaces Quick Reference

### IComponent

```python
id: str                                    # Unique identifier
metadata: dict[str, Any]                   # Auxiliary data (cuboid_tracks, etc.)
transformation_contexts: List[ITransformContext]  # Contexts for transforms
```

### IScene

```python
add_component(offset, component)    # Copy tensors, store metadata
get_component(component_id)         # Return views with restored metadata
delete_component(component_id)      # Remove and compact buffers
write_component(component)          # Persist external component data
```

### ITransformContext

```python
merge(contexts) -> ITransformContext  # Merge contexts from multiple components
```

### TransformFunction

```python
context_type: Type[ITransformContext]  # Required context type
__call__(timestamps, component)        # Apply transformation in-place
```

---

## Detailed Documentation

- **Component Module**: [component.md](component.md) - Data container design, view semantics, metadata API
- **Scene Module**: [scene.md](scene.md) - Buffer management, component lifecycle, capacity design
- **Transform Module**: [transforms.md](transforms.md) - Context hierarchy, transform functions, TransformationStack
