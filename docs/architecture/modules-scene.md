# Scene Module Architecture

**Module Path:** `nre/libs/scene/`

---

## 1. Overview

The Scene library provides a unified representation for Gaussian splatting scenes with component-based semantic partitioning. It combines the performance benefits of contiguous GPU buffers with intuitive component-level operations for rendering, optimization, and scene manipulation.

### 1.1 Key Components

1. **Component**: Data containers holding tensor parameters (positions, rotations, scales, densities) as views into Scene buffers, plus metadata and transformation contexts

2. **Scene**: Unified buffer manager that owns contiguous GPU memory for all primitives and provides O(1) component access via offset/size mappings

3. **Transformation Stack**: Composable transformation pipeline with typed contexts, stateless transform functions, and batched execution via TransformationStack

4. **Stage**: Unified rendering interface combining Scene, TransformationStack, and Renderer into a single `render(ImageFrame)` API

---

## 2. Documentation Structure

Detailed documentation is split across multiple files for readability:

| Document                                   | Description                                   |
| ------------------------------------------ | --------------------------------------------- |
| [scene/README.md](scene/README.md)         | Architecture overview and data flow           |
| [scene/component.md](scene/component.md)   | Component data structure, view semantics, API |
| [scene/scene.md](scene/scene.md)           | Buffer management, component lifecycle        |
| [scene/transforms.md](scene/transforms.md) | Transform functions, TransformationStack      |
| [scene/stage.md](scene/stage.md)           | Unified interface into the scene and renderer |

**Start here:** [scene/README.md](scene/README.md) for the high-level architecture and visual diagrams.
