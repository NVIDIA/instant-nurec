# Metrics

This directory contains the metrics framework for NuRec. The framework provides a unified interface for computing, aggregating, and managing various evaluation metrics used in NuRec tasks.

# Core Classes

## Metric

Metric is an abstract library that is intended to be inherited by all metrics. It provides an interface for computing a metric and internally aggregating multiple values.

The `BaseMetric` class defines the core interface that all metrics must implement:

- `validate_inputs()`: Validate input parameters before computation
- `_compute()`: Core computation logic (abstract method)
- `compute()`: Public interface that validates inputs and calls `_compute()`
- `aggregate()`: Aggregate accumulated metric results using specified methods
- `clear()`: Clear accumulated values
- `reset()`: Reset any internal state (if any) in the metric

## MetricResult

`MetricResult` is a unified return type for all Metric classes which is a return type of `compute`. `MetricResult` supports a dictionary of `torch.Tensor` as well as any metadata associated with a given metric computation. This unification standardizes the output of all metrics which can vastly vary from metric to metric.

```python
@dataclass
class MetricResult:
    values: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Example:**

```python
# PSNR metric result
values = {"psnr": torch.tensor(25.5)}
metadata = {"data_range": 255.0, "input_shape": [3, 1024, 1024]}
result = MetricResult(values=values, metadata=metadata)

# Access values
psnr_value = result.get_value("psnr")  # torch.tensor(25.5)
available_values = result.get_available_values()  # ["psnr"]
```

## MetricManager

`MetricManager` is a class for handling multiple metrics and storing their aggregation under-the-hood. This allows abstracting away the handling of individual metrics to the `MetricManager` while the caller only needs to make calls to the `MetricManager`. The caller can register metrics to the manager and call perform and aggregate specific computations using `MetricManager.compute(...)`.

**Key Features:**

- Register and manage multiple metrics
- Automatic computation and storage
- Aggregation across multiple results
- Device management
- YAML export capabilities

## MetricStorage

MetricStorage maintains state of specific metric computations in a serializable format to support exporting to yaml file.

# Constructing a MetricManager

The `MetricManager` serves as the central coordinator for all metrics in your evaluation pipeline. It provides a unified interface for managing multiple metrics, handling their lifecycle, and aggregating results across different computation runs.

When constructing a `MetricManager`, you can specify:

- **Device**: The target device (CPU/GPU) for metric computations
- **DataSource**: Optional data source for automatic metadata handling (used for exporting)

The manager acts as a factory that maintains references to all registered metrics and provides high-level operations like batch computation, aggregation, and result export.

```python
from nre.metrics import MetricManager

# Create a metric manager for GPU computations
manager = MetricManager(device="cuda")

# Create a metric manager with data source integration
manager = MetricManager(device="cuda", datasource=my_datasource)
```

# Creating a New Metric

Creating custom metrics involves inheriting from the `BaseMetric` abstract class and implementing the required interface methods. This ensures consistency across all metrics and enables seamless integration with the `MetricManager`.

Refer to `psnr.py` for an example.

# Adding a Metric to Metric Manager

Once you have created or selected your metrics, you need to register them with the `MetricManager` to enable centralized management and aggregation.

## Creating Metric Instances

The standard way to create a metric is to first instantiate it using the factory in `factory.py`. You can also manually create it by directly importing the class.

### Using the Factory (Recommended)

```python
from nre.metrics import MetricType, MetricFactory

# Create metric using factory
psnr_metric = MetricFactory[MetricType.PSNR](data_range=1.0, device="cuda")
```

### Manual Instantiation

```python
from nre.metrics import PSNRMetric

# Create metrics directly
psnr_metric = PSNRMetric(data_range=1.0, device="cuda")
```

## Registration Process

The registration process involves:

- **Naming**: Assign a unique name to each metric for identification
- **Registration**: Add the metric instance to the manager
- **Computation**: Use the manager to compute and store results
- **Retrieval**: Access individual or aggregated results

```python
# Register a metric with a unique name
manager.register_metric("psnr", psnr_metric)

# Compute and store results
manager.compute("psnr", pred, target)

# Retrieve results
last_result = manager.get_last("psnr")
all_results = manager.get_all("psnr")

# Aggregate results across multiple computations
aggregated = manager.aggregate("psnr")
```

## Computing Metrics with Metadata Collection

The `compute` method supports two signatures for backward compatibility and enhanced functionality:

### Basic Usage (String Parameter)

```python
# Simple computation without metadata collection
manager.compute("psnr", pred, target)
```

### Advanced Usage (ComputeEntry Parameter)

For more advanced use cases, you can use `ComputeEntry` to pass additional metadata for metric collection:

```python
from nre.metrics import ComputeEntry
from nre.utils.types import RayFlags
from nre.utils.batch import FrameMeta

frame_meta = FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)

# Create ComputeEntry with metadata
compute_entry = ComputeEntry(
    name="psnr",
    # Optional metadata dictionary that can be passed in ComputeEntry
    {
        "datasource": my_datasource, # Optional: for automatic sensor ID extraction
        "frame_meta": frame_meta, # Optional: for unique sensor and frame index
        "sequence_id": "sequence_001", # Optional: sequence identifier
    }
    include_metadata=True       # Optional: whether to include metadata in collection
)

# Compute with metadata collection
manager.compute(compute_entry, pred, target)
```

### ComputeEntry Parameters

- **name** (str): The name of the metric to compute (required)
- **metadata** (dict[str, Any] | None): Optional metadata to provide as part of input. This can be used for computing run specific information. Refer to the following keys that are currently utilized:
  - **datasource** (BaseDataSource | None): Optional data source for automatic sensor ID extraction
  - **sequence_id** (list[str] | str | None): Optional sequence identifier for organizing results
- **include_metadata** (bool): Whether to include metadata in the metric collection (default: True)

### Example: Computing Metrics with Sensor Information

```python
from nre.metrics import ComputeEntry
from nre.utils.types import RayFlags

# Compute metric with sensor metadata
compute_entry = ComputeEntry(
    name="psnr",
    {
        "sequence_id": ["frame_001", "frame_002"],
    },
    include_metadata=True
)

manager.compute(compute_entry, pred, target)
```

This approach allows for rich metadata collection while maintaining backward compatibility with the simple string-based API.

## Create tests for all functions that are implemented/overriden

When implementing a new metric, create comprehensive tests for your metric. Refer to `psnr_test.py` or other implementation examples for reference.
