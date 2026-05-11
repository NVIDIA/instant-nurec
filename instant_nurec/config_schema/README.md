<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Config schema

The runtime config is constructed in code from `pydantic` models. Field
defaults are encoded directly on each schema; `cli.py` only specifies the
fields it overrides via CLI flags (input dataset path, output directory,
merge mode, camera id, max chunks).

Model parameters are sourced by the runtime, not via these schemas.

## What's in scope here

| File | Schemas |
| --- | --- |
| `instantnurec.py` | `InstantNuRecConfig` (top-level), `GaussiansInstantNuRecSystemConfig` |
| `dataset.py` | `InstantNuRecSplitsConfig`, `NCoreInstantNuRecDatasetConfig`, `AdaptiveSequentialFrameBatchSamplerConfig`, `NCoreInstantNuRecCuboidTracksParamsConfig` |
| `predict.py` | `PredictConfig`, `PrimitiveMergeConfig` |
| `models.py` | `KelvinModelConfig` (slim — only `export_preprocess`), `PrimitiveExportPreprocessConfig` |

## BaseConfigSchema

`base_schema.py:BaseConfigSchema` is the base for every config struct. It
behaves like a `pydantic.BaseModel` with strict validation and supports
literals and discriminated unions.

```python
class SubconfigA(BaseConfigSchema):
    name: Literal["subconfig-a"]
    val1: str

class SubconfigB(BaseConfigSchema):
    name: Literal["subconfig-b"]
    val2: int

class MainConfig(BaseConfigSchema):
    regular_str_field: str
    literal_field: Literal["train", "validate"]
    nested_struct_field: SubconfigA
    discriminated_union_field: SubconfigA | SubconfigB = Field(discriminator="name")
    free_form_field: Any
```

### Regular fields

A field declared as `regular_str_field: str` validates/converts the input
to `str` at construction time.

### Literal fields

`literal_field: Literal["train", "validate"]` enforces one of the listed
string values.

### Nested struct fields

`nested_struct_field: SubconfigA` recursively applies the rules of
`SubconfigA`.

### Discriminated unions

`discriminated_union_field: SubconfigA | SubconfigB = Field(discriminator="name")`
selects which sub-schema to use based on the value of the `name` field.

### Free-form fields (`Any`)

Fields annotated as `typing.Any` remain untyped — useful for gradual
adoption of strongly-typed config. `Any` can hold nested sub-configs.
