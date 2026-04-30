// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

#pragma once

namespace constants {

[ForceInline]
__generic<T: __BuiltinFloatingPointType>
T pi() { return (T)3.14159265358979323846264338327950288; }

} // namespace constants