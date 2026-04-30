// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
module utils;

namespace utils
{

// FIXME: cuda backward does not compile with this function
//
// [BackwardDifferentiable]
// vector<float, Dim> safe_normalize<let Dim : int>(vector<float, Dim> v)
// {
//     float len = length(v);
//     if (len <= 0.0f)
//     {
//         vector<float, Dim> result = vector<float, Dim>(0);
//         result[0] = 1.0f;
//         return result;
//     }
//     return v/len;
// }

[BackwardDifferentiable]
float3 safe_normalize(float3 v)
{
    float len = length(v);
    if (len <= 0.0f)
    {
        return float3(1.0f, 0.0f, 0.0f);
    }
    return v / len;
}

}
