// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
module transforms;

namespace transforms
{

[BackwardDifferentiable]
float3x3 rotationMatrixTranspose(float4 quaternion)
{
    // quaternion represented as (r,x,y,z)
    const float xx = quaternion.y * quaternion.y;
    const float yy = quaternion.z * quaternion.z;
    const float zz = quaternion.w * quaternion.w;
    const float xy = quaternion.y * quaternion.z;
    const float xz = quaternion.y * quaternion.w;
    const float yz = quaternion.z * quaternion.w;
    const float rx = quaternion.x * quaternion.y;
    const float ry = quaternion.x * quaternion.z;
    const float rz = quaternion.x * quaternion.w;

    // Compute rotation matrix from quaternion
    return float3x3(float3((1.f - 2.f * (yy + zz)), 2.f * (xy + rz), 2.f * (xz - ry)),
                    float3(2.f * (xy - rz), (1.f - 2.f * (xx + zz)), 2.f * (yz + rx)),
                    float3(2.f * (xz + ry), 2.f * (yz - rx), (1.f - 2.f * (xx + yy))));
}

}
