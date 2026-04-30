// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
#ifndef __sphericalHarmonics_slang
#define __sphericalHarmonics_slang

module sphericalHarmonics;

namespace sphericalHarmonics
{

[BackwardDifferentiable]
vector<float, Dim> decode<let Dim : int, let MaxNumCoefficients : int, let Clamped : bool>(
    no_diff in int degree,
    in vector<float, Dim> coefficients[MaxNumCoefficients], 
    in float3 direction)
{
    static const float C0 = 0.28209479177387814;
    static const float C1 = 0.4886025119029199;
    static const float C2[] = { 1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                                   -1.0925484305920792, 0.5462742152960396 };
    static const float C3[] = { -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                                   -0.4570457994644658, 1.445305721320277, -0.5900435899266435 };

    vector<float, Dim> features = C0 * coefficients[0];
    if (MaxNumCoefficients >= 4 && degree > 0)
    {
        const float x = direction.x;
        const float y = direction.y;
        const float z = direction.z;
        features = features - C1 * y * coefficients[1] + C1 * z * coefficients[2] -
              C1 * x * coefficients[3];

        if (MaxNumCoefficients >= 9 && degree > 1)
        {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            features = features + C2[0] * xy * coefficients[4] + C2[1] * yz * coefficients[5] +
                  C2[2] * (2.0 * zz - xx - yy) * coefficients[6] +
                  C2[3] * xz * coefficients[7] + C2[4] * (xx - yy) * coefficients[8];

            if (MaxNumCoefficients >= 16 && degree > 2)
            {
                features = features + C3[0] * y * (3.0 * xx - yy) * coefficients[9] +
                      C3[1] * xy * z * coefficients[10] +
                      C3[2] * y * (4.0 * zz - xx - yy) * coefficients[11] +
                      C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coefficients[12] +
                      C3[4] * x * (4.0 * zz - xx - yy) * coefficients[13] +
                      C3[5] * z * (xx - yy) * coefficients[14] +
                      C3[6] * x * (xx - 3.0 * yy) * coefficients[15];
            }
        }
    }
    if (Clamped)
    {
        return max(features + 0.5, vector<float, Dim>(0));
    }
    else 
    {
        return features + 0.5;
    }
}

[BackwardDifferentiable]
void decode<let Dim : int, let MaxNumCoefficients : int, let Clamped : bool>(
    no_diff in int degree,
    in float coefficients[Dim * MaxNumCoefficients],
    in float3 direction,
    out float features[Dim])
{
    static const float C0 = 0.28209479177387814;
    static const float C1 = 0.4886025119029199;
    static const float C2[] = { 1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                                -1.0925484305920792, 0.5462742152960396 };
    static const float C3[] = { -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                                -0.4570457994644658, 1.445305721320277, -0.5900435899266435 };

    [ForceUnroll] for (int i = 0; i < Dim; ++i) {
        features[i] = C0 * coefficients[i];
    }
    if (MaxNumCoefficients >= 4 && degree > 0)
    {
        const float x = direction.x;
        const float y = direction.y;
        const float z = direction.z;
        [ForceUnroll] for (int i = 0; i < Dim; ++i) {
            features[i] = features[i] - C1 * y * coefficients[i + 1*Dim] + C1 * z * coefficients[i + 2*Dim] -
                   C1 * x * coefficients[i + 3*Dim];
        }
        if (MaxNumCoefficients >= 9 && degree > 1)
        {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            [ForceUnroll] for (int i = 0; i < Dim; ++i) {
                features[i] = features[i] + C2[0] * xy * coefficients[i + 4*Dim] + C2[1] * yz * coefficients[i + 5*Dim] +
                       C2[2] * (2.0 * zz - xx - yy) * coefficients[i + 6*Dim] +
                       C2[3] * xz * coefficients[i + 7*Dim] + C2[4] * (xx - yy) * coefficients[i + 8*Dim];
            }

            if (MaxNumCoefficients >= 16 && degree > 2)
            {
                [ForceUnroll] for (int i = 0; i < Dim; ++i) {
                    features[i] = features[i] + C3[0] * y * (3.0 * xx - yy) * coefficients[i + 9*Dim] +
                           C3[1] * xy * z * coefficients[i + 10*Dim] +
                           C3[2] * y * (4.0 * zz - xx - yy) * coefficients[i + 11*Dim] +
                           C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coefficients[i + 12*Dim] +
                           C3[4] * x * (4.0 * zz - xx - yy) * coefficients[i + 13*Dim] +
                           C3[5] * z * (xx - yy) * coefficients[i + 14*Dim] +
                           C3[6] * x * (xx - 3.0 * yy) * coefficients[i + 15*Dim];
                }
            }
        }
    }
    if (Clamped)
    {
        [ForceUnroll] for (int i = 0; i < Dim; ++i) {
            features[i] = max(features[i] + 0.5, 0.0);
        }
    }
    else
    {
        [ForceUnroll] for (int i = 0; i < Dim; ++i) {
            features[i] = features[i] + 0.5;
        }
    }
}

}
#endif