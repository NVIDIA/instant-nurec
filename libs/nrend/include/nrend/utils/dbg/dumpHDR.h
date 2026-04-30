// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <algorithm> // for std::clamp
#include <cmath>     // for std::ceil, std::log2, std::pow
#include <cstdint>   // for uint8_t, uint32_t
#include <cuda_runtime.h>
#include <fstream>
#include <iostream> // for std::cout
#include <string>
#include <vector>

namespace nrend {

/**
 * @brief Writes a float4 buffer to a simple HDR file
 *
 * @param filename Output HDR filename
 * @param buffer Pointer to float4 host buffer (RGBA format)
 * @param width Image width
 * @param height Image height
 * @return true if successful, false otherwise
 */
inline bool dumpFloat4HostBufferToHDR(const std::string& filename,
                                      const float4* buffer,
                                      int width,
                                      int height) {
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>> Writing float4 buffer to HDR file: " << filename << std::endl;

    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        return false;
    }

    // Write Radiance HDR header
    outFile << "#?RADIANCE\n";
    outFile << "# Simple HDR format\n";
    outFile << "FORMAT=32-bit_rle_rgbe\n";
    outFile << "\n";
    outFile << "-Y " << height << " +X " << width << "\n";

    // Write RGBE data
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float4& pixel = buffer[y * width + x];

            // Convert RGB to RGBE
            float r = std::max(0.0f, pixel.x);
            float g = std::max(0.0f, pixel.y);
            float b = std::max(0.0f, pixel.z);

            float max_component = std::max(std::max(r, g), b);
            if (max_component < 1e-32f) {
                // Write black pixel
                outFile.put(0);
                outFile.put(0);
                outFile.put(0);
                outFile.put(0);
            } else {
                // Normalize and convert to RGBE
                int e       = static_cast<int>(std::ceil(std::log2(max_component))) + 128;
                float scale = std::pow(2.0f, e - 128);

                uint8_t r_byte = static_cast<uint8_t>(std::clamp(r / scale * 255.0f, 0.0f, 255.0f));
                uint8_t g_byte = static_cast<uint8_t>(std::clamp(g / scale * 255.0f, 0.0f, 255.0f));
                uint8_t b_byte = static_cast<uint8_t>(std::clamp(b / scale * 255.0f, 0.0f, 255.0f));

                outFile.put(r_byte);
                outFile.put(g_byte);
                outFile.put(b_byte);
                outFile.put(static_cast<uint8_t>(e));
            }
        }
    }

    return true;
}

/**
 * @brief Writes a float4 buffer to a simple HDR file
 *
 * @param filename Output HDR filename
 * @param buffer Pointer to float4 devicebuffer (RGBA format)
 * @param width Image width
 * @param height Image height
 * @param cudaStream CUDA stream
 * @return true if successful, false otherwise
 */
inline bool dumpFloat4DeviceBufferToHDR(const std::string& filename,
                                        const float4* buffer,
                                        int width,
                                        int height,
                                        cudaStream_t cudaStream) {
    std::vector<float4> bufferHost(width * height);
    cudaMemcpyAsync(bufferHost.data(), buffer, width * height * sizeof(float4), cudaMemcpyDeviceToHost, cudaStream);
    cudaStreamSynchronize(cudaStream);
    return dumpFloat4HostBufferToHDR(filename, bufferHost.data(), width, height);
}

} // namespace nrend