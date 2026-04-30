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

#include <algorithm>
#include <functional>
#include <vector>

// Magic numbers
#define OBFUSCATION_HDR 0x50419213AF923791
#define OBFUSCATION_KEY 0x3E2713D4992C61A5

namespace obfuscation {

class Decoder {

    std::vector<char*> headers;

public:
    Decoder() = default;
    ~Decoder() {
        for (auto el : headers) {
            free(el);
        }
    }

    // Decodes the input if the initial 32 bits match the HDR.
    // Otherwise, returns it as is.
    // To decode, it will malloc and copy the input before xoring it with the KEY.
    // Note: This function signature is constrained by the __nvrtcCPEx API
    const char* decode(const char* in) {
        uint64_t* input_64 = (uint64_t*)in;

        // Verify that the sequence starts with [ header ^ length ]
        if (input_64[0] >> 32 != OBFUSCATION_HDR >> 32) {
            return in;
        }

        const uint64_t len   = input_64[0] ^ OBFUSCATION_HDR;
        const size_t keysize = sizeof(uint64_t);

        // We pad the result so its length is divisible by the length of the xorkey
        const size_t padding = (keysize - ((len + 1) % keysize)) % keysize;

        char* decoded_result = (char*)malloc(len + 1 + padding);
        headers.push_back(decoded_result);

        uint64_t* result_64 = (uint64_t*)decoded_result;
        std::fill_n(result_64, (len + 1 + padding) / keysize, OBFUSCATION_KEY);

        auto obfuscated_begin = in + sizeof(uint64_t); // skip header
        auto obfuscated_end   = obfuscated_begin + len + 1;
        std::transform(obfuscated_begin, obfuscated_end, decoded_result, decoded_result, std::bit_xor<char>());

        return (const char*)decoded_result;
    }
};

std::vector<char> encode(const char* in, size_t maxlen);

} // namespace obfuscation
