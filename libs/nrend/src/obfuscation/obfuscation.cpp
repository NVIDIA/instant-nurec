// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "obfuscation.h"

namespace fs = std::filesystem;

// Simple obfuscator function. Just XOR.
std::vector<char> obfuscation::encode(const char* in, size_t maxlen) {
    const uint64_t len   = strnlen(in, maxlen);
    const size_t keysize = sizeof(uint64_t);
    const size_t padding = (keysize - ((len + 1) % keysize)) % keysize;

    // result is [header ^ len | code ^ xorkey]
    // We pad the result so its length is divisible by the length of the xorkey
    // We use a raw pointer here because that is the input to the decoder function as defined by __nvrtcCPEx
    std::vector<char> obfuscated_result(len + 1 + keysize + padding);
    uint64_t* result_64 = (uint64_t*)obfuscated_result.data();

    result_64[0] = len ^ OBFUSCATION_HDR;
    std::fill_n(result_64 + 1, (len + 1 + padding) / keysize, OBFUSCATION_KEY);

    auto obfuscated_begin = obfuscated_result.data() + sizeof(uint64_t); // skip header
    std::transform(in, in + len + 1, obfuscated_begin, obfuscated_begin, std::bit_xor<char>());

    return obfuscated_result;
}

int main(int nargs, char** argv) {
    // First argument is the destination path. Subsequent arguments are the input files (relative paths!)
    //
    // Multiple file usage:
    // $ obfuscator obfuscated include/foo.cuh include/bar.cuh include/abc/xyz.cuh
    // Single file usage:
    // $ obfuscator obfuscated/include/foo.cuh include/foo.cuh
    fs::path destination(argv[1]);

    // Decoder class that manages memory
    obfuscation::Decoder decoder;

    for (int fid = 2; fid < nargs; ++fid) {
        fs::path filepath(argv[fid]);

        // NOTE: This is commented out to reduce the verbosity of the program.
        // std::cout << "Encoding header " << filepath << std::endl;

        std::ifstream inputFile(filepath, std::ios::in | std::ios::binary | std::ios::ate);
        if (!inputFile.is_open()) {
            std::cerr << "\nerror: unable to open " << filepath << " for reading!\n";
            exit(1);
        }

        std::streampos pos = inputFile.tellg();
        size_t inputSize   = pos;
        char* memBlock     = new char[inputSize + 1];
        inputFile.seekg(0, std::ios::beg);
        inputFile.read(memBlock, inputSize);
        inputFile.close();
        memBlock[inputSize] = '\x0';

        // Note: We are using raw char * pointers to match the usage of the decoder callback in __nvrtcCPEx
        // we setup our decoder as a `char * fn(const void *, void *)` which takes in the obfuscated headers
        // and decodes them.
        auto obfuscated_result = obfuscation::encode(memBlock, inputSize);
        const auto result      = decoder.decode(obfuscated_result.data());

        for (size_t j = 0; j <= inputSize; ++j) {
            if (result[j] != memBlock[j]) {
                std::cerr << "Error: encoding/decoding failed at char " << j << std::endl;
                std::cout << result[j] << " != " << memBlock[j] << std::endl;
                return -1;
            }
        }

        // single input: destination is the full output path
        // multiple inputs: destination is the subpath
        const fs::path outfilepath = nargs == 3 ? destination : destination / filepath;
        fs::create_directories(outfilepath.parent_path());
        std::ofstream outputFile(outfilepath, std::ios::out | std::ios::binary | std::ios::ate);
        if (!outputFile.is_open()) {
            std::cerr << "\nerror: unable to open " << outfilepath << " for writing!\n";
            exit(1);
        }

        outputFile.write(obfuscated_result.data(), inputSize + 1 + sizeof(uint64_t));
        outputFile.close();
    }

    return 0;
}
