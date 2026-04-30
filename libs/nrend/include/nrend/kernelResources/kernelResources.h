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

#include <nrend/kernelResources/kernelMemory.h>
#include <nrend/utils/status.h>

#include <array>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/vec.h>

namespace nrend {
struct KernelMemoryBindings {

    static constexpr int InvalidMemoryIndex = -1;

    enum BindingsFlag {
        Parameters,
        ParameterGradients,
        Internal,
        Num,
    } flag = BindingsFlag::Parameters;

    typedef KernelMemoryType MemoryType;

    Status registerMemory(BindingsFlag flag,
                          const std::string& name,
                          MemoryType memoryType,
                          const Logger& logger) const {

        LOG_DEBUG(logger, "KernelMemoryBindings : registering memory : %s.", name.c_str());

        if (name == s_parametersValueBufferName) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "KernelMemoryBindings : memory name %s is reserved.", name.c_str());
        }

        const int memoryIndex = registeredMemoryIndex(flag, name);
        if (memoryIndex != -1) {
            if (memoryType != registeredMemoryType(flag, memoryIndex)) {
                RETURN_ERROR(logger, ErrorCode::BadInput, "KernelMemoryBindings : memory %s already registered with a different definition.", name.c_str());
            }
            return Status();
        }
        m_bindingsTables[flag].push_back(std::make_pair(name, memoryType));
        return Status();
    }

    inline int registeredMemoryIndex(BindingsFlag flag,
                                     const std::string& name) const {

        for (size_t index = 0; index < m_bindingsTables[flag].size(); ++index) {
            if (m_bindingsTables[flag][index].first == name) {
                return index;
            }
        }
        return -1;
    }

    inline int registeredValuesMemoryIndex(BindingsFlag flag = Parameters) const {
        return registeredMemoryIndex(flag, s_parametersValueBufferName);
    }

    inline int numRegisteredMemory(BindingsFlag flag) const {
        return static_cast<int>(m_bindingsTables[flag].size());
    }

    inline MemoryType registeredMemoryType(BindingsFlag flag, int index) const {
        return ((index >= 0) && (index < static_cast<int>(m_bindingsTables[flag].size()))) ? m_bindingsTables[flag][index].second : MemoryType();
    }

    inline std::string registeredMemoryName(BindingsFlag flag, int index) const {
        return (index >= 0) && (index < static_cast<int>(m_bindingsTables[flag].size())) ? m_bindingsTables[flag][index].first : "";
    }

    struct ValueBinding {
        size_t offset = 0;
        size_t size   = 0;

        bool operator!=(const ValueBinding& rhs) const {
            return (offset != rhs.offset) || (size != rhs.size);
        }
    };

    template <typename T>
    Status registerValue(const std::string& name,
                         const T& initialValue,
                         const Logger& logger) const {

        LOG_DEBUG(logger, "KernelMemoryBindings : registering value : %s.", name.c_str());

        const ValueBinding binding = ValueBinding{m_parametersValue.size(), sizeof(T)};
        const int parameterIndex   = registeredValueIndex(name);
        if (parameterIndex != -1) {
            const ValueBinding previousBinding = registeredValueBinding(parameterIndex);
            if (binding != previousBinding) {
                RETURN_ERROR(logger, ErrorCode::BadInput, "KernelParameterBindings : binding %s already registered with a different definition [(%d,%d) / (%d,%d) ].",
                             name.c_str(), static_cast<int>(binding.offset), static_cast<int>(binding.size),
                             static_cast<int>(previousBinding.offset), static_cast<int>(previousBinding.size));
            }
            return Status();
        }
        m_parametersValueBindingsTable.push_back(std::make_pair(name, binding));
        m_parametersValue.resize(binding.offset + binding.size);
        std::memcpy(m_parametersValue.data() + binding.offset, &initialValue, binding.size);

        const int valueMemoryIndex = registeredMemoryIndex(Parameters, s_parametersValueBufferName);
        if (valueMemoryIndex == -1) {
            m_bindingsTables[Parameters].push_back(std::make_pair(s_parametersValueBufferName, MemoryType::Buffer));
        }
        return Status();
    }

    inline int registeredValueIndex(const std::string& name) const {
        for (size_t index = 0; index < m_parametersValueBindingsTable.size(); ++index) {
            if (m_parametersValueBindingsTable[index].first == name) {
                return index;
            }
        }
        return -1;
    }

    inline int numRegisteredValues() const {
        return static_cast<int>(m_parametersValueBindingsTable.size());
    }

    inline std::string registeredValueName(int index) const {
        return (index >= 0) && (index < static_cast<int>(m_parametersValueBindingsTable.size())) ? m_parametersValueBindingsTable[index].first : "";
    }

    inline const ValueBinding registeredValueBinding(int index) const {
        return (index >= 0) && (index < static_cast<int>(m_parametersValueBindingsTable.size())) ? m_parametersValueBindingsTable[index].second : ValueBinding{};
    }

    template <typename T>
    inline Status getRegisteredValue(std::string name, T& value, const Logger& logger) const {
        const int valueIndex = registeredValueIndex(name);
        if (valueIndex < 0) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "KernelParameterBindings : get value %s not registered.", name.c_str());
        }
        const ValueBinding binding = registeredValueBinding(valueIndex);
        if (binding.size != sizeof(T)) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "KernelParameterBindings : get value %s wrong size [%zu / %zu].",
                         name.c_str(), binding.size, sizeof(T));
        }
        std::memcpy(&value, m_parametersValue.data() + binding.offset, binding.size);
        return Status();
    }

    inline Status setRegisteredValue(int valueIndex, const char* value, const Logger& logger) const {
        const ValueBinding binding = registeredValueBinding(valueIndex);
        std::memcpy(m_parametersValue.data() + binding.offset, value, binding.size);
        return Status();
    }

    inline const std::vector<uint8_t>& parametersValueBuffer() const {
        return m_parametersValue;
    }

private:
    static constexpr char s_parametersValueBufferName[] = ".___parameterValues___";
    mutable std::vector<std::pair<std::string, ValueBinding>> m_parametersValueBindingsTable;
    mutable std::vector<uint8_t> m_parametersValue;
    using BindingsTable = std::vector<std::pair<std::string, MemoryType>>;
    mutable std::array<BindingsTable, BindingsFlag::Num> m_bindingsTables;
};

struct KernelSourceCodeTable {
    enum Idiom {
        Slang,
        Cuda,
        Num
    } idiom;

    void registerKernel(Idiom idiom, const std::string& sourceCode) const {
        m_sourceCodeTables[idiom].push_back(sourceCode);
    }

    inline std::string sourceCode(Idiom idiom) const {
        std::string sourceCode;
        for (const auto& sc : m_sourceCodeTables[idiom]) {
            sourceCode.append(sc);
        }
        return sourceCode;
    }

private:
    using SourceCodeTable = std::vector<std::string>;
    mutable std::array<SourceCodeTable, Idiom::Num> m_sourceCodeTables;
};

} // namespace nrend
