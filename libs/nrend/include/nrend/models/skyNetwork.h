// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

/** @file   sky_network.h
 */
#pragma once

#include <nrend/models/networkWithInputEncoding.h>

namespace nrend {

using namespace fmt::literals;

template <typename T>
class SkyNetwork : public NetworkWithInputEncoding<T> {
public:
    SkyNetwork(
        uint32_t n_dims_to_encode,
        uint32_t n_extra_dims,
        uint32_t n_output_dims,
        const tcnn::json& encoding,
        const tcnn::json& network)
        : NetworkWithInputEncoding<T>::NetworkWithInputEncoding{
              std::shared_ptr<tcnn::Encoding<T>>{tcnn::create_encoding<T>(n_dims_to_encode, encoding)},
              n_output_dims,
              network,
              n_extra_dims}
        , m_n_extra_dims(n_extra_dims) {}

    virtual ~SkyNetwork() = default;

    std::string generate_device_function(const std::string& name) const override {
        std::string encoding = name + "_encoding";
        std::string network  = name + "_network";

        std::ostringstream preamble;
        preamble << NetworkWithInputEncoding<T>::m_network->generate_device_function(network) << "\n\n"
                 << NetworkWithInputEncoding<T>::m_encoding->generate_device_function(encoding) << "\n\n";

        std::string body = tcnn::dfmt(
            1,
            R"(
                {MLP_IN} mlp_in;
		 		mlp_in.slice<0, {POS_ENC_DIMS_OUT}>() = {ENC}(input.slice<0, {POS_ENC_DIMS_IN}>(), params + {ENC_PARAMS_OFFSET}, fwd_ctx ? fwd_ctx + WARP_SIZE * {ENC_FWD_CTX_OFFSET} : nullptr);
                TCNN_PRAGMA_UNROLL
                for (uint32_t i = 0; i < {N_EXTRA_DIMS}; ++i) {{
                     mlp_in[{POS_ENC_DIMS_OUT} + i] = 0;
                }}
				return {MLP}(mlp_in, params, fwd_ctx);
			)",
            "ENC"_a                = encoding,
            "POS_ENC_DIMS_IN"_a    = nrend::NetworkWithInputEncoding<T>::m_encoding->input_width(),
            "MLP_IN"_a             = nrend::NetworkWithInputEncoding<T>::m_network->generate_vec_in(),
            "POS_ENC_DIMS_OUT"_a   = nrend::NetworkWithInputEncoding<T>::m_encoding->padded_output_width(),
            "N_EXTRA_DIMS"_a       = m_n_extra_dims,
            "ENC_PARAMS_OFFSET"_a  = nrend::NetworkWithInputEncoding<T>::m_network->n_params(),
            "ENC_FWD_CTX_OFFSET"_a = nrend::NetworkWithInputEncoding<T>::m_network->device_function_fwd_ctx_bytes(),
            "MLP"_a                = network);

        return fmt::format("{}{}", preamble.str(), this->generate_device_function_from_body(name, body));
    }

private:
    uint32_t m_n_extra_dims = 0;
};

} // namespace nrend
