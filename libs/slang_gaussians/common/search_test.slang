// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

#include "search.slang"

// This kernel receives a flat array of values representing arrays.
// Each array is identified by a pair of values, start index and length.
// This allow us to test several arrays of different lengths in a single
// kernel call.
void search_kernel<SearchStrategy : search::ISearchStrategy>(
    TensorView<SearchStrategy.Value> values,            // Flat array of values.
    TensorView<SearchStrategy.Index> arrays,            // N pairs of start index and length of arrays
    TensorView<SearchStrategy.Value> values_to_search,  // N values to search in this array
    TensorView<SearchStrategy.Index> out_indices,       // N output pairs of surrounding indices
    TensorView<bool> out_inside,                        // N boolean values
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= values_to_search.size(0)) {
        return;
    }

    let start_index = arrays.load(uint2(tid,0));
    let length = arrays.load(uint2(tid,1));

    let value = values_to_search.load(tid);
    let data = ImmutablePtr<SearchStrategy.Value>(values.data_ptr_at(start_index.toUInt()));

    let indices = search::find_surrounding_indices<SearchStrategy>(value, data, length);

    out_indices.store(uint2(tid,0), indices._0);
    out_indices.store(uint2(tid,1), indices._1);
    out_inside.store(tid, indices._2);
}

[CUDAKernel]
[AutoPyBindCUDA]
void search_linear_int64(
    TensorView<int64_t> values,
    TensorView<int32_t> arrays,
    TensorView<int64_t> values_to_search,
    TensorView<int32_t> out_indices,
    TensorView<bool> out_inside,
) {
    typedef search::LinearSearch<int64_t, int32_t> SearchStrategy;
    search_kernel<SearchStrategy>(values, arrays, values_to_search, out_indices, out_inside);
}

[CUDAKernel]
[AutoPyBindCUDA]
void search_binary_int64(
    TensorView<int64_t> values,
    TensorView<int32_t> arrays,
    TensorView<int64_t> values_to_search,
    TensorView<int32_t> out_indices,
    TensorView<bool> out_inside,
) {
    typedef search::BinarySearch<int64_t, int32_t> SearchStrategy;
    search_kernel<SearchStrategy>(values, arrays, values_to_search, out_indices, out_inside);
}

[CUDAKernel]
[AutoPyBindCUDA]
void search_linear_float(
    TensorView<float> values,
    TensorView<int64_t> arrays,
    TensorView<float> values_to_search,
    TensorView<int64_t> out_indices,
    TensorView<bool> out_inside,
) {
    typedef search::LinearSearch<float, int64_t> SearchStrategy;
    search_kernel<SearchStrategy>(values, arrays, values_to_search, out_indices, out_inside);
}

[CUDAKernel]
[AutoPyBindCUDA]
void search_binary_float(
    TensorView<float> values,
    TensorView<int64_t> arrays,
    TensorView<float> values_to_search,
    TensorView<int64_t> out_indices,
    TensorView<bool> out_inside,
) {
    typedef search::BinarySearch<float, int64_t> SearchStrategy;
    search_kernel<SearchStrategy>(values, arrays, values_to_search, out_indices, out_inside);
}

// This kernel triggers a compilation error because it uses unsigned indices.
// We keep the tensors using signed values, otherwise it triggers errors in
// slangtorch.
#ifdef TRIGGER_UNSIGNED_COMPILATION_ERROR
[CUDAKernel]
[AutoPyBindCUDA]
void search_compilation_error_unsigned(
    TensorView<int64_t> values,
    TensorView<int32_t> arrays,
    TensorView<int64_t> values_to_search,
    TensorView<int32_t> out_indices,
    TensorView<bool> out_inside,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= values_to_search.size(0)) {
        return;
    }

    typedef uint32_t UnsignedIndex;
    typedef search::BinarySearch<int64_t, UnsignedIndex> SearchStrategy;

    let start_index = arrays.load(uint2(tid,0));
    UnsignedIndex length = arrays.load(uint2(tid,1));

    let value = values_to_search.load(tid);
    let data = ImmutablePtr<SearchStrategy.Value>(values.data_ptr_at(start_index.toUInt()));

    let indices = search::find_surrounding_indices<SearchStrategy>(value, data, length);

    out_indices.store(uint2(tid,0), indices._0);
    out_indices.store(uint2(tid,1), indices._1);
    out_inside.store(tid, indices._2);
}
#endif
