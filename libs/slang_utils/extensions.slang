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

__generic<T : __BuiltinFloatingPointType>
extension TensorView<T> {
    // M is the number of elements to load
    // R is the running dimension of the index vector (R < N)
    // N is the number of dimensions of the index vector
    __generic <let M : int, let R : int, let N : int>
    vector<T, M> loadVec(vector<uint, N> x) {
        vector<T, M> result;
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            result[j] = this.load(x);
            x[R] += 1;
        }
        return result;
    }

    // By default the last dimension is the running dimension
    __generic <let M : int, let N : int>
    vector<T, M> loadVec(vector<uint, N> x) {
        return this.loadVec<M, N - 1, N>(x);
    }

    // K is the number of rows to load
    // L is the number of columns to load
    // R is the running dimension of the index vector (R < N)
    // N is the number of dimensions of the index vector
    __generic <let K : int, let L : int, let R : int, let N : int>
    matrix<T, K, L> loadMatrix(vector<uint, N> x) {
        matrix<T, K, L> result;
        [ForceUnroll]
        for (int i = 0; i < K; i++) {
            result[i] = this.loadVec<L, R, N>(x);
            x[R] += L;
        }
        return result;
    }

    // By default the last dimension is the running dimension
    __generic <let K : int, let L : int, let N : int>
    matrix<T, K, L> loadMatrix(vector<uint, N> x) {
        return this.loadMatrix<K, L, N - 1, N>(x);
    }

    // M is the number of elements to store
    // R is the running dimension of the index vector (R < N)
    // N is the number of dimensions of the index vector
    __generic <let M : int, let R : int, let N : int>
    void storeVec(vector<uint, N> x, vector<T, M> value) {
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            this.store(x, value[j]);
            x[R] += 1;
        }
    }

    // By default the last dimension is the running dimension
    __generic <let M : int, let N : int>
    void storeVec(vector<uint, N> x, vector<T, M> value) {
        this.storeVec<M, N - 1, N>(x, value);
    }

    // K is the number of rows to store
    // L is the number of columns to store
    // R is the running dimension of the index vector (R < N)
    // N is the number of dimensions of the index vector
    __generic <let K : int, let L : int, let R : int, let N : int>
    void storeMatrix(vector<uint, N> x, matrix<T, K, L> value) {
        [ForceUnroll]
        for (int i = 0; i < K; i++) {
            this.storeVec<L, R, N>(x, value[i]);
            x[R] += L;
        }
    }

    // By default the last dimension is the running dimension
    __generic <let K : int, let L : int, let N : int>
    void storeMatrix(vector<uint, N> x, matrix<T, K, L> value) {
        this.storeMatrix<K, L, N - 1, N>(x, value);
    }
}

// Extension to add loadUniform functionality to DiffTensorView
__generic<T : __BuiltinFloatingPointType, A : IDiffTensorWrapper>
extension DiffTensorView<T, A>
{
    // loadUniform - indicates that all threads will load from the same indices
    [ForwardDerivative(__loadExUniform_forward)]
    [BackwardDerivative(__loadUniform_backward)]
    [NoDiffThis]
    __generic<let N : int>
    T loadUniform(vector<uint, N> x) { return primal.load(x); }

    __generic<let N : int>
    DifferentialPair<T> __loadExUniform_forward(vector<uint, N> x)
    {
        return diffPair(primal.load(x), reinterpret<T.Differential, T>(diff.load_forward<T, N>(x)));
    }

    __generic<let N : int>
    void __loadUniform_backward(vector<uint, N> x, T.Differential dOut)
    {
        // Since all threads access the same index, we need to perform a reduction
        // to accumulate all gradients. Use WaveActiveSum for efficient parallel reduction
        T reducedGrad = WaveActiveSum(reinterpret<T, T.Differential>(dOut));

        // Only the first lane in the wave stores the reduced gradient to avoid races
        if (WaveIsFirstLane())
        {
            diff.load_backward<T, N>(x, reducedGrad);
        }
    }

    // loadBatchUniform - Load M consecutive uniform floats with block-level backward reduction.
    // Instead of M independent WaveActiveSum + global atomicAdd ops (one per parameter per warp),
    // this batches all M parameters into a single shared-memory reduction pass, yielding
    // ONE global atomicAdd per parameter per BLOCK (~warps_per_block x fewer global atomics).
    [ForwardDerivative(__loadBatchUniform_forward)]
    [BackwardDerivative(__loadBatchUniform_backward)]
    [NoDiffThis]
    __generic<let M : int>
    T[M] loadBatchUniform(uint base_offset) {
        T[M] result;
        [ForceUnroll]
        for (int i = 0; i < M; i++) {
            result[i] = primal.load(base_offset + uint(i));
        }
        return result;
    }

    __generic<let M : int>
    DifferentialPair<T[M]> __loadBatchUniform_forward(uint base_offset) {
        T[M] y;
        T.Differential[M] dy;
        [ForceUnroll]
        for (int i = 0; i < M; i++) {
            y[i] = primal.load(base_offset + uint(i));
            dy[i] = reinterpret<T.Differential, T>(diff.load_forward<T, 1>(base_offset + uint(i)));
        }
        return diffPair(y, reinterpret<T[M].Differential, T.Differential[M]>(dy));
    }

    __generic<let M : int>
    void __loadBatchUniform_backward(uint base_offset, T.Differential[M] dOut) {
        // REQUIRED: _MAX_WARPS is the maximum number of warps that can be used in this kernel's blocks.
        // This must be enforced upstream. If num_threads > 1024 (32 warps) this will overrun the shared memory allocated.

        static const int _MAX_WARPS = 32;
        static groupshared T s_batch_grad[_MAX_WARPS * M];

        uint threadId = cudaThreadIdx().x;
        uint warpSize = WaveGetLaneCount();
        uint warpId = threadId / warpSize;
        uint numWarps = (cudaBlockDim().x + warpSize - 1) / warpSize;

        // Phase 1: Warp-reduce each gradient component via shuffle, store to per-warp slot
        [ForceUnroll]
        for (int i = 0; i < M; i++) {
            T reduced = WaveActiveSum(reinterpret<T, T.Differential>(dOut[i]));
            if (WaveIsFirstLane()) {
                s_batch_grad[warpId * M + i] = reduced;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Phase 2: First M threads sum across warp slots → single global atomicAdd per param
        if (threadId < uint(M)) {
            T total = T(0);
            for (uint w = 0; w < numWarps; w++) {
                total += s_batch_grad[w * M + threadId];
            }
            diff.load_backward<T, 1>(base_offset + threadId, total);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // loadEx - Like load but will pivot to loadUniform's backward if all threads are accessing the same indices
    [ForwardDerivative(__loadExUniform_forward)]
    [BackwardDerivative(__loadEx_backward)]
    [NoDiffThis]
    __generic<let N : int>
    T loadEx(vector<uint, N> x) { return primal.load(x); }

    __generic<let N : int>
    void __loadEx_backward(vector<uint, N> x, T.Differential dOut)
    {
        if (WaveActiveAllEqual(x)) {
            __loadUniform_backward(x, dOut);
        } else {
            // If the values are not all the same, we need to load the backward gradient for each index
            diff.load_backward<T, N>(x, reinterpret<T, T.Differential>(dOut));
        }
    }

    // M is the number of elements to load
    // R is the running dimension (R < N)
    // N is the number of dimensions of the index vector
    [ForwardDerivative(__loadVecExUniform_forward)]
    [BackwardDerivative(__loadVecUniform_backward)]
    [NoDiffThis]
    __generic<let M : int, let R : int, let N : int>
    vector<T, M> loadVecUniform(vector<uint, N> x) {
        vector<T, M> result;
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            result[j] = primal.load(x);
            x[R] += 1;
        }
        return result;
    }

    // By default the last dimension is the running dimension
    [Differentiable]
    __generic<let M : int, let N : int>
    vector<T, M> loadVecUniform(vector<uint, N> x) {
        return this.loadVecUniform<M, N - 1, N>(x);
    }

    __generic<let M : int, let R : int, let N : int>
    DifferentialPair<vector<T, M>> __loadVecExUniform_forward(vector<uint, N> x)
    {
        vector<T, M> y;
        vector<T, M> dy;
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            y[j] = primal.load(x);
            dy[j] = diff.load_forward<T, N>(x);
            x[R] += 1;
        }
        return diffPair(y, reinterpret<vector<T, M>.Differential, vector<T, M>>(dy));
    }

    __generic<let M : int, let R : int, let N : int>
    void __loadVecUniform_backward(vector<uint, N> x, vector<T, M>.Differential dOut)
    {
        // Since all threads access the same index, we need to perform a reduction
        // to accumulate all gradients. Use WaveActiveSum for efficient parallel reduction
        vector<T, M> reducedGrad = WaveActiveSum(dOut);

        // Only the first lane in the wave stores the reduced gradient to avoid races
        if (WaveIsFirstLane())
        {
            [ForceUnroll]
            for (int j = 0; j < M; j++) {
                diff.load_backward<T, N>(x, reducedGrad[j]);
                x[R] += 1;
            }
        }
    }

    // loadVecEx - Will pivot to loadVecUniform's backward if all threads are accessing the same indices
    // M is the number of elements to load
    // R is the running dimension (R < N)
    // N is the number of dimensions of the index vector
    [ForwardDerivative(__loadVecExUniform_forward)]
    [BackwardDerivative(__loadVecEx_backward)]
    [NoDiffThis]
    __generic<let M : int, let R : int, let N : int>
    vector<T, M> loadVecEx(vector<uint, N> x) {
        vector<T, M> result;
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            result[j] = primal.load(x);
            x[R] += 1;
        }
        return result;
    }

    // By default the last dimension is the running dimension
    [Differentiable]
    __generic<let M : int, let N : int>
    vector<T, M> loadVecEx(vector<uint, N> x) {
        return this.loadVecEx<M, N - 1, N>(x);
    }

    __generic<let M : int, let R : int, let N : int>
    void __loadVecEx_backward(vector<uint, N> x, vector<T, M>.Differential dOut)
    {
        if (WaveActiveAllEqual(x)) {
            __loadVecUniform_backward<M, R, N>(x, dOut);
        } else {
            // If the values are not all the same, we need to load the backward gradient for each index
            [ForceUnroll]
            for (int j = 0; j < M; j++) {
                diff.load_backward<T, N>(x, dOut[j]);
                x[R] += 1;
            }
        }
    }

    // M is the number of elements to load
    // R is the running dimension (R < N)
    // N is the number of dimensions of the index vector
    [Differentiable]
    __generic<let M : int, let R : int, let N : int>
    vector<T, M> loadVecOnce(vector<uint, N> x) {
        vector<T, M> result;
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            result[j] = this.loadOnce(x);
            x[R] += 1;
        }
        return result;
    }

    // By default the last dimension is the running dimension
    [Differentiable]
    __generic<let M : int, let N : int>
    vector<T, M> loadVecOnce(vector<uint, N> x) {
        return this.loadVecOnce<M, N - 1, N>(x);
    }

    // M is the number of elements to store
    // R is the running dimension (R < N)
    // N is the number of dimensions of the index vector
    [Differentiable]
    __generic<let M : int, let R : int, let N : int>
    void storeVecOnce(vector<uint, N> x, vector<T, M> y) {
        [ForceUnroll]
        for (int j = 0; j < M; j++) {
            this.storeOnce(x, y[j]);
            x[R] += 1;
        }
    }

    // By default the last dimension is the running dimension
    [Differentiable]
    __generic<let M : int, let N : int>
    void storeVecOnce(vector<uint, N> x, vector<T, M> y) {
        this.storeVecOnce<M, N - 1, N>(x, y);
    }

    // K is the number of rows to load
    // L is the number of columns to load
    // R is the running dimension (R < N)
    // N is the number of dimensions of the index vector
    [Differentiable]
    __generic<let K : int, let L : int, let R : int, let N : int>
    matrix<T, K, L> loadMatrixUniform(vector<uint, N> x) {
        matrix<T, K, L> result;
        [ForceUnroll]
        for (int i = 0; i < K; i++) {
            result[i] = this.loadVecUniform<L, R, N>(x);
            x[R] += L;
        }
        return result;
    }

    // By default the last dimension is the running dimension
    [Differentiable]
    __generic<let K : int, let L : int, let N : int>
    matrix<T, K, L> loadMatrixUniform(vector<uint, N> x) {
        return this.loadMatrixUniform<K, L, N - 1, N>(x);
    }

    // loadMatrixEx - Will pivot to loadMatrixUniform's backward if all threads are accessing the same indices
    // K is the number of rows to load
    // L is the number of columns to load
    // R is the running dimension (R < N)
    // N is the number of dimensions of the index vector
    [Differentiable]
    __generic<let K : int, let L : int, let R : int, let N : int>
    matrix<T, K, L> loadMatrixEx(vector<uint, N> x) {
        matrix<T, K, L> result;
        [ForceUnroll]
        for (int i = 0; i < K; i++) {
            result[i] = this.loadVecEx<L, R, N>(x);
            x[R] += L;
        }
        return result;
    }

    // By default the last dimension is the running dimension
    [Differentiable]
    __generic<let K : int, let L : int, let N : int>
    matrix<T, K, L> loadMatrixEx(vector<uint, N> x) {
        return this.loadMatrixEx<K, L, N - 1, N>(x);
    }

    // K is the number of rows to load
    // L is the number of columns to load
    // R is the running dimension (R < N)
    // N is the number of dimensions of the index vector
    [Differentiable]
    __generic<let K : int, let L : int, let R : int, let N : int>
    matrix<T, K, L> loadMatrixOnce(vector<uint, N> x) {
        matrix<T, K, L> result;
        [ForceUnroll]
        for (int i = 0; i < K; i++) {
            result[i] = this.loadVecOnce<L, R, N>(x);
            x[R] += L;
        }
        return result;
    }

    // By default the last dimension is the running dimension
    [Differentiable]
    __generic<let K : int, let L : int, let N : int>
    matrix<T, K, L> loadMatrixOnce(vector<uint, N> x) {
        return this.loadMatrixOnce<K, L, N - 1, N>(x);
    }

    // K is the number of rows to store
    // L is the number of columns to store
    // R is the running dimension (R < N)
    // N is the number of dimensions of the index vector
    [Differentiable]
    __generic<let K : int, let L : int, let R : int, let N : int>
    void storeMatrixOnce(vector<uint, N> x, matrix<T, K, L> value) {
        [ForceUnroll]
        for (int i = 0; i < K; i++) {
            this.storeVecOnce<L, R, N>(x, value[i]);
            x[R] += L;
        }
    }

    // By default the last dimension is the running dimension
    [Differentiable]
    __generic<let K : int, let L : int, let N : int>
    void storeMatrixOnce(vector<uint, N> x, matrix<T, K, L> value) {
        this.storeMatrixOnce<K, L, N - 1, N>(x, value);
    }
};
