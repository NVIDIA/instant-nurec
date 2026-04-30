// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

#include <slang_utils/extensions.slang>
#include <geometry/kernels/transform.slang>

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void compose_rotations(
    no_diff uint count,
    DiffTensorView a,
    DiffTensorView b,
    DiffTensorView output,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    const var r_a = transform::SO3.from_vec(a.loadVecOnce<4>(uint2(tid, 0)));
    const var r_b = transform::SO3.from_vec(b.loadVecOnce<4>(uint2(tid, 0)));

    const var out_r = r_a * r_b;

    output.storeVecOnce<4>(uint2(tid,0), out_r.to_vec());
}


[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void interpolate(
    no_diff uint count,
    DiffTensorView a,
    DiffTensorView b,
    DiffTensorView x,
    DiffTensorView output,
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    const var se3_a = transform.SE3.from_vec(
        a.loadVecOnce<3>(uint2(tid, 0)),
        a.loadVecOnce<4>(uint2(tid, 3)),
    );
    const var se3_b = transform.SE3.from_vec(
        b.loadVecOnce<3>(uint2(tid, 0)),
        b.loadVecOnce<4>(uint2(tid, 3)),
    );

    const float alpha = x.loadOnce(tid);

    const var trans_out = transform::interpolate(se3_a, se3_b, alpha).to_vec();

    output.storeVecOnce<3>(uint2(tid,0), trans_out._0);
    output.storeVecOnce<4>(uint2(tid,3), trans_out._1);
}
