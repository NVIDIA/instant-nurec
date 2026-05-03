"""Phase A.1 (se3pose_from_matrix → torch helper) is deferred.

Reason: a faithful slang-equivalent torch impl still produces ~1-ULP
floating-point drift relative to the bazel-built kernel on GPU, which
propagates through the post-A.1 ``image_points_to_world_rays_shutter_pose``
call (still slang) into a non-deterministic Gaussian cull boundary —
breaking the exact vertex-count contract.

Once Phase A.6 lands a torch ray-gen, the FP precision of the whole pose →
ray chain is controlled in Python and the kernel can be replaced without
relying on bit-identical f32 ops between slang and torch. Until then the
``libs.geometry.kernels.pose.se3pose_from_matrix`` import in
``instant_nurec/_pkg/utils/batch.py`` stays.
"""
