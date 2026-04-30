"""Repository rule for downloading GenMO/HMR4D inputs and checkpoints."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "get_auth")

def _genmo_inputs_impl(ctx):
    """Download and extract multiple archives into the same repository."""

    all_urls = [
        ctx.attr.checkpoints_url,
        ctx.attr.smpl_data_url,
        ctx.attr.nvhuman_data_url,
        ctx.attr.hmr4d_url,
        ctx.attr.genmo_checkpoint_url,
    ]

    auth = get_auth(ctx, all_urls)

    # Download and extract checkpoints into checkpoints/ subdirectory
    ctx.download_and_extract(
        url = ctx.attr.checkpoints_url,
        sha256 = ctx.attr.checkpoints_sha256,
        output = "inputs/checkpoints",
        type = "tar.gz",
        auth = auth,
    )

    # Download and extract smpl_data into smpl_data/ subdirectory
    ctx.download_and_extract(
        url = ctx.attr.smpl_data_url,
        sha256 = ctx.attr.smpl_data_sha256,
        output = "inputs/smpl_data",
        type = "tar.gz",
        auth = auth,
    )

    # Download and extract nvhuman_data into nvhuman_data/ subdirectory
    ctx.download_and_extract(
        url = ctx.attr.nvhuman_data_url,
        sha256 = ctx.attr.nvhuman_data_sha256,
        output = "inputs/nvhuman_data",
        type = "tar.gz",
        auth = auth,
    )

    # Download and extract hmr4d into hmr4d/ subdirectory
    ctx.download_and_extract(
        url = ctx.attr.hmr4d_url,
        sha256 = ctx.attr.hmr4d_sha256,
        output = "hmr4d",
        type = "tar.gz",
        auth = auth,
    )

    # Download GenMO checkpoint (single file, not archive)
    ctx.download(
        url = ctx.attr.genmo_checkpoint_url,
        sha256 = ctx.attr.genmo_checkpoint_sha256,
        output = "inputs/mocap_mixed_v1/genmo/genmo_lg_nvhuman_v4+v9/version_0/checkpoints/last.ckpt",
        auth = auth,
    )

    # Create BUILD file
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))

genmo_inputs_repository = repository_rule(
    implementation = _genmo_inputs_impl,
    attrs = {
        "checkpoints_url": attr.string(mandatory = True),
        "checkpoints_sha256": attr.string(mandatory = True),
        "smpl_data_url": attr.string(mandatory = True),
        "smpl_data_sha256": attr.string(mandatory = True),
        "nvhuman_data_url": attr.string(mandatory = True),
        "nvhuman_data_sha256": attr.string(mandatory = True),
        "hmr4d_url": attr.string(mandatory = True),
        "hmr4d_sha256": attr.string(mandatory = True),
        "genmo_checkpoint_url": attr.string(mandatory = True),
        "genmo_checkpoint_sha256": attr.string(mandatory = True),
        "build_file": attr.label(mandatory = True),
    },
)
