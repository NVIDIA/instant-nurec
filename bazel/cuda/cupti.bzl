"""Simple repository rule to use CUPTI_PATH environment variable."""

def _cupti_impl(repository_ctx):
    """Use CUPTI_PATH environment variable or default path."""
    cupti_path = repository_ctx.os.environ.get("CUPTI_PATH", "/usr/local/cuda/extras/CUPTI")

    # Create symlink to the CUPTI installation
    repository_ctx.symlink(cupti_path, "cupti")

    # Create BUILD file
    build_content = """
cc_library(
    name = "cupti",
    srcs = glob([
        "cupti/lib64/libcupti.so*",     # Linux shared libraries
        "cupti/lib/libcupti.so*",       # Alternative Linux path
        "cupti/lib/x64/cupti.lib",      # Windows import library
    ], allow_empty = True),
    hdrs = glob(["cupti/include/**/*.h"], allow_empty = True),
    includes = ["cupti/include"],
    visibility = ["//visibility:public"],
)
"""

    repository_ctx.file("BUILD", build_content)

cupti = repository_rule(
    implementation = _cupti_impl,
    environ = ["CUPTI_PATH"],
    local = True,
)
