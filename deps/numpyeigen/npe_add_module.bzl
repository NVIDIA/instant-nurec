# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

"""Rules for numpyeigen function / module codegeneration"""

def _npe_codegen_function_impl(ctx):
    # The list of arguments we pass to the script.
    args = [ctx.files.src[0].path] + ["/usr/bin/g++"] + ["-o=" + ctx.outputs.out.path] + ["--c-preprocessor-args=-w -E"]

    # Action to call the script.
    ctx.actions.run(
        inputs = ctx.files.src,
        outputs = [ctx.outputs.out],
        arguments = args,
        progress_message = "NPE codegen function %s" % ctx.outputs.out.short_path,
        executable = ctx.executable.codegen_function,
    )

npe_codegen_function = rule(
    implementation = _npe_codegen_function_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = [".cpp"]),
        "out": attr.output(mandatory = True),
        "codegen_function": attr.label(
            executable = True,
            cfg = "exec",
            default = Label("@numpyeigen//:codegen_function"),
        ),
    },
)

def _npe_codegen_module_impl(ctx):
    # The list of arguments we pass to the script.
    args = ["-m=" + ctx.attr.module] + ["-f=" + ctx.files.src[0].path] + ["-e"] + ["-o=" + ctx.outputs.out.path]

    # Action to call the script.
    ctx.actions.run(
        inputs = ctx.files.src,
        outputs = [ctx.outputs.out],
        arguments = args,
        progress_message = "NPE codegen module %s" % ctx.outputs.out.short_path,
        executable = ctx.executable.codegen_module,
    )

npe_codegen_module = rule(
    implementation = _npe_codegen_module_impl,
    attrs = {
        "module": attr.string(mandatory = True),
        "src": attr.label(mandatory = True, allow_single_file = [".cpp"]),
        "out": attr.output(mandatory = True),
        "codegen_module": attr.label(
            executable = True,
            cfg = "exec",
            default = Label("@numpyeigen//:codegen_module"),
        ),
    },
)
