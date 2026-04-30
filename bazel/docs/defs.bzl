# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

""" Execute wrapped sphinx-build to build/view HTML docs """

load("@bazel_skylib//lib:paths.bzl", "paths")

SphinxInfo = provider(
    doc = "Info pertaining to Sphinx build.",
    fields = ["open_uri"],
)

def _sphinx_html_impl(ctx):
    sandbox = ctx.actions.declare_directory(ctx.label.name + "_sandbox")
    output_dir = ctx.actions.declare_directory(ctx.label.name + "_html")

    root_dir = paths.dirname(paths.join(sandbox.path, ctx.file.config.short_path))

    # Sphinx expects the config and index files to be in the root directory with the canonical
    # names.  This possibly renames and relocates the config and index files in the sandbox.
    shell_cmds = [
        "mkdir -p {}".format(root_dir),
        "cp {} {}".format(ctx.file.config.path, paths.join(root_dir, "conf.py")),
        "cp {} {}".format(ctx.file.index.path, paths.join(root_dir, "index.rst")),
    ]

    for f in ctx.files.srcs:
        dest = paths.join(sandbox.path, f.short_path)
        shell_cmds.append("mkdir -p {}; cp {} {}".format(paths.dirname(dest), f.path, dest))

    ctx.actions.run_shell(
        outputs = [sandbox],
        inputs = ctx.files.config + ctx.files.index + ctx.files.srcs,
        mnemonic = "SphinxCollect",
        command = "; ".join(shell_cmds),
        progress_message = "Collecting Sphinx source documents for {}".format(ctx.label.name),
    )

    args = ctx.actions.args()
    args.add("-b", "html")
    args.add("-j", "8")
    args.add("-W")
    args.add_all(ctx.attr.args)
    args.add(root_dir)
    args.add(output_dir.path)

    ctx.actions.run(
        outputs = [output_dir],
        inputs = [sandbox],
        executable = ctx.executable._sphinx_build,
        arguments = [args],
        mnemonic = "SphinxBuild",
        progress_message = "Building Sphinx HTML documentation for {}".format(ctx.label.name),
    )

    return [
        DefaultInfo(files = depset([output_dir])),
        SphinxInfo(open_uri = paths.join(output_dir.short_path, "index.html")),
    ]

sphinx_html_gen = rule(
    implementation = _sphinx_html_impl,
    doc = "Sphinx HTML documentation.",
    attrs = {
        "args": attr.string_list(
            doc = "sphinx-build argument list.",
        ),
        "config": attr.label(
            doc = "Sphinx project config file.",
            allow_single_file = True,
            mandatory = True,
        ),
        "index": attr.label(
            doc = "Sphinx project index.",
            allow_single_file = True,
            mandatory = True,
        ),
        "srcs": attr.label_list(
            doc = "Sphinx source and include files.",
            allow_files = True,
            mandatory = True,
            allow_empty = False,
        ),
        "_sphinx_build": attr.label(
            doc = "sphinx-build wrapper.",
            default = Label("//bazel/docs:sphinx_wrapper"),
            executable = True,
            cfg = "exec",
        ),
    },
)

def _sphinx_view_impl(ctx):
    shell_cmd = ctx.attr.open_cmd.format(ctx.attr.generator[SphinxInfo].open_uri)

    script = ctx.actions.declare_file("{}.sh".format(ctx.label.name))
    ctx.actions.write(script, shell_cmd, is_executable = True)

    runfiles = ctx.runfiles(files = ctx.files.generator)

    return [DefaultInfo(executable = script, runfiles = runfiles)]

sphinx_view = rule(
    implementation = _sphinx_view_impl,
    doc = "View Sphinx documentation.",
    attrs = {
        "generator": attr.label(
            doc = "Sphinx documentation generation target",
            mandatory = True,
            providers = [SphinxInfo],
        ),
        "open_cmd": attr.string(
            doc = "Shell open command for Sphinx URI",
            default = "xdg-open {} 1> /dev/null",
        ),
    },
    executable = True,
)
