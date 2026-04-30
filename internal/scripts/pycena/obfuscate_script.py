# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import ast
import logging
import marshal
import re

from ast import Module
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Pattern, Set, Tuple, Union

import click
import yaml

from jinja2 import Environment, FileSystemLoader, Template

from internal.scripts.pycena.node_transformer import ObfuscationTransformer


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)


#######################################################################################################################
@dataclass
class ObfuscateArgs:
    modules: Dict[str, Path]
    debug: bool
    skip_script: bool
    dump_ast: bool
    priority_pass_regex: List[Pattern]


#######################################################################################################################
@dataclass
class BuildArgs:
    debug: bool
    script_prefix: str
    python_version: str
    python_build_root: str
    python_run_root: str

    # explicit output target files for bazel rules
    header_output_path: Optional[Path] = None
    source_output_path: Optional[Path] = None


#######################################################################################################################
@dataclass
class DependencyNode:
    module: str
    path: Path
    ast: Optional[Module] = None
    deps: List["DependencyNode"] = field(default_factory=list)
    node_map: ClassVar[Dict[str, "DependencyNode"]] = dict()
    to_probe: ClassVar[List["DependencyNode"]] = list()
    visited: ClassVar[Set[str]] = set()


#######################################################################################################################
@dataclass
class ByteCodeInfo:
    module_var: str
    module_byte_var: str
    header_path: Path


#######################################################################################################################
@dataclass
class ByteCodeDump:
    build_dir: Path
    import_order: List[str]
    byte_code_map: Dict[str, ByteCodeInfo]


#######################################################################################################################
def check_for_skip(node: Union[ast.Import, ast.ImportFrom], script_lines: List[str]) -> bool:
    if isinstance(node.end_lineno, int):
        for i in range(node.lineno, node.end_lineno + 1):
            if re.search("pycena\:\W*skip", script_lines[i - 1]):
                return True

    return False


#######################################################################################################################
def find_module_imports(obfuscation_modules: Dict[str, Path], import_node: DependencyNode) -> Optional[DependencyNode]:
    logger.info(f"Finding import for {import_node.module} via {import_node.path} {import_node.deps}")
    if not import_node.path.exists():
        logger.error(f"{str(import_node.path)} does not exist")
        return None

    with open(import_node.path, "r") as fp:
        script_source = fp.read()
        script_source_lines = script_source.split("\n")

    import_node.ast = ast.parse(script_source, filename="<unknown>")

    if import_node.ast is None:
        raise Exception(f"Parsing {import_node.path} yielded a None ast")

    # TODO: handle mods that use __init__
    for node in ast.walk(import_node.ast):
        if isinstance(node, ast.Import):
            if check_for_skip(node, script_source_lines):
                continue

            mod_imports = [n.name for n in node.names if n.name.split(".")[0] in obfuscation_modules]

            for m in mod_imports:
                mod_parts = m.split(".")
                mod = mod_parts[0]
                path_parts = mod_parts[1:]

                module_dir = obfuscation_modules[mod]
                import_exists = False
                import_path = module_dir.joinpath(*path_parts).with_suffix(".py")

                import_exists = import_path.exists()

                # This import might be an <module>/__init__.py
                if not import_exists:
                    import_path = import_path.with_suffix("")
                    if import_path.is_dir() and (import_path / "__init__.py").exists():
                        import_exists = True
                        import_path = import_path / "__init__.py"

                if not import_exists:
                    logger.error(f"Could not find module import via: {str(import_path)}(/.py)")
                    return None

                if m in DependencyNode.node_map:
                    mod_node = DependencyNode.node_map[m]
                else:
                    mod_node = DependencyNode(m, import_path)
                    DependencyNode.node_map[m] = mod_node

                import_node.deps.append(mod_node)

                if m not in DependencyNode.visited:
                    DependencyNode.to_probe.append(mod_node)

        elif isinstance(node, ast.ImportFrom):
            # TODO: add all files in this dir or yell at the person to fix the import.

            if check_for_skip(node, script_source_lines):
                continue

            module_name = node.module

            if module_name is None:
                logger.warning(f"{import_node.path} continues an local from '.' import")
                continue

            mod_parts = module_name.split(".")

            if mod_parts[0] not in obfuscation_modules:
                continue

            module_dir = obfuscation_modules[mod_parts[0]]

            path_parts = mod_parts[1:]

            import_exists = False
            import_path = module_dir.joinpath(*path_parts).with_suffix(".py")

            import_exists = import_path.exists()

            # This import might be an <module>/__init__.py
            if not import_exists:
                import_path = import_path.with_suffix("")

                if import_path.is_dir() and (import_path / "__init__.py").exists():
                    import_exists = True
                    import_path = import_path / "__init__.py"

            if not import_exists:
                for name in node.names:
                    if isinstance(name, ast.alias):
                        possible_module = name.name
                        possible_path = (import_path / possible_module).with_suffix(".py")

                        if possible_path.exists():
                            import_exists = True
                            import_path = possible_path
                            module_name = possible_module

                            break  # THIS only takes the first one

            if not import_exists:
                logger.error(f"Could not find module import via: {str(import_path)}")
                return None

            if module_name in DependencyNode.node_map:
                mod_node = DependencyNode.node_map[module_name]
            else:
                mod_node = DependencyNode(module_name, import_path)
                DependencyNode.node_map[module_name] = mod_node

            import_node.deps.append(mod_node)

            # TODO: Find the correct way to prob this module
            if module_name not in DependencyNode.visited:
                DependencyNode.to_probe.append(mod_node)

    DependencyNode.visited.add(import_node.module)

    return import_node


#######################################################################################################################
def dfs_mangle_modules(obf_args: ObfuscateArgs, script_node: DependencyNode) -> Tuple[DependencyNode, Dict[str, str]]:
    obfuscation_transformer = ObfuscationTransformer(obf_args.modules, obf_args.debug, obf_args.skip_script, logger)

    # Init DFS
    DependencyNode.to_probe = list()
    DependencyNode.to_probe.append(script_node)
    DependencyNode.to_probe.extend(script_node.deps)
    DependencyNode.visited = set()
    transformed_nodes = set()

    def transform_node(node):
        transformed_nodes.add(node.module)
        obfuscation_transformer.set_current_module(node.module)
        obfuscation_transformer.add_remap_module(node.module)

        # Once pass for the definitions another for the calls
        obfuscation_transformer.definition_pass = True
        node.ast = obfuscation_transformer.visit(node.ast)

        if node.ast is None:
            raise Exception(f"Definition pass ast transform yielded a None ast: {node.module}")

        obfuscation_transformer.definition_pass = False
        node.ast = obfuscation_transformer.visit(node.ast)

        if node.ast is None:
            raise Exception(f"Regular pass ast transform yielded a None ast: {node.module}")

    # We have minor circular deps near leaf nodes (non circular at runtime).
    # Do a priority pass to handle the problematic nodes.

    priority_visited = set()

    #####################
    # Priority Pass
    #####################
    while len(obf_args.priority_pass_regex) > 0 and len(DependencyNode.to_probe) > 0:
        node = DependencyNode.to_probe[-1]

        if len(node.deps) > 0 and node.module not in DependencyNode.visited:
            DependencyNode.visited.add(node.module)
            DependencyNode.to_probe.extend(node.deps)
            continue

        DependencyNode.to_probe.pop()

        # Only dump priority modules in this pass
        if not matches_any_regex(node.module, obf_args.priority_pass_regex):
            continue

        priority_visited.add(node.module)

        if node.module in transformed_nodes:
            continue

        transformed_nodes.add(node.module)
        transform_node(node)

    DependencyNode.to_probe = list()
    DependencyNode.to_probe.append(script_node)
    DependencyNode.to_probe.extend(script_node.deps)
    DependencyNode.visited = priority_visited

    #####################
    # Normal Pass
    #####################
    while len(DependencyNode.to_probe) > 0:
        node = DependencyNode.to_probe[-1]

        if len(node.deps) > 0 and node.module not in DependencyNode.visited:
            DependencyNode.visited.add(node.module)
            DependencyNode.to_probe.extend(node.deps)
            continue

        DependencyNode.to_probe.pop()

        if node.module in transformed_nodes:
            continue

        transformed_nodes.add(node.module)
        transform_node(node)

        if node.ast is None:
            raise Exception(f"Regular pass ast transform yielded a None ast: {node.module}")

    return script_node, obfuscation_transformer.get_remapping()


#######################################################################################################################
def matches_any_regex(string, regex_list):
    for regex_pattern in regex_list:
        if re.match(regex_pattern, string):
            return True
    return False


#######################################################################################################################
def dfs_dump_byte_code(
    script_node: DependencyNode, build_dir: Path, module_mapping: Dict[str, str], obf_args: ObfuscateArgs
) -> ByteCodeDump:
    # Init DFS
    DependencyNode.to_probe = list()
    DependencyNode.to_probe.append(script_node)
    DependencyNode.to_probe.extend(script_node.deps)
    DependencyNode.visited = set()
    node_dumped = set()

    import_order = []
    byte_code_map = {}

    def dump_node(node):
        mode = "exec"

        # Some NRE functions, in particular hydra's "search-path"-based config lookup from relative paths,
        # depend on having a valid relative path-like string to be functional. Hence, don't remove the
        # full node-path from the compiled byte-code, but only obfuscate the actual filename
        trace_path = node.path if obf_args.debug else node.path.with_name("<unknown>")

        compiled_code = compile(node.ast, str(trace_path), mode)

        if obf_args.dump_ast:
            dumped_ast = build_dir / f"{node.module.replace('.', '_')}_ast.txt"
            with open(dumped_ast, "w") as fp:
                fp.write(ast.dump(node.ast, indent=4))

        bytecode = marshal.dumps(compiled_code)
        header_prefix = node.module.replace(".", "-")
        header_prefix_val = node.module.replace(".", "_")

        header_path = build_dir / f"{header_prefix}.h"
        build_dir.mkdir(parents=True, exist_ok=True)

        with open(header_path, "w") as header_file:
            header_file.write("#include <cstdint>\n")
            header_file.write("#include <string>\n\n")

            header_file.write(f'std::string {header_prefix_val}_module = "{module_mapping[node.module]}";\n')
            header_file.write(f"uint8_t {header_prefix_val}_bytecode[] = {{")
            header_file.write(", ".join(f"0x{byte:02x}" for byte in bytecode))
            header_file.write("};\n")

            import_order.append(node.module)
            byte_code_map[node.module] = ByteCodeInfo(
                f"{header_prefix_val}_module", f"{header_prefix_val}_bytecode", Path(f"{header_prefix}.h")
            )

    # There can near-leaf circular deps (ast level not runtime)
    # This does a first pass that only takes the problematic node in
    # a first first pass so it can be safely ignore in the later pass.

    priority_visited = set()

    #####################
    # Priority Pass
    #####################
    while len(obf_args.priority_pass_regex) > 0 and len(DependencyNode.to_probe) > 0:
        node = DependencyNode.to_probe[-1]

        if len(node.deps) > 0 and node.module not in DependencyNode.visited:
            DependencyNode.visited.add(node.module)
            DependencyNode.to_probe.extend(node.deps)
            continue

        DependencyNode.to_probe.pop()

        # Only dump priority modules in this pass
        if not matches_any_regex(node.module, obf_args.priority_pass_regex):
            continue

        priority_visited.add(node.module)

        if node.module in node_dumped:
            continue

        node_dumped.add(node.module)
        dump_node(node)

    DependencyNode.to_probe = list()
    DependencyNode.to_probe.append(script_node)
    DependencyNode.to_probe.extend(script_node.deps)
    DependencyNode.visited = priority_visited

    #####################
    # Normal Pass
    #####################
    while len(DependencyNode.to_probe) > 0:
        node = DependencyNode.to_probe[-1]

        if len(node.deps) > 0 and node.module not in DependencyNode.visited:
            DependencyNode.visited.add(node.module)
            DependencyNode.to_probe.extend(node.deps)
            continue

        DependencyNode.to_probe.pop()

        if node.module in node_dumped:
            continue

        node_dumped.add(node.module)
        dump_node(node)

    return ByteCodeDump(build_dir, import_order, byte_code_map)


#######################################################################################################################
def generate_cpp_source(code_dump: ByteCodeDump, template: Template, build_args: BuildArgs, header_file_name: str):
    module_prefix = [m.replace(".", "_") for m in code_dump.import_order if m != "__script__"]

    rendered_template = template.render(
        header_file_name=header_file_name, module_prefix=module_prefix, python_run_root=build_args.python_run_root
    )

    if build_args.source_output_path is None:
        source_path = code_dump.build_dir / (build_args.script_prefix + "_source.cpp")
    else:
        source_path = build_args.source_output_path

    with open(source_path, "w") as fp:
        fp.write(rendered_template)

    logger.info(f"Wrote {str(source_path)}")


#######################################################################################################################
def gen_build_files(code_dump: ByteCodeDump, template_path: Path, build_args: BuildArgs):
    env = Environment(loader=FileSystemLoader(template_path))

    source_template = "cpp_source_debug_template.j2" if build_args.debug else "cpp_source_template.j2"

    cpp_source_template = env.get_template(source_template)

    # Merge individual headers into single header file (simpler to consume by bazel build rules)
    header_output_path = build_args.header_output_path
    if header_output_path is None:
        header_output_path = code_dump.build_dir / (build_args.script_prefix + "_header.h")

    header_file_paths = code_dump.build_dir.glob("*.h")
    with open(header_output_path, "w", encoding="utf-8") as script_header_file:
        for header_file_path in header_file_paths:
            with open(header_file_path, "r", encoding="utf-8") as header_file:
                script_header_file.write(header_file.read())
            Path(header_file_path).unlink()  # remove header file
        logger.info(f"Wrote {str(header_output_path)}")

    generate_cpp_source(code_dump, cpp_source_template, build_args, str(header_output_path.name))


#######################################################################################################################
@click.command()
@click.option("--config", required=True, help="Path to config yaml to parse arguments")
@click.argument("header_source_overwrite", nargs=-1)
def obfuscate_script(
    config: str,
    header_source_overwrite: list[str],
):
    with open(config, "r") as fp:
        config_data = yaml.safe_load(fp)

    script_path = Path(config_data["script"])
    build_dir = Path(config_data["build_dir"])
    template_dir = Path(config_data["template_dir"])
    obfuscation_modules = {
        n["mod"]: Path(n["mod_dir"]) for n in config_data["obfuscation_modules"] if "mod" in n and "mod_dir" in n
    }

    if not script_path.exists():
        logger.error(f"{script_path=} does not exist")
        exit(1)

    missing_dir = False

    for _, mod_dir in obfuscation_modules.items():
        if not mod_dir.exists():
            logger.error(f"{mod_dir=} does not exist.")
            missing_dir = True

    if missing_dir:
        exit(1)

    priority_pass_regex = [re.compile(pattern) for pattern in config_data.get("priority_pass_modules_regex", [])]
    obf_args = ObfuscateArgs(
        obfuscation_modules,
        config_data["debug"],
        config_data["skip_script"],
        config_data["dump_ast"],
        priority_pass_regex,
    )

    build_args = BuildArgs(
        config_data["debug"],
        config_data["script_prefix"],
        config_data["python_version"],
        config_data["python_build_root"],
        config_data["python_run_root"],
    )
    if n_overwrites := len(header_source_overwrite):
        assert n_overwrites == 2, "Expecting both header and source overwrites"
        # Incorporate extra overwrites from bazel rules
        build_args.header_output_path = Path(header_source_overwrite[0])
        build_args.source_output_path = Path(header_source_overwrite[1])

    # this call will populate imports nodes in to_probe
    script_node = find_module_imports(obf_args.modules, DependencyNode("__script__", script_path))
    assert script_node is not None

    while len(DependencyNode.to_probe) > 0:
        probe_node = DependencyNode.to_probe.pop()

        if probe_node.module in DependencyNode.visited:
            logger.info(f"Skipping {probe_node.module}")
            continue

        import_node = find_module_imports(obf_args.modules, probe_node)
        if import_node is None:
            logger.error(
                f"Failed to lookup all imports for '{probe_node.module}' - "
                "check previous errors - are (bazel) dependencies listed correctly?"
            )
            exit(1)

    script_node, module_mapping = dfs_mangle_modules(obf_args, script_node)
    byte_code_dump = dfs_dump_byte_code(script_node, build_dir, module_mapping, obf_args)
    gen_build_files(byte_code_dump, template_dir, build_args)

    return 0


if __name__ == "__main__":
    obfuscate_script()
