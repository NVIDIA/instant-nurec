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
import random

from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ObfuscationTransformer(ast.NodeTransformer):
    ###################################################################################################################
    def __init__(self, obfuscation_modules: Dict[str, Path], debug: bool, skip_script: bool, logger: Logger):
        self.debug = debug
        self.logger = logger
        self.skip_script = skip_script

        self.current_transform_module: Optional[str] = None
        self.current_from_imports: Dict[
            str, str
        ] = {}  # Maps objects imported via from x import y (y -> x) for current module
        self.obfuscation_modules = obfuscation_modules  # Which modules we are obfuscating

        self.module_import_remap: Dict[str, str] = {}  # Mapping modules names "import a.b.c" -> "import asdf"
        self.module_import_as_remap: Dict[
            str, Dict[str, str]
        ] = {}  # if given new name as x map x -> to obfuscated name.
        self.module_func_remap: Dict[
            str, Dict[str, str]
        ] = {}  # maps modules to bidirectional dict of func to obfuscated name
        self.skip_func_defs: Set[Any] = set()  # set of function ids to avoid obfuscating.

        self.definition_pass: bool = True

        self.class_to_module: Dict[str, List[str]] = {}
        # Map script to itself
        self.module_import_remap["__script__"] = "__script__"

    ###################################################################################################################
    def set_current_module(self, module: str):
        self.logger.info(f"Setting current Module to {module}")
        self.current_transform_module = module

        if module not in self.module_func_remap:
            self.module_func_remap[module] = {}

        self.module_import_as_remap = {}

    ###################################################################################################################
    def random_name(self, original_name: str):
        if self.debug:
            return f"fake_{original_name}".replace(".", "_")
        else:
            return "".join(random.choice("lI1") for _ in range(25))

    ###################################################################################################################
    def visit_ClassDef(self, node):
        if not self.definition_pass:
            return self.generic_visit(node)

        inner_func_defs = self.find_inner_func_defs(node)
        for inner_func in inner_func_defs:
            self.skip_func_defs.add(inner_func.name)

        return self.generic_visit(node)

    ###################################################################################################################
    def visit_Import(self, node):
        # We skip this visit on pass
        if not self.definition_pass:
            return self.generic_visit(node)

        # Go through each import
        for name in node.names:
            original_name = name.name

            # Already been remapped. Module add themselves to this map.
            if name.name in self.module_import_remap:
                name.name = self.module_import_remap[name.name]

            # If we are using as alias map to original module name.
            if name.asname is not None and original_name in self.module_import_remap:
                self.module_import_as_remap[name.asname] = name.name

        return self.generic_visit(node)

    ###################################################################################################################
    def find_name_in_obfuscation(self, name: str):
        return [mod for mod, mod_remaps in self.module_func_remap.items() if name in mod_remaps]

    ###################################################################################################################
    def visit_ImportFrom(self, node):
        # Skip if not this pass
        if not self.definition_pass:
            return self.generic_visit(node)

        original_module_name = node.module
        module_func_map = self.module_func_remap.get(original_module_name, {})

        if node.module is None:
            return self.generic_visit(node)

        # Simple remapping import X -> import new_name
        if node.module in self.module_import_remap:
            node.module = self.module_import_remap[node.module]

        # Go through each import
        for name in node.names:
            # Check if the name we are importing has been remapped.
            if name.name in module_func_map:
                # Only include it locally if not renamed.
                if name.asname is None:
                    self.module_func_remap[self.current_transform_module][name.name] = module_func_map[name.name]

                    if self.skip_script and self.current_transform_module == "__script__":
                        name.asname = name.name

                name.name = module_func_map[name.name]

            else:
                # It might be the import from a public -> private impl.
                # this code checks if we can find a single impl.
                found_mods = self.find_name_in_obfuscation(name.name)

                if len(found_mods) == 1:
                    store_name = name.name if name.asname is None else name.asname

                    obfuscated_name = self.module_func_remap[found_mods[0]][name.name]
                    self.module_func_remap[self.current_transform_module][store_name] = obfuscated_name
                    name.name = module_func_map[name.name] = obfuscated_name

        return self.generic_visit(node)

    ###################################################################################################################
    @staticmethod
    def find_inner_func_defs(node) -> List[ast.FunctionDef]:
        inner_func_defs = list()
        search_space = list()
        search_space.extend(node.body)

        while len(search_space) > 0:
            current_node = search_space.pop()

            if isinstance(current_node, ast.FunctionDef):
                inner_func_defs.append(current_node)
                search_space.extend(current_node.body)
            elif isinstance(current_node, ast.AST) and hasattr(current_node, "body"):
                search_space.extend(current_node.body)

        return inner_func_defs

    ###################################################################################################################
    def visit_FunctionDef(self, node):
        # Skip this pass.
        if not self.definition_pass:
            return self.generic_visit(node)

        # We might not want to mangle func for CLIs
        if self.skip_script and self.current_transform_module == "__script__":
            return self.generic_visit(node)

        # Skipping local __<func>__ and <func>_decorator functions
        if (len(node.name) > 4 and node.name[:2] == "__" and node.name[-2:] == "__") or (
            len(node.name) > 10 and node.name.endswith("_decorator")
        ):
            inner_func_defs = self.find_inner_func_defs(node)
            for inner_func in inner_func_defs:
                self.skip_func_defs.add(inner_func.name)

            return self.generic_visit(node)

        # We want to strip cache keywords from numba jitted functions.
        if len(node.decorator_list) > 0:
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                    if decorator.func.id == "jit":
                        if hasattr(decorator, "keywords"):
                            decorator.keywords = [k for k in decorator.keywords if k.arg != "cache"]
                            return self.generic_visit(node)

        # Skip functions we marked:
        #   FuncDefs in FuncDefs
        #   Class methods (it a pain to track)
        if node.name in self.skip_func_defs:
            return self.generic_visit(node)

        # functions open to available to to current module
        func_map = self.module_func_remap[self.current_transform_module]

        random_name = self.random_name(node.name)
        original_name = node.name

        # bi-directional map.
        func_map[node.name] = random_name
        func_map[random_name] = original_name
        node.name = random_name

        # We support a single level scope level for function defs.
        # this can be improved.
        # We don't want to deal with func defs within a closer. Yet?
        inner_func_defs = self.find_inner_func_defs(node)
        for inner_func in inner_func_defs:
            self.skip_func_defs.add(inner_func.name)

        return self.generic_visit(node)

    ###################################################################################################################
    @staticmethod
    def get_func_module(node) -> Optional[Tuple[str, str]]:
        if isinstance(node.func, ast.Attribute):
            module_node = node.func.value
            func = node.func.attr

            module = []

            while isinstance(module_node, ast.Attribute):
                module.append(module_node.attr)
                module_node = module_node.value

            if isinstance(module_node, ast.Name) and hasattr(module_node, "id"):
                module.append(module_node.id)

            # build back the module by reversing
            return ".".join(module[::-1]), func

        return None

    ###################################################################################################################
    def visit_Call(self, node):
        if self.definition_pass:
            return self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            module, func_name = self.get_func_module(node)

            if (
                module not in self.module_import_as_remap
                and module not in self.module_import_remap
                and module not in self.class_to_module
                and module != "self"
            ):
                # It's possible the func is called like a module via decorators.
                # e.g click.pass_context
                if module in self.module_func_remap[self.current_transform_module]:
                    node.func.value.id = self.module_func_remap[self.current_transform_module][module]

                return self.generic_visit(node)

            # function is in this module. Aliasing might be an issue.
            # We don't do classes yet so this will be skipped.
            if module == "self" and func_name in self.module_func_remap[self.current_transform_module]:
                node.func.attr = self.module_func_remap[self.current_transform_module][func_name]
                return self.generic_visit(node)

            # Remapping as
            if module in self.module_import_as_remap:
                # 1) as_module -> obfuscated_module -> orginial_module
                # 2) module func -> obfuscated func name

                import_module = self.module_import_as_remap[module]
                original_module = self.module_import_remap[import_module]
                module_func_mapping = self.module_func_remap[original_module]

                original_func_name = node.func.attr
                node.func.attr = module_func_mapping.get(node.func.attr, original_func_name)
            elif module in self.module_import_remap:
                module_func_mapping = self.module_func_remap[module]
                node.func.value = ast.Name(id=self.module_import_remap[module], ctx=ast.Load(), lineno=0, col_offset=0)

                node.func.attr = module_func_mapping.get(node.func.attr, node.func.attr)

        elif isinstance(node.func, ast.Name):
            call_name = node.func.id
            # When skip_script, the script keeps import aliases as original names (e.g. unpack_optional),
            # so we must not rewrite call sites in the script or we'd call fake_unpack_optional which is not bound.
            if self.skip_script and self.current_transform_module == "__script__":
                pass
            else:
                # Note: current_module include imports from other modules.
                module_func_mapping = self.module_func_remap[self.current_transform_module]
                if call_name in module_func_mapping:
                    node.func.id = module_func_mapping[call_name]

        return self.generic_visit(node)

    ###################################################################################################################
    def get_remapping(self):
        return self.module_import_remap

    ###################################################################################################################
    def add_remap_module(self, module: str):
        if module not in self.module_import_remap:
            random_module = self.random_name(module)
            self.module_import_remap[module] = random_module
            self.module_import_remap[random_module] = module

    ###################################################################################################################
    def generic_visit(self, node):
        if not self.debug:
            # Remove file tracing offsets.
            if hasattr(node, "lineno"):
                node.lineno = 0
            if hasattr(node, "col_offset"):
                node.col_offset = 0
            if hasattr(node, "end_lineno"):
                node.end_lineno = 0
            if hasattr(node, "end_col_offset"):
                node.end_col_offset = 0

            # Removing doc strings
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (
                    node.body
                    and len(node.body) > 1
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Str)
                ):
                    node.body.pop(0)

        return super().generic_visit(node)
