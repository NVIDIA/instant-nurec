<!-- Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved. -->

# Obfuscation in NRE Repo

This file covers some details on how obfuscation works in this repo.

## How it works

Currently obfuscation works by going through the script you wish to obfuscate and parsing the AST of that file then identifying any modules we wish to obfuscate recursively.

One we have all the modules, we mangle the AST of each module by removing certain source code information. Additionally we mangle the import names and any free-standing function names.

We then compile each AST into a python PyCodeObject using the compile function and marshal the bytes into a C++ header. Next we create a C++ binary that loads these PyCodeObject using the Python C-api.

Finally we trigger the main script via the compiled PyCodeObject also using the C-api.

## How to obfuscate a script

The obfuscation script takes in a yaml config that defines what script and modules we want to obfuscate and the path to those files.

The current configs can be found in:

    scripts/pycena/configs

To obfuscate the script we run:

    python3 obfuscate_script.py --config <path_to_config>

## Debugging

If for some reason the obfuscated binaries have issues with imports it will cryptically fail (which makes sense). To debug these issues make sure to set the `debug: True` and `dump_ast: True` in the config.

This will cause the binary to build with debug messages and objects which have their names mangled will have it done in a fairly human readable form of `fake_<original name>` where `<original_name>` might have some character spawned with '\_'.

Additionally an `<module>_ast.txt` will be generated for each PyCodeObject. This might be helpful when debugging why some mangled values cannot be found.

## Example of Building and Running Obfuscated Targets in Images

In this example we will show the obfuscation of the `//apps:nre_tools` target.

The first step is to build the image containing the obfuscated target:

    bazel run //apps:load_obfuscated_nre_tools_image_oci

This will register the image in the local system's docker daemon as the tag `nvcr.io/nvidian/ct-toronto-ai/nre_obfuscated_tools:latest`.

The obfuscated target within the image is registered as the entry-point of the image and can be run with, e.g.,

    docker run -it --rm --gpus all nvcr.io/nvidian/ct-toronto-ai/nre_obfuscated_tools:latest --help
