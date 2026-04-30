"Wrapper macro to define default layer groups for python applications"

def default_layer_groups():
    return {
        "000_interpreter": "\\\\.runfiles/[^/]+\\\\+\\\\+python\\\\+python_[^/]+/",  # Python interpreter, typically at .runfiles/rules_python++python+python_3_11_x86_64-unknown-linux-gnu
        "001_base_system": "\\\\.runfiles/.*/site-packages/(numpy|scipy|pillow|requests|urllib3|setuptools|wheel|pip)",  # Stable system packages
        "002_ml_frameworks": "\\\\.runfiles/.*/site-packages/(torch|torchvision|pytorch_lightning|kornia|einops)",  # ML frameworks (large, stable)
        "003_cuda_libs": "\\\\.runfiles/.*/site-packages/nvidia_.*",  # NVIDIA CUDA libraries (large, stable)
        "004_model_files_trt": "\\\\.runfiles/.*trt_pretrained_models_repo/.*\\\\.(engine|ckpt|pt|pth|onnx)$$",  # TensorRT model files (large, stable)
        "005_model_files": "\\\\.runfiles/.*pretrained_models_repo/.*\\\\.(engine|ckpt|pt|pth|onnx)$$",  # Other model files (large, stable)
        "006_rest_packages_a_p": "\\\\.runfiles/.*/site-packages/[a-p].*",  # Rest of the packages, package names starting with A-P
        "007_rest_packages_q_z": "\\\\.runfiles/.*/site-packages/[q-z].*",  # Rest of the packages, package names starting with Q-Z
        "008_libtorch": "\\\\.runfiles/.*/libtorch.*",  # LibTorch
        "009_system_so": "\\\\.runfiles/_main/_solib_.*$$",  # System shared libraries
        "010_nre_libs": "\\\\.runfiles/_main/.*\\\\.(slang-module|so)$$",  # Compiled NRE libraries and modules
        "011_nre_code_proto": "\\\\.runfiles/_main/.*\\\\.(proto)$$",  # NRE Proto code (changes frequently)
        "012_nre_code_slang": "\\\\.runfiles/_main/.*\\\\.slang$$",  # NRE Slang code (changes frequently)
        "013_nre_markdown": "\\\\.runfiles/_main/.*\\\\.(md)$$",  # NRE Markdown code (changes frequently)
        "014_nre_code_cu": "\\\\.runfiles/_main/.*\\\\.(cu|cuh)$$",  # NRE CUDA code (changes frequently)
        "015_nre_code_cpp": "\\\\.runfiles/_main/.*\\\\.(cpp|h|hpp)$$",  # NRE C++ code (changes frequently)
        "016_nre_code_py": "\\\\.runfiles/_main/.*\\\\.(py|pyi)$$",  # NRE Python code (changes frequently)
        "017_nre_code_sh": "\\\\.runfiles/_main/.*\\\\.(sh|bash)$$",  # NRE Shell code (changes frequently)
        "018_configs": "\\\\.runfiles/_main/configs/.*",  # Configuration files (changes frequently)
        "019_nre_other": "\\\\.runfiles/_main/.*",  # Remaining NRE files, ex. obfuscated code (changes frequently)
    }
