set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CROSSCOMPILING TRUE)

set(compiler_name gcc)
set(target_arch aarch64-linux-gnu)
set(CMAKE_LIBRARY_ARCHITECTURE ${target_arch} CACHE STRING "" FORCE)

set(CMAKE_FIND_ROOT_PATH "/usr/${target_arch}")

find_program(CMAKE_C_COMPILER NAMES ${target_arch}-gcc NO_CMAKE_FIND_ROOT_PATH)
if(NOT CMAKE_C_COMPILER)
    message(FATAL_ERROR "Cannot find a C cross-compiler targetting ${CMAKE_LIBRARY_ARCHITECTURE}")
endif()
find_program(CMAKE_CXX_COMPILER NAMES ${target_arch}-g++ NO_CMAKE_FIND_ROOT_PATH)
if(NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "Cannot find a C++ cross-compiler targetting ${CMAKE_LIBRARY_ARCHITECTURE}")
endif()

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/${target_arch}-g++)
set(CUDAToolkit_ROOT /usr/local/cuda/targets/aarch64-linux/)
set(CMAKE_CUDA_FLAGS "-ccbin ${CMAKE_CXX_COMPILER} -Xcompiler -fPIC" CACHE STRING "" FORCE)

set(CMAKE_AR ${target_arch}-ar CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB ${target_arch}-ranlib)
set(CMAKE_LINKER ${target_arch}-ld)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
