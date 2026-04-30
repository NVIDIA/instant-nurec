@echo off

if defined VisualStudioVersion (
    echo ======== Skipping VS env setup
    goto vsdone
)

echo ======== Setting VS 2019 x64 build env

set VCVARS64="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist %VCVARS64% (
    call %VCVARS64%
    goto vsdone
)

set VCVARS64="C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS64%

:vsdone

set CUDA_PATH=%CUDA_PATH_V11_8%
set CUDA_HOME=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

set ROOT_DIR=%CD%
set OUT_DIR=%ROOT_DIR%\out
set DEP_DIR=%OUT_DIR%\dependencies
set SLANG_DIR=%DEP_DIR%\slang
set TCNN_DIR=%DEP_DIR%\tcnn
set REPO_DIR=%DEP_DIR%\repo_man
set GENERATOR=Ninja
set FLAVOUR=RelWithDebInfo

echo ======== Config for windows build
echo ROOT_DIR:  %ROOT_DIR%
echo OUT_DIR:   %OUT_DIR%
echo DEP_DIR:   %DEP_DIR%
echo SLANG_DIR: %SLANG_DIR%
echo TCNN_DIR:  %TCNN_DIR%
echo REPO_DIR:  %REPO_DIR%
echo GENERATOR: %GENERATOR%
echo FLAVOUR:   %FLAVOUR%
echo CUDA_PATH: %CUDA_PATH%

echo ======== CUDA version
where nvcc
nvcc --version

echo ======== cl version
where cl
cl 2>&1 | findstr "Version"

echo ======== git version
where git
git --version

echo ======== python version
where python
python --version

echo ======== ninja version
where ninja
ninja --version

echo ======== cmake version
where cmake
cmake --version 2>&1 | findstr "version"
