@echo off
setlocal enabledelayedexpansion

echo ======== NREND deps build starting: %time%

call %~dp0\setenv.bat

echo ======== Extracting Slang version from MODULE.bazel: %time%
for /f "tokens=*" %%p in ('git rev-parse --show-toplevel') do set NRE_PATH=%%p
set MODULE_FILE=%NRE_PATH%\MODULE.bazel

if not exist "%MODULE_FILE%" (
    echo Error: MODULE.bazel not found at %MODULE_FILE%
    exit /b 1
)

for /f "tokens=*" %%i in ('findstr /R "releases/download/v.*slang.*tar.gz" "%MODULE_FILE%"') do (
    set LINE=%%i
    set "LINE=!LINE:*releases/download/=!"
    for /f "tokens=1 delims=/" %%v in ("!LINE!") do (
        set SLANG_VERSION=%%v
        goto :found_version
    )
)
:found_version
if not defined SLANG_VERSION (
    echo Error: Could not extract Slang version from %MODULE_FILE%
    exit /b 1
)
echo Extracted Slang version: %SLANG_VERSION%

echo ======== Fetching Slang: %time%
git clone ^
    -c advice.detachedHead=false ^
    --no-tags ^
    --single-branch ^
    --branch %SLANG_VERSION% ^
    https://github.com/shader-slang/slang.git ^
    "%SLANG_DIR%" || exit /b 1

echo ======== Fetching tiny-cuda-nn: %time%
git clone ^
    --no-tags ^
    --single-branch ^
    --branch publish/nrend ^
    https://%GITLAB_USER%:%GITLAB_TOKEN%@gitlab-master.nvidia.com/nrs/nre_external/tiny-cuda-nn.git ^
    "%TCNN_DIR%" || exit /b 1

echo ======== Updating git submodules: %time%
git -C "%SLANG_DIR%" submodule update --init --recursive || exit /b 1
git -C "%TCNN_DIR%" submodule update --init --recursive || exit /b 1

echo ======== Configuring Slang: %time%
cmake ^
    -G "%GENERATOR%" ^
    -DCMAKE_BUILD_TYPE=%FLAVOUR% ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" ^
    -DSLANG_ENABLE_EXAMPLES=OFF ^
    -DSLANG_SLANG_LLVM_FLAVOR=DISABLE ^
    -DSLANG_ENABLE_SPLIT_DEBUG_INFO=OFF ^
    -DSLANG_ENABLE_RELEASE_DEBUG_INFO=OFF ^
    -DSLANG_LIB_TYPE=STATIC ^
    -S "%SLANG_DIR%" ^
    -B "%SLANG_DIR%\build" || exit /b 1

echo ======== Configuring tiny-cuda-nn: %time%
cmake ^
    -G "%GENERATOR%" ^
    -DCMAKE_BUILD_TYPE=%FLAVOUR% ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" ^
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON ^
    -DTCNN_BUILD_WITH_UNALIGNED_PARAMS=ON ^
    -DTCNN_CUDA_ARCHITECTURES="75;80;86;89;90" ^
    -DTCNN_BUILD_EXAMPLES=OFF ^
    -DTCNN_BUILD_BENCHMARK=OFF ^
    -DTCNN_BUILD_TESTS=OFF ^
    -S "%TCNN_DIR%" ^
    -B "%TCNN_DIR%\build" || exit /b 1

echo ======== Building Slang: %time%
:: limit number of parallel jobs to avoid memory exhaustion
:: tested against VM running docker container with 4 CPUs and 12GB reserved
cmake --build "%SLANG_DIR%\build" -j 2 || exit /b 1

echo ======== Building tiny-cuda-nn: %time%
cmake --build "%TCNN_DIR%\build" || exit /b 1

echo ======== NREND deps build finished: %time%
