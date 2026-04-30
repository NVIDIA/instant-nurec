@echo off

echo ======== NREND build starting: %time%

call %~dp0\setenv.bat

echo ======== Fetching repo_man: %time%
git clone ^
    --no-tags ^
    --single-branch ^
    https://%GITLAB_USER%:%GITLAB_TOKEN%@gitlab-master.nvidia.com/omniverse/repo/repo_man.git ^
    "%REPO_DIR%" || exit /b 1

echo ======== Updating git submodules: %time%
git -C "%REPO_DIR%" submodule update --init --recursive || exit /b 1

echo ======== Configuring NREND: %time%
cmake ^
    -G "%GENERATOR%" ^
    -DCMAKE_BUILD_TYPE=%FLAVOUR% ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" ^
    -DCMAKE_INSTALL_PREFIX="%OUT_DIR%\nrend" ^
    -DNREND_DEPS_BUILD_DIR="build" ^
    -DNREND_TCNN_DIR="%TCNN_DIR%" ^
    -DNREND_OBFUSCATE_HEADERS=ON ^
    -DSLANG_DIR="%SLANG_DIR%" ^
    -S "%ROOT_DIR%" ^
    -B "%ROOT_DIR%\build" || exit /b 1

echo ======== Building NREND: %time%
cmake --build "%ROOT_DIR%\build" || exit /b 1
cmake --install "%ROOT_DIR%\build" || exit /b 1

echo ======== Computing VERSION: %time%
python "%ROOT_DIR%\ci\windows\version_string.py" > "%ROOT_DIR%\VERSION"
type "%ROOT_DIR%\VERSION"

echo ======== Copying repo_man tools: %time%
cmake -E copy "%REPO_DIR%\repo.bat" .  || exit /b 1
cmake -E copy_directory "%REPO_DIR%\tools" tools || exit /b 1

echo ======== Uploading debug symbols: %time%
call repo.bat symstore
cmake -E remove "%OUT_DIR%\nrend\lib\nrend.pdb" || exit /b 1

echo ======== Creating repo_man package: %time%
call repo.bat upload -p || exit /b 1

echo ======== Contents of final package: %time%
7z l _build\packages\*.7z

echo ======== NREND build finished: %time%
