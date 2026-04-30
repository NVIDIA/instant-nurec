@echo off

:: Update these with your own gitlab credentials
if not defined GITLAB_USER set GITLAB_USER=???
if not defined GITLAB_TOKEN set GITLAB_TOKEN=???

set NRE_ROOT=%CD%\..\..
set DOCKER_IMAGE=nrendimage
set DOCKER_CONTAINER=nrendbuilder

echo ======== Building docker image: %time%
docker build ^
    -t %DOCKER_IMAGE% ^
    -f "%NRE_ROOT%\libs\nrend\ci\windows\Dockerfile.builder" ^
    "%NRE_ROOT%\libs\nrend\ci\windows" || exit /b 1

echo ======== Creating docker container: %time%
docker stop %DOCKER_CONTAINER%
docker rm %DOCKER_CONTAINER%
docker create ^
    --cpus 8 ^
    --memory 16GB ^
    -it ^
    --name %DOCKER_CONTAINER% ^
    %DOCKER_IMAGE% || exit /b 1

echo ======== Copying NRE into container: %time%
docker cp "%NRE_ROOT%" %DOCKER_CONTAINER%:/workdir || exit /b 1

echo ======== Starting docker container: %time%
docker start %DOCKER_CONTAINER% || exit /b 1

echo ======== Setting gitlab token in container: %time%
docker exec ^
    %DOCKER_CONTAINER% ^
    cmd /c "setx GITLAB_USER %GITLAB_USER% && setx GITLAB_TOKEN %GITLAB_TOKEN%"

echo ======== Starting docker build of NREND deps: %time%
docker exec ^
    %DOCKER_CONTAINER% ^
    cmd /c "cd nre\libs\nrend && ci\windows\build_deps.bat" || exit /b 1

echo ======== Starting docker build of NREND library: %time%
docker exec ^
    %DOCKER_CONTAINER% ^
    cmd /c "cd nre\libs\nrend && ci\windows\build_nrend.bat" || exit /b 1

echo ======== Copying package out of container: %time%
docker stop %DOCKER_CONTAINER%
docker cp %DOCKER_CONTAINER%:/workdir/nre/libs/nrend/_build/ %CD%/ || exit /b 1

echo ======== Removing docker container: %time%
docker rm %DOCKER_CONTAINER%
