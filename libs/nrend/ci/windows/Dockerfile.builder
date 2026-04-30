# Use the official Windows Server Core image
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Set default shell to cmd
SHELL ["cmd", "/S", "/C"]

# Set environment variables to enable Windows compression for Chocolatey
ENV ChocolateyUseWindowsCompression=true

# Download and install Chocolatey
RUN powershell -NoProfile -ExecutionPolicy Bypass -Command " \
    $env:chocolateyUseWindowsCompression='true'; \
    Invoke-WebRequest -Uri https://community.chocolatey.org/install.ps1 -OutFile install.ps1; \
    Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File install.ps1' -Wait -NoNewWindow; \
    Remove-Item -Force install.ps1"

# Set path to include choco and cmake
RUN setx PATH "%PATH%;C:\ProgramData\chocolatey\bin;C:\Program Files\Cmake\bin"

# Install git, cmake, python, and 7zip
RUN choco install git -y --force && \
    choco install cmake --version=3.29.8 -y --force && \
    choco install python -y --force && \
    choco install 7zip -y --force

# Install Visual Studio 2019 via choco
RUN choco install visualstudio2019-workload-vctools -y --no-progress

# Install CUDA 11.8 via choco
RUN choco install cuda --version=11.8.0.52206 -y --force

# Set default work directory
WORKDIR "/workdir"

# Set default command to cmd
CMD ["cmd"]
