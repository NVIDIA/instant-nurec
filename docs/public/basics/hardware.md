# Hardware Setup and Requirements

To use Neural Reconstruction (NuRec), you need **at least one (1) NVIDIA GPU
with CUDA support (version 12.8 or higher)** and more than 24GB of memory on a **Linux x86_64 operating system**.

We recommend a system with **more than 48GB of memory**. Currently, we do not support Linux aarch64 systems.

NuRec supports the following GPU architectures and corresponding recommended drivers:

**Ampere:**

- Boards: A100 | A10 | A40 | RTX A6000
- GPUs (Codenames): GA100 | GA102
- Drivers: R550 or later required, R570 or later recommended

**Ada Lovelace:**

- Boards: L20 | L40 | L40S
- GPU (Codename): AD102
- Drivers: R550 or later required, R570 or later recommended

**Grace Hopper:**

- Boards: H20 | H100
- GPU (Codename): GH100
- Drivers: R550 or later required, R570 or later recommended

**Blackwell:**

- Board: RTX Pro 6000D
- GPU (Codename): GB202
- Drivers: R580 or later required

See [software setup and requirements](software) for more details on software and driver requirements.
