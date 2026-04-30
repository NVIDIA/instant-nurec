<!--
  -- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  -- SPDX-License-Identifier: LicenseRef-NvidiaProprietary
  --
  -- NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
  -- property and proprietary rights in and to this material, related
  -- documentation and any modifications thereto. Any use, reproduction,
  -- disclosure or distribution of this material and related documentation
  -- without an express license agreement from NVIDIA CORPORATION or
  -- its affiliates is strictly prohibited.
  -->

# Modular Ansible Setup for NuRec

This directory contains modular Ansible playbooks for setting up machines for NuRec development and CI.

## Table of Contents

- [Development Setup](#development-setup)
  - [Compatibility Strategy](#compatibility-strategy)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
    - [Local Setup](#local-setup)
    - [Remote Setup](#remote-setup)
  - [Authentication Variables](#authentication-variables)
  - [Notes](#notes)
- [CI Setup](#ci-setup)
  - [Prerequisites](#prerequisites-1)
    - [Colossus target machines leases](#colossus-target-machines-leases)
    - [Control machine setup](#control-machine-setup)
    - [SSH access to target machines](#ssh-access-to-target-machines)
    - [GitLab setup](#gitlab-setup)
  - [Usage](#usage-1)
  - [Post-Installation Steps](#post-installation-steps)
- [Files](#files)

---

# Development Setup

## Compatibility Strategy

The development playbook (`setup-dev.yml`) aims to be compatible with supported NuRec versions (s. below) to minimize the need for machine reconfiguration when switching between branches/commits. The current compatibility approach:

- **Main and release branches**: Supported with the playbook version at the given commit, starting from v25.09. When switching between branches with different dependency requirements, re-running the `setup-dev.yml` playbook ensures environment compatibility. The playbook maintains backward compatibility by supporting dependencies from the previous release cycle alongside current requirements. Dependencies are deprecated and removed after one full release cycle to ensure seamless transitions between supported versions.
- **Feature branches**: Inherit compatibility from their base commit; branch owners are responsible for documenting any additional setup requirements

The playbook is designed to install minimum required versions rather than exact versions where possible (e.g., GPU drivers), and supports side-by-side installation of different versions for components like CUDA. This allows developers to work across multiple branches without frequent machine reconfiguration, while additional version-specific configuration (like setting CUDA versions in `.bazelrc`) remains the developer's responsibility.

## Prerequisites

- Target machine running Ubuntu 22.04
- Ansible installed on control machine
- SSH access to the target machine (for remote setup)

## Usage

### Local Setup

For setting up your current machine directly:

```bash
# Run development machine setup locally (basic)
ansible-playbook -i 'localhost,' --connection=local internal/scripts/ansible/setup-dev.yml

# Run locally with authentication tokens
ansible-playbook -i 'localhost,' --connection=local internal/scripts/ansible/setup-dev.yml \
  -e gitlab_token="your_gitlab_token" \
  -e urm_token="your_urm_token" \
  -e nvidia_username="your_nvidia_username"

# Run locally with only some authentication tokens (others will be skipped)
ansible-playbook -i 'localhost,' --connection=local internal/scripts/ansible/setup-dev.yml \
  -e gitlab_token="your_gitlab_token"
```

The trailing comma is required for Ansible to treat it as a list.

### Remote Setup

You can run the playbook directly on a single host using the `-i` flag with an inline inventory.
For setting up a remote machine over SSH:

```bash
# Test connection
ansible all -i '<hostname>,' --ask-pass -m ping

# Run development machine setup (basic)
ansible-playbook -i '<hostname>,' --ask-pass internal/scripts/ansible/setup-dev.yml

# Run development machine setup with authentication tokens
ansible-playbook -i '<hostname>,' --ask-pass internal/scripts/ansible/setup-dev.yml \
  -e gitlab_token="your_gitlab_token" \
  -e urm_token="your_urm_token" \
  -e nvidia_username="your_nvidia_username"
```

Replace `<hostname>` with your target machine's address, e.g. IP or FQDN. The trailing comma is required for Ansible to treat it as a list.

**Note:** You can use SSH key authentication instead of password authentication. Run `ssh-copy-id <username>@<hostname>` to add your public key to the list of authorized keys on the remote machine and drop `--ask-pass`.

## Authentication Variables

The development setup supports optional authentication configuration:

- `gitlab_token` - GitLab Personal Access Token for accessing GitLab Container Registry and packages
- `urm_token` - URM/Artifactory Personal Access Token for accessing NVIDIA internal packages
- `nvidia_username` - Your NVIDIA domain username (required for URM authentication)

These variables are optional. If not provided, the corresponding `.netrc` entries will be skipped.

**Note:** When `gitlab_token` is provided, the playbook will automatically perform `docker login` to the GitLab Container Registry (`gitlab-master.nvidia.com:5005`), enabling immediate access to Docker images.

## Notes

- The playbook will install NVIDIA drivers, Docker, the container toolkit, CUDA toolkit, Bazel, build dependencies, and configure authentication.
- The machine will be rebooted at the end to ensure all drivers and services are properly loaded.

---

# CI Setup

## Prerequisites

### Colossus target machines leases

- Colossus target machine leases need to be created through NCA account `nurec-ci`
  - Ask other CI admins to grant you access to this account if necessary
- Machines should be provisioned with OS image `ubuntu-22.04-x86_64-standard-uefi`

### Control machine setup

- Need Ansible installed. This will be installed by default if the control machine is itself on Colossus
- GitLab API access with token stored in .netrc per [authentication instructions](../../../README.md#authentication)

#### SSH access to target machines

Colossus target machines are set up with a local account for `nurec-ci`. Its username is `local-HvujS85MaqVjpd`, relying
on a shortened form of the NCA account ID. The password for each machine can be read from the Colossus "My Leases"
dashboard of user `nurec-ci`.

Ansible supports explicitly passing a single password per playbook invocation with the `--ask-pass` command line
argument. This approach can be used for configuring machines one-by-one.

However to ease runner setup and debugging access, it is recommended that individual CI admins set up SSH public key
authentication from their control machine for this local account.

Example `.ssh/config` entry:

```
Host 1u1g-spr-0164.ipp2a2.colossus.nvidia.com <other machines>
    User local-HvujS85MaqVjpd
    IdentityFile ~/.ssh/id_ecdsa_nurec-ci
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
```

Example creation and installation of the SSH key:

```bash
ssh-keygen -t ecdsa -f ~/.ssh/id_ecdsa_nurec-ci
ssh-copy-id -o IdentitiesOnly=yes -i ~/.ssh/id_ecdsa_nurec-ci.pub \
  1u1g-spr-0164.ipp2a2.colossus.nvidia.com
<repeat ssh-copy-id for other machines>
```

### GitLab setup

For registration of new runners, or renewing registration after wiping out the OS:

- The user running the script must have Owner role for the NRS group and Maintainer (or higher) on the Alpasim and Sauron projects.
  - Only machines marked `nrs_and_alpasim` have Alpasim runners, requiring Maintainer permissions on the Alpasim project.
  - Only machines marked `nrs_and_sauron` have Sauron runners, requiring Maintainer permissions on the Sauron project.

In case of SW update for machines already registered as GitLab runners:

- Set the machines to "Disabled" in Settings -> CI/CD -> Runners for the NRS group and, if the Alpasim or Sauron runner is enabled on those machines, for the respective project as well. Disabling only on the NRS side is not sufficient when multiple runners share the same host.
- Wait for jobs to complete before running the playbook.

## Usage

### 1. Run the Playbook

```bash
# Test connection to all machines in the inventory (using SSH keys)
ansible all -i inventory-ci.yml -m ping

# Test connection to a single machine named `colossus-gitlab-runner-l40s-1` in the inventory (with SSH password prompt)
ansible all -i inventory-ci.yml -m ping --limit colossus-gitlab-runner-l40s-1 --ask-pass

# Run the playbook on the whole inventory
ansible-playbook -i inventory-ci.yml setup-ci.yml

# Run the playbook, limiting processing to the machine named `colossus-gitlab-runner-l40s-1`
ansible-playbook -i inventory-ci.yml setup-ci.yml --limit colossus-gitlab-runner-l40s-1
```

### 2. Post-Installation Steps

The playbook handles SW installation and runner registration automatically.

After the playbook completes successfully it remains the user's responsibility to validate the setup
and enable the runner in production by replacing the original tag `test` defined by the automation.

---

# Files

- `setup-common.yml` - Common playbook for basic setup (NVIDIA drivers, Docker, container toolkit)
- `setup-ci.yml` - Complete CI setup (includes common setup + GitLab runner installation)
- `setup-dev.yml` - Complete development setup (includes common setup + dev-setup tasks)
- `register_gitlab_runner.yml` - Reusable flow to register a runner (used by setup-ci)
- `inventory-ci.yml` - Inventory file for CI machines
- `vars-ansible.yml` - Common Ansible configuration variables
- `README.md` - This file
