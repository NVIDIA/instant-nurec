# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-torch SE3/SO3 shim — small subset of the ``lietorch`` API.

We *would* have used ``pip install lietorch``, but the public PyPI
``lietorch==0.8.2`` (the latest release) hard-pins ``torch==2.6.*`` and
``torchvision==0.21.*``. This project pins ``torch==2.7.0+cu128`` for
compute-stack stability; pulling lietorch would force a torch downgrade
and break the cu128 wheel choice.

To avoid the version clash we re-implement only the small subset of the
lietorch API the predict path actually exercises:

* ``SE3(data)`` / ``SE3.InitFromVec(data)`` — construct from
  ``(..., 7)`` ``[tx, ty, tz, qx, qy, qz, qw]``.
* ``.data`` / ``.vec()`` — return underlying tquat tensor.
* ``.shape`` / ``.dtype`` / ``.device`` / ``__getitem__``.
* ``.inv()``.
* ``SE3 * SE3`` (composition) and ``SE3 * (..., 3) tensor`` (transform points).
* ``SO3(data)`` / ``SO3.InitFromVec(data)`` — quaternion XYZW.
* ``SO3.exp(omega)`` / ``SO3.log()``.
* ``SO3 * SO3`` (composition).
* ``SO3.inv()``.

Quaternion convention: XYZW (matches lietorch and ncore).

All intellectual credit for the lietorch interface, the SE3/SO3 group
algebra and the underlying numerics belongs to the original lietorch
authors:

    Zachary Teed and Jia Deng
    https://github.com/princeton-vl/lietorch

If you need the full lietorch surface (Lie-group derivatives, autograd
hooks, optimizer interop), prefer the upstream package on a torch
version it supports.
"""

from __future__ import annotations

import torch

from instant_nurec.utils.geometry import quat_mult_xyzw


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply XYZW quaternion ``q`` to vector ``v``: ``v' = q v q^{-1}``."""
    q = _quat_normalize(q)
    qv = q[..., :3]
    qw = q[..., 3:]
    t = 2 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_xyzw_slerp(
    quat_s: torch.Tensor, quat_e: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Batched SLERP between two unit quaternions (XYZW).

    Mirrors ``ncore.impl.sensors.common.unitquat_slerp`` (shortest-arc).
    """
    cos_omega = torch.sum(quat_s * quat_e, dim=-1)
    quat_e = torch.where((cos_omega < 0).unsqueeze(-1), -quat_e, quat_e)
    cos_omega = torch.abs(cos_omega)

    nearby = cos_omega > (1.0 - 1e-3)
    cos_omega = torch.clamp(cos_omega, -1.0 + 1e-6, 1.0 - 1e-6)
    omega = torch.acos(cos_omega)
    alpha = torch.sin((1 - t) * omega)
    beta = torch.sin(t * omega)
    alpha = torch.where(nearby, (1 - t), alpha)
    beta = torch.where(nearby, t, beta)
    quat = alpha.unsqueeze(-1) * quat_s + beta.unsqueeze(-1) * quat_e
    return quat / torch.norm(quat, dim=-1, keepdim=True)


def quat_xyzw_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """XYZW unit quaternion → 3x3 rotation matrix."""
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    R = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)
    R[..., 0, 0] = x2 - y2 - z2 + w2
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 1, 1] = -x2 + y2 - z2 + w2
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 2] = -x2 - y2 + z2 + w2
    return R


class SO3:
    """XYZW unit-quaternion rotation. Drop-in for ``lietorch.SO3``."""

    __slots__ = ("data",)

    def __init__(self, data: torch.Tensor):
        self.data = _quat_normalize(data) if data.numel() > 0 else data

    @classmethod
    def InitFromVec(cls, data: torch.Tensor) -> "SO3":
        return cls(data)

    @staticmethod
    def exp(omega: torch.Tensor) -> "SO3":
        """``omega: (..., 3)`` axis-angle vector → SO3 unit quaternion XYZW."""
        theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
        small = theta < 1e-6
        # Taylor expansion for small angle: q = (omega/2 - omega*theta^2/48, 1 - theta^2/8).
        half_theta = theta / 2
        sin_half = torch.where(small, half_theta - (half_theta**3) / 6, torch.sin(half_theta))
        cos_half = torch.where(small, 1 - (half_theta**2) / 2, torch.cos(half_theta))
        axis = omega / theta.clamp_min(1e-12)
        q_xyz = sin_half * torch.where(small, omega / 2, axis * sin_half / sin_half.clamp_min(1e-30) * sin_half)
        # Simpler: q_xyz = axis * sin_half — but axis is undefined when theta=0.
        # Use the "small" mask to fall back to the Taylor form.
        q_xyz = torch.where(small.expand_as(omega), omega * 0.5, axis * sin_half)
        q = torch.cat([q_xyz, cos_half], dim=-1)
        return SO3(q)

    def log(self) -> torch.Tensor:
        """Inverse of ``exp``: SO3 unit quaternion XYZW → ``(..., 3)`` axis-angle."""
        q = _quat_normalize(self.data)
        # Ensure shortest-arc: flip sign if w < 0 so the angle is in [0, pi].
        q = torch.where((q[..., 3:4] < 0).expand_as(q), -q, q)
        v = q[..., :3]
        w = q[..., 3:4]
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        small = v_norm < 1e-6
        # angle = 2 * atan2(|v|, w); axis = v / |v|; omega = angle * axis.
        # For small v, use Taylor: omega ≈ 2 * v / w.
        angle = 2 * torch.atan2(v_norm, w)
        axis = v / v_norm.clamp_min(1e-12)
        omega = torch.where(small.expand_as(v), 2 * v / w.clamp_min(1e-12), axis * angle)
        return omega

    def inv(self) -> "SO3":
        return SO3(_quat_conj(self.data))

    def vec(self) -> torch.Tensor:
        return self.data

    @property
    def shape(self) -> torch.Size:
        return self.data.shape[:-1]

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device

    def __getitem__(self, idx) -> "SO3":
        return SO3(self.data[idx])

    def __mul__(self, other):
        if isinstance(other, SO3):
            return SO3(quat_mult_xyzw(self.data, other.data))
        if isinstance(other, torch.Tensor) and other.shape[-1] == 3:
            return _quat_rotate(self.data, other)
        return NotImplemented

    def to(self, *args, **kwargs) -> "SO3":
        return SO3(self.data.to(*args, **kwargs))


class SE3:
    """SE(3) rigid transform stored as ``(..., 7)`` ``[tx, ty, tz, qx, qy, qz, qw]``.
    Drop-in for ``lietorch.SE3``.
    """

    __slots__ = ("data",)

    def __init__(self, data: torch.Tensor):
        # Normalize the rotation part if there's any data.
        if data.numel() == 0:
            self.data = data
            return
        t = data[..., :3]
        q = _quat_normalize(data[..., 3:])
        self.data = torch.cat([t, q], dim=-1)

    @classmethod
    def InitFromVec(cls, data: torch.Tensor) -> "SE3":
        return cls(data)

    def vec(self) -> torch.Tensor:
        return self.data

    def translation(self) -> torch.Tensor:
        """``(..., 3)`` translation component."""
        return self.data[..., :3]

    def rotation(self) -> "SO3":
        """``SO3`` rotation component."""
        return SO3(self.data[..., 3:])

    @property
    def shape(self) -> torch.Size:
        return self.data.shape[:-1]

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device

    def __getitem__(self, idx) -> "SE3":
        return SE3(self.data[idx])

    def inv(self) -> "SE3":
        t = self.data[..., :3]
        q = self.data[..., 3:]
        q_inv = _quat_conj(q)
        t_inv = -_quat_rotate(q_inv, t)
        return SE3(torch.cat([t_inv, q_inv], dim=-1))

    def __mul__(self, other):
        if isinstance(other, SE3):
            t1 = self.data[..., :3]
            q1 = self.data[..., 3:]
            t2 = other.data[..., :3]
            q2 = other.data[..., 3:]
            # Composition: T1 * T2 has translation t1 + R1 * t2, rotation q1 * q2.
            t_out = t1 + _quat_rotate(q1, t2)
            q_out = quat_mult_xyzw(q1, q2)
            return SE3(torch.cat([t_out, q_out], dim=-1))
        if isinstance(other, torch.Tensor) and other.shape[-1] == 3:
            t = self.data[..., :3]
            q = self.data[..., 3:]
            return t + _quat_rotate(q, other)
        return NotImplemented

    def to(self, *args, **kwargs) -> "SE3":
        return SE3(self.data.to(*args, **kwargs))

    def cuda(self) -> "SE3":
        return SE3(self.data.cuda())

    def cpu(self) -> "SE3":
        return SE3(self.data.cpu())

    def detach(self) -> "SE3":
        return SE3(self.data.detach())
