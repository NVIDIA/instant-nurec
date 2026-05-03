"""Pure-torch drop-in shim for the lietorch SE3/SO3 surface used by the
predict pipeline.

After Phase A.8 dropped ``libs/`` and Phase B aimed to drop ``bazel``,
``lietorch`` became the last native dep blocking ``pip install -e .``
on systems whose glibc is older than the pip wheel's requirement
(``GLIBC_2.36``). This module replaces the API surface used by
``tracks.py``, ``types.py``, and ``motion.py`` with pure torch:

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

Quaternion convention: XYZW (matches lietorch and the slang/ncore code).
"""

from __future__ import annotations

import torch


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def _quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``q1 * q2`` for XYZW quaternions."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply XYZW quaternion ``q`` to vector ``v``: ``v' = q v q^{-1}``."""
    q = _quat_normalize(q)
    qv = q[..., :3]
    qw = q[..., 3:]
    t = 2 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


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
            return SO3(_quat_mul_xyzw(self.data, other.data))
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
            q_out = _quat_mul_xyzw(q1, q2)
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
