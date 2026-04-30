# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import math

from typing import Callable, Literal, Optional

import torch
import torch.nn as nn

from omegaconf import DictConfig
from torch.autograd.function import once_differentiable

from libs.vren.interface import vren  # type: ignore
from nre.utils.misc import dataclass_items, get_pack_info_from_n, torch_interp1d, unpack_optional
from nre.utils.packed_ops import (
    linstep_interleave,
    packed_cumsum,
    packed_diff,
    packed_div,
    packed_invert_cdf,
    packed_weighted_sum,
)
from nre.utils.types import ExtraSignal, RadianceEmbeddingType, VolumeRenderingReturn


class RayAABBIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and axis-aligned voxels.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        aabb_blb: (N_voxels, 3) coordinates of the bottom-left-back corner of the AABBs
        aabb_trf: (N_voxels, 3) coordinates of the top-right-front corner of the AABBs

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit) \
            Note that the entering depth `hits_t[:, :, 0]` could be negative as they are the results of extended intersection compute.
    """

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, rays_o: torch.Tensor, rays_d: torch.Tensor, aabb_blb: torch.Tensor, aabb_trf: torch.Tensor):
        return vren.ray_aabb_intersect(rays_o, rays_d, aabb_blb, aabb_trf)


class RaySphereIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and spheres.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_spheres, 3) sphere centers
        radii: (N_spheres, 3) radii
        max_hits: maximum number of intersected spheres to keep for one ray

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit) \
            Note that the entering depth `hits_t[:, :, 0]` could be negative as they are the results of extended intersection compute.
        hits_sphere_idx: (N_rays, max_hits) hit sphere indices (-1 if no hit)
    """

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, rays_o, rays_d, center, radii, max_hits):
        return vren.ray_sphere_intersect(rays_o, rays_d, center, radii, max_hits)


class _TruncExp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, x, log_max: float = 15.0):
        ctx.save_for_backward(x)
        ctx.log_max = log_max
        return torch.exp(x.clamp(max=log_max))

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(max=ctx.log_max)), None


class TruncExp:
    """
    Truncate the forward and backward exponential operation by exp(x.clamp_max(log_max))
    - ZipNeRF uses np.log(np.float32(np.finfo(np.float32).max)) ~= 88.72
    - np.log(np.float32(np.finfo(np.float6).max)) ~= 11.08
    """

    @staticmethod
    def __call__(x, log_max: float = 15.0):
        return _TruncExp.apply(x, log_max)


trunc_exp = TruncExp()


class _TruncBCE(torch.autograd.Function):
    """
    Perform no-gradient-dead-zone truncated binary cross entropy.

    This function is similar to pytorch `binary_cross_entropy(x.clip(eps, 1 - eps), y)`,
    but it still carries truncated gradients on those clipped values to avoid dead zone.
    """

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, x, y, eps=0.001):
        # In pytorch BCE:
        # - log_eps = -100.0 => eps = exp(-100) = 3.7e-44
        #   It will lead to x.grad = -2.7e+43, which is extremely large.
        # In here:
        # - eps = 0.1, log_eps = -2.3 -> leads to x.grad = -9.9 when y=1 & x=0
        # - eps = 0.01, log_eps = -4.6 -> leads to x.grad = -99.9 when y=1 & x=0
        # - eps = 0.001, log_eps = -6.9 -> leads to x.grad = -999.9 when y=1 & x=0
        # For reference:
        # when y=1 & x=1, or y=0 & x=0, x.grad=-1 or +1 respectively
        log_eps = math.log(eps)
        x = torch.clip(x, 0, 1)
        y = torch.clip(y, 0, 1)
        ctx.save_for_backward(x, y)
        ctx.eps = eps
        return -1 * torch.where(y == 0, torch.log(1 - x).clamp_min_(log_eps), torch.log(x).clamp_min_(log_eps))

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        eps = ctx.eps
        grad = None
        if ctx.needs_input_grad[0]:
            # On y == 0, grad_x = 1 / (1 - x.clip(0, 1 - eps))
            # On y == 1, grad_x = -1 / x.clip(eps, 1)
            grad = torch.where(y == 0, 1 / (1 - torch.clip(x, 0, 1 - eps)), -1 / torch.clip(x, eps, 1))
            # No grad when x matches y
            grad = torch.logical_not(x == y) * grad * grad_output
        return grad, None, None


trunc_bce = _TruncBCE.apply


class AlphaCompositing(torch.autograd.Function):
    """
    Performs the alpha compositing of the values along the ray

    Inputs:
        alphas: (N)
        features: (N, D)
        ts: (N)
        pack_info: (N_rays, 2) start_idx, N_samples meaning each entry corresponds to the i-th ray, whose samples are [start_idx:start_idx+N_samples]
        transmittance_threshold: float, stop the ray if the transmittance is below it

    Outputs:
        n_samples: int, total effective samples
        opacity: (N_rays)
        distance: (N_rays)
        feature: (N_rays, D)
        ws: (N_rays) sample point weights
        samples: samples per ray (N_rays)
    """

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, alphas, features, ts, pack_info, transmittance_threshold):
        pack_info = pack_info.int()

        if pack_info.size(0) == 0:
            n_samples = 0
            opacity = distance = ws = torch.empty((0,), dtype=alphas.dtype, device=alphas.device)
            samples = torch.empty((0,), dtype=torch.long, device=alphas.device)
            feature = torch.empty((0, features.size(-1)), dtype=alphas.dtype, device=alphas.device)
        else:
            samples, opacity, distance, feature, ws = vren.alpha_composite_train_fw(
                alphas, features, ts, pack_info, transmittance_threshold
            )
            n_samples = samples.sum()

        ctx.save_for_backward(alphas, features, ts, pack_info, opacity, distance, feature, ws)
        ctx.transmittance_threshold = transmittance_threshold
        return n_samples, opacity, distance, feature, ws, samples

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dL_dn_samples, dL_dopacity, dL_ddistance, dL_dfeature, dL_dws, dL_dsamples):
        alphas, features, ts, pack_info, opacity, distance, feature, ws = ctx.saved_tensors

        if pack_info.size(0) == 0:
            # NOTE: [JG] Materialize to zeros instead of None, to allow for autograd.grad() usage
            dL_dalphas = torch.zeros_like(alphas)
            dL_dfeatures = torch.zeros_like(features)
        else:
            dL_dalphas, dL_dfeatures = vren.alpha_composite_train_bw(
                dL_dopacity,
                dL_ddistance,
                dL_dfeature,
                dL_dws,
                alphas,
                features,
                ws,
                ts,
                pack_info,
                opacity,
                distance,
                feature,
                ctx.transmittance_threshold,
            )
        return dL_dalphas, dL_dfeatures, None, None, None, None


def vol_rend_from_alphas(
    alphas: torch.Tensor,
    radiance_embedding_type: RadianceEmbeddingType,
    radiance_embedding_samples: torch.Tensor,
    ts: torch.Tensor,
    pack_info: torch.Tensor,
    transmittance_threshold: float = 1e-4,
    extra_ray_signals: Optional[ExtraSignal] = None,
    extra_ray_detach_weights: dict[str, bool] = {},
) -> VolumeRenderingReturn:
    """
    Perform volume rendering using precomputed alpha values
    """

    n_vr_samples, opacity, distance, radiance_embedding, weights, vr_samples = AlphaCompositing.apply(
        alphas, radiance_embedding_samples.to(alphas.dtype), ts, pack_info, transmittance_threshold
    )

    # Volume render extra ray signals if present
    # TODO: create a generalized volume renderer kernel to alpha-composite
    #       these in a single kernel invocation - see issue #78
    vr_extra_ray_signal: ExtraSignal | None = None
    if extra_ray_signals is not None:
        vr_extra_ray_signal = ExtraSignal(
            **{
                k: packed_weighted_sum(
                    v.contiguous(), weights.detach() if extra_ray_detach_weights.get(k, False) else weights, pack_info
                )
                for k, v in dataclass_items(extra_ray_signals)
                if isinstance(v, torch.Tensor)
            }
        )

    return VolumeRenderingReturn(
        n_vr_samples=n_vr_samples,
        pack_info=pack_info,
        opacity=opacity,
        distance=distance,
        radiance_embedding_type=radiance_embedding_type,
        radiance_embedding=radiance_embedding,
        sample_weights=weights,
        sample_transmittance=torch.empty_like(weights)
        if weights.numel() == 0
        else (weights / torch.clip(alphas, min=1e-10)),
        vr_samples=vr_samples,
        extra_ray_signals=vr_extra_ray_signal,
    )


def vol_rend_from_densities(
    sigmas: torch.Tensor,
    radiance_embedding_type: RadianceEmbeddingType,
    radiance_embedding_samples: torch.Tensor,
    deltas: torch.Tensor,
    ts: torch.Tensor,
    pack_info: torch.Tensor,
    transmittance_threshold: float,
    extra_ray_signals: Optional[ExtraSignal] = None,
    extra_ray_detach_weights: dict[str, bool] = {},
) -> VolumeRenderingReturn:
    """
    Perform volume rendering using precomputed alpha values
    """

    alphas = 1 - torch.exp(-sigmas * deltas)

    return vol_rend_from_alphas(
        alphas,
        radiance_embedding_type,
        radiance_embedding_samples,
        ts,
        pack_info,
        transmittance_threshold,
        extra_ray_signals,
        extra_ray_detach_weights,
    )


def ray_samples_in_distranges_masks(
    rays_samples_packinfo: torch.Tensor,
    rays_samples_t: torch.Tensor,
    rays_distranges_packinfo: torch.Tensor,
    rays_distranges_ts: torch.Tensor,
) -> torch.Tensor:
    """
    Given per-ray samples (encoded as distances along the ray) and per-ray distance ranges (encoded as start/end distances)
    compute a binary mask for each sample indicating if it is inside of *any* of the distance ranges

    Inputs:
    - rays_samples_packinfo: per ray sample packinfo with [sample_start_idx, N_samples_of_ray] N_rays x 2 [int]
    - rays_samples_t: distances of individual ray samples N_total_samples [float]
    - rays_distranges_packinfo: per ray distranges packinfo with [distrange_start_idx, N_distranges_of_ray] N_rays x 2 [int]
    - rays_distranges_ts: individual distranges given as start_distance/end_distance along the associated ray N_total_distranges x 2 [float]

    Returns:
    - rays_samples_distranges_cover: binary mask for each individual sample indicating if the sample is within at least a single distrange along it's ray, N_total_samples [bool]
    """

    return vren.ray_samples_in_distranges_masks(
        rays_samples_packinfo, rays_samples_t, rays_distranges_packinfo, rays_distranges_ts
    )


class WeightsFromAlphas(torch.autograd.Function):
    """
    Computes the weights of each sample from its alpha value using the NeRF volume rendering formulation

    Inputs:
        alphas: (N,) alphas for each sample along each ray in a packed representation
        pack_info: (N_rays, 2) start_idx, N_samples
        meaning each entry corresponds to the i-th ray,
        whose samples are [start_idx:start_idx+N_samples]
        transmittance_threshold: (float) transmittance threshold used for early stopping, when the remaining transmittance
        falls below this value, the alpha compositing will stop (weights 0 after this sample)

    Outputs:
        samples: (N_rays) total number of samples used for each ray (n_samples before eventual early stopping)
        opacity: (N_rays) accumulated opacity of each ray
        weights: (N) weights for each sample along each ray
    """

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, alphas, pack_info, transmittance_threshold):
        if pack_info.size(0) == 0:
            samples = opacity = weights = torch.empty((0,), dtype=alphas.dtype, device=alphas.device)
        else:
            samples, opacity, weights = vren.weights_from_alphas_fw(alphas, pack_info, transmittance_threshold)

        ctx.save_for_backward(alphas, pack_info, opacity, weights)
        ctx.transmittance_threshold = transmittance_threshold

        return samples, opacity, weights

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dL_dsamples, dL_dopacity, dL_dws):
        alphas, pack_info, opacity, weights = ctx.saved_tensors

        if pack_info.size(0) == 0:
            # NOTE: [JG] Materialize to zeros instead of None, to allow for autograd.grad() usage
            dL_dalphas = torch.zeros_like(alphas)
        else:
            dL_dalphas = vren.weights_from_alphas_bw(
                dL_dws, dL_dopacity, alphas, weights, opacity, pack_info, ctx.transmittance_threshold
            )

        return dL_dalphas, None, None


class Embedding(nn.Embedding):
    """
    `nn.Embedding` with more initialization methods and nearest method
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float,
        requires_grad: bool = True,
        weight_init_config: DictConfig | None = None,
        **kwargs,
    ):
        init_config = (
            DictConfig(dict(method="randn", mean=0.0, std=1.0)) if weight_init_config is None else weight_init_config
        )
        self.weight_init_config = init_config
        weight = torch.empty([num_embeddings, embedding_dim], dtype=dtype, device=device)

        match init_method := init_config.method:
            case "zero":
                weight.zero_()
            case "meshgrid":
                # Scatter meshgrid points uniformly distributed in N-D hyperspace
                if embedding_dim == 1:  # 1-D meshgrid is just linspace
                    weight = torch.linspace(
                        init_config.from_, init_config.to_, num_embeddings, dtype=dtype, device=device
                    ).view(num_embeddings, embedding_dim)
                else:
                    side_length = int(math.ceil(num_embeddings ** (1.0 / embedding_dim)))
                    points = torch.stack(
                        torch.meshgrid(
                            [torch.linspace(init_config.from_, init_config.to_, side_length, dtype=dtype)],
                            indexing="ij",
                        ),
                        dim=-1,
                    ).view(-1, embedding_dim)
                    weight = points[torch.randperm(num_embeddings)].contiguous().to(device)
            case "linspace":
                # Scatter points uniformly distributed on a 1D line
                assert embedding_dim == 1, (
                    f"{self.__class__.__name__} init_method linspace is only suitable for 1-D embeddings"
                )
                weight = torch.linspace(
                    init_config.from_, init_config.to_, num_embeddings, dtype=dtype, device=device
                ).view(num_embeddings, embedding_dim)
            case "random_normal" | "randn":
                # Scatter points normally distributed in N-D hyperspace
                weight.normal_(init_config.mean, init_config.std)
            case "random_uniform":
                # Scatter points drawn from a uniform distribution in N-D hyperspace
                weight.uniform_(-init_config.from_, init_config.to_)
            case "random_bernoulli":
                # Scatter points drawn from a Bernoulli distribution in N-D space
                weight.bernoulli_(init_config.p)
            case "bypass":
                data = torch.tensor(init_config.data, dtype=dtype, device=device)
                weight.fill_(data)
            case _:
                raise ValueError(f"{self.__class__.__name__} invalid init_method {init_method}")

        # Call nn.Embedding.__init__()
        super().__init__(num_embeddings, embedding_dim, _weight=weight, dtype=dtype, device=device, **kwargs)

        if not requires_grad:
            self.weight.requires_grad = False

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @torch.no_grad()
    def get_nearest_idx(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.embedding_dim
        inds = (x.unsqueeze(-2) - self.weight.data).norm(dim=-1).argmin(dim=-1)
        return inds


class SequentialEmbedding(Embedding):
    """
    `nn.Embedding` with more initialization methods and 1D interpolation along given `t_keyframes`.
    """

    t_keyframes: torch.Tensor

    def __init__(
        self,
        t_keyframes: torch.Tensor,
        embedding_dim: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float,
        requires_grad: bool = True,
        weight_init_config: DictConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            len(t_keyframes),
            embedding_dim,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            weight_init_config=weight_init_config,
            **kwargs,
        )

        self.t_keyframes = nn.Parameter(
            unpack_optional(t_keyframes).to(dtype=dtype, device=device),
            requires_grad=False,
        )

    def interp(self, t: torch.Tensor) -> torch.Tensor:
        v = torch_interp1d(self.t_keyframes, self.weight, t)
        return v

    def nearest(self, t: torch.Tensor, mode: Literal["nearest", "ceil", "floor"] = "nearest"):
        inds = torch.searchsorted(self.t_keyframes, t)  # in range [0, len]
        match mode:
            case "nearest":
                inds = torch.clamp(inds, 1, len(self.t_keyframes) - 1)
                prev_dis = (self.t_keyframes[inds - 1] - t).abs()
                next_dis = (self.t_keyframes[inds] - t).abs()
                inds = torch.where(prev_dis < next_dis, inds - 1, inds)
            case "ceil":
                inds = torch.clamp(inds, 1, len(self.t_keyframes)) - 1
            case "floor":
                inds = torch.clamp(inds, 0, len(self.t_keyframes))
            case _:
                raise ValueError(f"{self.__class__.__name__} invalid mode {mode}")
        return self.weight[inds]


def weights_from_alphas(
    alphas: torch.Tensor, pack_info: torch.Tensor, transmittance_threshold: float = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the weights of each sample from its alpha value using the NeRF volume rendering formulation

    Inputs:
        alphas: (N,) alphas for each sample along each ray in a packed representation
        pack_info: (N_rays, 2) start_idx, N_samples
        meaning each entry corresponds to the i-th ray,
        whose samples are [start_idx:start_idx+N_samples]
        transmittance_threshold: (float) transmittance threshold used for early stopping, when the remaining transmittance
        falls below this value, the alpha compositing will stop (weights 0 after this sample)

    Outputs:
        samples: (N_rays) total number of samples used for each ray (n_samples before eventual early stopping)
        opacity: (N_rays) accumulated opacity of each ray
        weights: (N) weights for each sample along each ray
    """
    return WeightsFromAlphas.apply(alphas, pack_info.int(), transmittance_threshold)


@torch.no_grad()
def linear_sampling(
    starts: torch.Tensor,
    stops: torch.Tensor,
    step_size: float | torch.Tensor = 0.01,
    stratified_perturb: bool = False,
    single_jitter: bool = False,
):
    """
    Linear (uniform) sample between given start values and stop values

    Examples:
        >>> starts = [0.1, 1.12, 2.1]
        >>> stops = [0.15, 1.35, 2.4]
        >>> step_size = 0.1

        - For `stratified_perturb=False`:
        >>> t_samples = [
        >>>     0.10, 0.15,
        >>>     1.12, 1.22, 1.32, 1.35,
        >>>     2.10, 2.20, 2.30, 2.40, 2.40]
        - For `stratified_perturb=True` and `single_jitter=True` (one example of random):
        >>> t_samples = [
        >>>     0.1049, 0.1500,
        >>>     1.1402, 1.2402, 1.3402, 1.3500,
        >>>     2.1867, 2.2867, 2.3867, 2.4000, 2.4000]
        - For `stratified_perturb=True` and `single_jitter=False`(one example of random):
        >>> t_samples = [
        >>>     0.1471, 0.1500,
        >>>     1.2015, 1.2729, 1.3500, 1.3500,
        >>>     2.1804, 2.2708, 2.3817, 2.4000, 2.400]
    Args:
        starts (torch.Tensor): The starting value of each pack
        stops (torch.Tensor): The stopping value of each pack
        step_size (float | torch.Tensor, optional): The step size of uniform sampling. \
            Potentially being a per-ray tensor for different sampling distance per-ray. Defaults to 0.01.
        stratified_perturb (bool, optional): Set if apply random perturbation when sampling. Defaults to False.
        single_jitter (bool, optional): Set to use the same random jitter for all samples along one pack. Defaults to False.

    Returns:
        t_samples (torch.Tensor): The packed sample results
        deltas (torch.Tensor): The length of each sampled interval
        ridx (torch.Tensor): The ray(pack) indices of each sample point
        pack_info (torch.Tensor): Pack info of the sampled results
    """

    steps_1 = ((stops - starts) / step_size).floor_().long() + 1  # Including starts, at least 1
    t_samples_1 = linstep_interleave(starts, steps_1, step_size, return_idx=False).values
    steps = steps_1 + 1  # Including stops
    pack_info = get_pack_info_from_n(steps.int())
    ridx = torch.arange(len(starts), device=starts.device).repeat_interleave(steps)

    if stratified_perturb:
        # Increase every sample by a random jitter.
        if single_jitter:
            t_samples_1 += (step_size * torch.rand_like(starts)).repeat_interleave(steps_1)
        else:
            t_samples_1 += (
                step_size.repeat_interleave(steps_1) if isinstance(step_size, torch.Tensor) else step_size
            ) * torch.rand_like(t_samples_1)

    # Fill the full samples with adding `stops` to each pack
    t_samples = torch.zeros_like(ridx, dtype=starts.dtype)
    indices_1 = linstep_interleave(pack_info[:, 0], steps_1, 1, return_idx=False).values
    t_samples[indices_1] = t_samples_1
    if stratified_perturb:
        second_last = pack_info[:, 0] + pack_info[:, 1] - 2
        # The second last of each pack should not exceed `stops` (Indices always valid, since steps >= 2)
        t_samples[second_last] = t_samples[second_last].clamp_max(stops)
    # Add `stops` to the end of each pack.
    t_samples[pack_info[:, 0] + pack_info[:, 1] - 1] = stops

    # Compute the interval length
    deltas = packed_diff(t_samples, pack_info)
    return t_samples, deltas, ridx, pack_info


@torch.no_grad()
def invert_cdf_sampling(
    bins: torch.Tensor,
    cdfs: torch.Tensor,
    pack_info: torch.Tensor,
    num_to_sample: int | torch.Tensor,
    stratified_perturb: bool = False,
    single_jitter: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Conduct invert CDF sampling on packed bins.
    NOTE: Tested and Supports 0 entries in `pack_info`

    Args:
        bins (torch.Tensor): Packed tensor, intervals' boundary points
        cdfs (torch.Tensor): Packed tensor, CDF value corresponding to each boundary point (with leading zero on each pack)
        pack_info (torch.Tensor): `pack_info` for `bins`/`cdfs`
        num_to_sample (int | torch.Tensor): Could either be a float or per-ray tensor. \
            Giving a tensor will suggest different number of samples per ray.
        stratified_perturb (bool, optional): Set to true to have stratified randomness. Defaults to False.
        single_jitter (bool, optional): \
            Set to true to have the same amount of random jittering for each ray respectively. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        - t_samples: The newly sampled bins
        - ridx: The corresponding ray-idx each sample belongs to
        - out_pack_info: Pack info for the output
    """

    assert [*bins.shape] == [*cdfs.shape], "bins and cdfs should have the same shape"
    num_packs, device, dtype = pack_info.shape[0], bins.device, bins.dtype

    # Only sample u on non-empty packs
    non_empty_idx = (pack_info[:, 1] > 0).nonzero().long()[..., 0]

    if (n_non_empty := len(non_empty_idx)) == 0:
        u, ridx, t_samples, out_pack_info = (
            torch.empty([0], device=device),
            torch.empty([0], device=device),
            torch.empty([0], device=device),
            torch.tensor([[0, 0]], dtype=torch.long, device=device),
        )
    else:
        u, _, ridx, u_pack_info = linear_sampling(
            torch.zeros((n_non_empty,), dtype=dtype, device=device),
            torch.ones((n_non_empty,), dtype=dtype, device=device),
            (1.0 / num_to_sample[non_empty_idx]) if isinstance(num_to_sample, torch.Tensor) else (1.0 / num_to_sample),
            stratified_perturb=stratified_perturb,
            single_jitter=single_jitter,
        )
        n_per_pack = torch.zeros((num_packs,), dtype=u_pack_info.dtype, device=device)
        n_per_pack[non_empty_idx] = u_pack_info[:, 1]
        out_pack_info = get_pack_info_from_n(n_per_pack.int())
        t_samples, _ = packed_invert_cdf(bins, cdfs.to(dtype), pack_info, u.contiguous(), out_pack_info)

    return t_samples, ridx, out_pack_info


def packed_pdf_to_cdf(
    pdfs: torch.Tensor,
    pack_info: torch.Tensor,
) -> torch.Tensor:
    """
    Supports 0 entries in `pack_info`
    """
    # Right-shifted cumsum, resulting in CDF with leading 0.0
    cdfs = packed_cumsum(pdfs, pack_info, exclusive=True)
    # Normalize the CDF, such that for all non empty packs there's always an ending 1.0
    last_cdf = torch.ones((pack_info.size(0),), dtype=pdfs.dtype, device=pdfs.device)
    non_empty_ridx = (pack_info[:, 1] > 0).nonzero().long()[..., 0]
    last_cdf[non_empty_ridx] = cdfs[pack_info[non_empty_ridx].sum(-1).sub(1)]
    cdfs = packed_div(cdfs, last_cdf.clamp_min(1e-5), pack_info)
    return cdfs


@torch.no_grad()
def packed_pdf_sampling(
    bins: torch.Tensor,
    pdfs: torch.Tensor,
    pack_info: torch.Tensor,
    num_to_sample: int | torch.Tensor,
    stratified_perturb: bool = False,
    single_jitter: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cdfs = packed_pdf_to_cdf(pdfs, pack_info)
    return invert_cdf_sampling(
        bins, cdfs, pack_info, num_to_sample, stratified_perturb=stratified_perturb, single_jitter=single_jitter
    )
