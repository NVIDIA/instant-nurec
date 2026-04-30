import torch

from nre.utils.misc import unpack_optional
from nre.utils.types import GaussiansRenderReturn


def distance_based_filter(
    rendered: GaussiansRenderReturn,
    model_elements: torch.Tensor,
    valid_mask_pred: torch.Tensor,
    filter_threshold: float,
    n_vertical_bins: int,
    n_horizontal_bins: int,
) -> torch.Tensor:
    """Apply distance-based filtering to LiDAR rays.

    Args:
        rendered: Rendered LiDAR data with distance and optional raydrop signals
        model_elements: 2D indices for ray elements
        valid_mask_pred: Boolean mask indicating which rays returned a valid measurement
        filter_threshold: Threshold for distance discontinuity filtering
        n_vertical_bins: Number of vertical bins in the LiDAR range view
        n_horizontal_bins: Number of horizontal bins in the LiDAR range view

    Returns:
        Boolean mask indicating which rays passed the filter
    """
    device = model_elements.device
    dist_pred = unpack_optional(rendered.distance)

    rangeview_range_pred = torch.full((n_vertical_bins, n_horizontal_bins), torch.inf, device=device)
    rangeview_range_factor = torch.full((n_vertical_bins, n_horizontal_bins), torch.inf, device=device)

    rangeview_range_pred[model_elements[..., 0][valid_mask_pred], model_elements[..., 1][valid_mask_pred]] = dist_pred[
        valid_mask_pred
    ]
    rangeview_range_factor[model_elements[..., 0][valid_mask_pred], model_elements[..., 1][valid_mask_pred]] = 1.0 / (
        dist_pred[valid_mask_pred] + 1e-6
    )
    rangeview_range_factor[rangeview_range_factor == torch.inf] = 0

    range_diff_left = torch.abs(rangeview_range_pred[:, 1:-1] - rangeview_range_pred[:, :-2])
    range_diff_right = torch.abs(rangeview_range_pred[:, 1:-1] - rangeview_range_pred[:, 2:])
    range_diff = torch.min(range_diff_left, range_diff_right)
    range_diff[range_diff == torch.inf] = 0
    range_diff *= rangeview_range_factor[:, 1:-1]
    range_filter_mask = range_diff > filter_threshold
    range_filter_mask_full = torch.full((n_vertical_bins, n_horizontal_bins), 0.0, device=device)
    range_filter_mask_full[:, 1:-1] = range_filter_mask

    filter_mask = range_filter_mask_full.reshape(-1) == 1.0

    return filter_mask
