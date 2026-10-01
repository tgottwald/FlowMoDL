import math

import torch
import torch.nn.functional as F


def extract_velocity(p_k, v_enc):
    """Velocity field from the complex image: v_i = v_enc_i / pi * angle(x_i conj(x_0)).

    p_k:   (B, 4, Nt, Z, Y, X) complex, encoding 0 is the reference.
    v_enc: (B, 3) tensor or scalar.
    returns (B, 3, Nt, Z, Y, X)
    """
    phase_diff = torch.angle(p_k[:, 1:4] * torch.conj(p_k[:, 0:1]))
    if isinstance(v_enc, torch.Tensor):
        v_enc = v_enc.view(-1, 3, 1, 1, 1, 1)
    return (v_enc / math.pi) * phase_diff


def _expand_mask(mask, like):
    """Broadcast a (B, Z, Y, X) mask to the shape of ``like``, (B, ..., Z, Y, X)."""
    while mask.ndim < like.ndim:
        mask = mask.unsqueeze(1)
    return mask.expand_as(like)


class SSIM3D(torch.nn.Module):
    """3D SSIM map with an 11^3 Gaussian window (sigma = 1.5)."""

    def __init__(self, window_size=11):
        super().__init__()
        self.pad = window_size // 2

        gauss = torch.tensor(
            [
                math.exp(-((x - self.pad) ** 2) / (2 * 1.5**2))
                for x in range(window_size)
            ],
            dtype=torch.float32,
        )
        gauss /= gauss.sum()
        window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window_3d = gauss.view(window_size, 1, 1) * window_2d.unsqueeze(0)
        self.register_buffer("kernel", window_3d.unsqueeze(0).unsqueeze(0))

    def forward(self, img1, img2):
        channels = img1.shape[1]
        if self.kernel.device != img1.device:
            self.kernel = self.kernel.to(img1.device)
        window = self.kernel.expand(channels, 1, -1, -1, -1)

        def filt(x):
            return F.conv3d(x, window, padding=self.pad, groups=channels)

        mu1, mu2 = filt(img1), filt(img2)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.square(), mu2.square(), mu1 * mu2
        sigma1_sq = filt(img1 * img1) - mu1_sq
        sigma2_sq = filt(img2 * img2) - mu2_sq
        sigma12 = filt(img1 * img2) - mu1_mu2

        C1, C2 = 0.0001, 0.0009
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )


_ssim_engine = SSIM3D(window_size=11)


def compute_nrmse(pred_mag, target_mag, mask, eps=1e-12):
    """RMSE inside the ROI normalised by the peak target magnitude in the ROI.

    pred_mag, target_mag: (B, Nv, Nt, Z, Y, X); mask: (B, Z, Y, X)
    """
    mask_expanded = _expand_mask(mask, pred_mag)
    reduce_dims = tuple(range(1, pred_mag.ndim))

    mse = torch.sum(
        ((pred_mag - target_mag) * mask_expanded).square(), dim=reduce_dims
    ) / (torch.sum(mask_expanded, dim=reduce_dims) + eps)
    denom = (
        torch.amax(
            target_mag.masked_fill(mask_expanded == 0, 0.0), dim=reduce_dims
        ).clamp_min(0.0)
        + eps
    )
    return torch.mean(torch.sqrt(mse) / denom)


def compute_velocity_metrics(pred_vel, target_vel, mask, eps_rel=1e-12, eps_ang=1e-8):
    """Relative speed error and mean angular error [deg] inside the ROI.

    pred_vel, target_vel: (B, 3, Nt, Z, Y, X); mask: (B, Z, Y, X)
    """
    p_mag = torch.linalg.norm(pred_vel, dim=1)  # (B, Nt, Z, Y, X)
    t_mag = torch.linalg.norm(target_vel, dim=1)
    mask_expanded = _expand_mask(mask, t_mag)
    reduce_dims = tuple(range(1, t_mag.ndim))

    numerator = torch.sum((t_mag - p_mag).square() * mask_expanded, dim=reduce_dims)
    denominator = torch.sum(t_mag.square() * mask_expanded, dim=reduce_dims) + eps_rel
    rel_err = torch.mean(torch.sqrt(numerator / denominator))

    dot_product = torch.sum(pred_vel * target_vel, dim=1)
    cos_theta = torch.clamp(dot_product / (p_mag * t_mag + eps_ang), -1.0, 1.0)
    theta = torch.acos(cos_theta)
    ang_err_per_case = (
        torch.sum(theta * mask_expanded, dim=reduce_dims)
        / (torch.sum(mask_expanded, dim=reduce_dims) + 1e-12)
        * (180.0 / math.pi)
    )
    return rel_err, torch.mean(ang_err_per_case)


def compute_ssim(pred_mag, target_mag, mask):
    """3D SSIM averaged over the ROI, after scaling both images by the peak
    target magnitude in the ROI.

    pred_mag, target_mag: (B, Nv, Nt, Z, Y, X); mask: (B, Z, Y, X)
    """
    mask_expanded = _expand_mask(mask, target_mag)
    reduce_dims = tuple(range(1, target_mag.ndim))

    t_max = torch.amax(
        target_mag.masked_fill(mask_expanded == 0, 0.0), dim=reduce_dims, keepdim=True
    ).clamp_min(1e-12)
    p_scaled = (pred_mag * mask_expanded) / t_max
    t_scaled = (target_mag * mask_expanded) / t_max

    B, Nv, Nt, Z, Y, X = p_scaled.shape
    total_roi_sum = 0.0
    total_roi_cnt = torch.sum(mask_expanded, dim=reduce_dims)

    # One cardiac frame at a time to bound memory.
    for t in range(Nt):
        p_frame = p_scaled[:, :, t].reshape(B * Nv, 1, Z, Y, X)
        t_frame = t_scaled[:, :, t].reshape(B * Nv, 1, Z, Y, X)
        ssim_map = _ssim_engine(p_frame, t_frame).view(B, Nv, Z, Y, X)
        total_roi_sum += torch.sum(ssim_map * mask_expanded[:, :, t], dim=(1, 2, 3, 4))

    return torch.mean(total_roi_sum / total_roi_cnt.clamp_min(1.0))


def compute_complex_diff_err(pred, target, mask, eps=1e-12):
    """Symmetric bounded complex difference error in [0, 1] inside the ROI:
    ||pred - ref||_2 / (||pred||_2 + ||ref||_2).

    pred, target: (B, Nv, Nt, Z, Y, X) complex; mask: (B, Z, Y, X)
    """
    mask_expanded = _expand_mask(mask, pred)
    reduce_dims = tuple(range(1, pred.ndim))

    def masked_norm(x):
        return torch.sqrt(torch.sum(x.abs().square() * mask_expanded, dim=reduce_dims))

    err = masked_norm(pred - target)
    return torch.mean(err / (masked_norm(pred) + masked_norm(target) + eps))


# -------------------------------------------------------------------------
# Checkpoint selection
# -------------------------------------------------------------------------
# Checkpoints are selected with the CMRx4DFlow Task 1 ranking rule applied to
# the validated epochs: every epoch is ranked separately on each metric
# (standard competition ranking, ties share the best rank), the ranks are
# summed, and the lowest rank sum wins.

# name -> True if lower is better, False if higher is better.
CHECKPOINT_METRIC_DIRECTIONS = {
    "nRMSE": True,
    "SSIM": False,
    "RelErr": True,
    "AngErr": True,
}


def _competition_ranks(values, lower_is_better):
    """Standard competition ranking: ties share the best rank and the next
    distinct value skips ahead by the size of the tie group."""
    order = sorted(values) if lower_is_better else sorted(values, reverse=True)
    rank_of_value = {}
    for i, v in enumerate(order):
        rank_of_value.setdefault(v, i + 1)
    return [rank_of_value[v] for v in values]


def composite_checkpoint_score(history, metric_directions=None):
    """Rank-sum score of every validated epoch.

    history: list of dicts, one per validated epoch (the current epoch last),
        each holding the keys of ``metric_directions``.

    Returns (composite_scores, is_best):
        composite_scores: rank sums rescaled to (0, 1], where 1.0 means rank 1
            on every metric. Rescaling keeps the value comparable as the history
            grows while preserving the ordering of the raw rank sums.
        is_best: True if the newest epoch has the highest score (the newest
            epoch wins ties).
    """
    if metric_directions is None:
        metric_directions = CHECKPOINT_METRIC_DIRECTIONS
    if not history:
        return [], False

    n = len(history)
    ranks_by_metric = {
        name: _competition_ranks([h[name] for h in history], lower_is_better)
        for name, lower_is_better in metric_directions.items()
    }
    rank_sums = [
        sum(ranks_by_metric[name][i] for name in metric_directions) for i in range(n)
    ]

    num_metrics = len(metric_directions)
    best_possible = num_metrics
    worst_possible = num_metrics * n
    if worst_possible == best_possible:
        composite_scores = [1.0] * n
    else:
        composite_scores = [
            1.0 - (rs - best_possible) / (worst_possible - best_possible)
            for rs in rank_sums
        ]

    # `>=` so that the last epoch reaching the top score wins ties.
    best_idx = 0
    for i, s in enumerate(composite_scores):
        if s >= composite_scores[best_idx]:
            best_idx = i
    return composite_scores, best_idx == n - 1
