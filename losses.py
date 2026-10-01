"""Training objectives.

All losses share the call signature

    criterion(predictions, target_image, tau, K, target_velocity=None,
              segmask=None, v_enc=None, epoch=None)

where ``predictions`` is the list of per-cascade estimates returned by the
model in training mode ([B, Nv, Nt, Z, Y, X] complex each), ``tau`` sets the
deep-supervision weights w_k = exp(-tau * (K - k)) and ``K`` is the number of
cascades. Losses ignore the keyword arguments they do not need.

Losses with ``separable_over_encodings = True`` are per-element means and may be
evaluated on subsets of the velocity encodings (see train.py); losses that
couple the encodings through the velocity field set it to False.
"""

import math

import torch
import torch.nn as nn

from utils.metrics import extract_velocity


def _cascade_weights(num_predictions, tau, K):
    """Deep-supervision weights w_k = exp(-tau * (K - k)), k = 1..num_predictions."""
    return [math.exp(-tau * (K - k)) for k in range(1, num_predictions + 1)]


def _linear_ramp(epoch, start, length):
    """Linear 0 -> 1 ramp over ``length`` epochs beginning at epoch ``start``.
    Returns 1.0 when the epoch is unknown or the ramp is disabled (length <= 0)."""
    if epoch is None or not length or length <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - start + 1) / float(length))))


class LayerwiseWeightedL1Loss(nn.Module):
    """Deep-supervision L1 loss sum_k w_k * mean|x_k - x| on the complex image."""

    separable_over_encodings = True

    def forward(self, predictions, target_image, tau, K, **_):
        weights = _cascade_weights(len(predictions), tau, K)
        return sum(
            w * torch.nn.functional.l1_loss(p_k, target_image)
            for w, p_k in zip(weights, predictions)
        )


class FinalPredictionL1Loss(nn.Module):
    """mean|x_K - x| on the final estimate only (used for FlowVN)."""

    separable_over_encodings = True

    def forward(self, predictions, target_image, tau=None, K=None, **_):
        return (predictions[-1] - target_image).abs().mean()


class FinalPredictionMSELoss(nn.Module):
    """mean|x_K - x|^2 on the final estimate only: the objective of the original
    MoDL (Aggarwal et al., IEEE TMI 2019, Eq. 13), used for the MoDL baseline."""

    separable_over_encodings = True

    def forward(self, predictions, target_image, tau=None, K=None, **_):
        return (predictions[-1] - target_image).abs().square().mean()


class CompositeFlowLoss(nn.Module):
    """FlowMoDL's deep-supervision composite loss (Eqs. 5-7 of the paper):

        L = sum_k w_k [ L_img + lambda_vel * a_v(e) * L_vel
                              + lambda_rel * a_v(e) * L_rel
                              + lambda_ang * a_ang(e) * L_ang ]

    * L_img: L1 on the complex image; ROI voxels have weight 1, background
      voxels ``mag_bg_weight``.
    * L_vel: L1 on the velocity field inside the ROI.
    * L_rel: differentiable version of the RelErr metric,
      sqrt(sum_ROI (|v| - |v_k|)^2 / sum_ROI |v|^2), per sample.
    * L_ang: cosine distance 1 - cos(theta) between predicted and true velocity
      vectors inside the ROI, a smooth surrogate of the AngErr metric (whose
      acos has an unbounded gradient at perfect alignment).

    Curriculum: a_v(e) ramps linearly from 0 to 1 over ``vel_warmup_epochs``
    epochs starting at ``vel_warmup_start``; a_ang(e) ramps over
    ``ang_warmup_epochs`` starting at ``ang_warmup_start`` (by default where the
    velocity ramp ends). Before the ramps only the image term is optimised.
    """

    separable_over_encodings = False

    def __init__(
        self,
        lambda_vel=0.1,
        lambda_rel=0.1,
        lambda_ang=0.1,
        mag_bg_weight=0.1,
        vel_warmup_start=0,
        vel_warmup_epochs=0,
        ang_warmup_start=None,
        ang_warmup_epochs=0,
        eps_rel=1e-8,
        eps_ang=1e-8,
    ):
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_rel = lambda_rel
        self.lambda_ang = lambda_ang
        self.mag_bg_weight = mag_bg_weight
        self.vel_warmup_start = vel_warmup_start
        self.vel_warmup_epochs = vel_warmup_epochs
        self.ang_warmup_start = (
            ang_warmup_start
            if ang_warmup_start is not None
            else vel_warmup_start + vel_warmup_epochs
        )
        self.ang_warmup_epochs = ang_warmup_epochs
        self.eps_rel = eps_rel
        self.eps_ang = eps_ang
        # Most recent curriculum multipliers, logged by the training loop.
        self.curriculum_scales = {}

    def forward(
        self,
        predictions,
        target_image,
        tau,
        K,
        target_velocity=None,
        segmask=None,
        v_enc=None,
        epoch=None,
        **_,
    ):
        # segmask: (B, Z, Y, X) -> broadcast over encodings / velocity components and time
        mask_vel = segmask.unsqueeze(1).unsqueeze(2).expand_as(target_velocity)
        mask_img = segmask.unsqueeze(1).unsqueeze(2).expand_as(target_image)
        mag_weights = torch.where(mask_img > 0, 1.0, self.mag_bg_weight)
        m = mask_vel[:, 0]  # (B, Nt, Z, Y, X)
        reduce_dims = tuple(range(1, m.dim()))
        t_mag = torch.linalg.norm(target_velocity, dim=1)

        vel_scale = _linear_ramp(epoch, self.vel_warmup_start, self.vel_warmup_epochs)
        ang_scale = _linear_ramp(epoch, self.ang_warmup_start, self.ang_warmup_epochs)
        self.curriculum_scales = {"vel": vel_scale, "ang": ang_scale}

        loss_img = loss_vel = loss_rel = loss_ang = 0.0
        weights = _cascade_weights(len(predictions), tau, K)

        for weight, p_k in zip(weights, predictions):
            img_diff = torch.abs(p_k - target_image) * mag_weights
            loss_img += weight * (img_diff.sum() / (mag_weights.sum() + 1e-8))

            v_k = extract_velocity(p_k, v_enc)

            vel_diff = torch.abs(v_k - target_velocity) * mask_vel
            loss_vel += weight * (vel_diff.sum() / (mask_vel.sum() + 1e-8))

            p_mag = torch.linalg.norm(v_k, dim=1)

            rel_num = torch.sum((t_mag - p_mag).square() * m, dim=reduce_dims)
            rel_den = torch.sum(t_mag.square() * m, dim=reduce_dims) + self.eps_rel
            rel_per_sample = torch.sqrt(rel_num / rel_den + self.eps_rel)
            valid = (torch.sum(m, dim=reduce_dims) > 0).to(rel_per_sample.dtype)
            loss_rel += weight * (rel_per_sample * valid).mean()

            dot = torch.sum(v_k * target_velocity, dim=1)
            cos_theta = torch.clamp(dot / (p_mag * t_mag + self.eps_ang), -1.0, 1.0)
            ang_per_sample = torch.sum((1.0 - cos_theta) * m, dim=reduce_dims) / (
                torch.sum(m, dim=reduce_dims) + 1e-12
            )
            loss_ang += weight * ang_per_sample.mean()

        return (
            loss_img
            + self.lambda_vel * vel_scale * loss_vel
            + self.lambda_rel * vel_scale * loss_rel
            + self.lambda_ang * ang_scale * loss_ang
        )
