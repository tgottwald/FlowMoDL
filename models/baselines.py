"""Classical (non-learned) reconstruction baselines.

Both expose the same interface as the learned models,

    model(kspace_acq, undersampling_mask, forward_op, adjoint_op,
          usrate=None, training=False) -> [x]

where ``forward_op`` / ``adjoint_op`` are the SENSE operators of utils.sense.
"""

import torch
import torch.nn as nn

from utils.sense import conjugate_gradient


class ZeroFilled(nn.Module):
    """Coil-combined adjoint of the undersampled k-space."""

    def forward(
        self,
        kspace_acq,
        undersampling_mask,
        forward_op,
        adjoint_op,
        usrate=None,
        training=False,
    ):
        return [adjoint_op(kspace_acq * undersampling_mask)]


class CGSENSE(nn.Module):
    """Iterative SENSE (Pruessmann et al., 2001) solving the (optionally
    Tikhonov-regularised) normal equations

        (E^H E + lam * I) x = E^H b,   b = mask * kspace_acq

    with conjugate gradient, starting from zero.
    """

    def __init__(self, num_iter=20, tol=1e-6, lam=0.0):
        super().__init__()
        self.num_iter = num_iter
        self.tol = tol
        self.lam = lam

    def forward(
        self,
        kspace_acq,
        undersampling_mask,
        forward_op,
        adjoint_op,
        usrate=None,
        training=False,
    ):
        mask = undersampling_mask

        def normal(x):
            out = adjoint_op(mask * forward_op(x))
            return out + self.lam * x if self.lam != 0.0 else out

        rhs = adjoint_op(mask * kspace_acq)
        x = conjugate_gradient(
            normal, rhs, torch.zeros_like(rhs), self.num_iter, tol=self.tol
        )
        return [x]
