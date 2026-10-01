"""
Centered FFTs, the SENSE forward/adjoint operators and a batched conjugate
gradient solver shared by all reconstruction models.

Every model uses a *centered* FFT convention (DC in the middle of the array).
A centered transform along a set of axes can be computed in two equivalent
ways:

  1. Shift-based (any axis length):
         fftshift(fftn(ifftshift(x)))

  2. Checkerboard modulation (only valid when every transformed axis has even
     length), which replaces the two memory-bound shifts by elementwise sign
     flips:
         C * (-1)^k * fftn((-1)^n * x),   C = prod_over_axes((-1) ** (N // 2))

``centered_fftn`` / ``centered_ifftn`` use (2) whenever all transformed axes
are even-sized and fall back to (1) otherwise.
"""

from collections import OrderedDict

import torch

DIM = (-3, -2, -1)
# Phase-encode axes (SPE, PE) only; used when the fully sampled readout axis has
# already been moved to image space (see FlowMoDL(decouple_readout=True)).
DIM_DECOUPLED = (-3, -2)

_CACHE_MAXSIZE = 8
_checkerboard_cache = OrderedDict()


def _all_even(shape, dim):
    return all(shape[d] % 2 == 0 for d in dim)


def _get_checkerboard(shape, dim, device, real_dtype):
    key = (shape, dim, device, real_dtype)
    cached = _checkerboard_cache.get(key)
    if cached is not None:
        _checkerboard_cache.move_to_end(key)
        return cached

    pattern = None
    for d in dim:
        n = shape[d]
        idx = torch.arange(n, device=device)
        s = torch.where(idx % 2 == 0, 1, -1).to(real_dtype)
        view = [1] * len(shape)
        view[d] = n
        pattern = s.view(view) if pattern is None else pattern * s.view(view)

    _checkerboard_cache[key] = pattern
    if len(_checkerboard_cache) > _CACHE_MAXSIZE:
        _checkerboard_cache.popitem(last=False)  # evict least-recently-used
    return pattern


def _global_sign(shape, dim):
    sign = 1
    for d in dim:
        sign *= (-1) ** (shape[d] // 2)
    return sign


def _centered_transform(x, dim, fft_fn):
    if _all_even(x.shape, dim):
        checker = _get_checkerboard(x.shape, dim, x.device, x.real.dtype)
        sign = _global_sign(x.shape, dim)
        return sign * checker * fft_fn(x * checker, dim=dim, norm="ortho")

    x = torch.fft.ifftshift(x, dim=dim)
    x = fft_fn(x, dim=dim, norm="ortho")
    return torch.fft.fftshift(x, dim=dim)


def centered_fftn(x, dim=DIM):
    """Centered orthonormal FFT: fftshift(fftn(ifftshift(x)))."""
    return _centered_transform(x, tuple(dim), torch.fft.fftn)


def centered_ifftn(x, dim=DIM):
    """Centered orthonormal IFFT: fftshift(ifftn(ifftshift(x)))."""
    return _centered_transform(x, tuple(dim), torch.fft.ifftn)


def forward_op(p_k, smaps, dim=DIM):
    """
    SENSE forward operator (image -> coil k-space), without the sampling mask.

    p_k:   [B, Nv, Nt, Z, Y, X]
    smaps: [B, Nc, Z, Y, X]
    returns [B, Nv, Nc, Nt, Z, Y, X]

    ``dim`` selects the transformed axes: (Z, Y, X) for full 3D encoding, or
    (Z, Y) for the readout-decoupled hybrid representation.
    """
    coil_images = p_k.unsqueeze(2) * smaps.unsqueeze(1).unsqueeze(3)
    return centered_fftn(coil_images, dim=dim)


def adjoint_op(kspace, smaps, dim=DIM):
    """
    SENSE adjoint operator (coil k-space -> coil-combined image).

    kspace: [B, Nv, Nc, Nt, Z, Y, X]
    smaps:  [B, Nc, Z, Y, X]
    returns [B, Nv, Nt, Z, Y, X]
    """
    coil_images = centered_ifftn(kspace, dim=dim)
    return torch.sum(coil_images * smaps.unsqueeze(1).unsqueeze(3).conj(), dim=2)


def make_sense_ops(smaps, decouple_readout=False):
    """Bind ``smaps`` into the ``(forward_op, adjoint_op)`` pair every model's
    ``forward`` expects.

    With ``decouple_readout=True`` the operators transform only the two
    phase-encode axes. This must be paired with a model that moves the fully
    sampled readout axis into image space before its first iteration
    (``FlowMoDL(decouple_readout=True)``); together they are exactly equivalent
    to the full 3D problem, because the undersampling mask is constant along the
    readout axis.
    """
    dim = DIM_DECOUPLED if decouple_readout else DIM
    return (
        lambda x: forward_op(x, smaps, dim=dim),
        lambda x: adjoint_op(x, smaps, dim=dim),
    )


def _batched_dot(a, b):
    """Real part of <a, b> reduced over everything except the batch axis."""
    dims = tuple(range(1, a.ndim))
    out = torch.sum((a.conj() * b).real, dim=dims)
    return out.view(-1, *([1] * (a.ndim - 1)))


def conjugate_gradient(normal_op, rhs, x0, num_iter, tol=1e-5):
    """Solve ``normal_op(x) = rhs`` for a Hermitian positive (semi-)definite
    operator with ``num_iter`` CG iterations starting from ``x0``.

    Per-sample step sizes keep the solve correct for batch sizes > 1, and every
    operation is differentiable, so gradients propagate through the unrolled
    iterations.
    """
    x = x0
    r = rhs - normal_op(x)
    p = r
    rs_old = _batched_dot(r, r)

    for _ in range(num_iter):
        Ap = normal_op(p)
        alpha = rs_old / (_batched_dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = _batched_dot(r, r)
        if torch.sqrt(rs_new.max()) < tol:
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new

    return x
