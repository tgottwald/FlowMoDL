import torch

from utils.sense import centered_ifftn

NORM_FACTOR_FLOOR = 1e-12


def kspace_norm_factor(kspace_us, mask):
    """Scalar normalisation factor of one undersampled acquisition.

    The RMS magnitude of the *acquired* k-space samples: the L2 norm of the
    masked k-space divided by the square root of the number of acquired
    entries (mask points x Nv x Nc x readout samples).

    kspace_us: (..., Nv, Nc, Nt, Z, Y, X) masked k-space of a single sample.
    mask:      (..., 1, 1, Nt, Z, Y, 1) bool undersampling mask.
    """
    nv, nc, x_dim = kspace_us.shape[-6], kspace_us.shape[-5], kspace_us.shape[-1]
    kspace_l2 = torch.linalg.vector_norm(kspace_us, ord=2)
    total_nnz = mask.sum(dtype=torch.float32) * (nv * nc * x_dim)
    norm_factor = kspace_l2 / torch.clamp(torch.sqrt(total_nnz), min=1.0)
    return torch.where(
        norm_factor > NORM_FACTOR_FLOOR, norm_factor, torch.ones_like(norm_factor)
    )


def reconstruct_target_and_normalize(
    kdata_full, mask, smaps, temporal_chunk_size=None, spatial_ax=(-3, -2, -1)
):
    """Build the network input and the ground truth from fully sampled k-space.

    1. The target image is the coil-combined (SENSE-weighted) inverse FFT of the
       fully sampled k-space.
    2. The k-space is undersampled with ``mask``.
    3. Both are divided by ``kspace_norm_factor`` of the undersampled k-space.

    Works on CPU (called by the dataset) or GPU (called by the training loop
    when ``reconstruct_target_on_gpu`` is set).

    IMPORTANT: ``kdata_full`` is masked and normalised *in place* and returned
    as ``kspace_us``; do not reuse it afterwards.

    ``temporal_chunk_size`` bounds peak memory: the coil-resolved intermediate
    is Nc times larger than the coil-combined target, so it is built
    ``temporal_chunk_size`` cardiac frames at a time (None = all frames at once).

    Args:
        kdata_full: (1, Nv, Nc, Nt, Z, Y, X) complex64, unmasked and unnormalised.
        mask: (1, 1, 1, Nt, Z, Y, 1) bool undersampling mask.
        smaps: (1, Nc, Z, Y, X) complex64 sensitivity maps.
    Returns:
        kspace_us: (1, Nv, Nc, Nt, Z, Y, X) masked, normalised k-space
            (the same tensor object as ``kdata_full``).
        target_image: (1, Nv, Nt, Z, Y, X) normalised target image.
        norm_factor: scalar normalisation factor.
    """
    if kdata_full.shape[0] != 1:
        raise NotImplementedError(
            "reconstruct_target_and_normalize normalises per sample and assumes "
            f"batch_size=1, got batch size {kdata_full.shape[0]}."
        )

    B, Nv, Nc, Nt, Z, Y, X = kdata_full.shape
    chunk = max(1, min(temporal_chunk_size or Nt, Nt))

    smaps_conj = smaps.conj().unsqueeze(1).unsqueeze(3)  # (B, 1, Nc, 1, Z, Y, X)
    target_image = torch.empty(
        (B, Nv, Nt, Z, Y, X), dtype=kdata_full.dtype, device=kdata_full.device
    )
    for t0 in range(0, Nt, chunk):
        t1 = min(t0 + chunk, Nt)
        img_chunk = centered_ifftn(kdata_full[:, :, :, t0:t1], dim=spatial_ax)
        target_image[:, :, t0:t1] = torch.sum(img_chunk * smaps_conj, dim=2)
        del img_chunk

    kspace_us = kdata_full.mul_(mask)
    norm_factor = kspace_norm_factor(kspace_us, mask)
    kspace_us.div_(norm_factor)
    target_image.div_(norm_factor)

    return kspace_us, target_image, norm_factor
