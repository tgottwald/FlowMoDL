"""kt-Gaussian undersampling masks, following the CMRx4DFlow challenge's
mask generator: per cardiac frame, a Gaussian-weighted random sampling of the
(SPE, PE) plane with a minimum distance between samples and a decaying weight
for already sampled positions."""

import numpy as np
from numba import njit, prange


def generate_ktgaussian_mask(spe_dim, pe_dim, nt, R, seed=None):
    """kt-Gaussian mask for acceleration factor ``R``.

    Returns a float32 array of shape (Nt, SPE, PE) with (SPE * PE) // R sampled
    positions per cardiac frame. ``seed`` makes the draw reproducible.
    """
    if seed is not None:
        _seed_numba_rng(seed)
    masks_spe_pe_t = fun_mask_gen_2d_numba(
        width=pe_dim,
        height=spe_dim,
        total_points=(spe_dim * pe_dim) // R,
        pattern_num=nt,
        sigma_x=pe_dim / 5.0,
        sigma_y=spe_dim / 5.0,
        min_dist_factor=3.0,
        rep_decay_factor=0.5,
        center_radius_x=0.5,
        center_radius_y=0.5,
    )
    return np.transpose(masks_spe_pe_t, (2, 0, 1))


@njit(cache=True)
def _seed_numba_rng(seed):
    """Seed the RNG state Numba's nopython mode uses internally.

    Numba's nopython mode keeps an RNG state that is separate from NumPy's, so
    it can only be seeded by calling ``np.random.seed`` from inside jitted
    code.
    """
    np.random.seed(seed)


@njit(cache=True)
def create_gaussian_weight_matrix_numba(width, height, sigma_x, sigma_y):
    weight = np.zeros((height, width), dtype=np.float32)
    cx = (width + 1) / 2.0
    cy = (height + 1) / 2.0

    for y in range(height):
        y_val = (y + 1) - cy
        for x in range(width):
            x_val = (x + 1) - cx
            weight[y, x] = np.exp(
                -(x_val**2 / (2 * sigma_x**2) + y_val**2 / (2 * sigma_y**2))
            )

    return weight


@njit(cache=True)
def random_sampling_optimized_numba(
    width, height, total_points, weight, min_dist_lookup, existing_mask
):
    """Numba-compiled rejection sampling."""
    sampled_points_x = np.zeros(total_points, dtype=np.int32)
    sampled_points_y = np.zeros(total_points, dtype=np.int32)

    current_weight = weight * (1.0 - existing_mask)
    sw = np.sum(current_weight)

    if sw <= 0:
        return np.zeros((0, 2), dtype=np.int32)

    prob = (current_weight / sw).flatten()
    forbidden_mask = np.zeros((height, width), dtype=np.bool_)

    batch_size = max(total_points * 2, 1000)
    # np.random.choice with probabilities is not supported in Numba's njit currently,
    # so we use a compiled inverse transform sampling method.
    cumulative_prob = np.cumsum(prob)

    count = 0
    idx_ptr = 0

    # Generate initial batch
    rand_vals = np.random.rand(batch_size)
    indices = np.searchsorted(cumulative_prob, rand_vals)

    while count < total_points and idx_ptr < batch_size:
        idx = indices[idx_ptr]
        idx_ptr += 1

        y = idx // width
        x = idx % width

        if forbidden_mask[y, x] or existing_mask[y, x]:
            continue

        sampled_points_x[count] = x
        sampled_points_y[count] = y
        count += 1

        d = min_dist_lookup[y, x]
        if d > 0:
            y_min = max(0, int(y - d))
            y_max = min(height, int(y + d + 1))
            x_min = max(0, int(x - d))
            x_max = min(width, int(x + d + 1))

            for ry in range(y_min, y_max):
                for rx in range(x_min, x_max):
                    if (ry - y) ** 2 + (rx - x) ** 2 < d**2:
                        forbidden_mask[ry, rx] = True

        if idx_ptr >= batch_size and count < total_points:
            current_weight = weight * (1.0 - existing_mask) * (1.0 - forbidden_mask)
            sw = np.sum(current_weight)
            if sw <= 0:
                break
            prob = (current_weight / sw).flatten()
            cumulative_prob = np.cumsum(prob)
            rand_vals = np.random.rand(batch_size)
            indices = np.searchsorted(cumulative_prob, rand_vals)
            idx_ptr = 0

    out = np.empty((count, 2), dtype=np.int32)
    out[:, 0] = sampled_points_x[:count]
    out[:, 1] = sampled_points_y[:count]
    return out


@njit(cache=True)
def fun_mask_gen_2d_numba(
    width,
    height,
    total_points,
    pattern_num,
    sigma_x,
    sigma_y,
    min_dist_factor,
    rep_decay_factor,
    center_radius_x,
    center_radius_y,
):
    masks = np.zeros((height, width, pattern_num), dtype=np.float32)
    initial_weight = create_gaussian_weight_matrix_numba(
        width, height, sigma_x, sigma_y
    )

    min_dist_lookup = min_dist_factor * ((1.0 - initial_weight) / 2.0 + 0.5)

    center_ellipse = np.zeros((height, width), dtype=np.float32)
    if (center_radius_x <= 0.5) or (center_radius_y <= 0.5):
        cy = height // 2
        cx = width // 2
        center_ellipse[cy, cx] = 1.0
    else:
        cx_val = (width + 1) / 2.0
        cy_val = (height + 1) / 2.0
        for y in range(height):
            y_val = (y + 1) - cy_val
            for x in range(width):
                x_val = (x + 1) - cx_val
                if (x_val / center_radius_x) ** 2 + (
                    y_val / center_radius_y
                ) ** 2 <= 1.0:
                    center_ellipse[y, x] = 1.0

    # Force strict integer type
    num_center_points = int(np.sum(center_ellipse))

    for p in prange(pattern_num):
        mask = center_ellipse.copy()
        weight_local = initial_weight.copy()

        needed = int(total_points - num_center_points)
        if needed > 0:
            points = random_sampling_optimized_numba(
                width, height, needed, weight_local, min_dist_lookup, mask
            )
            for i in range(len(points)):
                x, y = points[i]
                mask[y, x] = 1.0
                weight_local[y, x] *= rep_decay_factor

        # Cast sum to int to avoid cascading float64 types to 'extra'
        curr_total = int(np.sum(mask))
        if curr_total < total_points:
            extra = int(total_points - curr_total)
            extra_points = random_sampling_optimized_numba(
                width, height, extra, weight_local, min_dist_lookup, mask
            )
            for i in range(len(extra_points)):
                x, y = extra_points[i]
                mask[y, x] = 1.0
                weight_local[y, x] *= rep_decay_factor

        masks[:, :, p] = mask

    return masks
