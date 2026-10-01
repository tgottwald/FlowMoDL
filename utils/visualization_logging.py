import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from utils.metrics import compute_nrmse, compute_velocity_metrics


def log_validation_visualizations(
    pred_image,
    target_image,
    segmask,
    epoch,
    R,
    pred_vel=None,
    target_vel=None,
    zf_image=None,
    batch_idx=0,
    v_idx=0,
):
    """
    Logs a side-by-side comparison of the Target, the zero-filled input, the
    Prediction and the ROI error heatmaps (nRMSE, angular error, relative
    error) to wandb.

    The scalar metrics of the visualized sample are attached as the image caption.

    Args:
        pred_image: Complex predicted tensor of shape (B, Nv, Nt, Z, Y, X)
        target_image: Complex ground truth tensor of shape (B, Nv, Nt, Z, Y, X)
        segmask: Binary mask tensor of shape (B, Z, Y, X)
        epoch: Current training epoch
        R: Acceleration factor
        pred_vel: Predicted velocity field (B, 3, Nt, Z, Y, X). Optional — the
            angular/relative error panels are skipped when it is not provided.
        target_vel: Ground truth velocity field, same shape as pred_vel
        zf_image: Complex zero-filled (adjoint SENSE) reconstruction of the
            undersampled input, same shape as pred_image. Optional — the
            zero-filled panel and its caption metric are skipped when it is
            not provided.
        batch_idx: Which sample in the batch to visualize (default: 0)
        v_idx: Velocity encoding index to visualize (0 is typically the anatomical reference)
    """
    # Detach and move to CPU
    pred_cpu = pred_image[batch_idx].detach().cpu()
    target_cpu = target_image[batch_idx].detach().cpu()
    mask_cpu = segmask[batch_idx].detach().cpu()  # (Z, Y, X)

    # Pick the Z-slice with maximum mask coverage — middle slice often has no vessel
    mask_coverage = mask_cpu.sum(dim=(-2, -1))  # (Z,)
    z_idx = int(mask_coverage.argmax().item())

    # Select the middle cardiac phase
    t_idx = pred_cpu.shape[1] // 2

    # Extract 2D slices and compute magnitude
    pred_slice_mag = torch.abs(pred_cpu[v_idx, t_idx, z_idx]).float().numpy()  # (Y, X)
    target_slice_mag = (
        torch.abs(target_cpu[v_idx, t_idx, z_idx]).float().numpy()
    )  # (Y, X)
    mask_slice = mask_cpu[z_idx].numpy()
    roi = mask_slice > 0

    # Normalize both images to the same scale so they are visually comparable.
    # Using the 99.5th percentile of the target to avoid a single bright outlier
    # collapsing the colorscale.
    vmax = float(np.percentile(target_slice_mag, 99.5))
    vmax = max(vmax, 1e-8)

    pred_slice_norm = np.clip(pred_slice_mag / vmax, 0.0, 1.0)
    target_slice_norm = np.clip(target_slice_mag / vmax, 0.0, 1.0)

    zf_slice_norm = None
    if zf_image is not None:
        zf_slice_mag = (
            torch.abs(zf_image[batch_idx].detach().cpu()[v_idx, t_idx, z_idx])
            .float()
            .numpy()
        )  # (Y, X)
        zf_slice_norm = np.clip(zf_slice_mag / vmax, 0.0, 1.0)

    # --- Per-pixel error maps (ROI only; NaN outside renders as white) --------
    error_map = np.abs(target_slice_mag - pred_slice_mag)

    # Per-pixel nRMSE: |pred - target| / peak_target_in_ROI.
    # Matches the normalisation used in compute_nrmse so the scale is meaningful.
    roi_peak = float(target_slice_mag[roi].max()) if roi.any() else 1.0
    roi_peak = max(roi_peak, 1e-8)
    nrmse_map = np.where(roi, error_map / roi_peak, np.nan)

    heatmaps = [(nrmse_map, "nRMSE Heatmap (ROI only)", "hot", None)]

    if pred_vel is not None and target_vel is not None:
        # (3, Y, X) velocity vectors at the visualized phase/slice
        p_vec = pred_vel[batch_idx][:, t_idx, z_idx].detach().cpu().float().numpy()
        t_vec = target_vel[batch_idx][:, t_idx, z_idx].detach().cpu().float().numpy()

        p_norm = np.linalg.norm(p_vec, axis=0)
        t_norm = np.linalg.norm(t_vec, axis=0)

        # Angular error in degrees between the predicted and true velocity vectors
        cos_theta = np.clip(
            np.sum(p_vec * t_vec, axis=0) / (p_norm * t_norm + 1e-8), -1.0, 1.0
        )
        ang_map = np.where(roi, np.degrees(np.arccos(cos_theta)), np.nan)

        # Per-pixel relative error: speed deviation normalised by the RMS target
        # speed inside the ROI — the local analogue of compute_velocity_metrics'
        # ||t - p|| / ||t|| ratio, without the blow-up of a per-pixel |t| divisor.
        t_rms = float(np.sqrt(np.mean(t_norm[roi] ** 2))) if roi.any() else 1.0
        t_rms = max(t_rms, 1e-8)
        rel_map = np.where(roi, np.abs(t_norm - p_norm) / t_rms, np.nan)

        heatmaps.append((ang_map, "Angular Error [deg] (ROI only)", "viridis", 180.0))
        heatmaps.append((rel_map, "Relative Error (ROI only)", "magma", None))

    # --- Figure ---------------------------------------------------------------
    # Magnitude panels, all on the same 0..1 scale so they are comparable.
    mag_panels = [(target_slice_norm, "Target Magnitude")]
    if zf_slice_norm is not None:
        mag_panels.append((zf_slice_norm, f"Zero-Filled Input (R={R})"))
    mag_panels.append((pred_slice_norm, f"Predicted Magnitude (R={R})"))

    n_panels = len(mag_panels) + len(heatmaps)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.suptitle(
        f"Epoch {epoch} | R={R}x | Z-Slice: {z_idx} (max ROI) | Phase: {t_idx}",
        fontsize=14,
    )

    for ax, (data, title) in zip(axes, mag_panels):
        im = ax.imshow(data, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Error heatmaps
    for ax, (data, title, cmap_name, vmax_map) in zip(
        axes[len(mag_panels) :], heatmaps
    ):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="white")
        im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=vmax_map)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # --- Caption: scalar metrics of this sample (full volume, not just the slice)
    sample_mask = segmask[batch_idx : batch_idx + 1].detach()
    nrmse = compute_nrmse(
        torch.abs(pred_image[batch_idx : batch_idx + 1].detach()),
        torch.abs(target_image[batch_idx : batch_idx + 1].detach()),
        sample_mask,
    ).item()
    caption = f"Sample {batch_idx} | R={R}x | nRMSE: {nrmse:.4f}"

    if zf_image is not None:
        zf_nrmse = compute_nrmse(
            torch.abs(zf_image[batch_idx : batch_idx + 1].detach()),
            torch.abs(target_image[batch_idx : batch_idx + 1].detach()),
            sample_mask,
        ).item()
        caption += f" | ZF nRMSE: {zf_nrmse:.4f}"

    if pred_vel is not None and target_vel is not None:
        rel_err, ang_err = compute_velocity_metrics(
            pred_vel[batch_idx : batch_idx + 1].detach(),
            target_vel[batch_idx : batch_idx + 1].detach(),
            sample_mask,
        )
        caption += f" | RelErr: {rel_err.item():.4f} | AngErr: {ang_err.item():.2f}°"

    wandb.log(
        {f"Visualizations/Reconstruction_R{R}": wandb.Image(fig, caption=caption)},
        commit=False,
    )

    plt.close(fig)
