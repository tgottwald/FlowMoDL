"""
Train FlowMoDL, MoDL or FlowVN.

    uv run train.py                              # FlowMoDL (configs/config_flowmodl.yaml)
    uv run train.py --config-name config_modl    # MoDL baseline
    uv run train.py --config-name config_flowvn  # FlowVN baseline

Any config value can be overridden on the command line, e.g.
``data.root=/path/to/TaskR1R2_compressed/TrainSet seed=43``.

Every ``validation.val_interval`` epochs the model is evaluated on the
validation cases at ``validation.val_accelerations`` (logged as val/*). On every
``validation.checkpoint_eval_every``-th of these validations (and the first
one), it is also evaluated at ``validation.checkpoint_accelerations``
(val_ckpt/*), and the checkpoint is replaced if the epoch has the best rank-sum
score so far (utils.metrics.composite_checkpoint_score).
"""

import gc
import os
import time

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_utils import resolve_array_seed, seed_everything, seed_worker
from data.dataset import CMRx4DFlowDataset
from utils.inference import model_call_flags, reconstruct
from utils.metrics import (
    compute_complex_diff_err,
    compute_nrmse,
    compute_ssim,
    compute_velocity_metrics,
    composite_checkpoint_score,
    extract_velocity,
)
from utils.sense import adjoint_op, make_sense_ops
from utils.transformations import reconstruct_target_and_normalize
from utils.visualization_logging import log_validation_visualizations

METRIC_KEYS = ("nRMSE", "SSIM", "RelErr", "AngErr", "ComplexErr")


def prepare_batch(batch, device, stage_cfg):
    """Move a batch to ``device``; build kspace_us / target_image from
    ``kdata_full`` if the dataset deferred that to the GPU."""
    data = {
        "mask": batch["undersampling_mask"].to(device, non_blocking=True),
        "smaps": batch["sensitivity_maps"].to(device, non_blocking=True),
        "segmask": batch["segmask"].to(device, non_blocking=True),
        "v_enc": batch["v_enc"].to(device, non_blocking=True),
        "usrate": batch["usrate_true"].to(device, non_blocking=True),
    }
    if stage_cfg.get("reconstruct_target_on_gpu", False):
        kdata_full = batch["kdata_full"].to(device, non_blocking=True).contiguous()
        data["kspace_us"], data["target_image"], _ = reconstruct_target_and_normalize(
            kdata_full,
            data["mask"],
            data["smaps"],
            temporal_chunk_size=stage_cfg.get("reconstruct_temporal_chunk_size", None),
        )
    else:
        data["kspace_us"] = batch["kspace_us"].to(device, non_blocking=True)
        data["target_image"] = batch["target_image"].to(device, non_blocking=True)
    data["target_velocity"] = extract_velocity(data["target_image"], data["v_enc"])
    return data


def compute_metrics(pred, data, include_complex_err=True):
    """Validation metrics of one batch as floats."""
    target = data["target_image"]
    segmask = data["segmask"]
    rel_err, ang_err = compute_velocity_metrics(
        extract_velocity(pred, data["v_enc"]), data["target_velocity"], segmask
    )
    metrics = {
        "nRMSE": compute_nrmse(pred.abs(), target.abs(), segmask).item(),
        "SSIM": compute_ssim(pred.abs(), target.abs(), segmask).item(),
        "RelErr": rel_err.item(),
        "AngErr": ang_err.item(),
    }
    if include_complex_err:
        metrics["ComplexErr"] = compute_complex_diff_err(pred, target, segmask).item()
    return metrics


# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------
def train_step(model, criterion, data, tau, K, epoch, cfg):
    """Forward and backward pass of one batch.

    Returns (unscaled loss, detached final prediction). Models with
    ``in_channels == 1`` (FlowVN) reconstruct the velocity encodings
    separately, ``training.nv_chunk_size`` encodings per model call.
    """
    per_encoding, decouple_readout = model_call_flags(cfg.model)
    accumulation_steps = cfg.training.accumulation_steps
    kspace_us, mask, smaps, usrate = (
        data["kspace_us"],
        data["mask"],
        data["smaps"],
        data["usrate"],
    )
    loss_context = dict(
        target_velocity=data["target_velocity"],
        segmask=data["segmask"],
        v_enc=data["v_enc"],
        epoch=epoch,
    )

    if not per_encoding:
        fwd, adj = make_sense_ops(smaps, decouple_readout)
        predictions = model(
            kspace_us, mask, forward_op=fwd, adjoint_op=adj, usrate=usrate, training=True
        )
        loss = criterion(predictions, data["target_image"], tau, K, **loss_context)
        (loss / accumulation_steps).backward()
        return loss.item(), predictions[-1].detach()

    # Per-encoding models: fold `g` encodings at a time into the batch axis.
    Nv = kspace_us.shape[1]
    g = max(1, int(cfg.training.get("nv_chunk_size", 1)))

    def run_chunk(start, end):
        n = end - start
        fwd, adj = make_sense_ops(smaps.repeat_interleave(n, dim=0), decouple_readout)
        preds = model(
            kspace_us[:, start:end].reshape(-1, 1, *kspace_us.shape[2:]),
            mask.repeat_interleave(n, dim=0),
            forward_op=fwd,
            adjoint_op=adj,
            usrate=usrate.repeat_interleave(n, dim=0),
            training=True,
        )
        return [p.view(-1, n, *p.shape[2:]) for p in preds]

    if not criterion.separable_over_encodings:
        # The loss couples the encodings (velocity terms): assemble all of them
        # before a single backward pass.
        chunks = [run_chunk(s, min(s + g, Nv)) for s in range(0, Nv, g)]
        predictions = [torch.cat(stage, dim=1) for stage in zip(*chunks)]
        loss = criterion(predictions, data["target_image"], tau, K, **loss_context)
        (loss / accumulation_steps).backward()
        return loss.item(), predictions[-1].detach()

    # Separable loss: back-propagate every chunk immediately so that only one
    # chunk's graph is alive at a time. Weighting by n / Nv makes the summed
    # gradient identical to the gradient of the loss over all encodings.
    total_loss = 0.0
    final_chunks = []
    for s in range(0, Nv, g):
        e = min(s + g, Nv)
        preds = run_chunk(s, e)
        loss = criterion(preds, data["target_image"][:, s:e], tau, K) * ((e - s) / Nv)
        (loss / accumulation_steps).backward()
        total_loss += loss.item()
        final_chunks.append(preds[-1].detach())
        del preds, loss
    return total_loss, torch.cat(final_chunks, dim=1)


def train_epoch(model, data_loader, optimizer, criterion, device, epoch, cfg, global_step):
    model.train()
    epoch_loss = 0.0
    K = cfg.model.num_layers
    accumulation_steps = cfg.training.accumulation_steps
    gradient_clip_val = cfg.training.get("gradient_clip_val", None)

    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch}")):
        data = prepare_batch(batch, device, cfg.training)
        # Deep-supervision weights w_k = exp(-tau (K - k)) sharpen towards the
        # last cascade as training progresses.
        tau = global_step * cfg.training.tau_multiplier
        loss, final_pred = train_step(model, criterion, data, tau, K, epoch, cfg)

        is_last = batch_idx + 1 == len(data_loader)
        if (batch_idx + 1) % accumulation_steps == 0 or is_last:
            if gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                metrics = compute_metrics(final_pred, data, include_complex_err=False)
            log_payload = {
                "train/loss": loss,
                "global_step": global_step,
                **{f"train/{k}": v for k, v in metrics.items()},
            }
            for name, scale in getattr(criterion, "curriculum_scales", {}).items():
                log_payload[f"train/curriculum_{name}_scale"] = scale
            wandb.log(log_payload)
            global_step += 1
            epoch_loss += loss

    return epoch_loss / len(data_loader), global_step


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------
def collect_reusable_metrics(metrics, accelerations, metrics_prefix="val"):
    """Per-R metrics of ``accelerations`` contained in a ``validate`` result, as
    {R: {metric: value}}."""
    return {
        R: {k: metrics[f"{metrics_prefix}/{k}_R{R}"] for k in METRIC_KEYS}
        for R in accelerations
        if f"{metrics_prefix}/nRMSE_R{R}" in metrics
    }


def validate(
    model, val_loaders, device, epoch, cfg, metrics_prefix="val", reuse_metrics=None
):
    """Evaluate ``model`` with one loader per acceleration factor.

    Accelerations in ``reuse_metrics`` ({R: {metric: value}}) were already
    evaluated on the same weights and are only re-logged under
    ``metrics_prefix``.
    """
    model.eval()
    per_encoding, decouple_readout = model_call_flags(cfg.model)
    compute_on_cpu = cfg.validation.get("compute_on_cpu", False)
    # Full volumes have case-dependent shapes, so cuDNN autotuning would re-run
    # for almost every case.
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False

    metrics = {}
    totals = {k: 0.0 for k in METRIC_KEYS}
    total_samples = 0

    with torch.inference_mode():
        for R, loader in val_loaders.items():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            n_cases = len(loader.dataset)

            if reuse_metrics and R in reuse_metrics:
                for key, value in reuse_metrics[R].items():
                    metrics[f"{metrics_prefix}/{key}_R{R}"] = value
                    totals[key] += value * n_cases
                total_samples += n_cases
                continue

            r_sums = {k: 0.0 for k in METRIC_KEYS}
            for i, batch in enumerate(
                tqdm(loader, desc=f"Validating Epoch {epoch} [R={R}]")
            ):
                data = prepare_batch(batch, device, cfg.validation)
                pred = reconstruct(
                    model,
                    data["kspace_us"],
                    data["mask"],
                    data["smaps"],
                    usrate=data["usrate"],
                    per_encoding=per_encoding,
                    decouple_readout=decouple_readout,
                )
                if compute_on_cpu:
                    pred = pred.cpu()
                    data = {k: v.cpu() for k, v in data.items()}

                if i == 0:
                    zero_filled = adjoint_op(data["kspace_us"] * data["mask"], data["smaps"])
                    log_validation_visualizations(
                        pred_image=pred,
                        target_image=data["target_image"],
                        segmask=data["segmask"],
                        epoch=epoch,
                        R=R,
                        pred_vel=extract_velocity(pred, data["v_enc"]),
                        target_vel=data["target_velocity"],
                        zf_image=zero_filled,
                    )
                    del zero_filled

                bz = pred.shape[0]
                for key, value in compute_metrics(pred, data).items():
                    r_sums[key] += value * bz
                total_samples += bz
                del data, pred

            for key, total in r_sums.items():
                metrics[f"{metrics_prefix}/{key}_R{R}"] = total / n_cases
                totals[key] += total

    for key, total in totals.items():
        metrics[f"{metrics_prefix}/{key}"] = total / total_samples
    metrics["epoch"] = epoch
    wandb.log(metrics)

    torch.backends.cudnn.benchmark = prev_cudnn_benchmark
    return metrics


# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------
def make_loader(dataset, stage_cfg, shuffle, seed):
    kwargs = dict(
        batch_size=stage_cfg.batch_size,
        shuffle=shuffle,
        num_workers=stage_cfg.num_workers,
        pin_memory=stage_cfg.pin_memory,
    )
    if seed is not None:
        kwargs["worker_init_fn"] = seed_worker
        if shuffle:
            kwargs["generator"] = torch.Generator().manual_seed(seed)
    if stage_cfg.num_workers > 0:
        kwargs["prefetch_factor"] = stage_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = stage_cfg.get("persistent_workers", False)
    return DataLoader(dataset, **kwargs)


def make_val_loaders(accelerations, cfg, seed, reuse=None):
    """One validation loader per acceleration factor; loaders already in
    ``reuse`` are shared instead of rebuilt."""
    loaders = {}
    for R in accelerations:
        if reuse and R in reuse:
            loaders[R] = reuse[R]
            continue
        dataset = CMRx4DFlowDataset(
            root_dir=cfg.data.root,
            split_yaml_path=cfg.data.split_path,
            mode="val",
            synthetic=cfg.validation.synthetic,
            data_format=cfg.data_format,
            val_acceleration=R,
            reconstruct_target_on_gpu=cfg.validation.get(
                "reconstruct_target_on_gpu", False
            ),
            seed=seed,
            cache_in_memory=cfg.validation.get("cache_in_memory", False),
        )
        loaders[R] = make_loader(dataset, cfg.validation, shuffle=False, seed=seed)
    return loaders


def save_checkpoint(path, model):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


@hydra.main(version_base=None, config_path="configs", config_name="config_flowmodl")
def main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = True
    seed = resolve_array_seed(cfg.get("seed", None))

    run_name = cfg.wandb.name if seed is None else f"{cfg.wandb.name}_seed{seed}"
    checkpoint_path = os.path.join(
        cfg.get("checkpoint_dir", "checkpoints"), f"{run_name}_best.pth"
    )
    if os.path.exists(checkpoint_path) and not cfg.get("overwrite_existing", False):
        print(
            f"Checkpoint '{checkpoint_path}' already exists -- skipping training. "
            "Set overwrite_existing=true to retrain."
        )
        return

    if seed is not None:
        seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = cfg.training.epochs
    val_interval = cfg.validation.val_interval
    train_on_all_data = cfg.data.get("train_on_all_data", False)

    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        group=cfg.wandb.name,
        dir=cfg.wandb.dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    model = instantiate(cfg.model).to(device)
    criterion = instantiate(cfg.loss).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler_cfg = cfg.training.get("lr_scheduler", None)
    scheduler = (
        instantiate(lr_scheduler_cfg, optimizer=optimizer)
        if lr_scheduler_cfg is not None
        else None
    )

    if train_on_all_data:
        print(
            "data.train_on_all_data=true: training on every case of the split "
            "manifest. Checkpoints are snapshots of the latest weights; val/* "
            "metrics are only a fit monitor."
        )

    train_dataset = CMRx4DFlowDataset(
        root_dir=cfg.data.root,
        split_yaml_path=cfg.data.split_path,
        mode="train",
        ignore_split=train_on_all_data,
        data_format=cfg.data_format,
        spatial_crop_size=tuple(cfg.data.spatial_crop_size),
        spatial_edge_crop_prob=cfg.data.get("spatial_edge_crop_prob", 0.0),
        temporal_crop_size=cfg.data.temporal_crop_size,
        assume_temporal_cycle=cfg.data.assume_temporal_cycle,
        reconstruct_target_on_gpu=cfg.training.get("reconstruct_target_on_gpu", False),
        seed=seed,
    )
    train_loader = make_loader(train_dataset, cfg.training, shuffle=True, seed=seed)

    val_loaders = make_val_loaders(cfg.validation.val_accelerations, cfg, seed)
    checkpoint_accelerations = list(cfg.validation.checkpoint_accelerations)
    checkpoint_eval_every = cfg.validation.checkpoint_eval_every
    # No checkpoint selection when training on all data.
    checkpoint_val_loaders = (
        {}
        if train_on_all_data
        else make_val_loaders(checkpoint_accelerations, cfg, seed, reuse=val_loaders)
    )

    global_step = 0
    validation_count = 0
    val_history = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()
        train_dataset.set_epoch(epoch)
        wandb.log({"epoch": epoch})

        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, cfg, global_step
        )
        train_time_s = time.perf_counter() - epoch_start

        # The scheduler steps once per epoch.
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else None
        if scheduler is not None:
            scheduler.step()

        val_time_s = None
        if epoch % val_interval == 0:
            val_start = time.perf_counter()
            val_metrics = validate(model, val_loaders, device, epoch, cfg)
            validation_count += 1

            is_checkpoint_eval = not train_on_all_data and (
                validation_count == 1 or validation_count % checkpoint_eval_every == 0
            )
            if is_checkpoint_eval:
                ckpt_metrics = validate(
                    model,
                    checkpoint_val_loaders,
                    device,
                    epoch,
                    cfg,
                    metrics_prefix="val_ckpt",
                    reuse_metrics=collect_reusable_metrics(
                        val_metrics, checkpoint_accelerations
                    ),
                )
                val_history.append(
                    {"epoch": epoch}
                    | {
                        k: ckpt_metrics[f"val_ckpt/{k}"]
                        for k in ("nRMSE", "SSIM", "RelErr", "AngErr")
                    }
                )
                composite_scores, is_best = composite_checkpoint_score(val_history)
                wandb.log(
                    {"val_ckpt/composite_score": composite_scores[-1], "epoch": epoch}
                )
                if is_best:
                    save_checkpoint(checkpoint_path, model)
                    print(
                        f"Epoch {epoch}: new best model (composite="
                        f"{composite_scores[-1]:.4f}, R={checkpoint_accelerations}) "
                        + " ".join(
                            f"{k}={ckpt_metrics[f'val_ckpt/{k}']:.4f}"
                            for k in ("nRMSE", "SSIM", "RelErr", "AngErr")
                        )
                        + f" -> {checkpoint_path}"
                    )
            val_time_s = time.perf_counter() - val_start

        if train_on_all_data and (epoch % val_interval == 0 or epoch == num_epochs):
            save_checkpoint(checkpoint_path, model)
            print(f"Epoch {epoch}: saved latest weights -> {checkpoint_path}")

        epoch_time_s = time.perf_counter() - epoch_start
        epoch_log = {
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "time/train_epoch_s": train_time_s,
            "time/epoch_total_s": epoch_time_s,
        }
        if current_lr is not None:
            epoch_log["lr"] = current_lr
        if val_time_s is not None:
            epoch_log["time/val_epoch_s"] = val_time_s
        for k, cascade in enumerate(getattr(model, "cascades", [])):
            epoch_log[f"model/lam_cascade_{k}"] = cascade.lam.item()
        wandb.log(epoch_log)
        print(
            f"Epoch {epoch}: loss={train_loss:.4f}, train={train_time_s:.1f}s, "
            + (f"val={val_time_s:.1f}s, " if val_time_s is not None else "")
            + f"total={epoch_time_s:.1f}s"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
