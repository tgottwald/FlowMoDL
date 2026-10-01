"""Model loading and inference helpers shared by training, evaluation and
prediction.

Every reconstruction model implements

    model(kspace_acq, undersampling_mask, forward_op, adjoint_op,
          usrate=None, training=False) -> [prediction, ...]

and two properties of its config decide how it has to be called:

* ``layer_params.in_channels == 1`` (FlowVN): the model reconstructs one
  velocity encoding at a time, so the Nv encodings are processed separately and
  concatenated afterwards.
* ``decouple_readout: true`` (FlowMoDL, MoDL): the model moves the fully sampled
  readout axis to image space itself, so the SENSE operators must transform
  only the two phase-encode axes (see ``utils.sense.make_sense_ops``).
"""

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils.sense import make_sense_ops


def model_call_flags(model_cfg):
    """(per_encoding, decouple_readout) for a model config (see module docstring)."""
    per_encoding = OmegaConf.select(model_cfg, "layer_params.in_channels") == 1
    decouple_readout = bool(OmegaConf.select(model_cfg, "decouple_readout", default=False))
    return per_encoding, decouple_readout


def load_model(ckpt_path, model_config, device):
    """Instantiate the model described by the Hydra model config file
    ``model_config`` (e.g. configs/model/flowmodl.yaml) and load its weights.

    Returns (model, per_encoding, decouple_readout).
    """
    cfg = OmegaConf.load(model_config)
    model = instantiate(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return (model, *model_call_flags(cfg))


def reconstruct(
    model,
    kspace_us,
    mask,
    smaps,
    usrate=None,
    per_encoding=False,
    decouple_readout=False,
):
    """Final reconstruction of ``model`` for one batch, (B, Nv, Nt, Z, Y, X)."""
    fwd, adj = make_sense_ops(smaps, decouple_readout)

    def run(kspace):
        return model(
            kspace, mask, forward_op=fwd, adjoint_op=adj, usrate=usrate, training=False
        )[-1]

    if per_encoding:
        return torch.cat(
            [run(kspace_us[:, v : v + 1]) for v in range(kspace_us.shape[1])], dim=1
        )
    return run(kspace_us)
