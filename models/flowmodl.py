import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from utils.sense import centered_ifftn, conjugate_gradient


# -------------------------------------------------------------------------
# Complex <-> real channel helpers
# -------------------------------------------------------------------------
def to_real(z):
    """[B, C, T, Z, Y, X] complex -> [B, 2*C, T, Z, Y, X] real ([Re; Im])."""
    return torch.cat([z.real, z.imag], dim=1)


def to_complex(r):
    """Inverse of ``to_real``: [B, 2*C, ...] real -> [B, C, ...] complex."""
    c = r.shape[1] // 2
    return torch.complex(r[:, :c].contiguous(), r[:, c:].contiguous())


def _spatial_conv(conv, x):
    """Apply a Conv3d over (Z, Y, X) to every cardiac phase of a
    [B, C, T, Z, Y, X] tensor by folding T into the batch axis."""
    b, c, t, z, y, xx = x.shape
    x = x.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, z, y, xx)
    x = conv(x)
    return x.view(b, t, -1, z, y, xx).permute(0, 2, 1, 3, 4, 5)


# Normalisation of log(R) so that R in [10, 50] maps to roughly [-1, 1].
_LOG_R_MEAN = math.log(30.0)
_LOG_R_SCALE = math.log(50.0 / 10.0) / 2.0


def _usrate_column(usrate, batch_size, device):
    """Acceleration factor(s) as a (B, 1) fp32 tensor; a missing usrate maps to R = 30."""
    if usrate is None:
        u = torch.full((1, 1), 30.0, device=device, dtype=torch.float32)
    else:
        u = torch.as_tensor(usrate, device=device, dtype=torch.float32).reshape(-1, 1)
    if u.shape[0] == 1 and batch_size > 1:
        u = u.expand(batch_size, 1)
    return u.clamp_min(1.0)


# -------------------------------------------------------------------------
# (3+1)D spatiotemporal denoiser
# -------------------------------------------------------------------------
class SpatioTemporalBlock(nn.Module):
    """Residual (3+1)D block on a [B, C, Nt, Z, Y, X] feature tensor.

    A spatial 3D convolution over (Z, Y, X) per cardiac phase, a ReLU, then a
    temporal 1D convolution over Nt per voxel, followed by the residual
    connection and a final ReLU.
    """

    def __init__(
        self,
        num_filters,
        spatial_kernel=3,
        temporal_kernel=3,
        temporal_padding_mode="zeros",
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        self.spatial = nn.Conv3d(
            num_filters,
            num_filters,
            spatial_kernel,
            padding=spatial_kernel // 2,
            padding_mode=spatial_padding_mode,
        )
        self.temporal = nn.Conv1d(
            num_filters,
            num_filters,
            temporal_kernel,
            padding=temporal_kernel // 2,
            padding_mode=temporal_padding_mode,
        )
        self.act = nn.ReLU(inplace=True)

    def _apply_temporal(self, x):
        # Fold the spatial grid into the batch so Conv1d sees each voxel's
        # temporal profile separately.
        b, c, t, z, y, xx = x.shape
        x = x.permute(0, 3, 4, 5, 1, 2).reshape(b * z * y * xx, c, t)
        x = self.temporal(x)
        return x.view(b, z, y, xx, c, t).permute(0, 4, 5, 1, 2, 3)

    def forward(self, x):
        residual = x
        x = self.act(_spatial_conv(self.spatial, x))
        x = self._apply_temporal(x)
        return self.act(x + residual)


class FlowDenoiser(nn.Module):
    """FlowMoDL's learned prior D_theta (Fig. 1 of the paper).

    The real and imaginary parts of all Nv velocity encodings of
    x: [B, Nv, Nt, Z, Y, X] (complex) are stacked along the channel axis, so the
    inter-encoding phase relationships the velocity is derived from are
    processed jointly. The backbone is

        head Conv3d -> FiLM -> [SpatioTemporalBlock -> FiLM] x num_blocks -> tail Conv3d

    with a global residual connection around it. The tail is zero-initialised,
    so every denoiser starts out as the identity map.

    FiLM conditioning: the normalised log acceleration factor (optionally
    concatenated with a learned cascade-index embedding, see ``num_cascades``)
    is mapped by a small MLP to per-channel scales and shifts (gamma, beta),
    applied as ``feat * (1 + gamma) + beta``. The MLP's output layer is
    zero-initialised, so training starts unconditioned.
    """

    def __init__(
        self,
        in_channels,
        num_filters=32,
        num_blocks=5,
        spatial_kernel=3,
        temporal_kernel=3,
        temporal_circular=False,
        spatial_padding_mode="zeros",
        use_amp=False,
        cond_hidden=16,
        num_cascades=None,
        cascade_emb_dim=8,
    ):
        super().__init__()
        # bf16 autocast for the conv stack only; the complex output and the CG
        # data consistency downstream stay in fp32/complex64.
        self.use_amp = use_amp
        self.num_filters = num_filters
        real_ch = 2 * in_channels
        conv3d_kwargs = dict(
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            padding_mode=spatial_padding_mode,
        )

        self.head = nn.Conv3d(real_ch, num_filters, **conv3d_kwargs)
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    num_filters,
                    spatial_kernel,
                    temporal_kernel,
                    temporal_padding_mode="circular" if temporal_circular else "zeros",
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_blocks)
            ]
        )
        self.tail = nn.Conv3d(num_filters, real_ch, **conv3d_kwargs)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

        # A cascade-index embedding lets a weight-shared denoiser specialise to
        # the unrolled step it is applied at.
        self.use_cascade_film = num_cascades is not None
        cond_in_dim = 1
        if self.use_cascade_film:
            self.cascade_emb = nn.Embedding(num_cascades, cascade_emb_dim)
            cond_in_dim += cascade_emb_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, cond_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cond_hidden, 2 * num_filters),
        )
        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)

    def _film_params(self, usrate, cascade_idx, batch_size, device, dtype):
        """(gamma, beta), each [B, num_filters, 1, 1, 1, 1]."""
        u = _usrate_column(usrate, batch_size, device)
        parts = [(torch.log(u) - _LOG_R_MEAN) / _LOG_R_SCALE]
        if self.use_cascade_film and cascade_idx is not None:
            idx = torch.full(
                (u.shape[0],), int(cascade_idx), device=device, dtype=torch.long
            )
            parts.append(self.cascade_emb(idx).to(torch.float32))

        out = self.cond_mlp(torch.cat(parts, dim=-1)).to(dtype)
        gamma, beta = out.chunk(2, dim=-1)
        view = (-1, self.num_filters, 1, 1, 1, 1)
        return gamma.view(view), beta.view(view)

    def forward(self, x, usrate=None, cascade_idx=None):
        r = to_real(x)
        with torch.autocast(
            device_type=r.device.type, dtype=torch.bfloat16, enabled=self.use_amp
        ):
            feat = _spatial_conv(self.head, r)

            film = None
            if usrate is not None or (self.use_cascade_film and cascade_idx is not None):
                gamma, beta = self._film_params(
                    usrate, cascade_idx, feat.shape[0], feat.device, feat.dtype
                )
                film = lambda f: f * (1.0 + gamma) + beta  # noqa: E731
                feat = film(feat)

            for block in self.blocks:
                feat = block(feat)
                if film is not None:
                    feat = film(feat)

            correction = _spatial_conv(self.tail, feat)

        # Back to fp32 so the complex output is complex64, as the SENSE
        # operators and the CG solver expect.
        return to_complex(r + correction.float())


class MoDLDenoiser(nn.Module):
    """The denoiser of the original MoDL (Aggarwal, Mani & Jacob, IEEE TMI 2019),
    used for the MoDL baseline.

    A plain CNN estimates the noise/alias pattern N_w(x) and the denoised
    estimate is the residual D_w(x) = x - N_w(x). The CNN has
    ``num_conv_layers`` conv -> batch norm -> ReLU layers with ``num_filters``
    channels, where the last layer has no ReLU. Real and imaginary parts of all
    Nv encodings are stacked along the channel axis. The convolutions are 3D over
    (Z, Y, X) and applied to every cardiac phase independently: there is no
    temporal convolution and no conditioning. ``usrate`` and ``cascade_idx``
    are accepted for interface compatibility and ignored.

    Under gradient checkpointing the forward pass runs twice per step, so the
    batch-norm running statistics are updated twice. This only affects eval
    mode; set ``use_checkpointing: False`` to avoid it at the usual memory cost.
    """

    def __init__(
        self,
        in_channels,
        num_filters=64,
        num_conv_layers=5,
        spatial_kernel=3,
        spatial_padding_mode="zeros",
        use_amp=False,
        zero_init_tail=False,
    ):
        super().__init__()
        if num_conv_layers < 2:
            raise ValueError(
                f"num_conv_layers must be at least 2 (got {num_conv_layers})."
            )
        self.use_amp = use_amp
        real_ch = 2 * in_channels

        widths = [real_ch] + [num_filters] * (num_conv_layers - 1) + [real_ch]
        self.convs = nn.ModuleList(
            [
                nn.Conv3d(
                    widths[i],
                    widths[i + 1],
                    spatial_kernel,
                    padding=spatial_kernel // 2,
                    padding_mode=spatial_padding_mode,
                )
                for i in range(num_conv_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.BatchNorm3d(widths[i + 1]) for i in range(num_conv_layers)]
        )
        self.act = nn.ReLU(inplace=True)

        if zero_init_tail:
            nn.init.zeros_(self.convs[-1].weight)
            nn.init.zeros_(self.convs[-1].bias)

    def forward(self, x, usrate=None, cascade_idx=None):
        r = to_real(x)
        b, c, t, z, y, xx = r.shape

        with torch.autocast(
            device_type=r.device.type, dtype=torch.bfloat16, enabled=self.use_amp
        ):
            feat = r.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, z, y, xx)
            last = len(self.convs) - 1
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                feat = norm(conv(feat))
                if i != last:
                    feat = self.act(feat)
            noise = feat.view(b, t, c, z, y, xx).permute(0, 2, 1, 3, 4, 5)

        return to_complex(r - noise.float())


# -------------------------------------------------------------------------
# Unrolled network
# -------------------------------------------------------------------------
class FlowMoDLCascade(nn.Module):
    """One unrolled cascade: z = D(x, R), then solve
    (A^H A + lam(R) I) x = A^H y + lam(R) z with CG."""

    def __init__(
        self,
        denoiser,
        dc_iters=6,
        lam_init=0.05,
        lam_usrate=False,
        lam_cond_hidden=8,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.dc_iters = dc_iters
        # Strictly positive DC weight, parameterised through softplus.
        self.lam_raw = nn.Parameter(torch.tensor(math.log(math.expm1(lam_init))))

        # Optional acceleration conditioning of the DC weight: an MLP maps
        # log(R) to an additive offset on lam_raw. Its output layer is
        # zero-initialised, so lam == softplus(lam_raw) at initialisation.
        self.lam_usrate = lam_usrate
        if lam_usrate:
            self.lam_mlp = nn.Sequential(
                nn.Linear(1, lam_cond_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(lam_cond_hidden, 1),
            )
            nn.init.zeros_(self.lam_mlp[-1].weight)
            nn.init.zeros_(self.lam_mlp[-1].bias)

    @property
    def lam(self):
        """Unconditioned DC weight softplus(lam_raw), used for logging."""
        return F.softplus(self.lam_raw)

    def _effective_lam(self, usrate, ref):
        """DC weight for the CG solve: a scalar when unconditioned, otherwise a
        per-sample tensor broadcastable over ``ref`` ([B, Nv, Nt, Z, Y, X])."""
        if not self.lam_usrate or usrate is None:
            return self.lam
        u = _usrate_column(usrate, ref.shape[0], ref.device)
        lam = F.softplus(self.lam_raw + self.lam_mlp(torch.log(u)).reshape(-1))
        return lam.view(-1, *([1] * (ref.ndim - 1)))

    def forward(
        self, x, kspace_acq, mask, forward_op, adjoint_op, usrate=None, cascade_idx=None
    ):
        z = self.denoiser(x, usrate=usrate, cascade_idx=cascade_idx)
        lam = self._effective_lam(usrate, z)

        def normal(v):
            return adjoint_op(mask * forward_op(v)) + lam * v

        rhs = adjoint_op(mask * kspace_acq) + lam * z
        return conjugate_gradient(normal, rhs, z, self.dc_iters)


class FlowMoDL(nn.Module):
    """Unrolled MoDL-style reconstruction network for 4D flow MRI.

    Starting from the zero-filled estimate x0 = A^H y, each of the
    ``num_layers`` cascades applies a learned denoiser followed by a
    conjugate-gradient data-consistency update built on the SENSE operators
    passed to ``forward``. During training the output of every cascade is
    returned (for deep supervision); at inference only the final estimate.

    Args:
        num_layers: number of unrolled cascades K.
        share_weights: use one denoiser for all cascades (original MoDL) instead
            of an independent denoiser per cascade (FlowMoDL).
        share_lam: use one DC weight for all cascades (original MoDL).
        dc_iters: CG iterations J per cascade.
        lam_init: initial DC weight.
        lam_usrate: condition the DC weight on the acceleration factor.
        cascade_film: feed a cascade-index embedding into the denoiser's FiLM
            generator. Only has an effect with ``share_weights=True``.
        denoiser_type: "cnn" (FlowDenoiser) or "modl" (MoDLDenoiser).
        layer_params: keyword arguments of the denoiser.
        decouple_readout: move the fully sampled readout axis to image space
            before the first cascade. The normal operator then decomposes into
            independent 2D problems per readout position, so every CG iteration
            uses 2D FFTs while solving exactly the same problem. Requires SENSE
            operators over the two phase-encode axes only
            (``utils.sense.make_sense_ops(smaps, decouple_readout=True)``).
        use_checkpointing: gradient checkpointing of every cascade in training.
        use_amp: bf16 autocast for the denoiser's convolutions.
    """

    DENOISERS = {
        "cnn": FlowDenoiser,
        "modl": MoDLDenoiser,
    }

    def __init__(
        self,
        num_layers=6,
        use_checkpointing=True,
        share_weights=True,
        dc_iters=6,
        lam_init=0.05,
        layer_params=None,
        use_amp=False,
        denoiser_type="cnn",
        decouple_readout=False,
        readout_dim=-1,
        cascade_film=False,
        lam_usrate=False,
        share_lam=False,
    ):
        super().__init__()
        if denoiser_type not in self.DENOISERS:
            raise ValueError(
                f"Unknown denoiser_type={denoiser_type!r}; "
                f"choose from {sorted(self.DENOISERS)}"
            )
        self.num_layers = num_layers
        self.use_checkpointing = use_checkpointing
        self.decouple_readout = decouple_readout
        self.readout_dim = readout_dim
        # The cascade-index embedding is only useful when one denoiser is
        # reused across cascades; untied denoisers each see a single index.
        self.cascade_film = cascade_film and share_weights

        denoiser_cls = self.DENOISERS[denoiser_type]
        denoiser_kwargs = dict(layer_params or {}, use_amp=use_amp)
        if self.cascade_film:
            denoiser_kwargs["num_cascades"] = num_layers

        shared_denoiser = denoiser_cls(**denoiser_kwargs) if share_weights else None
        self.cascades = nn.ModuleList(
            [
                FlowMoDLCascade(
                    (
                        shared_denoiser
                        if shared_denoiser is not None
                        else denoiser_cls(**denoiser_kwargs)
                    ),
                    dc_iters,
                    lam_init,
                    lam_usrate=lam_usrate,
                )
                for _ in range(num_layers)
            ]
        )

        # Rebinding lam_raw to one Parameter shares it across cascades; autograd
        # accumulates the gradient of every cascade into it.
        if share_lam:
            shared_lam_raw = self.cascades[0].lam_raw
            for cascade in self.cascades[1:]:
                cascade.lam_raw = shared_lam_raw

    def forward(
        self,
        kspace_acq,
        undersampling_mask,
        forward_op,
        adjoint_op,
        usrate=None,
        training=True,
    ):
        mask = undersampling_mask
        if self.decouple_readout:
            kspace_acq = centered_ifftn(kspace_acq, dim=(self.readout_dim,))

        x = adjoint_op(kspace_acq * mask)
        predictions = []

        for k, cascade in enumerate(self.cascades):
            cascade_idx = k if self.cascade_film else None
            args = (x, kspace_acq, mask, forward_op, adjoint_op, usrate, cascade_idx)
            if self.use_checkpointing and training:
                x = checkpoint.checkpoint(cascade, *args, use_reentrant=False)
            else:
                x = cascade(*args)
            if training:
                predictions.append(x)

        return predictions if training else [x]
