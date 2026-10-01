"""Re-implementation of FlowVN (Vishnevskiy, Walheim & Kozerke, Nature Machine
Intelligence 2020), used as a baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def _interpolate_knots(knots, x_scaled):
    """Piecewise-linear interpolation of per-row knot values.

    knots:    [R, K] knot values (one row per filter/output).
    x_scaled: [B, R, N] positions in knot units, already clamped to [0, K-1).
    """
    x_floor = torch.floor(x_scaled)
    k = x_scaled - x_floor
    idx_floor = x_floor.long()
    knots_view = knots.unsqueeze(0).expand(x_scaled.shape[0], -1, -1)
    y_floor = torch.gather(knots_view, 2, idx_floor)
    y_ceil = torch.gather(knots_view, 2, idx_floor + 1)
    return y_floor * (1 - k) + y_ceil * k


class AdaptiveLinearInterpolation(nn.Module):
    """Learnable per-filter activation function, parameterised by ``num_knots``
    equally spaced knots on [-init_range, init_range] and linear interpolation
    in between."""

    def __init__(self, num_filters, num_knots=91, init_range=3.0, stddev_init=0.05):
        super().__init__()
        self.num_filters = num_filters
        self.num_knots = num_knots

        self.register_buffer("activation_range", torch.tensor(float(init_range)))
        # Not used by the forward pass; persisted so that existing FlowVN
        # checkpoints keep loading with strict=True.
        self.register_buffer("running_max", torch.tensor(float(init_range)))

        knots_init = torch.empty(num_filters, num_knots)
        nn.init.trunc_normal_(
            knots_init,
            mean=0.0,
            std=stddev_init,
            a=-2.0 * stddev_init,
            b=2.0 * stddev_init,
        )
        self.knots = nn.Parameter(knots_init)

    @property
    def omega(self):
        return (self.activation_range * 2.0) / (self.num_knots - 1)

    def forward(self, x):
        original_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1)
        x_scaled = (x + self.activation_range) / self.omega
        x_scaled = torch.clamp(x_scaled, 0.00001, self.num_knots - 1.00001)
        return _interpolate_knots(self.knots, x_scaled).view(original_shape)


class USRateModulation(nn.Module):
    """Learnable, positive function of the acceleration factor on
    [min_rate, max_rate], used to modulate FlowVN's step weights."""

    def __init__(
        self, n_outputs=1, min_rate=9.0, max_rate=51.0, num_knots=91, stddev_init=0.1
    ):
        super().__init__()
        self.num_knots = num_knots
        self.n_outputs = n_outputs
        self.min_rate = float(min_rate)
        self.max_rate = float(max_rate)
        self.omega = (self.max_rate - self.min_rate) / (self.num_knots - 1)

        init_values = (
            torch.ones(self.n_outputs, self.num_knots)
            + torch.randn(self.n_outputs, self.num_knots) * stddev_init
        )
        self.knots = nn.Parameter(init_values)

    def forward(self, usrate):
        if usrate.dim() == 0:
            usrate = usrate.unsqueeze(0)
        usrate = usrate.view(-1, 1, 1).expand(-1, self.n_outputs, -1)
        x_scaled = (usrate - self.min_rate) / self.omega
        x_scaled = torch.clamp(x_scaled, 0.00001, self.num_knots - 1.00001)
        return F.softplus(_interpolate_knots(self.knots, x_scaled).squeeze(-1))


def init_zero_mean_norm_ball_(tensor, dim=(1, 2, 3, 4), eps=1e-6):
    """In-place initialisation of a kernel to zero mean and unit L2 norm across
    ``dim`` (in_channels, D, H, W)."""
    with torch.no_grad():
        tensor.sub_(tensor.mean(dim=dim, keepdim=True))
        tensor.div_(torch.sqrt(torch.sum(tensor**2, dim=dim, keepdim=True) + eps))
    return tensor


class FilterBank3D(nn.Module):
    """3D filter bank K, learned activation phi and the tied adjoint K^T,
    applied separately to the real and imaginary parts."""

    def __init__(
        self,
        in_channels,
        num_filters=8,
        kernel_size=5,
        init="zero_mean_norm_ball",
        init_std=0.01,
        bank_interp_params=None,
        use_amp=False,
    ):
        super().__init__()
        self.use_amp = use_amp
        self.conv = nn.Conv3d(
            in_channels, num_filters, kernel_size, padding="same", bias=False
        )
        if init == "zero_mean_norm_ball":
            nn.init.normal_(self.conv.weight, mean=0.0, std=init_std)
            init_zero_mean_norm_ball_(self.conv.weight, dim=(1, 2, 3, 4))

        self.conv_transpose = nn.ConvTranspose3d(
            num_filters, in_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        # Tie the weights of the filter bank and its adjoint.
        self.conv_transpose.weight = self.conv.weight

        self.activation = AdaptiveLinearInterpolation(
            num_filters, **(bank_interp_params or {})
        )

    def apply_complex(self, x_real, x_imag):
        device_type = x_real.device.type
        # Only the convolutions run under bf16 autocast. The knot-interpolated
        # activation stays in fp32: bf16 rounding can move a value into a
        # neighbouring knot bin.
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=self.use_amp
        ):
            dx_real = self.conv(x_real)
            dx_imag = self.conv(x_imag)

        psi_real = self.activation(dx_real.float())
        psi_imag = self.activation(dx_imag.float())

        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=self.use_amp
        ):
            out_real = self.conv_transpose(psi_real)
            out_imag = self.conv_transpose(psi_imag)

        return torch.complex(out_real.float(), out_imag.float())


class FlowVNRegularization(nn.Module):
    """Regulariser gradient built from four 3D filter banks, one per
    3D hyperplane of the (T, Z, Y, X) volume.

    Each bank convolves over three of the four axes; the remaining axis is
    folded into the batch dimension. With ``lowmem=True`` that axis is looped
    over instead, which is mathematically identical but keeps only one slice's
    activations alive at a time.
    """

    # (bank name, permutation of [B, C, T, Z, Y, X] into [B, batch axis, C, d1, d2, d3])
    BANKS = (
        ("bank_zyx", (0, 2, 1, 3, 4, 5)),  # convolves Z, Y, X; batch axis T
        ("bank_zyt", (0, 5, 1, 3, 4, 2)),  # convolves Z, Y, T; batch axis X
        ("bank_yxt", (0, 3, 1, 4, 5, 2)),  # convolves Y, X, T; batch axis Z
        ("bank_zxt", (0, 4, 1, 3, 5, 2)),  # convolves Z, X, T; batch axis Y
    )

    def __init__(
        self,
        in_channels,
        num_filters=8,
        filter_kernel_size=5,
        filter_init="zero_mean_norm_ball",
        filter_init_std=0.01,
        bank_interp_params=None,
        use_amp=False,
        lowmem=False,
    ):
        super().__init__()
        self.num_filters = num_filters
        self.lowmem = lowmem
        for name, _ in self.BANKS:
            setattr(
                self,
                name,
                FilterBank3D(
                    in_channels,
                    num_filters,
                    kernel_size=filter_kernel_size,
                    init=filter_init,
                    init_std=filter_init_std,
                    bank_interp_params=bank_interp_params,
                    use_amp=use_amp,
                ),
            )

    def _apply_bank(self, bank, x, perm):
        inverse = tuple(perm.index(i) for i in range(len(perm)))
        xp_real = x.real.permute(perm)
        xp_imag = x.imag.permute(perm)
        b, n = xp_real.shape[:2]

        if self.lowmem:
            out = torch.empty(xp_real.shape, dtype=x.dtype, device=x.device)
            for i in range(n):
                out[:, i] = bank.apply_complex(xp_real[:, i], xp_imag[:, i])
        else:
            flat_shape = (b * n, *xp_real.shape[2:])
            out = bank.apply_complex(
                xp_real.reshape(flat_shape), xp_imag.reshape(flat_shape)
            ).view(xp_real.shape)
        return out.permute(inverse)

    def forward(self, x):
        # x: [B, C, Nt, Z, Y, X] complex
        reg_total = None
        for name, perm in self.BANKS:
            reg = self._apply_bank(getattr(self, name), x, perm)
            if reg_total is None:
                reg_total = reg
            else:
                reg_total += reg
            del reg
        return reg_total / self.num_filters


class FlowVNLayer(nn.Module):
    """One variational-network step with momentum:

        g_k = lam_r * grad_reg(p_k) + lam_d * A^H phi_d(A p_k - y)
        s_{k+1} = alpha * s_k + g_k
        p_{k+1} = p_k - s_{k+1}
    """

    def __init__(
        self,
        in_channels,
        num_filters=8,
        use_usrate_mod=False,
        reg_params=None,
        interp_params=None,
        usrate_params=None,
        use_amp=False,
        lowmem=False,
    ):
        super().__init__()
        self.use_usrate_mod = use_usrate_mod

        self.regularization = FlowVNRegularization(
            in_channels,
            num_filters,
            use_amp=use_amp,
            lowmem=lowmem,
            **(reg_params or {}),
        )
        self.data_activation = AdaptiveLinearInterpolation(
            num_filters=in_channels, **(interp_params or {})
        )

        if self.use_usrate_mod:
            usrate_params = usrate_params or {}
            self.lamb_ru_mod = USRateModulation(n_outputs=1, **usrate_params)
            self.lamb_du_mod = USRateModulation(n_outputs=1, **usrate_params)
        else:
            self.lamb_ru = nn.Parameter(torch.tensor(1.0))
            self.lamb_du = nn.Parameter(torch.tensor(1.0))

        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, p_k, s_k, kspace_acq, undersampling_mask, forward_op, adjoint_op, usrate
    ):
        b, c = p_k.shape[:2]

        x_proj = forward_op(p_k) - kspace_acq
        x_proj_flat = x_proj.view(b, c, -1)
        x_proj_act = torch.complex(
            self.data_activation(x_proj_flat.real).view(x_proj.shape),
            self.data_activation(x_proj_flat.imag).view(x_proj.shape),
        )

        grad_data = adjoint_op(undersampling_mask * x_proj_act)
        grad_reg = self.regularization(p_k)

        if self.use_usrate_mod:
            lamb_ru = self.lamb_ru_mod(usrate).view(b, 1, 1, 1, 1, 1)
            lamb_du = self.lamb_du_mod(usrate).view(b, 1, 1, 1, 1, 1)
        else:
            lamb_ru = self.lamb_ru
            lamb_du = self.lamb_du

        g_k = (grad_reg * lamb_ru) + (grad_data * lamb_du)
        s_k_next = self.alpha * s_k + g_k
        return p_k - s_k_next, s_k_next


class FlowVN(nn.Module):
    """Unrolled variational network with ``num_layers`` untied steps, starting
    from the zero-filled estimate. Returns every step's estimate during training
    and only the final one at inference."""

    def __init__(
        self,
        num_layers=10,
        use_checkpointing=True,
        layer_params=None,
        use_amp=False,
        lowmem=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_checkpointing = use_checkpointing
        self.layers = nn.ModuleList(
            [
                FlowVNLayer(**(layer_params or {}), use_amp=use_amp, lowmem=lowmem)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        kspace_acq,
        undersampling_mask,
        forward_op,
        adjoint_op,
        usrate=None,
        training=True,
    ):
        p_k = adjoint_op(kspace_acq * undersampling_mask)
        s_k = torch.zeros_like(p_k)
        if usrate is None:
            usrate = torch.ones(p_k.shape[0], device=p_k.device)

        predictions = []
        for layer in self.layers:
            args = (p_k, s_k, kspace_acq, undersampling_mask, forward_op, adjoint_op, usrate)
            if self.use_checkpointing and training:
                p_k, s_k = checkpoint.checkpoint(layer, *args, use_reentrant=False)
            else:
                p_k, s_k = layer(*args)
            if training:
                predictions.append(p_k)

        return predictions if training else [p_k]
