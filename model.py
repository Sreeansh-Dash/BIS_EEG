"""
model.py
────────
DMK-I  –  Differential Mamba-KAN Integrator

Architecture
  1. Temporal Feature Fusion (TFF)  — depthwise-sep conv + adaptive weighting
  2. Bidirectional Mamba core        — 4 stacked BiMamba blocks
  3. KAN regression head             — Kolmogorov-Arnold Network → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Optional imports (graceful fallback for dev machines) ─────
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

try:
    from efficient_kan import KAN
except ImportError:
    try:
        from kan import KAN
    except ImportError:
        KAN = None


# ─────────────────────────────────────────────────────────────
# 1.  Temporal Feature Fusion (TFF)
# ─────────────────────────────────────────────────────────────
class TemporalFeatureFusion(nn.Module):
    """Raw + differential → depthwise-sep conv → adaptive gating."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        cat_ch = in_channels * 2           # raw + diff

        # Depthwise separable conv  (groups = channels)
        self.dw_conv = nn.Conv1d(
            cat_ch, cat_ch, kernel_size=3, padding=1, groups=cat_ch
        )
        # Pointwise conv (1×1) that mixes across channels
        self.pw_conv = nn.Conv1d(cat_ch, cat_ch, kernel_size=1)

        # Adaptive weighting branch
        self.gate_conv = nn.Conv1d(cat_ch, cat_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 1, T)
        diff = torch.zeros_like(x)
        diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]   # X_t − X_{t−1}

        h = torch.cat([x, diff], dim=1)                 # (B, 2, T)

        features = self.pw_conv(self.dw_conv(h))         # depthwise → pointwise
        gate     = torch.sigmoid(self.gate_conv(features))

        return features * gate                           # (B, 2, T)


# ─────────────────────────────────────────────────────────────
# 2.  Bidirectional Mamba block
# ─────────────────────────────────────────────────────────────
class BiMambaBlock(nn.Module):
    """Forward Mamba + reversed Mamba, outputs summed."""

    def __init__(self, d_model: int, mamba_cls=None):
        super().__init__()
        MambaCls = mamba_cls or Mamba
        if MambaCls is None:
            raise ImportError("mamba_ssm is required for BiMambaBlock.")

        self.fwd  = MambaCls(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd  = MambaCls(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, D)
        fwd_out = self.fwd(x)
        bwd_out = torch.flip(self.bwd(torch.flip(x, [1])), [1])
        return self.norm(fwd_out + bwd_out)


# ─────────────────────────────────────────────────────────────
# 3.  Full model
# ─────────────────────────────────────────────────────────────
class DMK_I(nn.Module):
    """Differential Mamba-KAN Integrator for continuous BIS prediction."""

    def __init__(self, seq_len: int = 640, d_model: int = 128,
                 mamba_cls=None, kan_cls=None):
        super().__init__()

        _mamba = mamba_cls or Mamba
        _kan   = kan_cls   or KAN

        # ── Front-end ────────────────────────────────────────
        self.tff        = TemporalFeatureFusion(in_channels=1)
        self.input_proj = nn.Linear(2, d_model)       # (B,T,2) → (B,T,D)

        # ── Sequence core ────────────────────────────────────
        self.layers = nn.ModuleList(
            [BiMambaBlock(d_model, mamba_cls=_mamba) for _ in range(4)]
        )

        # ── Regression head ──────────────────────────────────
        if _kan is None:
            raise ImportError(
                "efficient_kan (or pykan) is required for the KAN head."
            )
        self.kan = _kan([d_model, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 1, T)
        x = self.tff(x)                       # (B, 2, T)
        x = x.permute(0, 2, 1)                # (B, T, 2)
        x = self.input_proj(x)                # (B, T, D)

        for layer in self.layers:
            x = x + layer(x)                  # residual

        x = x.mean(dim=1)                     # avg-pool → (B, D)
        x = self.kan(x)                       # (B, 1)
        return torch.sigmoid(x)               # bound [0, 1]


# ─────────────────────────────────────────────────────────────
# Quick-test with mocks (runs on any machine)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mock_mamba = None
    mock_kan   = None

    if Mamba is None:
        print("⚠  Mocking Mamba (mamba_ssm not installed)")

        class _MockMamba(nn.Module):
            def __init__(self, d_model, **kw):
                super().__init__()
                self.fc = nn.Linear(d_model, d_model)
            def forward(self, x):
                return self.fc(x)

        mock_mamba = _MockMamba

    if KAN is None:
        print("⚠  Mocking KAN (efficient_kan not installed)")

        class _MockKAN(nn.Module):
            def __init__(self, widths):
                super().__init__()
                self.fc = nn.Linear(widths[0], widths[-1])
            def forward(self, x):
                return self.fc(x)

        mock_kan = _MockKAN

    # ── Instantiate & forward ────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = DMK_I(
        seq_len=640, d_model=128,
        mamba_cls=mock_mamba or Mamba,
        kan_cls=mock_kan or KAN,
    ).to(device)

    dummy = torch.randn(16, 1, 640, device=device)
    out   = net(dummy)

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")
    print(f"Params : {n_params:,}")
