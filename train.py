"""
train.py
────────
Training & evaluation loop for the DMK-I BIS predictor.

Key design choices
  • Loss         : L1 (MAE) — robust to outlier BIS noise.
  • AMP          : mixed-precision via torch.amp (auto-disabled on CPU).
  • Metrics      : MAE / RMSE rescaled to the clinical 0-100 BIS range.
  • Checkpointing: best Val MAE → best_dmk_i_model.pth
"""

import math
import os
import time

import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import DMK_I

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS     = 50
LR         = 1e-3
D_MODEL    = 128
SEQ_LEN    = 640            # 5 s × 128 Hz
DATA_DIR   = "data"
CKPT_PATH  = "best_dmk_i_model.pth"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    n_batches    = 0

    for x, y in loader:
        x = x.to(device)                          # (B, 1, 640)
        y = y.to(device).unsqueeze(-1)             # (B, 1)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, enabled=use_amp):
            pred = model(x)                        # (B, 1)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_batches    += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    all_preds, all_targets = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.amp.autocast(device.type, enabled=use_amp):
            pred = model(x).squeeze(-1)            # (B,)

        all_preds.append(pred)
        all_targets.append(y)

    # ── rescale to clinical 0-100 BIS ────────────
    preds   = torch.cat(all_preds)   * 100.0
    targets = torch.cat(all_targets) * 100.0

    mae  = torch.mean(torch.abs(preds - targets)).item()
    rmse = math.sqrt(torch.mean((preds - targets) ** 2).item())
    return mae, rmse


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Device  : {device}")
    print(f"AMP     : {'enabled' if use_amp else 'disabled (CPU)'}")

    # ── Data ──────────────────────────────────
    print("Loading data …")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    print(
        f"Batches → train {len(train_loader)}  "
        f"val {len(val_loader)}  test {len(test_loader)}"
    )

    # ── Model / Optimiser / Loss ──────────────
    model     = DMK_I(seq_len=SEQ_LEN, d_model=D_MODEL).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()                         # MAE
    scaler    = torch.amp.GradScaler(device.type, enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params  : {n_params:,}")

    # ── Training ──────────────────────────────
    best_mae = float("inf")
    hdr = f"{'Ep':>4}  {'Loss':>10}  {'MAE':>8}  {'RMSE':>8}  {'Best':>5}  {'Time':>6}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        loss     = train_one_epoch(model, train_loader, optimizer,
                                   criterion, scaler, device, use_amp)
        mae, rmse = validate(model, val_loader, device, use_amp)

        is_best = mae < best_mae
        if is_best:
            best_mae = mae
            torch.save(model.state_dict(), CKPT_PATH)

        dt = time.time() - t0
        tag = "  ★" if is_best else ""
        print(
            f"{epoch:4d}  {loss:10.6f}  {mae:8.3f}  {rmse:8.3f}  "
            f"{'yes' if is_best else '':>5}  {dt:5.1f}s{tag}"
        )

    print("=" * len(hdr))
    print(f"Done.  Best Val MAE = {best_mae:.3f}")
    print(f"Checkpoint → {os.path.abspath(CKPT_PATH)}")


if __name__ == "__main__":
    main()
