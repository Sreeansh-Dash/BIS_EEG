"""
dataset.py
──────────
PyTorch Dataset / DataLoader for the BIS-prediction pipeline.

• Windows 128 Hz EEG into 5-second segments (640 time-steps).
• Normalises BIS targets from 0-100 → 0-1.
• Chronological subject-level split (70 / 10 / 20 %).
• Optional *Harmonic Mix* frequency-domain augmentation.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class BISDataset(Dataset):
    """Lazy-loading dataset backed by memory-mapped .npy files."""

    def __init__(
        self,
        data_dir: str = "data",
        split: str = "train",
        window_size: int = 640,
        stride: int | None = None,
        augment: bool = False,
    ):
        self.data_dir    = data_dir
        self.window_size = window_size
        self.stride      = stride if stride is not None else window_size
        self.augment     = augment

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

        # ── collect & sort files ──────────────────────────────
        all_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        )
        try:
            all_files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            pass  # fall back to lexicographic order

        num_cases = len(all_files)
        if num_cases == 0:
            raise FileNotFoundError(f"No .npy files in '{data_dir}'.")

        # ── chronological 70 / 10 / 20 split ─────────────────
        n_train = max(1, int(0.7 * num_cases))
        n_val   = max(1, int(0.1 * num_cases)) if num_cases > 1 else 0

        if split == "train":
            self.files = all_files[:n_train]
        elif split == "val":
            self.files = all_files[n_train : n_train + n_val]
        elif split == "test":
            self.files = all_files[n_train + n_val :]
        else:
            self.files = []

        # ── build window index (file, start) ──────────────────
        self.indices: list[tuple[str, int]] = []

        for fname in self.files:
            path = os.path.join(data_dir, fname)
            try:
                data      = np.load(path, mmap_mode="r")
                n_samples = data.shape[0]
                if n_samples < window_size:
                    continue

                # .npy layout: col-0 = BIS, col-1 = EEG1
                bis = data[:, 0]

                starts  = np.arange(0, n_samples - window_size + 1, self.stride)
                targets = bis[starts + window_size - 1]

                valid = (targets > 0) & np.isfinite(targets)
                for s in starts[valid]:
                    self.indices.append((fname, int(s)))
            except Exception as exc:
                print(f"[dataset] skip {fname}: {exc}")

        print(
            f"[dataset] split='{split}'  files={len(self.files)}  "
            f"segments={len(self.indices)}"
        )

    # ──────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        fname, start = self.indices[idx]
        data = np.load(
            os.path.join(self.data_dir, fname), mmap_mode="r"
        )

        # .npy layout: col-0 = BIS, col-1 = EEG1
        eeg    = data[start : start + self.window_size, 1].astype(np.float32)
        target = float(data[start + self.window_size - 1, 0])

        # Replace any residual NaN with 0
        eeg = np.nan_to_num(eeg, nan=0.0)

        # Shape → (1, T)
        x = torch.from_numpy(eeg).unsqueeze(0)            # (1, 640)
        y = torch.tensor(target / 100.0, dtype=torch.float32)  # 0-1 scale

        # ── optional Harmonic-Mix augmentation ────────────────
        if self.augment and random.random() < 0.5:
            idx2          = random.randint(0, len(self.indices) - 1)
            fname2, st2   = self.indices[idx2]
            d2            = np.load(
                os.path.join(self.data_dir, fname2), mmap_mode="r"
            )
            eeg2   = np.nan_to_num(
                d2[st2 : st2 + self.window_size, 1].astype(np.float32), nan=0.0
            )
            x2     = torch.from_numpy(eeg2).unsqueeze(0)
            y2     = torch.tensor(
                float(d2[st2 + self.window_size - 1, 0]) / 100.0,
                dtype=torch.float32,
            )
            lam    = float(np.random.beta(1.0, 1.0))
            x      = self._harmonic_mix(x, x2, lam)
            y      = lam * y + (1 - lam) * y2

        return x, y

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _harmonic_mix(
        x1: torch.Tensor, x2: torch.Tensor, lam: float
    ) -> torch.Tensor:
        """Convex combination of frequency bands + harmonic LPF."""
        fft1 = torch.fft.rfft(x1, dim=-1)
        fft2 = torch.fft.rfft(x2, dim=-1)

        mixed = lam * fft1 + (1 - lam) * fft2

        # Low-pass: zero out bins above 45 Hz
        # freq resolution = 128 / 640 = 0.2 Hz  →  cutoff bin = 225
        cutoff = int(45.0 / 0.2)
        if cutoff < mixed.shape[-1]:
            mixed[..., cutoff:] = 0.0

        return torch.fft.irfft(mixed, n=x1.shape[-1], dim=-1)


# ─────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────
def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 0,
):
    train_ds = BISDataset(data_dir, split="train", augment=True)
    val_ds   = BISDataset(data_dir, split="val",   augment=False)
    test_ds  = BISDataset(data_dir, split="test",  augment=False)

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, **kw),
    )


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        tl, vl, tel = get_dataloaders()
        print(f"Train batches: {len(tl)}  Val: {len(vl)}  Test: {len(tel)}")
        for x, y in tl:
            print(f"  x.shape = {x.shape}  y.shape = {y.shape}  y[0] = {y[0]:.4f}")
            break
    except Exception as e:
        print(f"Verification error: {e}")
