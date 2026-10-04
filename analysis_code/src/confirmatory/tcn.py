from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.GELU(),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class CompactTCN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, channels=(32, 64, 64), kernel_size=5, scalar_dim=2):
        super().__init__()
        blocks = []
        c = in_channels
        for i, out_c in enumerate(channels):
            blocks.append(TemporalBlock(c, out_c, kernel_size, dilation=2 ** i))
            c = out_c
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(c + scalar_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x, scalars):
        h = self.tcn(x).mean(dim=2)
        return self.head(torch.cat([h, scalars], dim=1))


@dataclass
class SequenceScaler:
    mean: np.ndarray
    std: np.ndarray
    scalar_mean: np.ndarray
    scalar_std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, scalars: np.ndarray):
        # X: [N,C,T]
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True)
        std[std < 1e-6] = 1.0
        sm = scalars.mean(axis=0, keepdims=True)
        ss = scalars.std(axis=0, keepdims=True)
        ss[ss < 1e-6] = 1.0
        return cls(mean, std, sm, ss)

    def transform(self, X: np.ndarray, scalars: np.ndarray):
        return (X - self.mean) / self.std, (scalars - self.scalar_mean) / self.scalar_std


def prepare_sequence(X: np.ndarray, sensor_ids: tuple[int, ...]) -> np.ndarray:
    # [N,S,T,6] -> [N,S*6,T]
    return X[:, sensor_ids].transpose(0, 1, 3, 2).reshape(len(X), len(sensor_ids) * 6, X.shape[2]).astype(np.float32)


class TCNRegressor:
    def __init__(self, params: dict, seed: int, device: str | None = None):
        self.params = params
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.y_mean = None
        self.y_std = None
        self.best_epoch = None

    def fit(self, Xseq, scalars, Y, val_data=None):
        torch.manual_seed(self.seed)
        self.scaler = SequenceScaler.fit(Xseq, scalars)
        Xs, Ss = self.scaler.transform(Xseq, scalars)
        self.y_mean = Y.mean(axis=0, keepdims=True)
        self.y_std = Y.std(axis=0, keepdims=True)
        self.y_std[self.y_std < 1e-6] = 1.0
        Ys = (Y - self.y_mean) / self.y_std

        in_ch = Xs.shape[1]
        out_dim = Y.shape[1]
        self.model = CompactTCN(
            in_ch, out_dim,
            channels=tuple(self.params.get("channels", (32, 64, 64))),
            kernel_size=int(self.params.get("kernel_size", 5)),
        ).to(self.device)
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params.get("weight_decay", 1e-4)),
        )
        loss_fn = nn.MSELoss()
        ds = TensorDataset(
            torch.tensor(Xs, dtype=torch.float32),
            torch.tensor(Ss, dtype=torch.float32),
            torch.tensor(Ys, dtype=torch.float32),
        )
        gen = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(ds, batch_size=int(self.params.get("batch_size", 64)), shuffle=True, generator=gen)

        if val_data is not None:
            vx, vs, vy = val_data
            vx, vs = self.scaler.transform(vx, vs)
            vy = (vy - self.y_mean) / self.y_std
            vx = torch.tensor(vx, dtype=torch.float32, device=self.device)
            vs = torch.tensor(vs, dtype=torch.float32, device=self.device)
            vy = torch.tensor(vy, dtype=torch.float32, device=self.device)

        best_loss = np.inf
        best_state = None
        stale = 0
        max_epochs = int(self.params.get("max_epochs", 150))
        patience = int(self.params.get("patience", 15))
        min_delta = float(self.params.get("min_delta", 1e-4))

        for epoch in range(max_epochs):
            self.model.train()
            for xb, sb, yb in loader:
                xb, sb, yb = xb.to(self.device), sb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                pred = self.model(xb, sb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            if val_data is None:
                continue
            self.model.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self.model(vx, vs), vy).cpu())
            if vloss < best_loss - min_delta:
                best_loss = vloss
                best_state = deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        elif self.best_epoch is None:
            self.best_epoch = max_epochs
        return self

    def predict(self, Xseq, scalars):
        Xs, Ss = self.scaler.transform(Xseq, scalars)
        self.model.eval()
        preds = []
        bs = int(self.params.get("batch_size", 64))
        with torch.no_grad():
            for i in range(0, len(Xs), bs):
                xb = torch.tensor(Xs[i:i+bs], dtype=torch.float32, device=self.device)
                sb = torch.tensor(Ss[i:i+bs], dtype=torch.float32, device=self.device)
                preds.append(self.model(xb, sb).cpu().numpy())
        p = np.concatenate(preds, axis=0)
        return (p * self.y_std + self.y_mean).astype(np.float32)
