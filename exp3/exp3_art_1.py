#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import math
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ============================================================
# Config
# ============================================================
VAR_DIR     = "/home/Data/zhufuhua/MyData/para2/data/global"
LABEL_ROOT  = "/home/Data/zhufuhua/MyData/para2/data/global/thelabel"

# 这里建议优先指向你最终 full model 所在目录
# 如果你要用实验二里的 full ablation 模型，就填 theresult_ablation
CKPT_ROOT   = "/home/Data/zhufuhua/MyData/para2/data/global/theresult_ablation"

RESULT_ROOT = "/home/Data/zhufuhua/MyData/para2/data/global/theresult_exp3"

COARSE_LIST = [3, 5, 10]
MAX_DAYS    = 10

PATCH_SIZE  = 96
BATCH_SIZE_EVAL = 1

NUM_WORKERS = 0
USE_AMP = True

SCAPEX_WIDTH      = 64
SCAPEX_NBLOCKS    = 4
SCAPEX_AFNO_DEPTH = 2

SCAPEX_ALPHA_S_MAX_N3  = 0.55
SCAPEX_ALPHA_S_MAX_OTH = 0.90

EVAL_TILE = 192
EVAL_OVERLAP = 32

# 代表性样本：默认每个尺度取验证集第一个样本
REP_SAMPLE_INDEX = 0
# 主文代表性机制图默认使用哪个尺度
REP_FIG_N = 5

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
})

XR_OPEN_KW = dict(
    decode_cf=False,
    mask_and_scale=False,
    decode_times=False,
    cache=True,
)

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
DEVICE_IS_CUDA = str(DEVICE).startswith("cuda")

np.random.seed(42)
torch.manual_seed(42)
if DEVICE_IS_CUDA:
    torch.cuda.manual_seed_all(42)

from contextlib import nullcontext

def autocast_ctx():
    if DEVICE_IS_CUDA and USE_AMP:
        return torch.amp.autocast("cuda", enabled=True)
    return nullcontext()


# ============================================================
# Utilities
# ============================================================
def savefig_tight(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Saved:", path)


def resolve_label_dir_for_n(label_root: str, n: int) -> str:
    cand = os.path.join(label_root, f"label_{n}")
    return cand if os.path.isdir(cand) else label_root


def assert_4d_j_i(da: xr.DataArray, name: str):
    if da.dims != ("time", "lev", "j", "i"):
        raise RuntimeError(f"[{name}] dims expected ('time','lev','j','i') but got {da.dims}")


def assert_2d_j_i(da: xr.DataArray, name: str):
    if da.dims != ("j", "i"):
        raise RuntimeError(f"[{name}] dims expected ('j','i') but got {da.dims}")


def get2d_4dvar(ds: xr.Dataset, name: str, lev: int) -> np.ndarray:
    da = ds[name]
    assert_4d_j_i(da, name)
    return da.isel(time=0, lev=lev).values.astype(np.float32, copy=False)


def get_dxdy(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    dx_da = ds["dx"]
    dy_da = ds["dy"]
    assert_2d_j_i(dx_da, "dx")
    assert_2d_j_i(dy_da, "dy")
    return dx_da.values.astype(np.float32, copy=False), dy_da.values.astype(np.float32, copy=False)


def phys_grad_2d(f2d: np.ndarray, dx2d: np.ndarray, dy2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = f2d.shape
    dfdx = np.full_like(f2d, np.nan, dtype=np.float32)
    dfdy = np.full_like(f2d, np.nan, dtype=np.float32)

    if W >= 3:
        dfdx[:, 1:-1] = (f2d[:, 2:] - f2d[:, :-2]) / (2.0 * dx2d[:, 1:-1])
        dfdx[:, 0]    = (-3.0 * f2d[:, 0] + 4.0 * f2d[:, 1] - f2d[:, 2]) / (2.0 * dx2d[:, 0])
        dfdx[:, -1]   = ( 3.0 * f2d[:, -1] - 4.0 * f2d[:, -2] + f2d[:, -3]) / (2.0 * dx2d[:, -1])

    if H >= 3:
        dfdy[1:-1, :] = (f2d[2:, :] - f2d[:-2, :]) / (2.0 * dy2d[1:-1, :])
        dfdy[0, :]    = (-3.0 * f2d[0, :] + 4.0 * f2d[1, :] - f2d[2, :]) / (2.0 * dy2d[0, :])
        dfdy[-1, :]   = ( 3.0 * f2d[-1, :] - 4.0 * f2d[-2, :] + f2d[-3, :]) / (2.0 * dy2d[-1, :])

    return dfdx.astype(np.float32, copy=False), dfdy.astype(np.float32, copy=False)


def build_day_level_index(theta_files: List[str], label_dir: str, max_days: int) -> List[dict]:
    theta_files = theta_files[:max_days]
    items: List[dict] = []
    for file_id, theta_path in enumerate(theta_files):
        base = os.path.basename(theta_path)
        label_path = os.path.join(label_dir, base.replace("thetao_", "label_flux_"))
        if not os.path.exists(label_path):
            print("Warning: missing", label_path)
            continue

        ds = xr.open_dataset(label_path, **XR_OPEN_KW)
        try:
            da = ds["theta_lr"]
            assert_4d_j_i(da, "theta_lr")
            nlev = da.sizes["lev"]
            for lev in range(nlev):
                items.append({
                    "file_id": file_id,
                    "theta_path": theta_path,
                    "label_path": label_path,
                    "lev": int(lev),
                })
        finally:
            ds.close()
    return items


def estimate_norm_from_train(items: List[dict], n_samples: int = 2_000_000, seed: int = 0):
    rng = np.random.default_rng(seed)

    cnt = 0
    meanX = np.zeros(8, dtype=np.float64)
    m2X   = np.zeros(8, dtype=np.float64)
    meanY = np.zeros(4, dtype=np.float64)
    m2Y   = np.zeros(4, dtype=np.float64)

    per_item = max(10_000, n_samples // max(1, len(items)))

    for it in items:
        ds = xr.open_dataset(it["label_path"], **XR_OPEN_KW)
        try:
            lev = it["lev"]
            u = get2d_4dvar(ds, "uo_lr", lev)
            v = get2d_4dvar(ds, "vo_lr", lev)
            t = get2d_4dvar(ds, "theta_lr", lev)
            s = get2d_4dvar(ds, "so_lr", lev)
            qTx = get2d_4dvar(ds, "qT_x", lev)
            qTy = get2d_4dvar(ds, "qT_y", lev)
            qSx = get2d_4dvar(ds, "qS_x", lev)
            qSy = get2d_4dvar(ds, "qS_y", lev)

            dx, dy = get_dxdy(ds)
            dTdx, dTdy = phys_grad_2d(t, dx, dy)
            dSdx, dSdy = phys_grad_2d(s, dx, dy)

            X = np.stack([u, v, t, s, dTdx, dTdy, dSdx, dSdy], axis=-1)
            Y = np.stack([qTx, qTy, qSx, qSy], axis=-1)

            valid = np.isfinite(X).all(axis=-1) & np.isfinite(Y).all(axis=-1)
            idx = np.where(valid.reshape(-1))[0]
            if idx.size == 0:
                continue

            take = min(per_item, idx.size)
            sel = rng.choice(idx, size=take, replace=False)

            Xs = X.reshape(-1, 8)[sel].astype(np.float64, copy=False)
            Ys = Y.reshape(-1, 4)[sel].astype(np.float64, copy=False)

            nb = int(Xs.shape[0])
            bx = Xs.mean(axis=0)
            by = Ys.mean(axis=0)

            sumsqX = (Xs * Xs).sum(axis=0)
            sumsqY = (Ys * Ys).sum(axis=0)
            bM2x = sumsqX - nb * (bx * bx)
            bM2y = sumsqY - nb * (by * by)

            if cnt == 0:
                meanX = bx; m2X = bM2x
                meanY = by; m2Y = bM2y
                cnt = nb
            else:
                cnt_new = cnt + nb

                deltaX = bx - meanX
                meanX = meanX + deltaX * (nb / cnt_new)
                m2X = m2X + bM2x + (deltaX * deltaX) * (cnt * nb / cnt_new)

                deltaY = by - meanY
                meanY = meanY + deltaY * (nb / cnt_new)
                m2Y = m2Y + bM2y + (deltaY * deltaY) * (cnt * nb / cnt_new)

                cnt = cnt_new

            if cnt >= n_samples:
                break
        finally:
            ds.close()

    varX = m2X / max(cnt - 1, 1)
    varY = m2Y / max(cnt - 1, 1)

    X_mean = meanX.astype(np.float32)
    X_std  = np.sqrt(varX).astype(np.float32) + 1e-8
    Y_mean = meanY.astype(np.float32)
    Y_std  = np.sqrt(varY).astype(np.float32) + 1e-8
    return X_mean, X_std, Y_mean, Y_std


def fit_prior_K_vec_normalized(train_items, X_mean, X_std, Y_mean, Y_std, seed=0):
    rng = np.random.default_rng(seed)
    num = np.zeros(4, dtype=np.float64)
    den = np.zeros(4, dtype=np.float64)
    got = 0

    for it in train_items[:max(10, len(train_items)//6)]:
        ds = xr.open_dataset(it["label_path"], **XR_OPEN_KW)
        try:
            lev = it["lev"]

            t = get2d_4dvar(ds, "theta_lr", lev)
            s = get2d_4dvar(ds, "so_lr", lev)
            qTx = get2d_4dvar(ds, "qT_x", lev)
            qTy = get2d_4dvar(ds, "qT_y", lev)
            qSx = get2d_4dvar(ds, "qS_x", lev)
            qSy = get2d_4dvar(ds, "qS_y", lev)

            dx, dy = get_dxdy(ds)
            dTdx, dTdy = phys_grad_2d(t, dx, dy)
            dSdx, dSdy = phys_grad_2d(s, dx, dy)

            dTdx_n = (dTdx - X_mean[4]) / X_std[4]
            dTdy_n = (dTdy - X_mean[5]) / X_std[5]
            dSdx_n = (dSdx - X_mean[6]) / X_std[6]
            dSdy_n = (dSdy - X_mean[7]) / X_std[7]

            qTx_n = (qTx - Y_mean[0]) / Y_std[0]
            qTy_n = (qTy - Y_mean[1]) / Y_std[1]
            qSx_n = (qSx - Y_mean[2]) / Y_std[2]
            qSy_n = (qSy - Y_mean[3]) / Y_std[3]

            valid = np.isfinite(dTdx_n)&np.isfinite(qTx_n)&np.isfinite(dTdy_n)&np.isfinite(qTy_n)& \
                    np.isfinite(dSdx_n)&np.isfinite(qSx_n)&np.isfinite(dSdy_n)&np.isfinite(qSy_n)
            idx = np.where(valid.reshape(-1))[0]
            if idx.size == 0:
                continue

            take = min(120_000, idx.size)
            sel = rng.choice(idx, size=take, replace=False)

            dTdx_s = dTdx_n.reshape(-1)[sel]; qTx_s = qTx_n.reshape(-1)[sel]
            dTdy_s = dTdy_n.reshape(-1)[sel]; qTy_s = qTy_n.reshape(-1)[sel]
            dSdx_s = dSdx_n.reshape(-1)[sel]; qSx_s = qSx_n.reshape(-1)[sel]
            dSdy_s = dSdy_n.reshape(-1)[sel]; qSy_s = qSy_n.reshape(-1)[sel]

            num[0] += float(np.sum(dTdx_s * qTx_s))
            den[0] += float(np.sum(dTdx_s * dTdx_s) + 1e-12)
            num[1] += float(np.sum(dTdy_s * qTy_s))
            den[1] += float(np.sum(dTdy_s * dTdy_s) + 1e-12)
            num[2] += float(np.sum(dSdx_s * qSx_s))
            den[2] += float(np.sum(dSdx_s * dSdx_s) + 1e-12)
            num[3] += float(np.sum(dSdy_s * qSy_s))
            den[3] += float(np.sum(dSdy_s * dSdy_s) + 1e-12)

            got += take
            if got >= 600_000:
                break
        finally:
            ds.close()

    K_vec_n = -num / np.maximum(den, 1e-12)
    return K_vec_n.astype(np.float32)


# ============================================================
# Dataset
# ============================================================
class LazyLevFullEvalDataset(Dataset):
    def __init__(self, items: List[dict], X_mean, X_std, Y_mean, Y_std):
        self.items = items
        self.X_mean = X_mean.reshape(8,1,1).astype(np.float32)
        self.X_std  = X_std.reshape(8,1,1).astype(np.float32)
        self.Y_mean = Y_mean.reshape(4,1,1).astype(np.float32)
        self.Y_std  = Y_std.reshape(4,1,1).astype(np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        ds = xr.open_dataset(it["label_path"], **XR_OPEN_KW)
        try:
            lev = it["lev"]

            u = get2d_4dvar(ds, "uo_lr", lev)
            v = get2d_4dvar(ds, "vo_lr", lev)
            t = get2d_4dvar(ds, "theta_lr", lev)
            s = get2d_4dvar(ds, "so_lr", lev)
            qTx = get2d_4dvar(ds, "qT_x", lev)
            qTy = get2d_4dvar(ds, "qT_y", lev)
            qSx = get2d_4dvar(ds, "qS_x", lev)
            qSy = get2d_4dvar(ds, "qS_y", lev)

            dx, dy = get_dxdy(ds)
            dTdx, dTdy = phys_grad_2d(t, dx, dy)
            dSdx, dSdy = phys_grad_2d(s, dx, dy)

            X_phys = np.stack([u, v, t, s, dTdx, dTdy, dSdx, dSdy], axis=0).astype(np.float32, copy=False)
            Y_phys = np.stack([qTx, qTy, qSx, qSy], axis=0).astype(np.float32, copy=False)

            ocean = np.isfinite(Y_phys[0])
            Xn = (X_phys - self.X_mean) / self.X_std
            Xn = np.nan_to_num(Xn, nan=0.0)

            return (
                torch.from_numpy(Xn),
                torch.from_numpy(Y_phys),
                torch.from_numpy(ocean.astype(np.bool_)),
                int(it["file_id"]),
                int(it["lev"]),
                torch.from_numpy(X_phys),
            )
        finally:
            ds.close()


# ============================================================
# Model components
# ============================================================
class LayerNorm2d(nn.Module):
    def __init__(self, n_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.bias   = nn.Parameter(torch.zeros(n_channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1,-1,1,1) + self.bias.view(1,-1,1,1)


class CNOBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, groups=ch, bias=False)
        self.pw1 = nn.Conv2d(ch, ch*2, 1)
        self.pw2 = nn.Conv2d(ch*2, ch, 1)
        self.norm = nn.BatchNorm2d(ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        return x + y


class AFNOMixer(nn.Module):
    def __init__(self, width, num_blocks=8, hidden_mult=2.0):
        super().__init__()
        assert width % num_blocks == 0
        self.width = width
        self.num_blocks = num_blocks
        self.block = width // num_blocks
        hidden = int(self.block * hidden_mult)

        self.Wr1 = nn.Parameter(torch.randn(num_blocks, self.block, hidden) * 0.02)
        self.Wi1 = nn.Parameter(torch.randn(num_blocks, self.block, hidden) * 0.02)
        self.br1 = nn.Parameter(torch.zeros(num_blocks, 1, hidden))
        self.bi1 = nn.Parameter(torch.zeros(num_blocks, 1, hidden))

        self.Wr2 = nn.Parameter(torch.randn(num_blocks, hidden, self.block) * 0.02)
        self.Wi2 = nn.Parameter(torch.randn(num_blocks, hidden, self.block) * 0.02)
        self.br2 = nn.Parameter(torch.zeros(num_blocks, 1, self.block))
        self.bi2 = nn.Parameter(torch.zeros(num_blocks, 1, self.block))

    def forward(self, x):
        B, C, H, W = x.shape
        orig_dtype = x.dtype
        x_fft = x.float() if (x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)) else x

        x_ft = torch.fft.rfft2(x_fft, norm="ortho")
        xr = x_ft.real
        xi = x_ft.imag
        Wf = W // 2 + 1

        xr = xr.view(B, self.num_blocks, self.block, H, Wf)
        xi = xi.view(B, self.num_blocks, self.block, H, Wf)

        xr_ = xr.permute(0,1,3,4,2).contiguous().view(B, self.num_blocks, H*Wf, self.block)
        xi_ = xi.permute(0,1,3,4,2).contiguous().view(B, self.num_blocks, H*Wf, self.block)

        or1 = torch.einsum("bnpk,nkm->bnpm", xr_, self.Wr1) - torch.einsum("bnpk,nkm->bnpm", xi_, self.Wi1)
        oi1 = torch.einsum("bnpk,nkm->bnpm", xr_, self.Wi1) + torch.einsum("bnpk,nkm->bnpm", xi_, self.Wr1)
        or1 = F.gelu(or1 + self.br1)
        oi1 = F.gelu(oi1 + self.bi1)

        or2 = torch.einsum("bnpm,nmk->bnpk", or1, self.Wr2) - torch.einsum("bnpm,nmk->bnpk", oi1, self.Wi2)
        oi2 = torch.einsum("bnpm,nmk->bnpk", or1, self.Wi2) + torch.einsum("bnpm,nmk->bnpk", oi1, self.Wr2)
        or2 = or2 + self.br2
        oi2 = oi2 + self.bi2

        or2 = or2.view(B, self.num_blocks, H, Wf, self.block).permute(0,1,4,2,3).contiguous().view(B, C, H, Wf)
        oi2 = oi2.view(B, self.num_blocks, H, Wf, self.block).permute(0,1,4,2,3).contiguous().view(B, C, H, Wf)

        y_ft = torch.complex(or2, oi2)
        y = torch.fft.irfft2(y_ft, s=(H, W), norm="ortho")
        return y.to(orig_dtype)


class AFNOBlock(nn.Module):
    def __init__(self, width, num_blocks=8, hidden_mult=2.0):
        super().__init__()
        self.norm1 = LayerNorm2d(width)
        self.mixer = AFNOMixer(width, num_blocks=num_blocks, hidden_mult=hidden_mult)
        self.norm2 = LayerNorm2d(width)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, width*2, 1),
            nn.GELU(),
            nn.Conv2d(width*2, width, 1),
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FiniteDiff2D(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32).view(1,1,1,3)
        ky = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32).view(1,1,3,1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def grad(self, s):
        dx = F.conv2d(s, self.kx, padding=(0,1))
        dy = F.conv2d(s, self.ky, padding=(1,0))
        return dx, dy


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, n_feats, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.ca = CALayer(n_feats, reduction)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.ca(res)
        return x + res


class ResidualGroup(nn.Module):
    def __init__(self, n_feats, n_rcab=4):
        super().__init__()
        self.body = nn.Sequential(*[RCAB(n_feats) for _ in range(n_rcab)])
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)

    def forward(self, x):
        res = self.body(x)
        res = self.conv(res)
        return x + res


class HybridCNOAFNOFeatureBackbone(nn.Module):
    def __init__(self, in_channels=8, width=64, n_blocks=4, afno_depth=2):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 3, padding=1)

        self.enc1 = nn.Sequential(*[CNOBlock(width, dilation=1) for _ in range(n_blocks)])
        self.down1 = nn.Conv2d(width, width*2, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(*[CNOBlock(width*2, dilation=2) for _ in range(n_blocks)])
        self.down2 = nn.Conv2d(width*2, width*4, 3, stride=2, padding=1)

        mid = []
        for _ in range(n_blocks):
            mid.append(CNOBlock(width*4, dilation=2))
        for _ in range(max(0, afno_depth)):
            mid.append(AFNOBlock(width*4, num_blocks=8, hidden_mult=2.0))
        self.mid = nn.Sequential(*mid)

        self.up2 = nn.Conv2d(width*4, width*2, 1)
        self.dec2 = nn.Sequential(*[CNOBlock(width*2, dilation=2) for _ in range(n_blocks)])

        self.up1 = nn.Conv2d(width*2, width, 1)
        self.dec1 = nn.Sequential(*[CNOBlock(width, dilation=1) for _ in range(n_blocks)])

    def forward(self, x):
        x = self.lift(x)
        e1 = self.enc1(x)
        x  = self.down1(e1)

        e2 = self.enc2(x)
        x  = self.down2(e2)

        x  = self.mid(x)

        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x  = self.up2(x) + e2
        x  = self.dec2(x)

        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x  = self.up1(x) + e1
        x  = self.dec1(x)
        return x


class SCHOPNet:
    @staticmethod
    def _scale_to_id(n: int) -> int:
        if n == 3: return 0
        if n == 5: return 1
        if n == 10: return 2
        return 7


class SCAPEXNetDiagnostics(nn.Module):
    """
    与 full SCAPEXNet 保持一致，只额外返回诊断量
    """
    def __init__(
        self,
        n_coarse: int,
        X_mean, X_std, Y_mean, Y_std,
        K_vec_n: np.ndarray,
        width=64, n_blocks=4, afno_depth=2,
        alpha_s_max: float = 0.9,
    ):
        super().__init__()
        self.register_buffer("X_mean", torch.tensor(X_mean, dtype=torch.float32).view(1,8,1,1))
        self.register_buffer("X_std",  torch.tensor(X_std,  dtype=torch.float32).view(1,8,1,1))
        self.register_buffer("Y_mean", torch.tensor(Y_mean, dtype=torch.float32).view(1,4,1,1))
        self.register_buffer("Y_std",  torch.tensor(Y_std,  dtype=torch.float32).view(1,4,1,1))

        K_vec = torch.tensor(K_vec_n, dtype=torch.float32).view(4,1,1)
        self.register_buffer("K_vec", K_vec)

        self.alpha_s_max = float(alpha_s_max)

        self.scale_id = SCHOPNet._scale_to_id(n_coarse)
        self.register_buffer("scale_id_t", torch.tensor(self.scale_id, dtype=torch.long))
        self.scale_emb = nn.Embedding(8, 8)

        self.trunk = HybridCNOAFNOFeatureBackbone(
            in_channels=8, width=width, n_blocks=n_blocks, afno_depth=afno_depth
        )

        self.pot_head = nn.Conv2d(width, 4, 3, padding=1)
        self.gate_head = nn.Sequential(
            nn.Conv2d(width, max(16, width // 2), 1),
            nn.GELU(),
            nn.Conv2d(max(16, width // 2), 4, 1)
        )

        self.res_body = nn.Sequential(*[ResidualGroup(width, n_rcab=4) for _ in range(3)])
        self.res_conv = nn.Conv2d(width, width, 3, padding=1)
        self.res_out  = nn.Conv2d(width, 4, 3, padding=1)

        self.diff = FiniteDiff2D()

        self.alpha_diss_T = nn.Parameter(torch.tensor(1.0))
        self.alpha_rot_T  = nn.Parameter(torch.tensor(1.0))
        self.alpha_diss_S = nn.Parameter(torch.tensor(1.0))
        self.alpha_rot_S  = nn.Parameter(torch.tensor(1.0))

    def denorm_X(self, Xn):
        return Xn * self.X_std + self.X_mean

    def denorm_Y(self, Yn):
        return Yn * self.Y_std + self.Y_mean

    def _prior_flux(self, Xn):
        dTdx = Xn[:,4:5]; dTdy = Xn[:,5:6]
        dSdx = Xn[:,6:7]; dSdy = Xn[:,7:8]
        qK = torch.cat([
            -self.K_vec[0:1] * dTdx, -self.K_vec[1:2] * dTdy,
            -self.K_vec[2:3] * dSdx, -self.K_vec[3:4] * dSdy,
        ], dim=1)
        return qK

    def _pot_to_flux_parts(self, pot):
        chiT = pot[:, 0:1]; psiT = pot[:, 1:2]
        chiS = pot[:, 2:3]; psiS = pot[:, 3:4]

        dchiTdx, dchiTdy = self.diff.grad(chiT)
        dpsiTdx, dpsiTdy = self.diff.grad(psiT)
        dchiSdx, dchiSdy = self.diff.grad(chiS)
        dpsiSdx, dpsiSdy = self.diff.grad(psiS)

        qd_T = torch.cat([dchiTdx, dchiTdy], dim=1) * self.alpha_diss_T
        qd_S = torch.cat([dchiSdx, dchiSdy], dim=1) * self.alpha_diss_S
        q_diss = torch.cat([qd_T, qd_S], dim=1)

        qr_T = torch.cat([-dpsiTdy, dpsiTdx], dim=1) * self.alpha_rot_T
        qr_S = torch.cat([-dpsiSdy, dpsiSdx], dim=1) * self.alpha_rot_S
        q_rot = torch.cat([qr_T, qr_S], dim=1)
        q = q_diss + q_rot
        return q, q_diss, q_rot

    def forward_diagnostics(self, Xn):
        feat = self.trunk(Xn)

        emb = self.scale_emb(self.scale_id_t.to(feat.device)).view(1,8,1,1)
        gate_bias = emb[:,0:4]
        pot_bias  = emb[:,4:8]

        g_logits = self.gate_head(feat) + gate_bias
        g = torch.sigmoid(g_logits)

        alphaT = g[:,0:1]
        alphaS = torch.clamp(g[:,1:2], 0.0, self.alpha_s_max)
        betaT  = g[:,2:3]
        betaS  = g[:,3:4]

        alpha4 = torch.cat([alphaT, alphaT, alphaS, alphaS], dim=1)
        beta4  = torch.cat([betaT,  betaT,  betaS,  betaS ], dim=1)

        q_prior = self._prior_flux(Xn)
        pot = self.pot_head(feat) + pot_bias
        q_struct, qd_struct, qr_struct = self._pot_to_flux_parts(pot)

        rr = self.res_body(feat)
        rr = self.res_conv(rr) + feat
        q_res = self.res_out(rr)

        pred = alpha4 * q_prior + beta4 * q_struct + q_res
        q_diss_eff = alpha4 * q_prior + beta4 * qd_struct
        q_rot_eff  = beta4 * qr_struct

        return {
            "pred": pred,
            "q_prior": q_prior,
            "q_struct": q_struct,
            "q_res": q_res,
            "q_diss_eff": q_diss_eff,
            "q_rot_eff": q_rot_eff,
            "alphaT": alphaT,
            "alphaS": alphaS,
            "betaT": betaT,
            "betaS": betaS,
        }

    def forward(self, Xn):
        return self.forward_diagnostics(Xn)["pred"]


# ============================================================
# Checkpoint loading
# ============================================================
def load_state_flexible(model: nn.Module, path: str):
    obj = torch.load(path, map_location="cpu")
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        sd2 = {}
        for k, v in sd.items():
            if k.startswith("module."):
                sd2[k[len("module."):]] = v
            else:
                sd2[k] = v
        model.load_state_dict(sd2, strict=True)


def find_model_ckpt(ckpt_dir: str) -> str:
    candidates = [
        "SCAPEX_FULL_best.pt",
        "SCAPEX_best.pt",
        "SCAPEX_FULL.pt",
        "SCAPEX.pt",
    ]
    for name in candidates:
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No SCAPEX checkpoint found in {ckpt_dir}")


# ============================================================
# Tiled inference returning diagnostics
# ============================================================
def predict_scapex_diag_tiled(model: SCAPEXNetDiagnostics, Xn: torch.Tensor,
                              Y_mean: np.ndarray, Y_std: np.ndarray,
                              tile=192, overlap=32):
    assert Xn.dim() == 4 and Xn.size(0) == 1
    model.eval()

    B, C, H, W = Xn.shape
    stride = max(1, tile - overlap)
    device = Xn.device

    Y_mean_t = torch.tensor(Y_mean, device=device, dtype=torch.float32).view(1,4,1,1)
    Y_std_t  = torch.tensor(Y_std,  device=device, dtype=torch.float32).view(1,4,1,1)

    keys_4 = ["pred", "q_prior", "q_struct", "q_res", "q_diss_eff", "q_rot_eff"]
    keys_1 = ["alphaT", "alphaS", "betaT", "betaS"]

    out4 = {k: torch.zeros((1,4,H,W), device=device, dtype=torch.float32) for k in keys_4}
    out1 = {k: torch.zeros((1,1,H,W), device=device, dtype=torch.float32) for k in keys_1}
    w_acc = torch.zeros((1,1,H,W), device=device, dtype=torch.float32)

    if tile >= 2:
        wy = np.hanning(tile).astype(np.float32)
        wx = np.hanning(tile).astype(np.float32)
        ww_full = (wy[:,None] * wx[None,:]).astype(np.float32)
        ww_full = ww_full / (ww_full.max() + 1e-12)
    else:
        ww_full = np.ones((tile, tile), dtype=np.float32)
    ww_full_t = torch.from_numpy(ww_full).to(device=device, dtype=torch.float32).view(1,1,tile,tile)

    with torch.no_grad():
        for j0 in range(0, H, stride):
            for i0 in range(0, W, stride):
                j1 = min(j0 + tile, H)
                i1 = min(i0 + tile, W)
                js = max(0, j1 - tile)
                is_ = max(0, i1 - tile)
                je = min(js + tile, H)
                ie = min(is_ + tile, W)

                x_tile = Xn[:, :, js:je, is_:ie]

                with autocast_ctx():
                    d = model.forward_diagnostics(x_tile)

                ht = je - js
                wt = ie - is_
                ww = ww_full_t[:, :, :ht, :wt]

                for k in keys_4:
                    y = d[k].float()
                    if k == "pred":
                        y = y * Y_std_t + Y_mean_t
                    elif k in ["q_diss_eff", "q_rot_eff", "q_prior", "q_struct", "q_res"]:
                        y = y * Y_std_t + Y_mean_t
                    out4[k][:,:,js:je,is_:ie] += y * ww

                for k in keys_1:
                    out1[k][:,:,js:je,is_:ie] += d[k].float() * ww

                w_acc[:,:,js:je,is_:ie] += ww

    result = {}
    for k in keys_4:
        result[k] = (out4[k] / (w_acc + 1e-12)).detach().cpu().numpy()[0]
    for k in keys_1:
        result[k] = (out1[k] / (w_acc + 1e-12)).detach().cpu().numpy()[0,0]
    return result


# ============================================================
# Physics stats
# ============================================================
def safe_flat(a, mask):
    x = a[mask]
    x = x[np.isfinite(x)]
    return x.reshape(-1)


def analyze_one_scale(n_coarse: int):
    print("\n==============================")
    print(f"Analyze Experiment 3 | n={n_coarse}")
    print("==============================")

    label_dir = resolve_label_dir_for_n(LABEL_ROOT, n_coarse)
    ckpt_dir = os.path.join(CKPT_ROOT, f"n{n_coarse}", "checkpoints")
    out_dir = os.path.join(RESULT_ROOT, f"n{n_coarse}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    theta_files = sorted(glob.glob(os.path.join(VAR_DIR, "thetao_Oday_FGOALS-f3-H_omip2_r1i1p1f1_gn_*.nc")))
    items = build_day_level_index(theta_files, label_dir, max_days=MAX_DAYS)

    file_ids = np.array([it["file_id"] for it in items], dtype=int)
    unique_files = np.sort(np.unique(file_ids))
    split_idx = int(0.7 * len(unique_files))
    train_files = set(unique_files[:split_idx])
    val_files   = set(unique_files[split_idx:])

    train_items = [it for it in items if it["file_id"] in train_files]
    val_items   = [it for it in items if it["file_id"] in val_files]

    X_mean, X_std, Y_mean, Y_std = estimate_norm_from_train(train_items, n_samples=2_000_000, seed=0)
    K_vec_n = fit_prior_K_vec_normalized(train_items, X_mean, X_std, Y_mean, Y_std, seed=0)

    alpha_s_max = SCAPEX_ALPHA_S_MAX_N3 if n_coarse == 3 else SCAPEX_ALPHA_S_MAX_OTH

    model = SCAPEXNetDiagnostics(
        n_coarse=n_coarse,
        X_mean=X_mean, X_std=X_std,
        Y_mean=Y_mean, Y_std=Y_std,
        K_vec_n=K_vec_n,
        width=SCAPEX_WIDTH,
        n_blocks=SCAPEX_NBLOCKS,
        afno_depth=SCAPEX_AFNO_DEPTH,
        alpha_s_max=alpha_s_max,
    ).to(DEVICE)

    ckpt_path = find_model_ckpt(ckpt_dir)
    print("Loading checkpoint:", ckpt_path)
    load_state_flexible(model, ckpt_path)
    model.eval()

    ds = LazyLevFullEvalDataset(val_items, X_mean, X_std, Y_mean, Y_std)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    stats = {
        "PiT_diss": [], "PiT_rot": [], "PiT_total": [],
        "PiS_diss": [], "PiS_rot": [], "PiS_total": [],
        "Keff_T": [], "Keff_S": [],
        "alphaT": [], "alphaS": [], "betaT": [], "betaS": [],
        "rot_ratio_T": [], "rot_ratio_S": [],
    }

    rep_saved = False

    for i, batch in enumerate(loader):
        Xn, Y_phys, ocean, file_id, lev, X_phys = batch
        Xn = Xn.to(DEVICE)

        Xp = X_phys.numpy()[0]
        mask = ocean.numpy()[0].astype(bool)

        diag = predict_scapex_diag_tiled(
            model, Xn, Y_mean, Y_std,
            tile=EVAL_TILE, overlap=EVAL_OVERLAP
        )

        dTdx = Xp[4]
        dTdy = Xp[5]
        dSdx = Xp[6]
        dSdy = Xp[7]

        qd = diag["q_diss_eff"]
        qr = diag["q_rot_eff"]
        qp = diag["pred"]

        PiT_diss = -(qd[0] * dTdx + qd[1] * dTdy)
        PiT_rot  = -(qr[0] * dTdx + qr[1] * dTdy)
        PiT_total = -(qp[0] * dTdx + qp[1] * dTdy)

        PiS_diss = -(qd[2] * dSdx + qd[3] * dSdy)
        PiS_rot  = -(qr[2] * dSdx + qr[3] * dSdy)
        PiS_total = -(qp[2] * dSdx + qp[3] * dSdy)

        gT2 = dTdx**2 + dTdy**2
        gS2 = dSdx**2 + dSdy**2
        Keff_T = PiT_diss / (gT2 + 1e-12)
        Keff_S = PiS_diss / (gS2 + 1e-12)

        mag_qd_T = np.sqrt(qd[0]**2 + qd[1]**2)
        mag_qr_T = np.sqrt(qr[0]**2 + qr[1]**2)
        mag_qd_S = np.sqrt(qd[2]**2 + qd[3]**2)
        mag_qr_S = np.sqrt(qr[2]**2 + qr[3]**2)

        rot_ratio_T = mag_qr_T / (mag_qd_T + mag_qr_T + 1e-12)
        rot_ratio_S = mag_qr_S / (mag_qd_S + mag_qr_S + 1e-12)

        stats["PiT_diss"].append(safe_flat(PiT_diss, mask))
        stats["PiT_rot"].append(safe_flat(PiT_rot, mask))
        stats["PiT_total"].append(safe_flat(PiT_total, mask))

        stats["PiS_diss"].append(safe_flat(PiS_diss, mask))
        stats["PiS_rot"].append(safe_flat(PiS_rot, mask))
        stats["PiS_total"].append(safe_flat(PiS_total, mask))

        stats["Keff_T"].append(safe_flat(Keff_T, mask & np.isfinite(gT2) & (gT2 > 1e-12)))
        stats["Keff_S"].append(safe_flat(Keff_S, mask & np.isfinite(gS2) & (gS2 > 1e-12)))

        stats["alphaT"].append(safe_flat(diag["alphaT"], mask))
        stats["alphaS"].append(safe_flat(diag["alphaS"], mask))
        stats["betaT"].append(safe_flat(diag["betaT"], mask))
        stats["betaS"].append(safe_flat(diag["betaS"], mask))

        stats["rot_ratio_T"].append(safe_flat(rot_ratio_T, mask))
        stats["rot_ratio_S"].append(safe_flat(rot_ratio_S, mask))

        if (not rep_saved) and (i == REP_SAMPLE_INDEX):
            rep_saved = True

            # ---- representative mechanism maps for paper ----
            img_T_list = [mag_qd_T[mask], mag_qr_T[mask]]
            img_S_list = [mag_qd_S[mask], mag_qr_S[mask]]

            vmax_T = np.nanpercentile(np.concatenate(img_T_list), 99)
            vmax_S = np.nanpercentile(np.concatenate(img_S_list), 99)

            fig, axs = plt.subplots(2, 2, figsize=(8.8, 7.2))

            panels = [
                (mag_qd_T, r"$|Q_{\mathrm{diss}}^{T,*}|$", 0.0, vmax_T),
                (mag_qr_T, r"$|Q_{\mathrm{rot}}^{T,*}|$", 0.0, vmax_T),
                (mag_qd_S, r"$|Q_{\mathrm{diss}}^{S,*}|$", 0.0, vmax_S),
                (mag_qr_S, r"$|Q_{\mathrm{rot}}^{S,*}|$", 0.0, vmax_S),
            ]

            for ax, (img, ttl, vmin, vmax) in zip(axs.ravel(), panels):
                img_plot = np.where(mask, img, np.nan)
                im = ax.imshow(img_plot, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_title(ttl)
                ax.set_xticks([])
                ax.set_yticks([])

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4%", pad=0.08)
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=9)

            fig.tight_layout()
            savefig_tight(fig, os.path.join(fig_dir, f"exp3_rep_mechanism_maps_n{n_coarse}.png"))

    for k in stats:
        stats[k] = np.concatenate(stats[k], axis=0) if len(stats[k]) else np.array([], dtype=np.float32)

    # ---- histogram: production ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    axs[0].hist(stats["PiT_diss"], bins=100, density=True, alpha=0.65, label="diss")
    axs[0].hist(stats["PiT_rot"],  bins=100, density=True, alpha=0.65, label="rot")
    axs[0].axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    axs[0].set_title(r"Temperature production: $-\mathbf{q}\cdot\nabla T$")
    axs[0].set_xlabel("Production")
    axs[0].set_ylabel("PDF")
    axs[0].legend(frameon=True)

    axs[1].hist(stats["PiS_diss"], bins=100, density=True, alpha=0.65, label="diss")
    axs[1].hist(stats["PiS_rot"],  bins=100, density=True, alpha=0.65, label="rot")
    axs[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    axs[1].set_title(r"Salinity production: $-\mathbf{q}\cdot\nabla S$")
    axs[1].set_xlabel("Production")
    axs[1].set_ylabel("PDF")
    axs[1].legend(frameon=True)

    savefig_tight(fig, os.path.join(fig_dir, f"exp3_production_pdf_n{n_coarse}.png"))

    # ---- histogram: effective diffusivity ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    xT = stats["Keff_T"]
    xS = stats["Keff_S"]
    xT = xT[np.isfinite(xT)]
    xS = xS[np.isfinite(xS)]

    if xT.size > 0:
        lo, hi = np.quantile(xT, [0.01, 0.99])
        axs[0].hist(np.clip(xT, lo, hi), bins=100, density=True, alpha=0.8)
    axs[0].set_title(r"Effective diffusivity for temperature")
    axs[0].set_xlabel(r"$K_{\mathrm{eff}}^T$")
    axs[0].set_ylabel("PDF")

    if xS.size > 0:
        lo, hi = np.quantile(xS, [0.01, 0.99])
        axs[1].hist(np.clip(xS, lo, hi), bins=100, density=True, alpha=0.8)
    axs[1].set_title(r"Effective diffusivity for salinity")
    axs[1].set_xlabel(r"$K_{\mathrm{eff}}^S$")
    axs[1].set_ylabel("PDF")

    savefig_tight(fig, os.path.join(fig_dir, f"exp3_keff_pdf_n{n_coarse}.png"))

    # ---- save summary csv/json ----
    summary = {
        "n": n_coarse,
        "alphaT_mean": float(np.nanmean(stats["alphaT"])),
        "alphaS_mean": float(np.nanmean(stats["alphaS"])),
        "betaT_mean": float(np.nanmean(stats["betaT"])),
        "betaS_mean": float(np.nanmean(stats["betaS"])),
        "rot_ratio_T_mean": float(np.nanmean(stats["rot_ratio_T"])),
        "rot_ratio_S_mean": float(np.nanmean(stats["rot_ratio_S"])),
        "PiT_diss_positive_frac": float(np.mean(stats["PiT_diss"] > 0.0)),
        "PiT_rot_positive_frac": float(np.mean(stats["PiT_rot"] > 0.0)),
        "PiS_diss_positive_frac": float(np.mean(stats["PiS_diss"] > 0.0)),
        "PiS_rot_positive_frac": float(np.mean(stats["PiS_rot"] > 0.0)),
        "Keff_T_median": float(np.nanmedian(stats["Keff_T"])) if stats["Keff_T"].size else np.nan,
        "Keff_S_median": float(np.nanmedian(stats["Keff_S"])) if stats["Keff_S"].size else np.nan,
    }

    with open(os.path.join(out_dir, f"exp3_summary_n{n_coarse}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for k, v in summary.items():
        rows.append({"key": k, "value": v})
    with open(os.path.join(out_dir, f"exp3_summary_n{n_coarse}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return summary


# ============================================================
# Cross-scale summary plot
# ============================================================
def plot_cross_scale_summary(all_summary: Dict[int, Dict]):
    scales = sorted(all_summary.keys())

    alphaT = [all_summary[n]["alphaT_mean"] for n in scales]
    alphaS = [all_summary[n]["alphaS_mean"] for n in scales]
    betaT  = [all_summary[n]["betaT_mean"]  for n in scales]
    betaS  = [all_summary[n]["betaS_mean"]  for n in scales]

    rotT = [all_summary[n]["rot_ratio_T_mean"] for n in scales]
    rotS = [all_summary[n]["rot_ratio_S_mean"] for n in scales]

    # -------------------------------------------------
    # Figure 1: scale-dependent gating behavior
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)

    ax.plot(scales, alphaT, marker="o", linewidth=2.6, markersize=8, label=r"$\alpha_T$ (prior, temperature)")
    ax.plot(scales, alphaS, marker="o", linewidth=2.6, markersize=8, label=r"$\alpha_S$ (prior, salinity)")
    ax.plot(scales, betaT,  marker="s", linewidth=2.6, markersize=8, label=r"$\beta_T$ (structured, temperature)")
    ax.plot(scales, betaS,  marker="s", linewidth=2.6, markersize=8, label=r"$\beta_S$ (structured, salinity)")

    for x, y in zip(scales, alphaT):
        ax.text(x, y + 0.018, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    for x, y in zip(scales, alphaS):
        ax.text(x, y + 0.018, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    for x, y in zip(scales, betaT):
        ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    for x, y in zip(scales, betaS):
        ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Scale-dependent gating behavior")
    ax.set_xlabel(r"Coarse-graining factor $n$")
    ax.set_ylabel("Mean gate value")
    ax.set_xticks(scales)
    ax.set_ylim(0.0, 0.85)
    ax.legend(frameon=True, ncol=2)
    savefig_tight(fig, os.path.join(RESULT_ROOT, "exp3_gate_vs_scale.png"))

    # -------------------------------------------------
    # Figure 2: rotational contribution across scales
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    ax.plot(scales, rotT, marker="o", linewidth=2.8, markersize=8, label="Temperature")
    ax.plot(scales, rotS, marker="o", linewidth=2.8, markersize=8, label="Salinity")

    for x, y in zip(scales, rotT):
        ax.text(x, y + 0.003, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    for x, y in zip(scales, rotS):
        ax.text(x, y + 0.003, f"{y:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Rotational contribution across scales")
    ax.set_xlabel(r"Coarse-graining factor $n$")
    ax.set_ylabel(r"$|Q_{\mathrm{rot}}|/(|Q_{\mathrm{diss}}|+|Q_{\mathrm{rot}}|)$")
    ax.set_xticks(scales)
    ax.set_ylim(0.41, 0.51)
    ax.legend(frameon=True)
    savefig_tight(fig, os.path.join(RESULT_ROOT, "exp3_rotation_vs_scale.png"))


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(RESULT_ROOT, exist_ok=True)

    all_summary = {}
    for n in COARSE_LIST:
        summary = analyze_one_scale(n)
        all_summary[n] = summary

    with open(os.path.join(RESULT_ROOT, "exp3_all_scales_summary.json"), "w") as f:
        json.dump(all_summary, f, indent=2)

    plot_cross_scale_summary(all_summary)
    # copy representative mechanism map for the main paper figure
    src = os.path.join(
        RESULT_ROOT, f"n{REP_FIG_N}", "figures", f"exp3_rep_mechanism_maps_n{REP_FIG_N}.png"
    )
    dst = os.path.join(RESULT_ROOT, f"exp3_rep_mechanism_maps_main_n{REP_FIG_N}.png")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print("Saved:", dst)
    print("Done.")


if __name__ == "__main__":
    main()
