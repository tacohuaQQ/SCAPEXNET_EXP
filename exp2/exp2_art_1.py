#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import math
import time
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import xarray as xr

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================
# Global config
# ============================================================
VAR_DIR     = "/home/Data/zhufuhua/MyData/para2/data/global"
LABEL_ROOT  = "/home/Data/zhufuhua/MyData/para2/data/global/thelabel"
RESULT_ROOT = "/home/Data/zhufuhua/MyData/para2/data/global/theresult_ablation"

COARSE_LIST = [3, 5, 10]
MAX_DAYS    = 10

PATCH_SIZE  = 96
TRAIN_PATCHES_PER_ITEM = 4

BATCH_SIZE_IMG_TRAIN = 4
BATCH_SIZE_IMG_EVAL  = 1

NUM_WORKERS_TRAIN = 8
NUM_WORKERS_EVAL  = 0   # safer: avoid multiprocessing + CUDA teardown issue
PIN_MEMORY  = True
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

LR_SCAPEX        = 2e-4
N_EPOCHS_SCAPEX  = 50
CLIP_GRAD        = 1.0

SCAPEX_WIDTH      = 64
SCAPEX_NBLOCKS    = 4
SCAPEX_AFNO_DEPTH = 2

SCAPEX_LAM_DISS    = 0.03
SCAPEX_DISS_WARMUP = 2
SCAPEX_LAM_HF_T    = 0.05
SCAPEX_LAM_HF_S    = 0.00
SCAPEX_USE_HF      = True

SCAPEX_LAM_LF_N10  = 0.06
SCAPEX_LAM_DIV_N3  = 0.04

SCAPEX_ALPHA_S_MAX_N3  = 0.55
SCAPEX_ALPHA_S_MAX_OTH = 0.90

HF_LOSS_TYPE = "laplacian"   # "laplacian" or "dog"
DOG_KSIZE  = 7
DOG_SIGMA1 = 1.0
DOG_SIGMA2 = 2.0

EVAL_TILE = 192
EVAL_OVERLAP = 32

ABLATION_TAGS = [
    "SCAPEX_FULL",
    "SCAPEX_WO_PRIOR",
    "SCAPEX_WO_STRUCT",
    "SCAPEX_WO_RES",
    "SCAPEX_WO_SCALE",
    "SCAPEX_WO_ADAGATE",
    "SCAPEX_WO_SCAP",
]

ABLATION_COLORS = {
    "SCAPEX_FULL": "#d62728",
    "SCAPEX_WO_PRIOR": "#1f77b4",
    "SCAPEX_WO_STRUCT": "#ff7f0e",
    "SCAPEX_WO_RES": "#2ca02c",
    "SCAPEX_WO_SCALE": "#9467bd",
    "SCAPEX_WO_ADAGATE": "#8c564b",
    "SCAPEX_WO_SCAP": "#e377c2",
}

flux_names = ["qT_x", "qT_y", "qS_x", "qS_y"]


# ============================================================
# Device / AMP
# ============================================================
USE_AMP = True

CUDA_AVAILABLE = torch.cuda.is_available()
N_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
GPU_IDS = list(range(N_GPUS))
USE_DP = CUDA_AVAILABLE and (N_GPUS >= 2)

DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
DEVICE_IS_CUDA = str(DEVICE).startswith("cuda")

if USE_DP:
    print(f"[MultiGPU] Using DataParallel on GPUs: {GPU_IDS}")
else:
    print(f"[Device] Using: {DEVICE}")

from contextlib import nullcontext


def make_scaler():
    if DEVICE_IS_CUDA and USE_AMP:
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.amp.GradScaler("cuda", enabled=False)


def autocast_ctx():
    if DEVICE_IS_CUDA and USE_AMP:
        return torch.amp.autocast("cuda", enabled=True)
    return nullcontext()


np.random.seed(42)
torch.manual_seed(42)
if DEVICE_IS_CUDA:
    torch.cuda.manual_seed_all(42)


# ============================================================
# Plot style
# ============================================================
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
})


# ============================================================
# Logger
# ============================================================
LOGGER: Optional[logging.Logger] = None


def setup_logger(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{t}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    logger.info(f"Log file: {log_path}")
    return logger


def log(msg: str, level: str = "info"):
    if LOGGER is None:
        print(msg)
        return
    if level == "info":
        LOGGER.info(msg)
    elif level == "warning":
        LOGGER.warning(msg)
    elif level == "error":
        LOGGER.error(msg)
    else:
        LOGGER.info(msg)


# ============================================================
# xarray open config
# ============================================================
XR_OPEN_KW = dict(
    decode_cf=False,
    mask_and_scale=False,
    decode_times=False,
    cache=True,
)


# ============================================================
# IO utils
# ============================================================
def savefig_tight(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log(f"Saved: {path}")


def write_rows_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def metrics_to_rows(all_metrics, n=None, add_n=False):
    rows = []
    for method, md in all_metrics.items():
        for comp in ["qT_x", "qT_y", "qS_x", "qS_y"]:
            if comp not in md:
                continue
            d = md[comp]
            row = {
                "method": method,
                "component": comp,
                "rmse": d.get("rmse", np.nan),
                "corr": d.get("corr", np.nan),
                "r2":   d.get("r2", np.nan),
            }
            if add_n:
                row["n"] = n
            rows.append(row)

        if "mean" in md:
            d = md["mean"]
            row = {
                "method": method,
                "component": "mean(4)",
                "rmse": d.get("rmse", np.nan),
                "corr": d.get("corr", np.nan),
                "r2":   d.get("r2", np.nan),
            }
            if add_n:
                row["n"] = n
            rows.append(row)
    return rows


# ============================================================
# Checkpoint
# ============================================================
def ckpt_path(ckpt_dir: str, tag: str):
    safe = tag.replace(" ", "_").replace("/", "_")
    return os.path.join(ckpt_dir, f"{safe}.pt")


def _unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


def _strip_module_prefix(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def save_ckpt(ckpt_dir, tag, model, extra=None):
    obj = {"state_dict": _unwrap_model(model).state_dict()}
    if extra is not None:
        obj["extra"] = extra
    path = ckpt_path(ckpt_dir, tag)
    torch.save(obj, path)
    log(f"Saved ckpt: {path}")


def load_ckpt(ckpt_dir, tag, model):
    path = ckpt_path(ckpt_dir, tag)
    if os.path.exists(path):
        obj = torch.load(path, map_location="cpu")
        sd = obj["state_dict"]
        target = _unwrap_model(model)
        try:
            target.load_state_dict(sd, strict=True)
        except RuntimeError:
            target.load_state_dict(_strip_module_prefix(sd), strict=True)
        log(f"Loaded ckpt: {path}")
        return True
    return False


class BestCkptTracker:
    def __init__(self, ckpt_dir: str, tag: str, model: nn.Module, mode: str = "min", metric_name: str = "val"):
        assert mode in ("min", "max")
        self.ckpt_dir = ckpt_dir
        self.tag = tag
        self.model = model
        self.mode = mode
        self.metric_name = metric_name
        self.best_metric = float("inf") if mode == "min" else -float("inf")
        self.best_ep = 0

    def try_load(self) -> bool:
        return load_ckpt(self.ckpt_dir, self.tag + "_best", self.model)

    def _is_better(self, x: float) -> bool:
        if not np.isfinite(x):
            return False
        if self.mode == "min":
            return x < self.best_metric
        return x > self.best_metric

    def update(self, metric_value: float, epoch: int, extra_best: Optional[Dict[str, Any]] = None):
        metric_value = float(metric_value)
        if self._is_better(metric_value):
            self.best_metric = metric_value
            self.best_ep = int(epoch)
            extra = {} if extra_best is None else dict(extra_best)
            extra.update({"best_metric": self.best_metric, "best_ep": self.best_ep, "metric_name": self.metric_name})
            save_ckpt(self.ckpt_dir, self.tag + "_best", self.model, extra=extra)
            log(f"[{self.tag}] ✅ New best {self.metric_name}: {self.best_metric:.6f} @ epoch {self.best_ep}")

    def finalize(self, extra_last: Optional[Dict[str, Any]] = None, load_best: bool = True):
        best_path = ckpt_path(self.ckpt_dir, self.tag + "_best")
        if not os.path.exists(best_path):
            extra = {} if extra_last is None else dict(extra_last)
            extra.update({"best_metric": self.best_metric, "best_ep": self.best_ep, "metric_name": self.metric_name})
            save_ckpt(self.ckpt_dir, self.tag + "_best", self.model, extra=extra)
        if load_best:
            load_ckpt(self.ckpt_dir, self.tag + "_best", self.model)


# ============================================================
# Data helpers
# ============================================================
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
    if "dx" not in ds or "dy" not in ds:
        raise RuntimeError("label file missing dx/dy")
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


def phys_grad_core_center(f_h, dx_h, dy_h, h, ps):
    fx_p = f_h[h:h+ps, h+1:h+ps+1]
    fx_m = f_h[h:h+ps, h-1:h+ps-1]
    dx_c = dx_h[h:h+ps, h:h+ps]
    dfdx = (fx_p - fx_m) / (2.0 * dx_c)

    fy_p = f_h[h+1:h+ps+1, h:h+ps]
    fy_m = f_h[h-1:h+ps-1, h:h+ps]
    dy_c = dy_h[h:h+ps, h:h+ps]
    dfdy = (fy_p - fy_m) / (2.0 * dy_c)

    return dfdx.astype(np.float32, copy=False), dfdy.astype(np.float32, copy=False)


def build_day_level_index(theta_files: List[str], label_dir: str, max_days: int) -> List[dict]:
    theta_files = theta_files[:max_days]
    items: List[dict] = []
    for file_id, theta_path in enumerate(theta_files):
        base = os.path.basename(theta_path)
        label_path = os.path.join(label_dir, base.replace("thetao_", "label_flux_"))
        if not os.path.exists(label_path):
            log(f"⚠️ label missing: {label_path}", "warning")
            continue

        ds = xr.open_dataset(label_path, **XR_OPEN_KW)
        try:
            need = ["qT_x", "qT_y", "qS_x", "qS_y", "theta_lr", "so_lr", "uo_lr", "vo_lr", "dx", "dy"]
            for k in need:
                if k not in ds:
                    raise RuntimeError(f"label missing variable {k}: {label_path}")

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


def estimate_norm_from_train(
    items: List[dict],
    n_samples: int = 2_000_000,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    cnt = 0
    meanX = np.zeros(8, dtype=np.float64)
    m2X   = np.zeros(8, dtype=np.float64)
    meanY = np.zeros(4, dtype=np.float64)
    m2Y   = np.zeros(4, dtype=np.float64)

    per_item = max(10_000, n_samples // max(1, len(items)))
    t0 = time.perf_counter()

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
            if nb == 0:
                continue

            bx = Xs.mean(axis=0)
            by = Ys.mean(axis=0)

            sumsqX = (Xs * Xs).sum(axis=0)
            sumsqY = (Ys * Ys).sum(axis=0)
            bM2x = sumsqX - nb * (bx * bx)
            bM2y = sumsqY - nb * (by * by)

            if cnt == 0:
                meanX = bx
                m2X = bM2x
                meanY = by
                m2Y = bM2y
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

    log(f"✅ Norm estimated: cnt={cnt:,} | time={time.perf_counter()-t0:.1f}s")
    return X_mean, X_std, Y_mean, Y_std


# ============================================================
# Dataset cache mixin
# ============================================================
class _XrCacheMixin:
    def __init__(self):
        self._ds_cache: Dict[str, xr.Dataset] = {}

    def _open_cached(self, path: str) -> xr.Dataset:
        ds = self._ds_cache.get(path)
        if ds is None:
            ds = xr.open_dataset(path, **XR_OPEN_KW)
            self._ds_cache[path] = ds
        return ds

    def close_cache(self):
        for _, ds in self._ds_cache.items():
            try:
                ds.close()
            except Exception:
                pass
        self._ds_cache.clear()


# ============================================================
# Train patch dataset
# ============================================================
class LazyLevPatchDataset(Dataset, _XrCacheMixin):
    def __init__(self, items: List[dict], X_mean, X_std, Y_mean, Y_std,
                 patch_size: int, seed: int = 0, patches_per_item=4, halo: int = 1):
        Dataset.__init__(self)
        _XrCacheMixin.__init__(self)
        self.items = items
        self.X_mean = X_mean.reshape(8,1,1).astype(np.float32)
        self.X_std  = X_std.reshape(8,1,1).astype(np.float32)
        self.Y_mean = Y_mean.reshape(4,1,1).astype(np.float32)
        self.Y_std  = Y_std.reshape(4,1,1).astype(np.float32)
        self.patch_size = int(patch_size)
        self.halo = int(halo)
        self.rng = np.random.default_rng(seed)
        self.patches_per_item = int(patches_per_item)
        self._shape_cache = {}

    def __len__(self):
        return len(self.items) * self.patches_per_item

    def _get_hw(self, ds, label_path, lev):
        hw = self._shape_cache.get(label_path)
        if hw is None:
            da0 = ds["theta_lr"].isel(time=0, lev=lev)
            hw = (int(da0.sizes["j"]), int(da0.sizes["i"]))
            self._shape_cache[label_path] = hw
        return hw

    def __getitem__(self, idx):
        it = self.items[idx % len(self.items)]
        ds = self._open_cached(it["label_path"])
        lev = it["lev"]

        ps = self.patch_size
        h  = self.halo

        H, W = self._get_hw(ds, it["label_path"], lev)
        if (H < ps + 2*h) or (W < ps + 2*h):
            raise RuntimeError(f"Patch too large for halo: H={H}, W={W}, ps={ps}, halo={h}")

        j0 = self.rng.integers(h, H - ps - h + 1)
        i0 = self.rng.integers(h, W - ps - h + 1)

        sl_j  = slice(j0,     j0 + ps)
        sl_i  = slice(i0,     i0 + ps)
        sl_jh = slice(j0 - h, j0 + ps + h)
        sl_ih = slice(i0 - h, i0 + ps + h)

        u = ds["uo_lr"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)
        v = ds["vo_lr"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)

        qTx = ds["qT_x"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)
        qTy = ds["qT_y"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)
        qSx = ds["qS_x"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)
        qSy = ds["qS_y"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)

        t_h = ds["theta_lr"].isel(time=0, lev=lev, j=sl_jh, i=sl_ih).values.astype(np.float32, copy=False)
        s_h = ds["so_lr"].isel(time=0, lev=lev, j=sl_jh, i=sl_ih).values.astype(np.float32, copy=False)

        dx_h = ds["dx"].isel(j=sl_jh, i=sl_ih).values.astype(np.float32, copy=False)
        dy_h = ds["dy"].isel(j=sl_jh, i=sl_ih).values.astype(np.float32, copy=False)

        dTdx, dTdy = phys_grad_core_center(t_h, dx_h, dy_h, h, ps)
        dSdx, dSdy = phys_grad_core_center(s_h, dx_h, dy_h, h, ps)

        core = slice(h, h + ps)
        t = t_h[core, core]
        s = s_h[core, core]

        X = np.stack([u, v, t, s, dTdx, dTdy, dSdx, dSdy], axis=0).astype(np.float32, copy=False)
        Y = np.stack([qTx, qTy, qSx, qSy], axis=0).astype(np.float32, copy=False)

        ocean = np.isfinite(Y[0])

        Xn = (X - self.X_mean) / self.X_std
        Xn = np.nan_to_num(Xn, nan=0.0)

        Yn = (Y - self.Y_mean) / self.Y_std
        Yn[:, ~ocean] = np.nan

        return (
            torch.from_numpy(Xn),
            torch.from_numpy(Yn),
            torch.from_numpy(ocean.astype(np.bool_))
        )


# ============================================================
# Full eval dataset
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
# Loss helpers
# ============================================================
def masked_mse_torch_with_mask(pred, target, mask2d):
    m = mask2d.unsqueeze(1).expand_as(pred) & torch.isfinite(target)
    diff = pred[m] - target[m]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (diff ** 2).mean()


def _masked_mean_tensor(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x.device)
    return x[mask].mean()


def laplacian_filter(x):
    B, C, H, W = x.shape
    k = torch.tensor([[0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    k = k.repeat(C, 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=C)


def gaussian_kernel2d(ksize, sigma, device, dtype):
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-(xx*xx + yy*yy) / (2*sigma*sigma))
    ker = ker / ker.sum()
    return ker


def dog_filter(x, ksize=7, sigma1=1.0, sigma2=2.0):
    B, C, H, W = x.shape
    g1 = gaussian_kernel2d(ksize, sigma1, x.device, x.dtype).view(1,1,ksize,ksize).repeat(C,1,1,1)
    g2 = gaussian_kernel2d(ksize, sigma2, x.device, x.dtype).view(1,1,ksize,ksize).repeat(C,1,1,1)
    y1 = F.conv2d(x, g1, padding=ksize//2, groups=C)
    y2 = F.conv2d(x, g2, padding=ksize//2, groups=C)
    return y1 - y2


def _rfft2_safe(x: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    return torch.fft.rfft2(x, norm="ortho")


def make_freq_weight_low(H, W, k0_frac=0.22, p=1.0, device="cpu", dtype=torch.float32):
    ky = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype).view(H, 1)
    kx = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=dtype).view(1, W//2 + 1)
    kk = torch.sqrt(ky*ky + kx*kx)
    kk = kk / (kk.max() + 1e-12)
    gate = (kk <= k0_frac).float()
    w = gate * ((1.0 - kk) ** p)
    return w


def spectral_lf_loss_amp(pred, target, ocean_mask2d, k0_frac=0.22, p=1.0):
    B, C, H, W = pred.shape
    m = ocean_mask2d.unsqueeze(1).expand_as(pred).float()

    pred_m = pred * m
    targ_m = torch.nan_to_num(target, nan=0.0) * m

    Fp = _rfft2_safe(pred_m)
    Ft = _rfft2_safe(targ_m)
    Ap = torch.abs(Fp)
    At = torch.abs(Ft)

    w = make_freq_weight_low(H, W, k0_frac=k0_frac, p=p, device=pred.device, dtype=torch.float32)
    w = w.to(Ap.dtype).view(1,1,H,W//2+1)

    return (w * (Ap - At).pow(2)).mean()


def _div2d_pixel(qx, qy):
    kx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=qx.dtype, device=qx.device).view(1,1,1,3)
    ky = torch.tensor([[-0.5, 0.0, 0.5]], dtype=qy.dtype, device=qy.device).view(1,1,3,1)
    dqx_dx = F.conv2d(qx, kx, padding=(0,1))
    dqy_dy = F.conv2d(qy, ky, padding=(1,0))
    return dqx_dx + dqy_dy


def divergence_loss_pixel(pred, target, ocean_mask2d):
    qTx_p = pred[:,0:1]
    qTy_p = pred[:,1:2]
    qSx_p = pred[:,2:3]
    qSy_p = pred[:,3:4]

    qTx_t = torch.nan_to_num(target[:,0:1], nan=0.0)
    qTy_t = torch.nan_to_num(target[:,1:2], nan=0.0)
    qSx_t = torch.nan_to_num(target[:,2:3], nan=0.0)
    qSy_t = torch.nan_to_num(target[:,3:4], nan=0.0)

    divT_p = _div2d_pixel(qTx_p, qTy_p)
    divT_t = _div2d_pixel(qTx_t, qTy_t)
    divS_p = _div2d_pixel(qSx_p, qSy_p)
    divS_t = _div2d_pixel(qSx_t, qSy_t)

    div_p = torch.cat([divT_p, divS_p], dim=1)
    div_t = torch.cat([divT_t, divS_t], dim=1)
    return masked_mse_torch_with_mask(div_p, div_t, ocean_mask2d)


# ============================================================
# SCAPEX dependencies
# ============================================================
class LayerNorm2d(nn.Module):
    def __init__(self, n_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.bias   = nn.Parameter(torch.zeros(n_channels))
        self.eps = eps

    def forward(self, x):
        mu  = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1,-1,1,1) + self.bias.view(1,-1,1,1)


class CNOBlock(nn.Module):
    def __init__(self, ch, dilation=1, drop=0.0):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, groups=ch, bias=False)
        self.pw1 = nn.Conv2d(ch, ch*2, 1, bias=True)
        self.pw2 = nn.Conv2d(ch*2, ch, 1, bias=True)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.norm = nn.BatchNorm2d(ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.drop(y)
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

        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            x_fft = x.float()
        else:
            x_fft = x

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
        y = torch.fft.irfft2(y_ft, s=(H,W), norm="ortho")
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

    def grad(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.width = width
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
        if n == 3:
            return 0
        if n == 5:
            return 1
        if n == 10:
            return 2
        return 7


# ============================================================
# SCAPEX ablation model
# ============================================================
class SCAPEXNetAblation(nn.Module):
    """
    SCAPEX ablation model:
    prior + structured + residual + adaptive gate + scale conditioning
    """
    def __init__(
        self,
        n_coarse: int,
        X_mean, X_std, Y_mean, Y_std,
        K_vec_n: np.ndarray,
        width=64, n_blocks=4, afno_depth=2,
        alpha_s_max: float = 0.9,
        use_prior: bool = True,
        use_struct: bool = True,
        use_res: bool = True,
        use_scale: bool = True,
        use_adaptive_gate: bool = True,
        use_salinity_cap: bool = True,
    ):
        super().__init__()

        self.register_buffer("X_mean", torch.tensor(X_mean, dtype=torch.float32).view(1,8,1,1))
        self.register_buffer("X_std",  torch.tensor(X_std,  dtype=torch.float32).view(1,8,1,1))
        self.register_buffer("Y_mean", torch.tensor(Y_mean, dtype=torch.float32).view(1,4,1,1))
        self.register_buffer("Y_std",  torch.tensor(Y_std,  dtype=torch.float32).view(1,4,1,1))

        K_vec = torch.tensor(K_vec_n, dtype=torch.float32).view(4,1,1)
        self.register_buffer("K_vec", K_vec)

        self.alpha_s_max = float(alpha_s_max)

        self.use_prior = bool(use_prior)
        self.use_struct = bool(use_struct)
        self.use_res = bool(use_res)
        self.use_scale = bool(use_scale)
        self.use_adaptive_gate = bool(use_adaptive_gate)
        self.use_salinity_cap = bool(use_salinity_cap)

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

        res_groups = 3
        res_rcab = 4
        self.res_body = nn.Sequential(*[ResidualGroup(width, n_rcab=res_rcab) for _ in range(res_groups)])
        self.res_conv = nn.Conv2d(width, width, 3, padding=1)
        self.res_out  = nn.Conv2d(width, 4, 3, padding=1)

        self.diff = FiniteDiff2D()

        self.alpha_diss_T = nn.Parameter(torch.tensor(1.0))
        self.alpha_rot_T  = nn.Parameter(torch.tensor(1.0))
        self.alpha_diss_S = nn.Parameter(torch.tensor(1.0))
        self.alpha_rot_S  = nn.Parameter(torch.tensor(1.0))

    def denorm_X(self, Xn: torch.Tensor) -> torch.Tensor:
        return Xn * self.X_std + self.X_mean

    def denorm_Y(self, Yn: torch.Tensor) -> torch.Tensor:
        return Yn * self.Y_std + self.Y_mean

    def _prior_flux(self, Xn: torch.Tensor) -> torch.Tensor:
        dTdx = Xn[:,4:5]
        dTdy = Xn[:,5:6]
        dSdx = Xn[:,6:7]
        dSdy = Xn[:,7:8]
        qK = torch.cat([
            -self.K_vec[0:1] * dTdx, -self.K_vec[1:2] * dTdy,
            -self.K_vec[2:3] * dSdx, -self.K_vec[3:4] * dSdy,
        ], dim=1)
        return qK

    def _pot_to_flux_parts(self, pot: torch.Tensor):
        chiT = pot[:, 0:1]
        psiT = pot[:, 1:2]
        chiS = pot[:, 2:3]
        psiS = pot[:, 3:4]

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

    def forward(self, Xn: torch.Tensor, return_parts: bool = False):
        feat = self.trunk(Xn)

        if self.use_scale:
            emb = self.scale_emb(self.scale_id_t.to(feat.device)).view(1, 8, 1, 1)
            gate_bias = emb[:, 0:4]
            pot_bias  = emb[:, 4:8]
        else:
            gate_bias = torch.zeros((1, 4, 1, 1), device=feat.device, dtype=feat.dtype)
            pot_bias  = torch.zeros((1, 4, 1, 1), device=feat.device, dtype=feat.dtype)

        g_logits = self.gate_head(feat) + gate_bias
        g = torch.sigmoid(g_logits)

        if self.use_adaptive_gate:
            alphaT = g[:, 0:1]
            alphaS_raw = g[:, 1:2]
            betaT = g[:, 2:3]
            betaS = g[:, 3:4]
        else:
            alphaT = torch.ones_like(g[:, 0:1])
            alphaS_raw = torch.ones_like(g[:, 1:2])
            betaT = torch.ones_like(g[:, 2:3])
            betaS = torch.ones_like(g[:, 3:4])

        if self.use_salinity_cap:
            alphaS = torch.clamp(alphaS_raw, 0.0, self.alpha_s_max)
        else:
            alphaS = alphaS_raw

        alpha4 = torch.cat([alphaT, alphaT, alphaS, alphaS], dim=1)
        beta4  = torch.cat([betaT, betaT, betaS, betaS], dim=1)

        q_prior = self._prior_flux(Xn)
        pot = self.pot_head(feat) + pot_bias
        q_struct, qd_struct, qr_struct = self._pot_to_flux_parts(pot)

        rr = self.res_body(feat)
        rr = self.res_conv(rr) + feat
        q_res = self.res_out(rr)

        if not self.use_prior:
            q_prior = torch.zeros_like(q_prior)
            alpha4 = torch.zeros_like(alpha4)

        if not self.use_struct:
            q_struct = torch.zeros_like(q_struct)
            qd_struct = torch.zeros_like(qd_struct)
            qr_struct = torch.zeros_like(qr_struct)
            beta4 = torch.zeros_like(beta4)

        if not self.use_res:
            q_res = torch.zeros_like(q_res)

        pred = alpha4 * q_prior + beta4 * q_struct + q_res

        if return_parts:
            q_diss = alpha4 * q_prior + beta4 * qd_struct
            q_rot  = beta4 * qr_struct
            return pred, q_diss, q_rot

        return pred


def build_scapex_ablation_model(
    ablation_tag: str,
    n_coarse: int,
    X_mean, X_std, Y_mean, Y_std,
    K_vec_n: np.ndarray,
):
    if n_coarse == 3:
        alpha_s_max = SCAPEX_ALPHA_S_MAX_N3
    else:
        alpha_s_max = SCAPEX_ALPHA_S_MAX_OTH

    cfg = dict(
        n_coarse=n_coarse,
        X_mean=X_mean, X_std=X_std,
        Y_mean=Y_mean, Y_std=Y_std,
        K_vec_n=K_vec_n,
        width=SCAPEX_WIDTH,
        n_blocks=SCAPEX_NBLOCKS,
        afno_depth=SCAPEX_AFNO_DEPTH,
        alpha_s_max=alpha_s_max,
        use_prior=True,
        use_struct=True,
        use_res=True,
        use_scale=True,
        use_adaptive_gate=True,
        use_salinity_cap=True,
    )

    if ablation_tag == "SCAPEX_WO_PRIOR":
        cfg["use_prior"] = False
    elif ablation_tag == "SCAPEX_WO_STRUCT":
        cfg["use_struct"] = False
    elif ablation_tag == "SCAPEX_WO_RES":
        cfg["use_res"] = False
    elif ablation_tag == "SCAPEX_WO_SCALE":
        cfg["use_scale"] = False
    elif ablation_tag == "SCAPEX_WO_ADAGATE":
        cfg["use_adaptive_gate"] = False
    elif ablation_tag == "SCAPEX_WO_SCAP":
        cfg["use_salinity_cap"] = False

    return SCAPEXNetAblation(**cfg)


# ============================================================
# Dataloader helper
# ============================================================
def _make_loader(ds, batch_size, shuffle, num_workers, pin_memory):
    kw = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers and num_workers > 0:
        kw["prefetch_factor"] = PREFETCH_FACTOR
        kw["persistent_workers"] = PERSISTENT_WORKERS
    return DataLoader(ds, **kw)


# ============================================================
# K fit (normalized-space prior)
# ============================================================
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

            dTdx_s = dTdx_n.reshape(-1)[sel]
            qTx_s = qTx_n.reshape(-1)[sel]
            dTdy_s = dTdy_n.reshape(-1)[sel]
            qTy_s = qTy_n.reshape(-1)[sel]
            dSdx_s = dSdx_n.reshape(-1)[sel]
            qSx_s = qSx_n.reshape(-1)[sel]
            dSdy_s = dSdy_n.reshape(-1)[sel]
            qSy_s = qSy_n.reshape(-1)[sel]

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
# Training
# ============================================================
def train_scapex_ablation(
    tag,
    model,
    train_loader, val_loader,
    n_epochs, ckpt_dir,
    lr=2e-4, clip=1.0,
    lam_diss=0.03,
    diss_warmup=2,
    lam_hf_T=0.05, lam_hf_S=0.0, use_hf=True,
    lam_lf=0.0, lf_k0=0.22, lf_p=1.0,
    lam_div=0.0,
):
    model = model.to(DEVICE)
    if USE_DP and (not isinstance(model, nn.DataParallel)):
        model = nn.DataParallel(model, device_ids=GPU_IDS)

    tracker = BestCkptTracker(ckpt_dir, tag, model, mode="min", metric_name="val(mse)")
    if tracker.try_load():
        return model

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = make_scaler()

    for ep in range(1, n_epochs + 1):
        model.train()
        tot = 0.0
        t0 = time.perf_counter()

        if ep <= diss_warmup:
            lam_diss_eff = 0.0
        else:
            lam_diss_eff = lam_diss * (ep - diss_warmup) / max(1, (n_epochs - diss_warmup))

        for xb, yb, mb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast_ctx():
                pred_n, qd_n, _ = model(xb, return_parts=True)
                base = _unwrap_model(model)

                loss = masked_mse_torch_with_mask(pred_n, yb, mb)

                if lam_diss_eff > 0:
                    Xphys = base.denorm_X(xb)
                    dTdx = Xphys[:,4]
                    dTdy = Xphys[:,5]
                    dSdx = Xphys[:,6]
                    dSdy = Xphys[:,7]

                    qd_phys = base.denorm_Y(qd_n)
                    qTx = qd_phys[:,0]
                    qTy = qd_phys[:,1]
                    qSx = qd_phys[:,2]
                    qSy = qd_phys[:,3]

                    m2 = mb & torch.isfinite(dTdx) & torch.isfinite(dTdy) & torch.isfinite(dSdx) & torch.isfinite(dSdy)
                    prodT = -(qTx*dTdx + qTy*dTdy)
                    prodS = -(qSx*dSdx + qSy*dSdy)
                    diss = _masked_mean_tensor(F.relu(-prodT), m2) + _masked_mean_tensor(F.relu(-prodS), m2)
                    loss = loss + lam_diss_eff * diss

                if use_hf and (lam_hf_T > 0.0 or lam_hf_S > 0.0):
                    if HF_LOSS_TYPE.lower() == "laplacian":
                        hp_pred = laplacian_filter(pred_n)
                        hp_true = laplacian_filter(torch.nan_to_num(yb, nan=0.0))
                    else:
                        hp_pred = dog_filter(pred_n, ksize=DOG_KSIZE, sigma1=DOG_SIGMA1, sigma2=DOG_SIGMA2)
                        hp_true = dog_filter(torch.nan_to_num(yb, nan=0.0), ksize=DOG_KSIZE, sigma1=DOG_SIGMA1, sigma2=DOG_SIGMA2)

                    m4 = mb.unsqueeze(1).expand_as(hp_pred) & torch.isfinite(yb)
                    diff = (hp_pred - hp_true).abs()
                    hfT = _masked_mean_tensor(diff[:,0:2], m4[:,0:2])
                    hfS = _masked_mean_tensor(diff[:,2:4], m4[:,2:4])
                    loss = loss + lam_hf_T * hfT + lam_hf_S * hfS

                if lam_lf > 0.0:
                    loss_lf = spectral_lf_loss_amp(pred_n, yb, mb, k0_frac=lf_k0, p=lf_p)
                    loss = loss + lam_lf * loss_lf

                if lam_div > 0.0:
                    loss_div = divergence_loss_pixel(pred_n, yb, mb)
                    loss = loss + lam_div * loss_div

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            scaler.step(opt)
            scaler.update()

            tot += loss.item() * xb.size(0)

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                mb = mb.to(DEVICE, non_blocking=True)
                with autocast_ctx():
                    pred = model(xb)
                    vtot += masked_mse_torch_with_mask(pred, yb, mb).item() * xb.size(0)

        val_mse = vtot / len(val_loader.dataset)
        tracker.update(val_mse, ep, extra_best={"lam_diss_eff": float(lam_diss_eff)})

        dt = time.perf_counter() - t0
        log(f"[{tag}] Epoch {ep:03d} | train={tot/len(train_loader.dataset):.4f} | val(mse)={val_mse:.4f} | lam_diss={lam_diss_eff:.3f} | time={dt:.1f}s")

    tracker.finalize(
        extra_last={
            "lr": float(lr),
            "lam_diss": float(lam_diss),
            "lam_hf_T": float(lam_hf_T),
            "lam_hf_S": float(lam_hf_S),
            "lam_lf": float(lam_lf),
            "lam_div": float(lam_div),
        },
        load_best=True
    )
    return model


# ============================================================
# Inference
# ============================================================
def predict_cnn_full_tiled(model: nn.Module, Xn: torch.Tensor,
                           Y_mean: np.ndarray, Y_std: np.ndarray,
                           tile: int = 192, overlap: int = 32) -> np.ndarray:
    assert Xn.dim() == 4 and Xn.size(0) == 1
    model.eval()
    B, C, H, W = Xn.shape
    stride = max(1, tile - overlap)

    device = Xn.device
    Y_mean_t = torch.tensor(Y_mean, device=device, dtype=torch.float32).view(1,4,1,1)
    Y_std_t  = torch.tensor(Y_std,  device=device, dtype=torch.float32).view(1,4,1,1)

    out_acc = torch.zeros((1,4,H,W), device=device, dtype=torch.float32)
    w_acc   = torch.zeros((1,1,H,W), device=device, dtype=torch.float32)

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
                    yp_n = model(x_tile)

                yp_n = yp_n.float()
                yp = yp_n * Y_std_t + Y_mean_t

                ht = je - js
                wt = ie - is_
                ww = ww_full_t[:, :, :ht, :wt]

                out_acc[:,:,js:je,is_:ie] += yp * ww
                w_acc[:,:,js:je,is_:ie]   += ww

    out = out_acc / (w_acc + 1e-12)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


# ============================================================
# Metrics
# ============================================================
@dataclass
class OnlineStats1D:
    n: int = 0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    sum_p: float = 0.0
    sum_p2: float = 0.0
    sum_yp: float = 0.0
    sse: float = 0.0

    def update(self, y: np.ndarray, p: np.ndarray):
        self.n += int(y.size)
        self.sum_y  += float(np.sum(y))
        self.sum_y2 += float(np.sum(y*y))
        self.sum_p  += float(np.sum(p))
        self.sum_p2 += float(np.sum(p*p))
        self.sum_yp += float(np.sum(y*p))
        e = p - y
        self.sse += float(np.sum(e*e))

    def finalize(self):
        if self.n < 2:
            return {"rmse": np.nan, "corr": np.nan, "r2": np.nan}
        var_y  = self.sum_y2 - self.sum_y*self.sum_y / self.n
        var_p  = self.sum_p2 - self.sum_p*self.sum_p / self.n
        cov    = self.sum_yp - self.sum_y*self.sum_p / self.n
        rmse = math.sqrt(self.sse / self.n)
        corr = cov / (math.sqrt(var_y*var_p) + 1e-12) if (var_y > 0 and var_p > 0) else np.nan
        sst  = var_y
        r2   = 1.0 - (self.sse / (sst + 1e-12)) if sst > 0 else np.nan
        return {"rmse": float(rmse), "corr": float(corr), "r2": float(r2)}


def evaluate_models_stream_ablation(
    models: Dict[str, nn.Module],
    val_loader_full: DataLoader,
    Y_mean: np.ndarray, Y_std: np.ndarray,
):
    methods = list(models.keys())
    stats = {m: [OnlineStats1D() for _ in range(4)] for m in methods}

    t0 = time.perf_counter()

    for batch in val_loader_full:
        Xn, Y_phys, ocean, file_id, lev, X_phys = batch
        Xn = Xn.to(DEVICE, non_blocking=True)

        ocean2d = ocean.numpy()[0]
        Yt = Y_phys.numpy()[0]

        m2d = ocean2d & np.isfinite(Yt[0]) & np.isfinite(Yt[1]) & np.isfinite(Yt[2]) & np.isfinite(Yt[3])
        if m2d.sum() == 0:
            continue

        yt_flat = Yt[:, m2d]

        for name, model in models.items():
            yp = predict_cnn_full_tiled(model, Xn, Y_mean, Y_std, tile=EVAL_TILE, overlap=EVAL_OVERLAP)[0]
            yp_flat = yp[:, m2d]

            for c in range(4):
                y = yt_flat[c].reshape(-1)
                p = yp_flat[c].reshape(-1)
                good = np.isfinite(y) & np.isfinite(p)
                if np.any(good):
                    stats[name][c].update(y[good], p[good])

    log(f"✅ Ablation eval done | time={time.perf_counter()-t0:.1f}s")

    all_metrics = {}
    for m in methods:
        mm = {}
        rmse_list, corr_list, r2_list = [], [], []
        for c, name in enumerate(flux_names):
            out = stats[m][c].finalize()
            mm[name] = out
            rmse_list.append(out["rmse"])
            corr_list.append(out["corr"])
            r2_list.append(out["r2"])
        mm["mean"] = {
            "rmse": float(np.nanmean(rmse_list)),
            "corr": float(np.nanmean(corr_list)),
            "r2":   float(np.nanmean(r2_list)),
        }
        all_metrics[m] = mm
    return all_metrics


# ============================================================
# Plot ablation figures
# ============================================================
def plot_ablation_summary_one_scale(metrics: Dict[str, Dict], fig_dir: str, n_coarse: int):
    os.makedirs(fig_dir, exist_ok=True)

    methods = list(metrics.keys())
    mean_r2 = [metrics[m]["mean"]["r2"] for m in methods]
    mean_rmse = [metrics[m]["mean"]["rmse"] for m in methods]
    colors = [ABLATION_COLORS.get(m, None) for m in methods]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axs[0].bar(methods, mean_r2, color=colors)
    axs[0].set_title(f"Ablation mean(4) R² | n={n_coarse}")
    axs[0].set_ylabel("R²")
    axs[0].tick_params(axis="x", rotation=30)

    axs[1].bar(methods, mean_rmse, color=colors)
    axs[1].set_title(f"Ablation mean(4) RMSE | n={n_coarse}")
    axs[1].set_ylabel("RMSE")
    axs[1].tick_params(axis="x", rotation=30)

    savefig_tight(fig, os.path.join(fig_dir, f"ablation_bar_mean_n{n_coarse}.png"))

    comps = ["qT_x", "qT_y", "qS_x", "qS_y"]
    R2_matrix = np.array([[metrics[m][c]["r2"] for c in comps] for m in methods], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    im = ax.imshow(R2_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(comps)))
    ax.set_xticklabels(comps)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(f"Ablation R² heatmap | n={n_coarse}")

    for i in range(len(methods)):
        for j in range(len(comps)):
            val = R2_matrix[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if np.isfinite(val) and val > 0.55 else "black",
                fontsize=9
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("R²")
    savefig_tight(fig, os.path.join(fig_dir, f"ablation_heatmap_R2_n{n_coarse}.png"))


def plot_ablation_across_scales(all_scale_metrics: Dict[int, Dict], result_root: str):
    scales = sorted(all_scale_metrics.keys())
    methods = list(next(iter(all_scale_metrics.values())).keys())

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for m in methods:
        y = [all_scale_metrics[n][m]["mean"]["r2"] for n in scales]
        ax.plot(scales, y, marker="o", label=m, color=ABLATION_COLORS.get(m, None))
    ax.set_xlabel("Coarse-graining factor n")
    ax.set_ylabel("mean(4) R²")
    ax.set_title("SCAPEX ablation across scales")
    ax.legend(frameon=True, ncol=2)
    savefig_tight(fig, os.path.join(result_root, "ablation_skill_vs_scale_R2.png"))


# ============================================================
# Main pipeline for one scale
# ============================================================
def run_one_scale_ablation(n_coarse: int, var_dir: str, label_dir: str, result_dir: str) -> Dict:
    global LOGGER
    LOGGER = setup_logger(result_dir, name=f"ablation_n{n_coarse}")

    log("\n========================================")
    log(f"✅ Run SCAPEX ablation | n={n_coarse}")
    log(f"VAR_DIR   = {var_dir}")
    log(f"LABEL_DIR = {label_dir}")
    log(f"RESULT_DIR= {result_dir}")
    log(f"DEVICE    = {DEVICE} | USE_AMP={USE_AMP}")
    log("========================================\n")

    fig_dir  = os.path.join(result_dir, "figures")
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    theta_files = sorted(glob.glob(os.path.join(var_dir, "thetao_Oday_FGOALS-f3-H_omip2_r1i1p1f1_gn_*.nc")))
    if not theta_files:
        raise RuntimeError(f"Cannot find thetao files: {var_dir}")

    items = build_day_level_index(theta_files, label_dir, max_days=MAX_DAYS)
    if not items:
        raise RuntimeError(f"❌ n={n_coarse} has no valid items, please check label_dir={label_dir}")

    file_ids = np.array([it["file_id"] for it in items], dtype=int)
    unique_files = np.sort(np.unique(file_ids))
    split_idx = int(0.7 * len(unique_files))
    train_files = set(unique_files[:split_idx])
    val_files   = set(unique_files[split_idx:])

    train_items = [it for it in items if it["file_id"] in train_files]
    val_items   = [it for it in items if it["file_id"] in val_files]

    log(f"Items: {len(items)} | Train items: {len(train_items)} | Val items: {len(val_items)}")
    log(f"Days: {len(unique_files)} | Train days: {len(train_files)} | Val days: {len(val_files)}")

    # normalization
    X_mean, X_std, Y_mean, Y_std = estimate_norm_from_train(train_items, n_samples=2_000_000, seed=0)

    # datasets / loaders
    train_ds = LazyLevPatchDataset(
        train_items, X_mean, X_std, Y_mean, Y_std,
        patch_size=PATCH_SIZE, seed=1, patches_per_item=TRAIN_PATCHES_PER_ITEM
    )
    val_ds = LazyLevPatchDataset(
        val_items, X_mean, X_std, Y_mean, Y_std,
        patch_size=PATCH_SIZE, seed=2, patches_per_item=1
    )

    train_loader = _make_loader(
        train_ds,
        batch_size=BATCH_SIZE_IMG_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=PIN_MEMORY,
    )
    val_loader = _make_loader(
        val_ds,
        batch_size=BATCH_SIZE_IMG_TRAIN,
        shuffle=False,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=PIN_MEMORY,
    )

    val_full_ds = LazyLevFullEvalDataset(val_items, X_mean, X_std, Y_mean, Y_std)
    val_full_loader = DataLoader(
        val_full_ds,
        batch_size=BATCH_SIZE_IMG_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS_EVAL,
        pin_memory=False,
    )

    K_vec_n = fit_prior_K_vec_normalized(train_items, X_mean, X_std, Y_mean, Y_std, seed=0)
    log(f"✅ normalized prior K_vec = {K_vec_n}")

    models = {}

    for tag in ABLATION_TAGS:
        log(f"\n----- Build / Train {tag} -----")
        model = build_scapex_ablation_model(
            ablation_tag=tag,
            n_coarse=n_coarse,
            X_mean=X_mean, X_std=X_std,
            Y_mean=Y_mean, Y_std=Y_std,
            K_vec_n=K_vec_n,
        )

        if n_coarse == 3:
            lam_div = SCAPEX_LAM_DIV_N3
            lam_lf = 0.0
        elif n_coarse >= 10:
            lam_div = 0.0
            lam_lf = SCAPEX_LAM_LF_N10
        else:
            lam_div = 0.0
            lam_lf = 0.0

        model = train_scapex_ablation(
            tag,
            model,
            train_loader, val_loader,
            N_EPOCHS_SCAPEX, ckpt_dir,
            lr=LR_SCAPEX, clip=CLIP_GRAD,
            lam_diss=SCAPEX_LAM_DISS, diss_warmup=SCAPEX_DISS_WARMUP,
            lam_hf_T=SCAPEX_LAM_HF_T, lam_hf_S=SCAPEX_LAM_HF_S, use_hf=SCAPEX_USE_HF,
            lam_lf=lam_lf, lf_k0=0.22, lf_p=1.0,
            lam_div=lam_div,
        )
        models[tag] = model

    metrics = evaluate_models_stream_ablation(
        models=models,
        val_loader_full=val_full_loader,
        Y_mean=Y_mean, Y_std=Y_std,
    )

    plot_ablation_summary_one_scale(metrics, fig_dir, n_coarse)

    out_json = os.path.join(result_dir, f"ablation_metrics_n{n_coarse}.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved: {out_json}")

    rows = metrics_to_rows(metrics, n=n_coarse, add_n=True)
    out_csv = os.path.join(result_dir, f"ablation_metrics_n{n_coarse}.csv")
    write_rows_csv(out_csv, rows, fieldnames=["n", "method", "component", "rmse", "corr", "r2"])
    log(f"Saved: {out_csv}")

    try:
        train_ds.close_cache()
        val_ds.close_cache()
    except Exception:
        pass

    return metrics


# ============================================================
# Main
# ============================================================
def resolve_label_dir_for_n(label_root: str, n: int) -> str:
    cand = os.path.join(label_root, f"label_{n}")
    return cand if os.path.isdir(cand) else label_root


def main():
    os.makedirs(RESULT_ROOT, exist_ok=True)

    all_scale_metrics: Dict[int, Dict] = {}
    all_rows = []

    for n in COARSE_LIST:
        label_dir = resolve_label_dir_for_n(LABEL_ROOT, n)
        result_dir = os.path.join(RESULT_ROOT, f"n{n}")
        os.makedirs(result_dir, exist_ok=True)

        metrics = run_one_scale_ablation(n, VAR_DIR, label_dir, result_dir)
        all_scale_metrics[n] = metrics
        all_rows.extend(metrics_to_rows(metrics, n=n, add_n=True))

        out_csv = os.path.join(RESULT_ROOT, "scapex_ablation_all_scales.csv")
        write_rows_csv(out_csv, all_rows, fieldnames=["n", "method", "component", "rmse", "corr", "r2"])
        print("Saved (partial):", out_csv)

    plot_ablation_across_scales(all_scale_metrics, RESULT_ROOT)

    out_csv = os.path.join(RESULT_ROOT, "scapex_ablation_all_scales.csv")
    write_rows_csv(out_csv, all_rows, fieldnames=["n", "method", "component", "rmse", "corr", "r2"])
    print("Saved (final):", out_csv)

    out_json = os.path.join(RESULT_ROOT, "scapex_ablation_all_scales.json")
    with open(out_json, "w") as f:
        json.dump(all_scale_metrics, f, indent=2)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
