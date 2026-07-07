#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import math
import time
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


# =========================
# 全局配置（按需改）
# =========================
VAR_DIR     = "/home/Data/zhufuhua/MyData/para2/data/global"
LABEL_ROOT  = "/home/Data/zhufuhua/MyData/para2/data/global/thelabel"
RESULT_ROOT = "/home/Data/zhufuhua/MyData/para2/data/global/theresult"

COARSE_LIST = [3, 5, 10]
MAX_DAYS    = 10

PATCH_SIZE  = 96
TRAIN_PATCHES_PER_ITEM = 4

# batch
BATCH_SIZE_IMG_TRAIN = 4
BATCH_SIZE_IMG_EVAL  = 1
BATCH_SIZE_MLP       = 1024

NUM_WORKERS = 8
PIN_MEMORY  = True
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

# 训练超参
LR          = 1e-3
LR_RCAN     = 1e-4
LR_EDSR_FIX = 1e-4

LR_RESTORMER = 1e-4
LR_AFNO      = 1e-3
LR_CNO       = 1e-3
# ===== 新模型 SCHOPNet 超参 =====
LR_SCHOPNET       = 2e-4
N_EPOCHS_SCHOPNET = 25

SCHOP_WIDTH       = 64
SCHOP_NBLOCKS     = 4
SCHOP_AFNO_DEPTH  = 2   # bottleneck 里 AFNOBlock 个数

# Physics/regularization
SCHOP_LAM_DISS    = 0.03   # 耗散 hinge 权重（建议 0.01~0.05）
SCHOP_USE_HF      = True
SCHOP_LAM_HF_T    = 0.05   # HF 对温度通量权重
SCHOP_LAM_HF_S    = 0.00   # HF 对盐通量权重（默认关，避免 n=3 qS_x 崩）

# ===== SCHOPNet 变体：只对 SCHOP 系列生效 =====
SCHOP_PATCH_N10 = 192     # 只给 SCHOP 在 n=10 用更大 patch（其他模型仍用 PATCH_SIZE=96）
SCHOP_DISS_WARMUP = 3     # diss hinge 前 warmup 轮数（避免早期锁死）

# low-frequency spectral loss（n=10 重点）
SCHOP_LAM_LF   = 0.06     # 0.03~0.10 都可试
SCHOP_LF_K0    = 0.22     # 低频阈值（越小越“更低频”）
SCHOP_LF_P     = 1.0

# divergence loss（像素差分版；和 true 同算子，主要对齐结构）
SCHOP_LAM_DIV  = 0.04     # 0.02~0.08 可试

# Deep pyramid variant
SCHOP_DEEP_AFNO_DEPTH = 3  # deep 模型 bottleneck AFNOBlock 数

# ===== 新模型 SCAPEXNet 超参 =====
LR_SCAPEX        = 2e-4
N_EPOCHS_SCAPEX  = 50

SCAPEX_WIDTH      = 64
SCAPEX_NBLOCKS    = 4
SCAPEX_AFNO_DEPTH = 2

SCAPEX_LAM_DISS    = 0.03
SCAPEX_DISS_WARMUP = 2

# HF：温度开，盐度默认弱/关（避免 n=3 qS 爆）
SCAPEX_LAM_HF_T    = 0.05
SCAPEX_LAM_HF_S    = 0.00
SCAPEX_USE_HF      = True

# 分尺度建议：n=10 低频更关键；n=3 可加一点 div 稳盐度结构
SCAPEX_LAM_LF_N10  = 0.06
SCAPEX_LAM_DIV_N3  = 0.04

# 盐度先验门控上限（关键：n=3 防崩）
SCAPEX_ALPHA_S_MAX_N3  = 0.55
SCAPEX_ALPHA_S_MAX_OTH = 0.90


N_EPOCHS_MLP     = 25
N_EPOCHS_CNN     = 25
N_EPOCHS_UNET    = 25
N_EPOCHS_RESCNN  = 25
N_EPOCHS_FNO     = 25
N_EPOCHS_EDSRFIX = 25
N_EPOCHS_RCAN    = 25
N_EPOCHS_ATTUNET = 25

N_EPOCHS_RESTORMER = 25
N_EPOCHS_AFNO      = 25
N_EPOCHS_CNO       = 25

CLIP_GRAD = 1.0

RETRAIN_ALL  = False
RETRAIN_TAGS = set([])   # e.g. {"RCAN_HF"}

# RCAN 高频增强 loss
HF_LOSS_TYPE = "laplacian"  # "laplacian" or "dog"
HF_LAM       = 0.15
HF_RAMP      = True

DOG_KSIZE  = 7
DOG_SIGMA1 = 1.0
DOG_SIGMA2 = 2.0

ENABLE_SPEC_LOSS = False
SPEC_LAM         = 0.02
SPEC_P           = 2.0
SPEC_K0_FRAC     = 0.35

# MoE / FrontResidual
MOE_GATE_Q      = 0.90
HF_FRONT_BOOST  = 1.0

FR_GATE_Q     = 0.85
FR_GATE_TEMP  = 0.20
FR_ALPHA_INIT = 0.10

# 设备与随机种子
from contextlib import nullcontext

# ===== Multi-GPU config =====
CUDA_AVAILABLE = torch.cuda.is_available()
N_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
GPU_IDS = list(range(N_GPUS))  # 默认用所有可见GPU，如需固定可写 [0,1]
USE_DP = CUDA_AVAILABLE and (N_GPUS >= 2)

DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
DEVICE_IS_CUDA = str(DEVICE).startswith("cuda")
if USE_DP:
    print(f"[MultiGPU] Using DataParallel on GPUs: {GPU_IDS}")
else:
    print(f"[MultiGPU] Single device: {DEVICE}")

# ===== AMP helpers (new API, fix FutureWarning) =====
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

# AMP
USE_AMP = True

# Eval tiled 推理（防爆显存）
EVAL_TILE = 192       # 0 表示不 tiled
EVAL_OVERLAP = 32
TILED_METHODS = {
    "CNN", "UNet", "ResCNN", "EDSR_fix",
    "RCAN", "RCAN_HF", "RCAN_HF_MOE", "RCAN_HF_FR",
    "AttUNet",
    "CNO",
    "Restormer",
    "SCHOPNet",
    "SCHOPNet_BP",
    "SCHOPNet_LF",
    "SCHOPNet_LFDiv",
    "SCAPEX",

}

flux_names = ["qT_x", "qT_y", "qS_x", "qS_y"]

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

LOGGER: Optional[logging.Logger] = None


# =========================
# 日志
# =========================
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


# =========================
# xarray open 优化配置
# =========================
XR_OPEN_KW = dict(
    decode_cf=False,
    mask_and_scale=False,
    decode_times=False,
    cache=True,
)


# =========================
# 工具：保存图
# =========================
def savefig_tight(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log(f"Saved: {path}")


# =========================
# Checkpoint helpers
# =========================
def ckpt_path(ckpt_dir: str, tag: str):
    safe = tag.replace(" ", "_").replace("/", "_")
    return os.path.join(ckpt_dir, f"{safe}.pt")


def should_retrain(tag: str) -> bool:
    base = tag[:-5] if tag.endswith("_best") else tag
    return RETRAIN_ALL or (base in RETRAIN_TAGS)



def _unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, (nn.DataParallel,)) else m

def _strip_module_prefix(state: dict) -> dict:
    # 把 'module.xxx' -> 'xxx'
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out

def _add_module_prefix(state: dict) -> dict:
    # 把 'xxx' -> 'module.xxx'
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k] = v
        else:
            out["module." + k] = v
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
    if (not should_retrain(tag)) and os.path.exists(path):
        obj = torch.load(path, map_location="cpu")
        sd = obj["state_dict"]

        target = _unwrap_model(model)
        try:
            target.load_state_dict(sd, strict=True)
        except RuntimeError:
            # 处理 module. 前缀不一致
            try:
                target.load_state_dict(_strip_module_prefix(sd), strict=True)
            except RuntimeError:
                # 极端：反过来需要加 module.
                target.load_state_dict(_add_module_prefix(sd), strict=True)

        log(f"Loaded ckpt: {path}")
        return True
    return False

def load_ckpt_best_first(ckpt_dir: str, tag: str, model: nn.Module) -> bool:
    # 优先加载 best
    if load_ckpt(ckpt_dir, tag + "_best", model):
        return True
    # 其次加载 last
    if load_ckpt(ckpt_dir, tag, model):
        return True
    return False

COMPONENTS = ["qT_x", "qT_y", "qS_x", "qS_y", "mean(4)"]

def metrics_to_rows(all_metrics, n=None, add_n=False):
    """
    all_metrics[method] 结构假设：
      all_metrics[method]["qT_x"] = {"rmse":..,"corr":..,"r2":..}
      ...
      all_metrics[method]["mean"] = {"rmse":..,"corr":..,"r2":..}
    返回：list[dict] 每个 dict 对应 CSV 一行
    """
    rows = []
    for method, md in all_metrics.items():
        # 4 个分量
        for comp in ["qT_x", "qT_y", "qS_x", "qS_y"]:
            if comp not in md:
                continue
            d = md[comp]
            row = {
                "method": method,
                "component": comp,
                "rmse": d.get("rmse", np.nan),
                "corr": d.get("corr", np.nan),
                "r2":   d.get("r2",   np.nan),
            }
            if add_n:
                row["n"] = n
            rows.append(row)

        # mean(4)
        if "mean" in md:
            d = md["mean"]
            row = {
                "method": method,
                "component": "mean(4)",
                "rmse": d.get("rmse", np.nan),
                "corr": d.get("corr", np.nan),
                "r2":   d.get("r2",   np.nan),
            }
            if add_n:
                row["n"] = n
            rows.append(row)

    return rows
def write_rows_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

class BestCkptTracker:
    """
    统一 best/last checkpoint 管理：
    - try_load(): 优先加载 tag_best，其次 tag
    - update(metric): 若更好则保存 tag_best
    - finalize(): 保存 tag(last)，并把 best load 回模型（保证后续评估用 best）
    """
    def __init__(
        self,
        ckpt_dir: str,
        tag: str,
        model: nn.Module,
        mode: str = "min",          # "min" 用于 loss；"max" 用于 acc/r2
        metric_name: str = "val",   # 打日志用
        eps: float = 1e-12,
    ):
        assert mode in ("min", "max")
        self.ckpt_dir = ckpt_dir
        self.tag = tag
        self.model = model
        self.mode = mode
        self.metric_name = metric_name
        self.eps = float(eps)

        self.best_metric = float("inf") if mode == "min" else -float("inf")
        self.best_ep = 0
        self.last_metric = None

    def try_load(self) -> bool:
        # load_ckpt 内部已处理 should_retrain
        return load_ckpt_best_first(self.ckpt_dir, self.tag, self.model)

    def _is_better(self, x: float) -> bool:
        if not np.isfinite(x):
            return False
        if self.mode == "min":
            return x < (self.best_metric - self.eps)
        else:
            return x > (self.best_metric + self.eps)

    def update(self, metric_value: float, epoch: int, extra_best: Optional[Dict[str, Any]] = None) -> bool:
        self.last_metric = float(metric_value)
        if self._is_better(self.last_metric):
            self.best_metric = float(self.last_metric)
            self.best_ep = int(epoch)
            extra = {} if extra_best is None else dict(extra_best)
            extra.update({"best_metric": self.best_metric, "best_ep": self.best_ep, "metric_name": self.metric_name})
            save_ckpt(self.ckpt_dir, self.tag + "_best", self.model, extra=extra)
            log(f"[{self.tag}] ✅ New best {self.metric_name} | ep={self.best_ep} {self.best_metric:.6f}")
            return True
        return False

    def finalize(self, extra_last: Optional[Dict[str, Any]] = None, load_best: bool = True):
        # 不保存 last（tag.pt）
        best_path = ckpt_path(self.ckpt_dir, self.tag + "_best")

        # 如果训练中从没保存过 best，就在这里保存一次（兜底）
        if not os.path.exists(best_path):
            extra = {} if extra_last is None else dict(extra_last)
            extra.update({
                "best_metric": float(self.best_metric),
                "best_ep": int(self.best_ep),
                "metric_name": self.metric_name,
            })
            save_ckpt(self.ckpt_dir, self.tag + "_best", self.model, extra=extra)

        # 评估用 best 权重
        if load_best and os.path.exists(best_path):
            load_ckpt(self.ckpt_dir, self.tag + "_best", self.model)


# =========================
# 维名写死 + 断言
# =========================
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
        raise RuntimeError("label 文件缺少 dx/dy")
    dx_da = ds["dx"]; dy_da = ds["dy"]
    assert_2d_j_i(dx_da, "dx"); assert_2d_j_i(dy_da, "dy")
    return dx_da.values.astype(np.float32, copy=False), dy_da.values.astype(np.float32, copy=False)


# =========================
# numpy 版物理梯度（中心差分+边界单侧）
# =========================
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


# =========================
# 建索引：只存 (day, lev) 元信息
# =========================
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
                    raise RuntimeError(f"label 缺少变量 {k}: {label_path}")

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


# =========================
# 归一化参数：在线采样估计 mean/std（修正计数）
# =========================
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

            u = get2d_4dvar(ds, "uo_lr", lev); v = get2d_4dvar(ds, "vo_lr", lev)
            t = get2d_4dvar(ds, "theta_lr", lev); s = get2d_4dvar(ds, "so_lr", lev)
            qTx = get2d_4dvar(ds, "qT_x", lev); qTy = get2d_4dvar(ds, "qT_y", lev)
            qSx = get2d_4dvar(ds, "qS_x", lev); qSy = get2d_4dvar(ds, "qS_y", lev)

            dx, dy = get_dxdy(ds)
            dTdx, dTdy = phys_grad_2d(t, dx, dy)
            dSdx, dSdy = phys_grad_2d(s, dx, dy)

            X = np.stack([u, v, t, s, dTdx, dTdy, dSdx, dSdy], axis=-1)  # [H,W,8]
            Y = np.stack([qTx, qTy, qSx, qSy], axis=-1)                  # [H,W,4]

            valid = np.isfinite(X).all(axis=-1) & np.isfinite(Y).all(axis=-1)
            idx = np.where(valid.reshape(-1))[0]
            if idx.size == 0:
                continue

            take = min(per_item, idx.size)
            sel = rng.choice(idx, size=take, replace=False)
            Xs = X.reshape(-1, 8)[sel].astype(np.float64, copy=False)
            Ys = Y.reshape(-1, 4)[sel].astype(np.float64, copy=False)

            Xs64 = Xs.astype(np.float64, copy=False)
            Ys64 = Ys.astype(np.float64, copy=False)
            nb = int(Xs64.shape[0])
            if nb == 0:
                continue

            bx = Xs64.mean(axis=0)
            by = Ys64.mean(axis=0)

            # 更快更省内存的 batch M2
            sumsqX = (Xs64 * Xs64).sum(axis=0)
            sumsqY = (Ys64 * Ys64).sum(axis=0)
            bM2x = sumsqX - nb * (bx * bx)
            bM2y = sumsqY - nb * (by * by)

            if cnt == 0:
                meanX = bx;
                m2X = bM2x
                meanY = by;
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
    X_std  = (np.sqrt(varX).astype(np.float32) + 1e-8)
    Y_mean = meanY.astype(np.float32)
    Y_std  = (np.sqrt(varY).astype(np.float32) + 1e-8)

    log(f"✅ Norm estimated: cnt={cnt:,} | time={time.perf_counter()-t0:.1f}s")
    return X_mean, X_std, Y_mean, Y_std


# =========================
# Dataset cache mixin
# =========================
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


# =========================
# 训练点样本：MLP
# =========================
class BufferedPointDataset(Dataset, _XrCacheMixin):
    """
    关键：每次 refill 从一张图里抽很多点（points_per_refill），
    只算一次梯度/valid mask，然后 __getitem__ 超快。
    """
    def __init__(self, items, X_mean, X_std, Y_mean, Y_std,
                 length=500_000, seed=0, points_per_refill=20000):
        Dataset.__init__(self)
        _XrCacheMixin.__init__(self)
        self.items = items
        self.X_mean = X_mean.astype(np.float32)
        self.X_std  = X_std.astype(np.float32)
        self.Y_mean = Y_mean.astype(np.float32)
        self.Y_std  = Y_std.astype(np.float32)
        self.length = int(length)
        self.rng = np.random.default_rng(seed)
        self.points_per_refill = int(points_per_refill)

        self._bufX = np.empty((0, 8), np.float32)
        self._bufY = np.empty((0, 4), np.float32)
        self._pos = 0

    def __len__(self):
        return self.length

    def _refill(self):
        # 随机选一张图（day,lev）
        it = self.items[self.rng.integers(0, len(self.items))]
        ds = self._open_cached(it["label_path"])
        lev = it["lev"]

        # 读整图（这是必要的，但现在每次 refill 才做一次）
        u = get2d_4dvar(ds, "uo_lr", lev); v = get2d_4dvar(ds, "vo_lr", lev)
        t = get2d_4dvar(ds, "theta_lr", lev); s = get2d_4dvar(ds, "so_lr", lev)
        qTx = get2d_4dvar(ds, "qT_x", lev); qTy = get2d_4dvar(ds, "qT_y", lev)
        qSx = get2d_4dvar(ds, "qS_x", lev); qSy = get2d_4dvar(ds, "qS_y", lev)

        dx, dy = get_dxdy(ds)
        dTdx, dTdy = phys_grad_2d(t, dx, dy)
        dSdx, dSdy = phys_grad_2d(s, dx, dy)

        X = np.stack([u, v, t, s, dTdx, dTdy, dSdx, dSdy], axis=-1)  # [H,W,8]
        Y = np.stack([qTx, qTy, qSx, qSy], axis=-1)                  # [H,W,4]

        valid = np.isfinite(X).all(axis=-1) & np.isfinite(Y).all(axis=-1)
        idx = np.where(valid.reshape(-1))[0]
        if idx.size == 0:
            # 极端情况：没点可抽
            self._bufX = np.zeros((1,8), np.float32)
            self._bufY = np.zeros((1,4), np.float32)
            self._pos = 0
            return

        take = min(self.points_per_refill, idx.size)
        sel = self.rng.choice(idx, size=take, replace=False)

        Xs = X.reshape(-1, 8)[sel].astype(np.float32, copy=False)
        Ys = Y.reshape(-1, 4)[sel].astype(np.float32, copy=False)

        # normalize（点级别）
        Xs = (Xs - self.X_mean[None, :]) / self.X_std[None, :]
        Ys = (Ys - self.Y_mean[None, :]) / self.Y_std[None, :]

        self._bufX = Xs
        self._bufY = Ys
        self._pos = 0

    def __getitem__(self, idx):
        if self._pos >= self._bufX.shape[0]:
            self._refill()
        x = self._bufX[self._pos]
        y = self._bufY[self._pos]
        self._pos += 1
        return torch.from_numpy(x), torch.from_numpy(y)

# =========================
# 训练图样本：Lazy patch
# =========================
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
        self._shape_cache = {}  # key: label_path -> (H,W)
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

        # 先读一张变量来拿 H,W（只取 shape，不要整图 values）

        # H, W = int(da0.sizes["j"]), int(da0.sizes["i"])
        H, W = self._get_hw(ds, it["label_path"], lev)
        # 让 patch 留出 halo，确保梯度中心差分可用
        if (H < ps + 2*h) or (W < ps + 2*h):
            raise RuntimeError(f"Patch too large for halo: H={H}, W={W}, ps={ps}, halo={h}")

        j0 = self.rng.integers(h, H - ps - h + 1)
        i0 = self.rng.integers(h, W - ps - h + 1)

        sl_j  = slice(j0,     j0 + ps)
        sl_i  = slice(i0,     i0 + ps)
        sl_jh = slice(j0 - h, j0 + ps + h)
        sl_ih = slice(i0 - h, i0 + ps + h)

        # --- 1) 只读 patch：u,v,q ---
        u   = ds["uo_lr"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)
        v   = ds["vo_lr"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)

        qTx = ds["qT_x"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)
        qTy = ds["qT_y"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)
        qSx = ds["qS_x"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)
        qSy = ds["qS_y"].isel(time=0, lev=lev, j=sl_j, i=sl_i).values.astype(np.float32, copy=False)

        # --- 2) 温盐 & dxdy 读 halo 小块，用于算梯度 ---
        t_h = ds["theta_lr"].isel(time=0, lev=lev, j=sl_jh, i=sl_ih).values.astype(np.float32, copy=False)
        s_h = ds["so_lr"].isel(time=0, lev=lev, j=sl_jh, i=sl_ih).values.astype(np.float32, copy=False)

        dx_h = ds["dx"].isel(j=sl_jh, i=sl_ih).values.astype(np.float32, copy=False)
        dy_h = ds["dy"].isel(j=sl_jh, i=sl_ih).values.astype(np.float32, copy=False)
        #
        # dTdx_h, dTdy_h = phys_grad_2d(t_h, dx_h, dy_h)
        # dSdx_h, dSdy_h = phys_grad_2d(s_h, dx_h, dy_h)
        dTdx, dTdy = phys_grad_core_center(t_h, dx_h, dy_h, h, ps)
        dSdx, dSdy = phys_grad_core_center(s_h, dx_h, dy_h, h, ps)

        # t,s 也可以直接 core 切
        core = slice(h, h + ps)
        t = t_h[core, core];
        s = s_h[core, core]

        X = np.stack([u, v, t, s, dTdx, dTdy, dSdx, dSdy], axis=0).astype(np.float32, copy=False)
        Y = np.stack([qTx, qTy, qSx, qSy], axis=0).astype(np.float32, copy=False)

        ocean = np.isfinite(Y[0])

        Xn = (X - self.X_mean) / self.X_std
        Xn = np.nan_to_num(Xn, nan=0.0)

        Yn = (Y - self.Y_mean) / self.Y_std
        Yn[:, ~ocean] = np.nan

        return torch.from_numpy(Xn), torch.from_numpy(Yn), torch.from_numpy(ocean.astype(np.bool_))

# =========================
# 评估图样本：Lazy full map
# =========================
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

            u = get2d_4dvar(ds, "uo_lr", lev); v = get2d_4dvar(ds, "vo_lr", lev)
            t = get2d_4dvar(ds, "theta_lr", lev); s = get2d_4dvar(ds, "so_lr", lev)
            qTx = get2d_4dvar(ds, "qT_x", lev); qTy = get2d_4dvar(ds, "qT_y", lev)
            qSx = get2d_4dvar(ds, "qS_x", lev); qSy = get2d_4dvar(ds, "qS_y", lev)

            dx, dy = get_dxdy(ds)
            dTdx, dTdy = phys_grad_2d(t, dx, dy)
            dSdx, dSdy = phys_grad_2d(s, dx, dy)

            X_phys = np.stack([u, v, t, s, dTdx, dTdy, dSdx, dSdy], axis=0).astype(np.float32, copy=False)  # [8,H,W]
            Y_phys = np.stack([qTx, qTy, qSx, qSy], axis=0).astype(np.float32, copy=False)                  # [4,H,W]

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


# =========================
# Loss helpers
# =========================
def masked_mse_torch_with_mask(pred, target, mask2d):
    m = mask2d.unsqueeze(1).expand_as(pred) & torch.isfinite(target)
    diff = pred[m] - target[m]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (diff ** 2).mean()


def masked_l1_torch_with_mask(a, b, mask2d):
    m = mask2d.unsqueeze(1).expand_as(a) & torch.isfinite(b) & torch.isfinite(a)
    diff = (a[m] - b[m]).abs()
    if diff.numel() == 0:
        return torch.tensor(0.0, device=a.device)
    return diff.mean()


def masked_l1_weighted_torch_with_mask(a, b, mask2d, w2d):
    # w2d: [B,H,W] or [B,1,H,W]
    if w2d.dim() == 3:
        w = w2d.unsqueeze(1)
    else:
        w = w2d
    m = mask2d.unsqueeze(1).expand_as(a) & torch.isfinite(b) & torch.isfinite(a)
    diff = (a - b).abs()
    if m.sum() == 0:
        return torch.tensor(0.0, device=a.device)
    return (w.expand_as(a)[m] * diff[m]).mean()


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


def make_freq_weight(H, W, p=2.0, k0_frac=0.35, device="cpu", dtype=torch.float32):
    ky = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype).view(H,1)
    kx = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=dtype).view(1, W//2+1)
    kk = torch.sqrt(ky*ky + kx*kx)
    kk = kk / (kk.max() + 1e-12)
    gate = (kk >= k0_frac).float()
    w = gate * (kk ** p)
    return w

def _rfft2_safe(x: torch.Tensor) -> torch.Tensor:
    # cuFFT 对 half/bf16 的支持不稳定；统一转 float32 做 FFT
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    return torch.fft.rfft2(x, norm="ortho")

def spectral_hf_loss(pred, target, ocean_mask2d, wfreq):
    B, C, H, W = pred.shape
    m = ocean_mask2d.unsqueeze(1).expand_as(pred).float()

    pred_m = pred * m
    targ_m = torch.nan_to_num(target, nan=0.0) * m

    Fp = _rfft2_safe(pred_m)
    Ft = _rfft2_safe(targ_m)
    Ap = torch.abs(Fp)
    At = torch.abs(Ft)

    # wfreq 建议用 float32 创建，这里再对齐 dtype
    w = wfreq.to(Ap.dtype).view(1,1,H,W//2+1)
    return (w * (Ap - At).pow(2)).mean()


def make_freq_weight_low(H, W, k0_frac=0.22, p=1.0, device="cpu", dtype=torch.float32):
    ky = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype).view(H, 1)
    kx = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=dtype).view(1, W//2 + 1)
    kk = torch.sqrt(ky*ky + kx*kx)
    kk = kk / (kk.max() + 1e-12)
    gate = (kk <= k0_frac).float()
    w = gate * ((1.0 - kk) ** p)
    return w

def spectral_lf_loss_amp(pred, target, ocean_mask2d, k0_frac=0.22, p=1.0):
    """
    pred/target: [B,C,H,W]（建议 normalized flux）
    只惩罚低频（kk<=k0_frac）的幅值差
    """
    B, C, H, W = pred.shape
    m = ocean_mask2d.unsqueeze(1).expand_as(pred).float()

    pred_m = pred * m
    targ_m = torch.nan_to_num(target, nan=0.0) * m

    Fp = _rfft2_safe(pred_m)
    Ft = _rfft2_safe(targ_m)
    Ap = torch.abs(Fp)
    At = torch.abs(Ft)

    # 低频权重：固定用 float32 构造，再转成 Ap.dtype
    w = make_freq_weight_low(H, W, k0_frac=k0_frac, p=p, device=pred.device, dtype=torch.float32)
    w = w.to(Ap.dtype).view(1,1,H,W//2+1)

    return (w * (Ap - At).pow(2)).mean()


def _div2d_pixel(qx, qy):
    # qx,qy: [B,1,H,W]  -> div: [B,1,H,W]
    kx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=qx.dtype, device=qx.device).view(1,1,1,3)
    ky = torch.tensor([[-0.5, 0.0, 0.5]], dtype=qy.dtype, device=qy.device).view(1,1,3,1)
    dqx_dx = F.conv2d(qx, kx, padding=(0,1))
    dqy_dy = F.conv2d(qy, ky, padding=(1,0))
    return dqx_dx + dqy_dy

def divergence_loss_pixel(pred, target, ocean_mask2d):
    """
    对齐 div(q) 结构（像素差分），对粗尺度 n=10 很有用
    """
    # target 可能有 nan（陆地），我们用 ocean mask 控制
    qTx_p = pred[:,0:1]; qTy_p = pred[:,1:2]
    qSx_p = pred[:,2:3]; qSy_p = pred[:,3:4]

    qTx_t = torch.nan_to_num(target[:,0:1], nan=0.0); qTy_t = torch.nan_to_num(target[:,1:2], nan=0.0)
    qSx_t = torch.nan_to_num(target[:,2:3], nan=0.0); qSy_t = torch.nan_to_num(target[:,3:4], nan=0.0)

    divT_p = _div2d_pixel(qTx_p, qTy_p)
    divT_t = _div2d_pixel(qTx_t, qTy_t)
    divS_p = _div2d_pixel(qSx_p, qSy_p)
    divS_t = _div2d_pixel(qSx_t, qSy_t)

    div_p = torch.cat([divT_p, divS_p], dim=1)
    div_t = torch.cat([divT_t, divS_t], dim=1)
    return masked_mse_torch_with_mask(div_p, div_t, ocean_mask2d)


# =========================
# Baseline：K 拟合（流式） - ✅ 负号修正
# =========================
def fit_scalar_K_stream(train_items: List[dict], max_samples_per_item: int = 200_000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    numT = 0.0
    denT = 0.0
    numS = 0.0
    denS = 0.0

    t0 = time.perf_counter()
    for it in train_items:
        ds = xr.open_dataset(it["label_path"], **XR_OPEN_KW)
        try:
            lev = it["lev"]

            t = get2d_4dvar(ds, "theta_lr", lev); s = get2d_4dvar(ds, "so_lr", lev)
            qTx = get2d_4dvar(ds, "qT_x", lev); qTy = get2d_4dvar(ds, "qT_y", lev)
            qSx = get2d_4dvar(ds, "qS_x", lev); qSy = get2d_4dvar(ds, "qS_y", lev)

            dx, dy = get_dxdy(ds)
            dTdx, dTdy = phys_grad_2d(t, dx, dy)
            dSdx, dSdy = phys_grad_2d(s, dx, dy)

            valid = np.isfinite(qTx) & np.isfinite(qTy) & np.isfinite(qSx) & np.isfinite(qSy) & \
                    np.isfinite(dTdx) & np.isfinite(dTdy) & np.isfinite(dSdx) & np.isfinite(dSdy)
            idx = np.where(valid.reshape(-1))[0]
            if idx.size == 0:
                continue

            take = min(idx.size, max_samples_per_item)
            sel = rng.choice(idx, size=take, replace=False)

            dT_all = np.concatenate([dTdx.reshape(-1)[sel], dTdy.reshape(-1)[sel]])
            qT_all = np.concatenate([qTx.reshape(-1)[sel],  qTy.reshape(-1)[sel]])
            dS_all = np.concatenate([dSdx.reshape(-1)[sel], dSdy.reshape(-1)[sel]])
            qS_all = np.concatenate([qSx.reshape(-1)[sel],  qSy.reshape(-1)[sel]])

            numT += float(np.sum(dT_all * qT_all))
            denT += float(np.sum(dT_all * dT_all) + 1e-12)
            numS += float(np.sum(dS_all * qS_all))
            denS += float(np.sum(dS_all * dS_all) + 1e-12)
        finally:
            ds.close()

    # ✅ 负号修正：q ≈ -K grad  ->  K = -num/den
    K_T = - numT / denT if denT > 0 else 0.0
    K_S = - numS / denS if denS > 0 else 0.0
    log(f"✅ fit_scalar_K_stream done | time={time.perf_counter()-t0:.1f}s")
    return float(K_T), float(K_S)


# =========================
# Models
# =========================
class MLPFlux(nn.Module):
    def __init__(self, in_dim=8, out_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)


class CNNFlux(nn.Module):
    def __init__(self, in_channels=8, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
        )
    def forward(self, x): return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        out = self.conv1(x); out = self.relu(out); out = self.conv2(out)
        return self.relu(out + x)


class ResCNNFlux(nn.Module):
    """
    带 K 先验项的 ResCNN：q ≈ -K grad + delta
    注意：这里的输入/输出都是“归一化空间”的量，因此 K_vec 也应在归一化空间拟合。
    """
    def __init__(self, in_channels=8, out_channels=4, K_Tx=0.0, K_Ty=0.0, K_Sx=0.0, K_Sy=0.0):
        super().__init__()
        K_vec = torch.tensor([K_Tx, K_Ty, K_Sx, K_Sy], dtype=torch.float32).view(4, 1, 1)
        self.register_buffer("K_vec", K_vec)
        self.alpha = nn.Parameter(torch.ones(4, 1, 1))
        self.head = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.rb1 = ResBlock(64)
        self.rb2 = ResBlock(64)
        self.tail = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x):
        dTdx = x[:, 4:5]; dTdy = x[:, 5:6]
        dSdx = x[:, 6:7]; dSdy = x[:, 7:8]
        q_K = torch.cat([
            -self.K_vec[0:1]*dTdx, -self.K_vec[1:2]*dTdy,
            -self.K_vec[2:3]*dSdx, -self.K_vec[3:4]*dSdy
        ], dim=1)
        h = self.head(x); h = self.rb1(h); h = self.rb2(h)
        delta = self.tail(h)
        return self.alpha * q_K + delta


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNetFlux(nn.Module):
    def __init__(self, in_channels=8, out_channels=4):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64);         self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128);        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2);  self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2);   self.dec1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        x1 = self.enc1(x); x2 = self.pool1(x1)
        x3 = self.enc2(x2); x4 = self.pool2(x3)
        x5 = self.enc3(x4); x6 = self.pool3(x5)
        x7 = self.bottleneck(x6)

        x7u = self.up3(x7)
        x7u = F.pad(x7u, [0, x5.size(3)-x7u.size(3), 0, x5.size(2)-x7u.size(2)])
        x8 = self.dec3(torch.cat([x5, x7u], dim=1))

        x8u = self.up2(x8)
        x8u = F.pad(x8u, [0, x3.size(3)-x8u.size(3), 0, x3.size(2)-x8u.size(2)])
        x9 = self.dec2(torch.cat([x3, x8u], dim=1))

        x9u = self.up1(x9)
        x9u = F.pad(x9u, [0, x1.size(3)-x9u.size(3), 0, x1.size(2)-x9u.size(2)])
        x10 = self.dec1(torch.cat([x1, x9u], dim=1))
        return self.out_conv(x10)


class EDSRBlockFix(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_scale = float(res_scale)
    def forward(self, x):
        res = self.conv1(x); res = self.relu(res); res = self.conv2(res)
        return x + self.res_scale * res


class EDSRFluxFix(nn.Module):
    def __init__(self, in_channels=8, out_channels=4, n_feats=64, n_blocks=16, res_scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[EDSRBlockFix(n_feats, res_scale=res_scale) for _ in range(n_blocks)])
        self.body_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1)
    def forward(self, x):
        h = self.head(x)
        res = self.body(h)
        res = self.body_conv(res)
        h = h + res
        return self.tail(h)


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
        res = self.conv1(x); res = self.relu(res)
        res = self.conv2(res); res = self.ca(res)
        return x + res


class ResidualGroup(nn.Module):
    def __init__(self, n_feats, n_rcab=8):
        super().__init__()
        self.body = nn.Sequential(*[RCAB(n_feats) for _ in range(n_rcab)])
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
    def forward(self, x):
        res = self.body(x)
        res = self.conv(res)
        return x + res


class RCANFlux(nn.Module):
    def __init__(self, in_channels=8, out_channels=4, n_feats=64, n_groups=5, n_rcab=8):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResidualGroup(n_feats, n_rcab=n_rcab) for _ in range(n_groups)])
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1)
    def forward(self, x):
        h = self.head(x)
        res = self.body(h)
        res = self.conv(res)
        return self.tail(h + res)


class RCANTrunk(nn.Module):
    def __init__(self, in_channels=8, n_feats=64, n_groups=5, n_rcab=8):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResidualGroup(n_feats, n_rcab=n_rcab) for _ in range(n_groups)])
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
    def forward(self, x):
        h = self.head(x)
        res = self.body(h)
        res = self.conv(res)
        return h + res


class RCANFluxMoE(nn.Module):
    def __init__(self, gradT_scale, gradS_scale, n_feats=64, n_groups=5, n_rcab=8):
        super().__init__()
        self.trunk = RCANTrunk(8, n_feats, n_groups, n_rcab)
        self.tail_low  = nn.Conv2d(n_feats, 4, 3, padding=1)
        self.tail_high = nn.Conv2d(n_feats, 4, 3, padding=1)
        self.register_buffer("gradT_scale", torch.tensor(float(gradT_scale), dtype=torch.float32))
        self.register_buffer("gradS_scale", torch.tensor(float(gradS_scale), dtype=torch.float32))

    def compute_gate(self, x):
        dTdx = x[:,4:5]; dTdy = x[:,5:6]
        dSdx = x[:,6:7]; dSdy = x[:,7:8]
        gT = torch.sqrt(dTdx*dTdx + dTdy*dTdy + 1e-12) / (self.gradT_scale + 1e-12)
        gS = torch.sqrt(dSdx*dSdx + dSdy*dSdy + 1e-12) / (self.gradS_scale + 1e-12)
        g  = 0.5*(gT + gS)
        return torch.clamp(g, 0.0, 1.0)

    def forward(self, x):
        feat = self.trunk(x)
        y_low  = self.tail_low(feat)
        y_high = self.tail_high(feat)
        g = self.compute_gate(x)
        return (1.0 - g)*y_low + g*y_high


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


class RCANHF_FrontResidual(nn.Module):
    def __init__(self, base_model, tT, tS, tempT, tempS,
                 alpha_init=0.1, delta_groups=3, delta_rcab=4, n_feats=64, freeze_base=True):
        super().__init__()
        self.base = base_model
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self.delta = RCANFlux(8, 4, n_feats=n_feats, n_groups=delta_groups, n_rcab=delta_rcab)

        self.register_buffer("tT", torch.tensor(float(tT), dtype=torch.float32))
        self.register_buffer("tS", torch.tensor(float(tS), dtype=torch.float32))
        self.register_buffer("tempT", torch.tensor(float(tempT), dtype=torch.float32))
        self.register_buffer("tempS", torch.tensor(float(tempS), dtype=torch.float32))

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def compute_gate(self, x):
        dTdx = x[:,4:5]; dTdy = x[:,5:6]
        dSdx = x[:,6:7]; dSdy = x[:,7:8]
        gT = torch.sqrt(dTdx*dTdx + dTdy*dTdy + 1e-12)
        gS = torch.sqrt(dSdx*dSdx + dSdy*dSdy + 1e-12)
        gT = torch.sigmoid((gT - self.tT) / (self.tempT + 1e-12))
        gS = torch.sigmoid((gS - self.tS) / (self.tempS + 1e-12))
        return torch.maximum(gT, gS)

    def forward(self, x):
        with torch.no_grad():
            y0 = self.base(x)
        g  = self.compute_gate(x)
        dy = self.delta(x)
        return y0 + self.alpha * g * dy


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=12, modes2=12):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    def forward(self, x):
        B, C, H, W = x.shape
        # x_ft = torch.fft.rfft2(x, norm="ortho")
        orig_dtype = x.dtype

        # ✅ 关键：FFT 强制 float32，避开 ComplexHalf + cuFFT 限制
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            x_fft = x.float()
        else:
            x_fft = x

        x_ft = torch.fft.rfft2(x_fft, norm="ortho")
        # w = torch.view_as_complex(self.weights)
        # out_ft = torch.zeros(B, w.shape[1], H, W//2+1, device=x.device, dtype=torch.cfloat)
        # out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w)
        w = torch.view_as_complex(self.weights)  # weights 是 float32 参数，OK
        out_ft = torch.zeros(B, w.shape[1], H, W // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], w
        )

        y = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

        # ✅ 回到 autocast 的 dtype，避免后面 conv + 残差 dtype 不匹配
        return y.to(orig_dtype)
        # return torch.fft.irfft2(out_ft, s=(H,W), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, in_channels=8, out_channels=4, width=64, modes1=12, modes2=12, depth=4):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels, width, 1)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Conv2d(width, width, 1)
        self.fc2 = nn.Conv2d(width, out_channels, 1)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.fc0(x)
        for conv, w in zip(self.convs, self.ws):
            x = self.act(conv(x) + w(x))
        x = self.act(self.fc1(x))
        return self.fc2(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class AttUNetFlux(nn.Module):
    def __init__(self, in_channels=8, out_channels=4, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2);      self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4);    self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.att3 = AttentionGate(base*4, base*4, base*2)
        self.dec3 = DoubleConv(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.att2 = AttentionGate(base*2, base*2, base)
        self.dec2 = DoubleConv(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.att1 = AttentionGate(base, base, base//2)
        self.dec1 = DoubleConv(base*2, base)

        self.out_conv = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        x1 = self.enc1(x); x2 = self.pool1(x1)
        x3 = self.enc2(x2); x4 = self.pool2(x3)
        x5 = self.enc3(x4); x6 = self.pool3(x5)
        xb = self.bottleneck(x6)

        u3 = self.up3(xb)
        u3 = F.pad(u3, [0, x5.size(3)-u3.size(3), 0, x5.size(2)-u3.size(2)])
        x5a = self.att3(u3, x5)
        d3 = self.dec3(torch.cat([x5a, u3], dim=1))

        u2 = self.up2(d3)
        u2 = F.pad(u2, [0, x3.size(3)-u2.size(3), 0, x3.size(2)-u2.size(2)])
        x3a = self.att2(u2, x3)
        d2 = self.dec2(torch.cat([x3a, u2], dim=1))

        u1 = self.up1(d2)
        u1 = F.pad(u1, [0, x1.size(3)-u1.size(3), 0, x1.size(2)-u1.size(2)])
        x1a = self.att1(u1, x1)
        d1 = self.dec1(torch.cat([x1a, u1], dim=1))
        return self.out_conv(d1)


# ===== Restormer (light) =====
class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, 3, padding=1, groups=dim*3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = C // self.num_heads
        # [B, heads, head_dim, HW]
        q = q.view(B, self.num_heads, head_dim, H*W)
        k = k.view(B, self.num_heads, head_dim, H*W)
        v = v.view(B, self.num_heads, head_dim, H*W)

        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        # Channel-wise attention: [B,heads,head_dim,head_dim]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)  # [B,heads,head_dim,HW]
        out = out.view(B, C, H, W)
        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    def __init__(self, dim, expansion=2.66):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden*2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden*2, hidden*2, 3, padding=1, groups=hidden*2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn  = MDTA(dim, num_heads=num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = GDFN(dim, expansion=ffn_expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim*2, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim//2, 1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        return x


class RestormerFlux(nn.Module):
    def __init__(self, in_channels=8, out_channels=4,
                 dim=48, num_blocks=[2,2,2,2], num_heads=[1,2,4,8], ffn_expansion=2.66):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, dim, 3, padding=1, bias=False)

        self.enc1 = nn.Sequential(*[RestormerBlock(dim,   num_heads=num_heads[0], ffn_expansion=ffn_expansion) for _ in range(num_blocks[0])])
        self.down1 = Downsample(dim)

        self.enc2 = nn.Sequential(*[RestormerBlock(dim*2, num_heads=num_heads[1], ffn_expansion=ffn_expansion) for _ in range(num_blocks[1])])
        self.down2 = Downsample(dim*2)

        self.enc3 = nn.Sequential(*[RestormerBlock(dim*4, num_heads=num_heads[2], ffn_expansion=ffn_expansion) for _ in range(num_blocks[2])])
        self.down3 = Downsample(dim*4)

        self.latent = nn.Sequential(*[RestormerBlock(dim*8, num_heads=num_heads[3], ffn_expansion=ffn_expansion) for _ in range(num_blocks[3])])

        self.up3 = Upsample(dim*8)
        self.dec3 = nn.Sequential(*[RestormerBlock(dim*4, num_heads=num_heads[2], ffn_expansion=ffn_expansion) for _ in range(num_blocks[2])])

        self.up2 = Upsample(dim*4)
        self.dec2 = nn.Sequential(*[RestormerBlock(dim*2, num_heads=num_heads[1], ffn_expansion=ffn_expansion) for _ in range(num_blocks[1])])

        self.up1 = Upsample(dim*2)
        self.dec1 = nn.Sequential(*[RestormerBlock(dim,   num_heads=num_heads[0], ffn_expansion=ffn_expansion) for _ in range(num_blocks[0])])

        self.out = nn.Conv2d(dim, out_channels, 3, padding=1, bias=True)

    def forward(self, x):
        x = self.embed(x)

        e1 = self.enc1(x)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        e3 = self.enc3(x)
        x = self.down3(e3)

        x = self.latent(x)

        x = self.up3(x) + e3
        x = self.dec3(x)

        x = self.up2(x) + e2
        x = self.dec2(x)

        x = self.up1(x) + e1
        x = self.dec1(x)

        return self.out(x)


# ===== AFNO (simplified) =====
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
        # x_ft = torch.fft.rfft2(x, norm="ortho")  # [B,C,H,Wf]
        orig_dtype = x.dtype

        # ✅ FFT 用 float32
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            x_fft = x.float()
        else:
            x_fft = x

        x_ft = torch.fft.rfft2(x_fft, norm="ortho")
        xr = x_ft.real
        xi = x_ft.imag
        Wf = W//2 + 1

        xr = xr.view(B, self.num_blocks, self.block, H, Wf)
        xi = xi.view(B, self.num_blocks, self.block, H, Wf)

        xr_ = xr.permute(0,1,3,4,2).contiguous().view(B, self.num_blocks, H*Wf, self.block)
        xi_ = xi.permute(0,1,3,4,2).contiguous().view(B, self.num_blocks, H*Wf, self.block)

        # xr_, xi_: [B, n, p, k]  where p=H*Wf, k=block
        or1 = torch.einsum("bnpk,nkm->bnpm", xr_, self.Wr1) - torch.einsum("bnpk,nkm->bnpm", xi_, self.Wi1)
        oi1 = torch.einsum("bnpk,nkm->bnpm", xr_, self.Wi1) + torch.einsum("bnpk,nkm->bnpm", xi_, self.Wr1)
        or1 = F.gelu(or1 + self.br1)  # br1: [n,1,m] broadcast to [B,n,p,m]
        oi1 = F.gelu(oi1 + self.bi1)

        or2 = torch.einsum("bnpm,nmk->bnpk", or1, self.Wr2) - torch.einsum("bnpm,nmk->bnpk", oi1, self.Wi2)
        oi2 = torch.einsum("bnpm,nmk->bnpk", or1, self.Wi2) + torch.einsum("bnpm,nmk->bnpk", oi1, self.Wr2)
        or2 = or2 + self.br2  # br2: [n,1,k]
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


class AFNOFourCastNetFlux(nn.Module):
    def __init__(self, in_channels=8, out_channels=4, width=64, depth=6, num_blocks=8, hidden_mult=2.0):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.Sequential(*[AFNOBlock(width, num_blocks=num_blocks, hidden_mult=hidden_mult) for _ in range(depth)])
        self.proj = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        x = self.lift(x)
        x = self.blocks(x)
        x = self.proj(x)
        return x


# ===== CNO (light) =====
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


class CNOFlux(nn.Module):
    def __init__(self, in_channels=8, out_channels=4, width=64, n_blocks=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 3, padding=1)

        self.enc1 = nn.Sequential(*[CNOBlock(width, dilation=1) for _ in range(n_blocks)])
        self.down1 = nn.Conv2d(width, width*2, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(*[CNOBlock(width*2, dilation=2) for _ in range(n_blocks)])
        self.down2 = nn.Conv2d(width*2, width*4, 3, stride=2, padding=1)

        self.mid  = nn.Sequential(*[CNOBlock(width*4, dilation=2) for _ in range(n_blocks)])

        self.up2 = nn.Conv2d(width*4, width*2, 1)
        self.dec2 = nn.Sequential(*[CNOBlock(width*2, dilation=2) for _ in range(n_blocks)])

        self.up1 = nn.Conv2d(width*2, width, 1)
        self.dec1 = nn.Sequential(*[CNOBlock(width, dilation=1) for _ in range(n_blocks)])

        self.proj = nn.Conv2d(width, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.lift(x)

        e1 = self.enc1(x)
        x  = self.down1(e1)

        e2 = self.enc2(x)
        x  = self.down2(e2)

        x  = self.mid(x)

        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x  = self.up2(x)
        x  = x + e2
        x  = self.dec2(x)

        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x  = self.up1(x)
        x  = x + e1
        x  = self.dec1(x)

        x  = self.proj(x)
        return x

# ============================================================
# ✅ 新模型：SCHOPNet（Hybrid CNO-AFNO + Helmholtz-like decomposition）
#   - backbone: CNO encoder-decoder
#   - bottleneck: insert AFNOBlock for global spectral mixing
#   - head: output potentials [chi_T, psi_T, chi_S, psi_S]
#   - flux: q = ∇chi + k×∇psi
# ============================================================

class FiniteDiff2D(nn.Module):
    """可微差分：中心差分（像素单位）。用于从 χ/ψ potentials 生成通量。"""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32).view(1,1,1,3)
        ky = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32).view(1,1,3,1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def grad(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # s: [B,1,H,W]
        dx = F.conv2d(s, self.kx, padding=(0,1))
        dy = F.conv2d(s, self.ky, padding=(1,0))
        return dx, dy


class HybridCNOAFNOBackbone(nn.Module):
    """CNO encoder-decoder + bottleneck AFNO blocks。输出 potentials（4通道）"""
    def __init__(self, in_channels=8, width=64, n_blocks=4, afno_depth=2, out_channels=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 3, padding=1)

        self.enc1 = nn.Sequential(*[CNOBlock(width, dilation=1) for _ in range(n_blocks)])
        self.down1 = nn.Conv2d(width, width*2, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(*[CNOBlock(width*2, dilation=2) for _ in range(n_blocks)])
        self.down2 = nn.Conv2d(width*2, width*4, 3, stride=2, padding=1)

        mid_blocks = []
        for _ in range(n_blocks):
            mid_blocks.append(CNOBlock(width*4, dilation=2))
        for _ in range(max(0, afno_depth)):
            mid_blocks.append(AFNOBlock(width*4, num_blocks=8, hidden_mult=2.0))
        self.mid = nn.Sequential(*mid_blocks)

        self.up2 = nn.Conv2d(width*4, width*2, 1)
        self.dec2 = nn.Sequential(*[CNOBlock(width*2, dilation=2) for _ in range(n_blocks)])

        self.up1 = nn.Conv2d(width*2, width, 1)
        self.dec1 = nn.Sequential(*[CNOBlock(width, dilation=1) for _ in range(n_blocks)])

        self.proj = nn.Conv2d(width, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.lift(x)

        e1 = self.enc1(x)
        x  = self.down1(e1)

        e2 = self.enc2(x)
        x  = self.down2(e2)

        x  = self.mid(x)

        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x  = self.up2(x)
        x  = x + e2
        x  = self.dec2(x)

        x  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x  = self.up1(x)
        x  = x + e1
        x  = self.dec1(x)

        return self.proj(x)


class SCHOPNet(nn.Module):
    """
    输出 normalized flux（4通道）。
    potentials: [chi_T, psi_T, chi_S, psi_S]
    flux: q = ∇chi + k×∇psi = (dchi/dx, dchi/dy) + (-dpsi/dy, dpsi/dx)
    并加入 scale-conditioned bias（对 potentials 加 n-dependent bias）。
    """
    def __init__(self, n_coarse: int, X_mean, X_std, Y_mean, Y_std,
                 width=64, n_blocks=4, afno_depth=2):
        super().__init__()
        # 用于 diss loss：把输入/输出从 normalized 还原到 physical
        self.register_buffer("X_mean", torch.tensor(X_mean, dtype=torch.float32).view(1,8,1,1))
        self.register_buffer("X_std",  torch.tensor(X_std,  dtype=torch.float32).view(1,8,1,1))
        self.register_buffer("Y_mean", torch.tensor(Y_mean, dtype=torch.float32).view(1,4,1,1))
        self.register_buffer("Y_std",  torch.tensor(Y_std,  dtype=torch.float32).view(1,4,1,1))

        # scale conditioning：n -> potentials bias（论文点：尺度条件化）
        self.scale_bias = nn.Embedding(8, 4)
        self.scale_id = self._scale_to_id(n_coarse)
        self.register_buffer("scale_id_t", torch.tensor(self.scale_id, dtype=torch.long))

        self.backbone = HybridCNOAFNOBackbone(
            in_channels=8, width=width, n_blocks=n_blocks, afno_depth=afno_depth, out_channels=4
        )
        self.diff = FiniteDiff2D()

        # diss/rot mixing（论文点：可解释分解）
        self.alpha_diss_T = nn.Parameter(torch.tensor(1.0))
        self.alpha_rot_T  = nn.Parameter(torch.tensor(1.0))
        self.alpha_diss_S = nn.Parameter(torch.tensor(1.0))
        self.alpha_rot_S  = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _scale_to_id(n: int) -> int:
        if n == 3: return 0
        if n == 5: return 1
        if n == 10: return 2
        return 7

    def denorm_X(self, Xn: torch.Tensor) -> torch.Tensor:
        return Xn * self.X_std + self.X_mean

    def denorm_Y(self, Yn: torch.Tensor) -> torch.Tensor:
        return Yn * self.Y_std + self.Y_mean

    def _pot_to_flux_parts(self, pot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chiT = pot[:, 0:1]
        psiT = pot[:, 1:2]
        chiS = pot[:, 2:3]
        psiS = pot[:, 3:4]

        dchiTdx, dchiTdy = self.diff.grad(chiT)
        dpsiTdx, dpsiTdy = self.diff.grad(psiT)
        dchiSdx, dchiSdy = self.diff.grad(chiS)
        dpsiSdx, dpsiSdy = self.diff.grad(psiS)

        # diss part: grad chi
        qd_T = torch.cat([dchiTdx, dchiTdy], dim=1) * self.alpha_diss_T
        qd_S = torch.cat([dchiSdx, dchiSdy], dim=1) * self.alpha_diss_S
        q_diss = torch.cat([qd_T, qd_S], dim=1)  # [B,4,H,W]

        # rot part: k×∇psi = (-dpsi/dy, dpsi/dx)
        qr_T = torch.cat([-dpsiTdy, dpsiTdx], dim=1) * self.alpha_rot_T
        qr_S = torch.cat([-dpsiSdy, dpsiSdx], dim=1) * self.alpha_rot_S
        q_rot = torch.cat([qr_T, qr_S], dim=1)

        q = q_diss + q_rot
        return q, q_diss, q_rot

    def forward_parts(self, Xn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pot = self.backbone(Xn)  # [B,4,H,W]
        b = self.scale_bias(self.scale_id_t.to(pot.device)).view(1,4,1,1)
        pot = pot + b
        return self._pot_to_flux_parts(pot)

    def forward(self, Xn: torch.Tensor, return_parts: bool = False):
        q, q_diss, q_rot = self.forward_parts(Xn)
        if return_parts:
            return q, q_diss, q_rot
        return q


class HybridCNOAFNOBackbone3(nn.Module):
    """
    3-level encoder-decoder（1/8 bottleneck） + AFNO blocks
    输出 potentials（4通道）或其它 head（你后面 Kappa 会复用）
    """
    def __init__(self, in_channels=8, width=64, n_blocks=4, afno_depth=3, out_channels=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 3, padding=1)

        self.enc1 = nn.Sequential(*[CNOBlock(width, dilation=1) for _ in range(n_blocks)])
        self.down1 = nn.Conv2d(width, width*2, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(*[CNOBlock(width*2, dilation=2) for _ in range(n_blocks)])
        self.down2 = nn.Conv2d(width*2, width*4, 3, stride=2, padding=1)

        self.enc3 = nn.Sequential(*[CNOBlock(width*4, dilation=2) for _ in range(n_blocks)])
        self.down3 = nn.Conv2d(width*4, width*8, 3, stride=2, padding=1)

        mid = []
        for _ in range(n_blocks):
            mid.append(CNOBlock(width*8, dilation=2))
        for _ in range(max(0, afno_depth)):
            mid.append(AFNOBlock(width*8, num_blocks=8, hidden_mult=2.0))
        self.mid = nn.Sequential(*mid)

        self.up3 = nn.Conv2d(width*8, width*4, 1)
        self.dec3 = nn.Sequential(*[CNOBlock(width*4, dilation=2) for _ in range(n_blocks)])

        self.up2 = nn.Conv2d(width*4, width*2, 1)
        self.dec2 = nn.Sequential(*[CNOBlock(width*2, dilation=2) for _ in range(n_blocks)])

        self.up1 = nn.Conv2d(width*2, width, 1)
        self.dec1 = nn.Sequential(*[CNOBlock(width, dilation=1) for _ in range(n_blocks)])

        self.proj = nn.Conv2d(width, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.lift(x)

        e1 = self.enc1(x)
        x  = self.down1(e1)

        e2 = self.enc2(x)
        x  = self.down2(e2)

        e3 = self.enc3(x)
        x  = self.down3(e3)

        x = self.mid(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x) + e3
        x = self.dec3(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x) + e2
        x = self.dec2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x) + e1
        x = self.dec1(x)

        return self.proj(x)


class SCHOPNetDeep(SCHOPNet):
    """
    和 SCHOPNet 同样的 χ/ψ 头与差分，但 backbone 换成 3-level pyramid
    """
    def __init__(self, n_coarse: int, X_mean, X_std, Y_mean, Y_std,
                 width=64, n_blocks=4, afno_depth=3):
        super().__init__(n_coarse, X_mean, X_std, Y_mean, Y_std, width=width, n_blocks=n_blocks, afno_depth=0)
        self.backbone = HybridCNOAFNOBackbone3(
            in_channels=8, width=width, n_blocks=n_blocks, afno_depth=afno_depth, out_channels=4
        )

class SCHOPNetKappa(nn.Module):
    """
    物理分解：
      q = q_diff + q_rot + r
      q_diff = -kappa_T * ∇T  and  -kappa_S * ∇S   （用输入反归一化后的物理梯度）
      q_rot  = k×∇psi （像素差分生成，作为“非耗散/旋转”项）
      r      = small residual（直接回归修正）
    输出 pred_n（normalized flux），并返回 q_diff_n 作为耗散分量用于 diss hinge
    """
    def __init__(self, n_coarse: int, X_mean, X_std, Y_mean, Y_std,
                 width=64, n_blocks=4, afno_depth=2, kappa_floor=1e-6):
        super().__init__()
        self.register_buffer("X_mean", torch.tensor(X_mean, dtype=torch.float32).view(1,8,1,1))
        self.register_buffer("X_std",  torch.tensor(X_std,  dtype=torch.float32).view(1,8,1,1))
        self.register_buffer("Y_mean", torch.tensor(Y_mean, dtype=torch.float32).view(1,4,1,1))
        self.register_buffer("Y_std",  torch.tensor(Y_std,  dtype=torch.float32).view(1,4,1,1))

        self.scale_bias = nn.Embedding(8, 8)  # 给输出 8 通道一点尺度偏置
        self.scale_id = SCHOPNet._scale_to_id(n_coarse)
        self.register_buffer("scale_id_t", torch.tensor(self.scale_id, dtype=torch.long))

        # backbone 输出 8 通道：[logkT, logkS, psiT, psiS, r(4)]
        self.backbone = HybridCNOAFNOBackbone(
            in_channels=8, width=width, n_blocks=n_blocks, afno_depth=afno_depth, out_channels=8
        )
        self.kappa_floor = float(kappa_floor)
        self.diff = FiniteDiff2D()
        self.gamma_rot = nn.Parameter(torch.tensor(1.0))

    def denorm_X(self, Xn: torch.Tensor) -> torch.Tensor:
        return Xn * self.X_std + self.X_mean

    def denorm_Y(self, Yn: torch.Tensor) -> torch.Tensor:
        return Yn * self.Y_std + self.Y_mean

    def _kappa(self, logk: torch.Tensor) -> torch.Tensor:
        return F.softplus(logk) + self.kappa_floor

    def forward_parts(self, Xn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.backbone(Xn)  # [B,8,H,W]
        b = self.scale_bias(self.scale_id_t.to(out.device)).view(1,8,1,1)
        out = out + b

        logkT = out[:,0:1]
        logkS = out[:,1:2]
        psiT  = out[:,2:3]
        psiS  = out[:,3:4]
        res_n = out[:,4:8]        # residual already in normalized space（网络自己学尺度）

        kT = self._kappa(logkT)
        kS = self._kappa(logkS)

        Xp = self.denorm_X(Xn)
        dTdx = Xp[:,4:5]; dTdy = Xp[:,5:6]
        dSdx = Xp[:,6:7]; dSdy = Xp[:,7:8]

        qdiff_phys = torch.cat([-kT*dTdx, -kT*dTdy, -kS*dSdx, -kS*dSdy], dim=1)
        qdiff_n = (qdiff_phys - self.Y_mean) / self.Y_std

        # rot part（normalized space）
        dpsiTdx, dpsiTdy = self.diff.grad(psiT)
        dpsiSdx, dpsiSdy = self.diff.grad(psiS)
        qrot_n = self.gamma_rot * torch.cat([-dpsiTdy, dpsiTdx, -dpsiSdy, dpsiSdx], dim=1)

        pred_n = qdiff_n + qrot_n + res_n
        return pred_n, qdiff_n, qrot_n

    def forward(self, Xn: torch.Tensor, return_parts: bool = False):
        pred_n, qdiff_n, qrot_n = self.forward_parts(Xn)
        if return_parts:
            return pred_n, qdiff_n, qrot_n
        return pred_n


class HybridCNOAFNOFeatureBackbone(nn.Module):
    """CNO encoder-decoder + AFNO bottleneck, 输出 feature map（不直接 proj 到 flux）"""
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

        return x  # [B,width,H,W]


class SCAPEXNet(nn.Module):
    """
    SCAPEXNet：把 ResCNN 的先验、SCHOP 的结构、RCAN 的高频残差融合到一个网络里。
    输入/输出：都在 normalized-space（与全局训练代码一致）
    """
    def __init__(
        self,
        n_coarse: int,
        X_mean, X_std, Y_mean, Y_std,
        K_vec_n: np.ndarray,                 # shape [4] in normalized space (sign-fixed)
        width=64, n_blocks=4, afno_depth=2,
        alpha_s_max: float = 0.9,
    ):
        super().__init__()
        # buffers for denorm in diss hinge
        self.register_buffer("X_mean", torch.tensor(X_mean, dtype=torch.float32).view(1,8,1,1))
        self.register_buffer("X_std",  torch.tensor(X_std,  dtype=torch.float32).view(1,8,1,1))
        self.register_buffer("Y_mean", torch.tensor(Y_mean, dtype=torch.float32).view(1,4,1,1))
        self.register_buffer("Y_std",  torch.tensor(Y_std,  dtype=torch.float32).view(1,4,1,1))

        # normalized-space prior K
        K_vec = torch.tensor(K_vec_n, dtype=torch.float32).view(4,1,1)
        self.register_buffer("K_vec", K_vec)

        self.alpha_s_max = float(alpha_s_max)

        # scale conditioning (即使每个 n 单独训练，也能学到 n-dependent bias；未来你想 multi-n 训练也能直接用)
        self.scale_id = SCHOPNet._scale_to_id(n_coarse)
        self.register_buffer("scale_id_t", torch.tensor(self.scale_id, dtype=torch.long))
        # emb: [alphaT, alphaS, betaT, betaS, pot(4)]
        self.scale_emb = nn.Embedding(8, 8)

        # shared feature trunk
        self.trunk = HybridCNOAFNOFeatureBackbone(in_channels=8, width=width, n_blocks=n_blocks, afno_depth=afno_depth)

        # heads
        self.pot_head  = nn.Conv2d(width, 4, 3, padding=1)   # χT, ψT, χS, ψS
        self.gate_head = nn.Sequential(
            nn.Conv2d(width, max(16, width//2), 1),
            nn.GELU(),
            nn.Conv2d(max(16, width//2), 4, 1)               # alphaT, alphaS, betaT, betaS (logits)
        )

        # RCAN-style residual head (operate on feature map)
        res_groups = 3
        res_rcab   = 4
        self.res_body = nn.Sequential(*[ResidualGroup(width, n_rcab=res_rcab) for _ in range(res_groups)])
        self.res_conv = nn.Conv2d(width, width, 3, padding=1)
        self.res_out  = nn.Conv2d(width, 4, 3, padding=1)

        self.diff = FiniteDiff2D()

        # allow small learnable scaling inside SCHOP part
        self.alpha_diss_T = nn.Parameter(torch.tensor(1.0))
        self.alpha_rot_T  = nn.Parameter(torch.tensor(1.0))
        self.alpha_diss_S = nn.Parameter(torch.tensor(1.0))
        self.alpha_rot_S  = nn.Parameter(torch.tensor(1.0))

    def denorm_X(self, Xn: torch.Tensor) -> torch.Tensor:
        return Xn * self.X_std + self.X_mean

    def denorm_Y(self, Yn: torch.Tensor) -> torch.Tensor:
        return Yn * self.Y_std + self.Y_mean

    def _prior_flux(self, Xn: torch.Tensor) -> torch.Tensor:
        # Xn is normalized; grad channels are normalized too -> prior flux is normalized
        dTdx = Xn[:,4:5]; dTdy = Xn[:,5:6]
        dSdx = Xn[:,6:7]; dSdy = Xn[:,7:8]
        qK = torch.cat([
            -self.K_vec[0:1]*dTdx, -self.K_vec[1:2]*dTdy,
            -self.K_vec[2:3]*dSdx, -self.K_vec[3:4]*dSdy,
        ], dim=1)
        return qK

    def _pot_to_flux_parts(self, pot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def forward(self, Xn: torch.Tensor, return_parts: bool = False):
        feat = self.trunk(Xn)

        # scale biases
        emb = self.scale_emb(self.scale_id_t.to(feat.device)).view(1,8,1,1)
        gate_bias = emb[:,0:4]   # [1,4,1,1]
        pot_bias  = emb[:,4:8]   # [1,4,1,1]

        # gates
        g_logits = self.gate_head(feat) + gate_bias
        g = torch.sigmoid(g_logits)
        alphaT = g[:,0:1]
        alphaS = torch.clamp(g[:,1:2], 0.0, self.alpha_s_max)  # ✅ 关键防崩
        betaT  = g[:,2:3]
        betaS  = g[:,3:4]

        alpha4 = torch.cat([alphaT, alphaT, alphaS, alphaS], dim=1)
        beta4  = torch.cat([betaT,  betaT,  betaS,  betaS ], dim=1)

        # experts
        q_prior = self._prior_flux(Xn)  # normalized
        pot = self.pot_head(feat) + pot_bias
        q_schop, qd_schop, qr_schop = self._pot_to_flux_parts(pot)  # normalized

        # residual expert
        rr = self.res_body(feat)
        rr = self.res_conv(rr) + feat
        q_res = self.res_out(rr)

        pred = alpha4 * q_prior + beta4 * q_schop + q_res

        if return_parts:
            # for diss hinge: dissipative part should include prior (purely diff) + schop-diss
            q_diss = alpha4 * q_prior + beta4 * qd_schop
            q_rot  = beta4 * qr_schop
            return pred, q_diss, q_rot

        return pred

# =========================
# DataLoader helper
# =========================
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


# =========================
# Train loops
# =========================
def train_points_mlp(train_items, X_mean, X_std, Y_mean, Y_std, ckpt_dir):
    tag = "MLP"
    model = MLPFlux().to(DEVICE)
    if load_ckpt(ckpt_dir, tag, model):
        return model

    ds = BufferedPointDataset(
        train_items, X_mean, X_std, Y_mean, Y_std,
        length=300_000, seed=0, points_per_refill=20000
    )
    loader = _make_loader(ds, batch_size=BATCH_SIZE_MLP, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    # scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda" and USE_AMP))
    scaler = make_scaler()

    for ep in range(1, N_EPOCHS_MLP + 1):
        model.train()
        tot = 0.0
        t0 = time.perf_counter()
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            # with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda" and USE_AMP)):
            with autocast_ctx():
                pred = model(xb)
                loss = crit(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * xb.size(0)

        dt = time.perf_counter() - t0
        log(f"[MLP] Epoch {ep:03d} | train={tot/len(loader.dataset):.4f} | time={dt:.1f}s")

    save_ckpt(ckpt_dir, tag, model)
    try:
        ds.close_cache()
    except Exception:
        pass
    return model


def train_img_model(tag, model, train_loader, val_loader, n_epochs, ckpt_dir, lr=1e-3, clip=1.0):
    model = model.to(DEVICE)
    # ✅ wrap after to(device)
    if USE_DP and (not isinstance(model, nn.DataParallel)):
        model = nn.DataParallel(model, device_ids=GPU_IDS)

    tracker = BestCkptTracker(ckpt_dir, tag, model, mode="min", metric_name="val(mse)")
    if tracker.try_load():
        return model

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda" and USE_AMP))
    scaler = make_scaler()
    for ep in range(1, n_epochs + 1):
        model.train()
        tot = 0.0
        t0 = time.perf_counter()

        for xb, yb, mb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            # with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda" and USE_AMP)):
            with autocast_ctx():
                pred = model(xb)
                loss = masked_mse_torch_with_mask(pred, yb, mb)

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
                # with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda" and USE_AMP)):
                with autocast_ctx():
                    pred = model(xb)
                    vtot += masked_mse_torch_with_mask(pred, yb, mb).item() * xb.size(0)

        val_mse = vtot / len(val_loader.dataset)
        tracker.update(val_mse, ep)
        dt = time.perf_counter() - t0
        log(f"[{tag}] Epoch {ep:03d} | train={tot/len(train_loader.dataset):.4f} | val={vtot/len(val_loader.dataset):.4f} | time={dt:.1f}s")

    tracker.finalize(extra_last={"lr": float(lr)}, load_best=True)
    return model


def _masked_mean_tensor(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x.device)
    return x[mask].mean()


def train_schopnet(tag, model: SCHOPNet, train_loader, val_loader, n_epochs, ckpt_dir,
                   lr=2e-4, clip=1.0, lam_diss=0.03, lam_hf_T=0.05, lam_hf_S=0.0, use_hf=True):
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
        diss_tot = 0.0
        hf_tot = 0.0
        t0 = time.perf_counter()

        for xb, yb, mb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx():

                pred_n, qd_n, _ = model(xb, return_parts=True)
                base = _unwrap_model(model)  # 只用来访问 denorm_X / buffers

                loss_mse = masked_mse_torch_with_mask(pred_n, yb, mb)

                # dissipation hinge（只对耗散分量 q_diss）
                Xphys = base.denorm_X(xb)
                dTdx = Xphys[:,4]; dTdy = Xphys[:,5]
                dSdx = Xphys[:,6]; dSdy = Xphys[:,7]

                qd_phys = base.denorm_Y(qd_n)
                qTx = qd_phys[:,0]; qTy = qd_phys[:,1]
                qSx = qd_phys[:,2]; qSy = qd_phys[:,3]

                m2 = mb & torch.isfinite(dTdx) & torch.isfinite(dTdy) & torch.isfinite(dSdx) & torch.isfinite(dSdy)
                prodT = -(qTx*dTdx + qTy*dTdy)  # should be > 0
                prodS = -(qSx*dSdx + qSy*dSdy)

                diss = _masked_mean_tensor(F.relu(-prodT), m2) + _masked_mean_tensor(F.relu(-prodS), m2)

                loss = loss_mse + lam_diss * diss

                # optional HF (split T/S)
                hf = torch.tensor(0.0, device=loss.device)
                if use_hf and (lam_hf_T > 0.0 or lam_hf_S > 0.0):
                    if HF_LOSS_TYPE.lower() == "laplacian":
                        hp_pred = laplacian_filter(pred_n)
                        hp_true = laplacian_filter(torch.nan_to_num(yb, nan=0.0))
                    elif HF_LOSS_TYPE.lower() == "dog":
                        hp_pred = dog_filter(pred_n, ksize=DOG_KSIZE, sigma1=DOG_SIGMA1, sigma2=DOG_SIGMA2)
                        hp_true = dog_filter(torch.nan_to_num(yb, nan=0.0), ksize=DOG_KSIZE, sigma1=DOG_SIGMA1, sigma2=DOG_SIGMA2)
                    else:
                        raise ValueError("HF_LOSS_TYPE must be 'laplacian' or 'dog'.")

                    m4 = mb.unsqueeze(1).expand_as(hp_pred) & torch.isfinite(yb)
                    diff = (hp_pred - hp_true).abs()

                    hfT = _masked_mean_tensor(diff[:,0:2], m4[:,0:2])
                    hfS = _masked_mean_tensor(diff[:,2:4], m4[:,2:4])
                    hf = lam_hf_T*hfT + lam_hf_S*hfS
                    loss = loss + hf

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            scaler.step(opt)
            scaler.update()

            tot += loss.item() * xb.size(0)
            diss_tot += float(diss.detach().item()) * xb.size(0)
            hf_tot += float(hf.detach().item()) * xb.size(0)

        # val only MSE
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
        tracker.update(
            val_mse, ep,
            extra_best={
                "lr": float(lr),
                "lam_diss": float(lam_diss),
                "lam_hf_T": float(lam_hf_T),
                "lam_hf_S": float(lam_hf_S),
            }
        )

        dt = time.perf_counter() - t0
        log(f"[{tag}] Epoch {ep:03d} | train={tot / len(train_loader.dataset):.4f} "
            f"| val(mse)={val_mse:.4f} | diss={diss_tot / len(train_loader.dataset):.4f} "
            f"| hf={hf_tot / len(train_loader.dataset):.4f} | time={dt:.1f}s")
    tracker.finalize(
        extra_last={
            "lr": float(lr),
            "lam_diss": float(lam_diss),
            "lam_hf_T": float(lam_hf_T),
            "lam_hf_S": float(lam_hf_S),
            "use_hf": bool(use_hf),
        },
        load_best=True
    )
    return model


def train_schopnet_plus(
    tag,
    model,
    train_loader, val_loader,
    n_epochs, ckpt_dir,
    lr=2e-4, clip=1.0,
    lam_diss=0.03,
    diss_warmup=3,
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

        # diss warmup+ramp
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

                # diss hinge：只对耗散分量 qd_n 做约束（用物理梯度）
                Xphys = base.denorm_X(xb) if hasattr(base, "denorm_X") else None
                if Xphys is not None and lam_diss_eff > 0:
                    dTdx = Xphys[:,4]; dTdy = Xphys[:,5]
                    dSdx = Xphys[:,6]; dSdy = Xphys[:,7]

                    # qd_n -> phys
                    if hasattr(base, "denorm_Y"):
                        qd_phys = base.denorm_Y(qd_n)
                    else:
                        # SCHOPNetKappa: qd_n 是 qdiff_n，用 buffers 反推
                        qd_phys = qd_n * base.Y_std + base.Y_mean

                    qTx = qd_phys[:,0]; qTy = qd_phys[:,1]
                    qSx = qd_phys[:,2]; qSy = qd_phys[:,3]

                    m2 = mb & torch.isfinite(dTdx) & torch.isfinite(dTdy) & torch.isfinite(dSdx) & torch.isfinite(dSdy)
                    prodT = -(qTx*dTdx + qTy*dTdy)
                    prodS = -(qSx*dSdx + qSy*dSdy)
                    diss = _masked_mean_tensor(F.relu(-prodT), m2) + _masked_mean_tensor(F.relu(-prodS), m2)
                    loss = loss + lam_diss_eff * diss

                # HF（可选）
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
                    loss = loss + lam_hf_T*hfT + lam_hf_S*hfS

                # Low-frequency spectral loss
                if lam_lf > 0.0:
                    loss_lf = spectral_lf_loss_amp(pred_n, yb, mb, k0_frac=lf_k0, p=lf_p)
                    loss = loss + lam_lf * loss_lf

                # divergence loss
                if lam_div > 0.0:
                    loss_div = divergence_loss_pixel(pred_n, yb, mb)
                    loss = loss + lam_div * loss_div

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * xb.size(0)

        # val：只看 MSE
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
        log(f"[{tag}] Epoch {ep:03d} | train={tot/len(train_loader.dataset):.4f} | val(mse)={vtot/len(val_loader.dataset):.4f} | lam_diss={lam_diss_eff:.3f} | time={dt:.1f}s")

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




def train_rcan_hf(tag, model, train_loader, val_loader, n_epochs, ckpt_dir, lr=1e-4, clip=1.0):
    model = model.to(DEVICE)
    if USE_DP and (not isinstance(model, nn.DataParallel)):
        model = nn.DataParallel(model, device_ids=GPU_IDS)

    tracker = BestCkptTracker(ckpt_dir, tag, model, mode="min", metric_name="val(mse)")
    if tracker.try_load():
        return model

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = make_scaler()

    wfreq = None

    for ep in range(1, n_epochs + 1):
        model.train()
        tot = 0.0
        t0 = time.perf_counter()

        lam_hf = HF_LAM * (ep / n_epochs) if HF_RAMP else HF_LAM
        lam_sp = SPEC_LAM * (ep / n_epochs) if (HF_RAMP and ENABLE_SPEC_LOSS) else (SPEC_LAM if ENABLE_SPEC_LOSS else 0.0)

        for xb, yb, mb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            # with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda" and USE_AMP)):
            with autocast_ctx():
                pred = model(xb)
                loss_mse = masked_mse_torch_with_mask(pred, yb, mb)

                if HF_LOSS_TYPE.lower() == "laplacian":
                    hp_pred = laplacian_filter(pred)
                    hp_true = laplacian_filter(torch.nan_to_num(yb, nan=0.0))
                elif HF_LOSS_TYPE.lower() == "dog":
                    hp_pred = dog_filter(pred, ksize=DOG_KSIZE, sigma1=DOG_SIGMA1, sigma2=DOG_SIGMA2)
                    hp_true = dog_filter(torch.nan_to_num(yb, nan=0.0), ksize=DOG_KSIZE, sigma1=DOG_SIGMA1, sigma2=DOG_SIGMA2)
                else:
                    raise ValueError("HF_LOSS_TYPE must be 'laplacian' or 'dog'.")

                loss_hf = masked_l1_torch_with_mask(hp_pred, hp_true, mb)

                if ENABLE_SPEC_LOSS:
                    B, C, H, W = pred.shape
                    if wfreq is None or (wfreq.shape[0] != H) or (wfreq.shape[1] != (W//2+1)):
                        wfreq = make_freq_weight(H, W, p=SPEC_P, k0_frac=SPEC_K0_FRAC, device=pred.device, dtype=pred.dtype)
                    loss_spec = spectral_hf_loss(pred, yb, mb, wfreq)
                else:
                    loss_spec = torch.tensor(0.0, device=pred.device)

                loss = loss_mse + lam_hf * loss_hf + lam_sp * loss_spec

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
                # with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda" and USE_AMP)):
                with autocast_ctx():
                    pred = model(xb)
                    vtot += masked_mse_torch_with_mask(pred, yb, mb).item() * xb.size(0)

        val_mse = vtot / len(val_loader.dataset)
        tracker.update(val_mse, ep, extra_best={"lam_hf": float(lam_hf), "lam_sp": float(lam_sp)})

        dt = time.perf_counter() - t0
        log(f"[{tag}] Epoch {ep:03d} | train={tot/len(train_loader.dataset):.4f} | val(mse)={vtot/len(val_loader.dataset):.4f} | lam_hf={lam_hf:.3f} | time={dt:.1f}s")

    tracker.finalize(extra_last={"lr": float(lr)}, load_best=True)
    return model


def train_rcan_hf_moe(tag, model, train_loader, val_loader, n_epochs, ckpt_dir, lr=1e-4, clip=1.0):
    model = model.to(DEVICE)
    if USE_DP and (not isinstance(model, nn.DataParallel)):
        model = nn.DataParallel(model, device_ids=GPU_IDS)

    tracker = BestCkptTracker(ckpt_dir, tag, model, mode="min", metric_name="val(mse)")
    if tracker.try_load():
        return model

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = make_scaler()

    wfreq = None

    for ep in range(1, n_epochs + 1):
        model.train()
        tot = 0.0
        t0 = time.perf_counter()

        lam_hf = HF_LAM * (ep / n_epochs) if HF_RAMP else HF_LAM
        lam_sp = SPEC_LAM * (ep / n_epochs) if (HF_RAMP and ENABLE_SPEC_LOSS) else (SPEC_LAM if ENABLE_SPEC_LOSS else 0.0)

        for xb, yb, mb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            # with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda" and USE_AMP)):
            with autocast_ctx():
                pred = model(xb)
                loss_mse = masked_mse_torch_with_mask(pred, yb, mb)

                base = _unwrap_model(model)
                with torch.no_grad():
                    g = base.compute_gate(xb)
                    w_hf = 1.0 + HF_FRONT_BOOST * g

                if HF_LOSS_TYPE.lower() == "laplacian":
                    hp_pred = laplacian_filter(pred)
                    hp_true = laplacian_filter(torch.nan_to_num(yb, nan=0.0))
                elif HF_LOSS_TYPE.lower() == "dog":
                    hp_pred = dog_filter(pred, ksize=DOG_KSIZE, sigma1=DOG_SIGMA1, sigma2=DOG_SIGMA2)
                    hp_true = dog_filter(torch.nan_to_num(yb, nan=0.0), ksize=DOG_KSIZE, sigma1=DOG_SIGMA1, sigma2=DOG_SIGMA2)
                else:
                    raise ValueError("HF_LOSS_TYPE must be 'laplacian' or 'dog'.")

                loss_hf = masked_l1_weighted_torch_with_mask(hp_pred, hp_true, mb, w_hf)

                if ENABLE_SPEC_LOSS:
                    B, C, H, W = pred.shape
                    if wfreq is None or (wfreq.shape[0] != H) or (wfreq.shape[1] != (W//2+1)):
                        wfreq = make_freq_weight(H, W, p=SPEC_P, k0_frac=SPEC_K0_FRAC, device=pred.device, dtype=pred.dtype)
                    loss_spec = spectral_hf_loss(pred, yb, mb, wfreq)
                else:
                    loss_spec = torch.tensor(0.0, device=pred.device)

                loss = loss_mse + lam_hf * loss_hf + lam_sp * loss_spec

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
                # with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda" and USE_AMP)):
                with autocast_ctx():
                    pred = model(xb)
                    vtot += masked_mse_torch_with_mask(pred, yb, mb).item() * xb.size(0)

        val_mse = vtot / len(val_loader.dataset)
        tracker.update(val_mse, ep, extra_best={"lam_hf": float(lam_hf), "lam_sp": float(lam_sp)})

        dt = time.perf_counter() - t0
        log(f"[{tag}] Epoch {ep:03d} | train={tot/len(train_loader.dataset):.4f} | val(mse)={vtot/len(val_loader.dataset):.4f} | lam_hf={lam_hf:.3f} | time={dt:.1f}s")

    tracker.finalize(extra_last={"lr": float(lr)}, load_best=True)
    return model


def train_rcan_hf_fr(tag, model, train_loader, val_loader, n_epochs, ckpt_dir, lr=1e-4, clip=1.0):

    model = model.to(DEVICE)
    if USE_DP and (not isinstance(model, nn.DataParallel)):
        model = nn.DataParallel(model, device_ids=GPU_IDS)

    tracker = BestCkptTracker(ckpt_dir, tag, model, mode="min", metric_name="val(mse)")
    if tracker.try_load():
        return model

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    scaler = make_scaler()

    wfreq = None

    for ep in range(1, n_epochs + 1):
        model.train()
        tot = 0.0
        t0 = time.perf_counter()

        lam_hf = HF_LAM * (ep / n_epochs) if HF_RAMP else HF_LAM
        lam_sp = SPEC_LAM * (ep / n_epochs) if (HF_RAMP and ENABLE_SPEC_LOSS) else (SPEC_LAM if ENABLE_SPEC_LOSS else 0.0)

        for xb, yb, mb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            # with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda" and USE_AMP)):
            with autocast_ctx():

                pred = model(xb)
                loss_mse = masked_mse_torch_with_mask(pred, yb, mb)

                base = _unwrap_model(model)
                with torch.no_grad():
                    g = base.compute_gate(xb)
                    w_hf = 1.0 + HF_FRONT_BOOST * g

                if HF_LOSS_TYPE.lower() == "laplacian":
                    hp_pred = laplacian_filter(pred)
                    hp_true = laplacian_filter(torch.nan_to_num(yb, nan=0.0))
                elif HF_LOSS_TYPE.lower() == "dog":
                    hp_pred = dog_filter(pred, ksize=DOG_KSIZE, sigma1=DOG_SIGMA1, sigma2=DOG_SIGMA2)
                    hp_true = dog_filter(torch.nan_to_num(yb, nan=0.0), ksize=DOG_KSIZE, sigma1=DOG_SIGMA1, sigma2=DOG_SIGMA2)
                else:
                    raise ValueError("HF_LOSS_TYPE must be 'laplacian' or 'dog'.")

                loss_hf = masked_l1_weighted_torch_with_mask(hp_pred, hp_true, mb, w_hf)

                if ENABLE_SPEC_LOSS:
                    B, C, H, W = pred.shape
                    if wfreq is None or (wfreq.shape[0] != H) or (wfreq.shape[1] != (W//2+1)):
                        wfreq = make_freq_weight(H, W, p=SPEC_P, k0_frac=SPEC_K0_FRAC, device=pred.device, dtype=pred.dtype)
                    loss_spec = spectral_hf_loss(pred, yb, mb, wfreq)
                else:
                    loss_spec = torch.tensor(0.0, device=pred.device)

                loss = loss_mse + lam_hf * loss_hf + lam_sp * loss_spec

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=clip)
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
                # with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda" and USE_AMP)):
                with autocast_ctx():
                    pred = model(xb)
                    vtot += masked_mse_torch_with_mask(pred, yb, mb).item() * xb.size(0)

        dt = time.perf_counter() - t0
        val_mse = vtot / len(val_loader.dataset)
        alpha_val = float(_unwrap_model(model).alpha.item())
        tracker.update(val_mse, ep, extra_best={"alpha": alpha_val})
        log(f"[{tag}] Epoch {ep:03d} | val(mse)={val_mse:.4f} | alpha={alpha_val:.3f} | time={dt:.1f}s")

    tracker.finalize(extra_last={"lr": float(lr)}, load_best=True)
    return model


# =========================
# Online metrics
# =========================
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


# =========================
# 推理：MLP 全图 chunk
# =========================
def predict_mlp_full(model_mlp: nn.Module, Xn: torch.Tensor, Y_mean: np.ndarray, Y_std: np.ndarray, chunk_pix: int = 100_000) -> np.ndarray:
    model_mlp.eval()
    with torch.no_grad():
        B, C, H, W = Xn.shape
        xflat = Xn.permute(0,2,3,1).reshape(-1, C).contiguous()
        yflat = torch.zeros((xflat.shape[0], 4), device=Xn.device, dtype=torch.float32)
        for st in range(0, xflat.shape[0], chunk_pix):
            ed = min(st + chunk_pix, xflat.shape[0])
            yflat[st:ed] = model_mlp(xflat[st:ed])
        Yn = yflat.reshape(B, H, W, 4).permute(0,3,1,2).cpu().numpy()
    Yp = Yn * Y_std.reshape(1,4,1,1) + Y_mean.reshape(1,4,1,1)
    return Yp.astype(np.float32, copy=False)


# =========================
# 推理：tiled（防爆显存）
# =========================
def predict_cnn_full_tiled(model: nn.Module, Xn: torch.Tensor,
                           Y_mean: np.ndarray, Y_std: np.ndarray,
                           tile: int = 192, overlap: int = 32) -> np.ndarray:
    """
    改进点：
    1) 每个 tile 不再 .cpu().numpy()，累积在 GPU 上
    2) 只在最后一次性搬回 CPU
    3) 支持边界 tile 小于 tile 的情况（避免 ww 形状不一致）
    """
    assert Xn.dim() == 4 and Xn.size(0) == 1
    model.eval()
    B, C, H, W = Xn.shape
    stride = max(1, tile - overlap)

    device = Xn.device

    # torch 版 mean/std（放 GPU）
    Y_mean_t = torch.tensor(Y_mean, device=device, dtype=torch.float32).view(1,4,1,1)
    Y_std_t  = torch.tensor(Y_std,  device=device, dtype=torch.float32).view(1,4,1,1)

    out_acc = torch.zeros((1,4,H,W), device=device, dtype=torch.float32)
    w_acc   = torch.zeros((1,1,H,W), device=device, dtype=torch.float32)

    # 预生成 full ww（numpy -> torch）
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

                x_tile = Xn[:, :, js:je, is_:ie]  # 可能小于 tile

                with autocast_ctx():
                    yp_n = model(x_tile)

                # 强制 float32 做累积，避免半精度累积误差
                yp_n = yp_n.float()
                yp   = yp_n * Y_std_t + Y_mean_t  # GPU 上 denorm

                ht = je - js
                wt = ie - is_
                ww = ww_full_t[:, :, :ht, :wt]     # 匹配边界 tile 尺寸

                out_acc[:,:,js:je,is_:ie] += yp * ww
                w_acc[:,:,js:je,is_:ie]   += ww

    out = out_acc / (w_acc + 1e-12)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


# =========================
# 评估：流式累计（rmse/corr/r2 + depth诊断 + grad分箱）
# =========================
# =========================
# 评估：流式累计（rmse/corr/r2 + depth诊断 + grad分箱）
# =========================
def evaluate_all_models_stream(
    models: Dict[str, Optional[nn.Module]],
    model_mlp: nn.Module,
    K_T: float, K_S: float,
    val_loader_full: DataLoader,
    Y_mean: np.ndarray, Y_std: np.ndarray,
    out_dir_fig: str,
    n_coarse: int,
    max_scatter_points: int = 120_000,
    grad_bins: int = 10,
    grad_sample_for_quantile: int = 500_000,
):
    # ---- methods：确保 Zero/Kgrad/MLP 都在列表里（并保持顺序）----
    base_methods = ["Zero", "Kgrad", "MLP"]
    methods = base_methods + [k for k in models.keys() if k not in base_methods]

    rng = np.random.default_rng(123)
    grad_samples = []
    got = 0
    max_batches_for_edges = 10  # 5~20 都行，先用 10
    for bi, batch in enumerate(val_loader_full):
        Xn, Y_phys, ocean, file_id, lev, X_phys = batch
        X_phys_np = X_phys.numpy()[0]
        dTdx = X_phys_np[4]
        dTdy = X_phys_np[5]
        g = np.sqrt(dTdx * dTdx + dTdy * dTdy).reshape(-1)
        m = np.isfinite(g)
        g = g[m]
        if g.size == 0:
            continue
        take = min(g.size, 80_000)
        sel = rng.choice(g.size, size=take, replace=False)
        grad_samples.append(g[sel])
        got += take

        if got >= grad_sample_for_quantile:
            break
        if bi + 1 >= max_batches_for_edges:
            break

    edges = np.quantile(np.concatenate(grad_samples, axis=0), np.linspace(0, 1, grad_bins + 1)) if grad_samples else None

    stats = {m: [OnlineStats1D() for _ in range(4)] for m in methods}

    lev_set = set()
    lev_acc = {m: {} for m in methods}
    def ensure_lev(m, levv):
        if levv not in lev_acc[m]:
            lev_acc[m][levv] = {
                "n": 0, "sum_y": 0.0, "sum_y2": 0.0, "sse": 0.0,
                "bias_sum": 0.0,
                "down_pos": 0, "down_n": 0,
                "grad_bin_sse": np.zeros(grad_bins, dtype=np.float64),
                "grad_bin_n":   np.zeros(grad_bins, dtype=np.int64)
            }
        return lev_acc[m][levv]

    scat_y = []
    scat_p = {m: [] for m in methods}

    t_eval0 = time.perf_counter()
    for batch in val_loader_full:
        Xn, Y_phys, ocean, file_id, lev, X_phys = batch

        levv = int(lev[0])
        lev_set.add(levv)

        Xn = Xn.to(DEVICE)
        ocean2d = ocean.numpy()[0]
        Yt = Y_phys.numpy()[0]
        Xp = X_phys.numpy()[0]

        m2d = ocean2d & np.isfinite(Yt[0]) & np.isfinite(Yt[1]) & np.isfinite(Yt[2]) & np.isfinite(Yt[3])
        if m2d.sum() == 0:
            continue

        # ---- 公共量 ----
        yt_flat = Yt[:, m2d]
        dTdx = Xp[4]; dTdy = Xp[5]
        dSdx = Xp[6]; dSdy = Xp[7]

        # ---- grad bins：每个样本只算一次 ----
        if edges is not None:
            gT = np.sqrt(Xp[4] * Xp[4] + Xp[5] * Xp[5])
            g_flat = gT[m2d].reshape(-1)
            bin_id = np.searchsorted(edges, g_flat, side="right") - 1
            bin_id = np.clip(bin_id, 0, grad_bins - 1)
        else:
            bin_id = None

        # ---- scatter：每个样本只抽一次索引，所有方法共用 ----
        sel_scatter = None
        if len(scat_y) < max_scatter_points:
            yt_qTx = Yt[0][m2d].reshape(-1)
            if yt_qTx.size > 0:
                take = min(2000, yt_qTx.size, max_scatter_points - len(scat_y))
                if take > 0:
                    sel_scatter = rng.choice(yt_qTx.size, size=take, replace=False)
                    scat_y.extend(list(yt_qTx[sel_scatter]))

        # ---- consume：边预测边更新（不保存 full preds_phys）----
        def consume_one_method(mname: str, Yp_m: np.ndarray):
            # Yp_m: [4,H,W] physical
            yp_flat = Yp_m[:, m2d]

            # 4通道 stats
            for c in range(4):
                y = yt_flat[c].reshape(-1)
                p = yp_flat[c].reshape(-1)
                good = np.isfinite(y) & np.isfinite(p)
                if np.any(good):
                    stats[mname][c].update(y[good], p[good])

            # lev 汇总（overall）
            acc = ensure_lev(mname, levv)
            y_all = yt_flat.reshape(-1)
            p_all = yp_flat.reshape(-1)
            good_all = np.isfinite(y_all) & np.isfinite(p_all)
            if np.any(good_all):
                y0 = y_all[good_all]
                p0 = p_all[good_all]
                acc["n"] += int(y0.size)
                acc["sum_y"]  += float(np.sum(y0))
                acc["sum_y2"] += float(np.sum(y0 * y0))
                e = p0 - y0
                acc["sse"] += float(np.sum(e * e))
                acc["bias_sum"] += float(np.sum(e))

            # down-gradient fraction（只用 T 分量）
            qTx_p = Yp_m[0][m2d]
            qTy_p = Yp_m[1][m2d]
            dTdx_v = dTdx[m2d]
            dTdy_v = dTdy[m2d]
            prod = -(qTx_p * dTdx_v + qTy_p * dTdy_v)
            prod = prod[np.isfinite(prod)]
            if prod.size > 0:
                acc["down_pos"] += int(np.sum(prod > 0.0))
                acc["down_n"]   += int(prod.size)

            # grad bin RMSE（如果启用）
            if bin_id is not None:
                err = (yp_flat[0:2] - yt_flat[0:2])  # [2, Npix]
                mse_pix = np.mean(err * err, axis=0)  # [Npix]
                for b in range(grad_bins):
                    mb = (bin_id == b)
                    if np.any(mb):
                        acc["grad_bin_sse"][b] += float(np.sum(mse_pix[mb]))
                        acc["grad_bin_n"][b]   += int(np.sum(mb))

            # scatter：只抽 qT_x
            if sel_scatter is not None:
                yp_qTx = Yp_m[0][m2d].reshape(-1)[sel_scatter]
                scat_p[mname].extend(list(yp_qTx))

        # ------------------ 依次处理各方法：不建 preds_phys dict ------------------

        # Zero
        Yp0 = np.zeros_like(Yt, dtype=np.float32)
        Yp0[:, ~m2d] = np.nan
        consume_one_method("Zero", Yp0)
        del Yp0

        # Kgrad
        Yk = np.stack(
            [-K_T * dTdx, -K_T * dTdy, -K_S * dSdx, -K_S * dSdy],
            axis=0
        ).astype(np.float32, copy=False)
        Yk[:, ~m2d] = np.nan
        consume_one_method("Kgrad", Yk)
        del Yk

        # MLP
        Ymlp = predict_mlp_full(model_mlp, Xn, Y_mean, Y_std)[0].astype(np.float32, copy=False)
        # 不 mask 也行（反正只取 m2d），但这里保持一致：
        Ymlp[:, ~m2d] = np.nan
        consume_one_method("MLP", Ymlp)
        del Ymlp

        # 其它模型
        for name, model in models.items():
            if name in ("Zero", "Kgrad", "MLP") or model is None:
                continue

            if (EVAL_TILE and EVAL_TILE > 0) and (name in TILED_METHODS):
                yp = predict_cnn_full_tiled(model, Xn, Y_mean, Y_std, tile=EVAL_TILE, overlap=EVAL_OVERLAP)[0]
            else:
                model.eval()
                with torch.no_grad():
                    with autocast_ctx():
                        yp_n = model(Xn).detach()  # [1,4,H,W] normalized tensor
                yp = (yp_n.float().cpu().numpy() * Y_std.reshape(1, 4, 1, 1) + Y_mean.reshape(1, 4, 1, 1))[0]
                yp = yp.astype(np.float32, copy=False)

            yp[:, ~m2d] = np.nan
            consume_one_method(name, yp)
            del yp

    log(f"✅ Eval streaming done | time={time.perf_counter()-t_eval0:.1f}s")

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

    os.makedirs(out_dir_fig, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes = axes.ravel()
    panels = flux_names + ["mean(4)"]
    for k, name in enumerate(panels):
        ax = axes[k]
        if name == "mean(4)":
            vals = [all_metrics[m]["mean"]["r2"] for m in methods]
            ttl = "R² (mean of 4 fluxes)"
        else:
            vals = [all_metrics[m][name]["r2"] for m in methods]
            ttl = f"R² ({name})"
        ax.bar(methods, vals)
        ax.set_ylim(-0.5, 1.0)
        ax.set_title(ttl)
        ax.set_ylabel("R²")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)

    ax = axes[5]
    vals_rmse = [all_metrics[m]["mean"]["rmse"] for m in methods]
    ax.bar(methods, vals_rmse)
    ax.set_title("RMSE (mean of 4 flux components)")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    savefig_tight(fig, os.path.join(out_dir_fig, f"panel_R2_components_plus_meanRMSE_n{n_coarse}.png"))

    R2_matrix = np.array([[all_metrics[m][f]["r2"] for f in flux_names] for m in methods], dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, max(4.6, 0.42 * len(methods))), constrained_layout=True)
    im = ax.imshow(R2_matrix, vmin=-0.5, vmax=1.0, cmap="viridis", aspect="auto")
    for i in range(len(methods)):
        for j in range(len(flux_names)):
            val = R2_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if (np.isfinite(val) and val > 0.55) else "black", fontsize=9)
    ax.set_xticks(np.arange(len(flux_names)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(flux_names)
    ax.set_yticklabels(methods)
    ax.set_title(f"R² heatmap | val(full) | n={n_coarse}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("R²")
    savefig_tight(fig, os.path.join(out_dir_fig, f"R2_heatmap_n{n_coarse}.png"))

    if len(scat_y) > 500:
        y_s = np.array(scat_y, dtype=np.float32)
        n = len(methods)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.6 * nrows),
                                constrained_layout=True, sharex=True, sharey=True)
        axs = np.array(axs).ravel()
        for k, name in enumerate(methods):
            ax = axs[k]
            p_s = np.array(scat_p[name], dtype=np.float32)
            ax.scatter(y_s, p_s, s=1.2, alpha=0.25)
            lo = float(np.nanmin([y_s.min(), p_s.min()]))
            hi = float(np.nanmax([y_s.max(), p_s.max()]))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_title(f"{name} (qT_x)")
            ax.set_xlabel("True"); ax.set_ylabel("Pred")
        for k in range(n, len(axs)):
            axs[k].axis("off")
        savefig_tight(fig, os.path.join(out_dir_fig, f"panel_scatter_qT_x_n{n_coarse}.png"))

    levs_sorted = np.array(sorted(list(lev_set)), dtype=int)
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True, constrained_layout=True)
    axs = axs.ravel()

    for m in methods:
        r2_lev = []
        for levv in levs_sorted:
            acc = lev_acc[m].get(int(levv), None)
            if acc is None or acc["n"] < 10:
                r2_lev.append(np.nan); continue
            n0 = acc["n"]
            sumy = acc["sum_y"]
            sumy2 = acc["sum_y2"]
            sse = acc["sse"]
            sst = sumy2 - (sumy * sumy) / max(n0, 1)
            r2 = 1.0 - sse / (sst + 1e-12) if sst > 0 else np.nan
            r2_lev.append(r2)
        axs[0].plot(levs_sorted, r2_lev, marker="o", label=m)
    axs[0].invert_xaxis()
    axs[0].set_ylim(-0.5, 1.0)
    axs[0].set_title(f"R² vs depth (overall, 4 comps) | n={n_coarse}")
    axs[0].set_ylabel("R²")

    for m in methods:
        rmse_lev = []
        for levv in levs_sorted:
            acc = lev_acc[m].get(int(levv), None)
            if acc is None or acc["n"] < 10:
                rmse_lev.append(np.nan); continue
            rmse_lev.append(math.sqrt(acc["sse"] / acc["n"]))
        axs[1].plot(levs_sorted, rmse_lev, marker="o", label=m)
    axs[1].invert_xaxis()
    axs[1].set_title("RMSE vs depth (overall)")
    axs[1].set_ylabel("RMSE")

    for m in methods:
        bias_lev = []
        for levv in levs_sorted:
            acc = lev_acc[m].get(int(levv), None)
            if acc is None or acc["n"] < 10:
                bias_lev.append(np.nan); continue
            bias_lev.append(acc["bias_sum"] / acc["n"])
        axs[2].plot(levs_sorted, bias_lev, marker="o", label=m)
    axs[2].axhline(0.0, color="k", linewidth=0.8)
    axs[2].invert_xaxis()
    axs[2].set_title("Bias vs depth (pred - true)")
    axs[2].set_ylabel("Bias")
    axs[2].set_xlabel("Vertical level index (lev)")

    for m in methods:
        frac_lev = []
        for levv in levs_sorted:
            acc = lev_acc[m].get(int(levv), None)
            if acc is None or acc["down_n"] < 10:
                frac_lev.append(np.nan); continue
            frac_lev.append(acc["down_pos"] / acc["down_n"])
        axs[3].plot(levs_sorted, frac_lev, marker="o", label=m)
    axs[3].invert_xaxis()
    axs[3].set_ylim(0, 1.0)
    axs[3].set_title("Down-gradient fraction vs depth")
    axs[3].set_ylabel("Fraction of (-q_T · ∇T) > 0")
    axs[3].set_xlabel("Vertical level index (lev)")

    handles, labels = axs[3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.02, 0.5), frameon=True)
    savefig_tight(fig, os.path.join(out_dir_fig, f"panel_depth_diagnostics_2x2_n{n_coarse}.png"))

    if edges is not None:
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        for m in methods:
            y = []
            for b in range(grad_bins):
                sse = 0.0
                nn_ = 0
                for levv in levs_sorted:
                    acc = lev_acc[m].get(int(levv), None)
                    if acc is None:
                        continue
                    sse += float(acc["grad_bin_sse"][b])
                    nn_ += int(acc["grad_bin_n"][b])
                if nn_ < 500:
                    y.append(np.nan)
                else:
                    y.append(math.sqrt(sse / nn_))
            ax.plot(centers, y, marker="o", label=m)
        ax.set_xlabel(r"$|\nabla T|$ (quantile-binned)")
        ax.set_ylabel("RMSE of [qT_x, qT_y]")
        ax.set_title(f"RMSE vs |∇T| regime | n={n_coarse}")
        ax.legend(ncol=2, frameon=True)
        savefig_tight(fig, os.path.join(out_dir_fig, f"RMSE_vs_gradT_bins_n{n_coarse}.png"))

    return all_metrics

# =========================
# 跨尺度总图
# =========================
def plot_skill_vs_scale(all_scale_metrics: Dict[int, Dict], result_root: str):
    scales = sorted(all_scale_metrics.keys())
    methods = list(next(iter(all_scale_metrics.values())).keys())

    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    for m in methods:
        r2s = [all_scale_metrics[n][m]["mean"]["r2"] for n in scales]
        ax.plot(scales, r2s, marker="o", label=m)
    ax.set_xlabel("Coarse-graining factor n")
    ax.set_ylabel("R² (mean of 4 flux components)")
    ax.set_title("Model skill vs coarse scale")
    ax.legend(ncol=2, frameon=True)
    savefig_tight(fig, os.path.join(result_root, "skill_vs_scale_R2_mean.png"))

    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    for m in methods:
        rmses = [all_scale_metrics[n][m]["mean"]["rmse"] for n in scales]
        ax.plot(scales, rmses, marker="o", label=m)
    ax.set_xlabel("Coarse-graining factor n")
    ax.set_ylabel("RMSE (mean of 4 flux components)")
    ax.set_title("Model error vs coarse scale")
    ax.legend(ncol=2, frameon=True)
    savefig_tight(fig, os.path.join(result_root, "skill_vs_scale_RMSE_mean.png"))

    out_csv = os.path.join(result_root, "summary_skill_vs_scale.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n","method","r2_mean","rmse_mean","corr_mean"])
        for n in scales:
            for m in methods:
                w.writerow([n, m,
                            all_scale_metrics[n][m]["mean"]["r2"],
                            all_scale_metrics[n][m]["mean"]["rmse"],
                            all_scale_metrics[n][m]["mean"]["corr"]])
    log(f"Saved: {out_csv}")


# =========================
# 单尺度主流程（方案A）
# =========================
def run_one_scale(n_coarse: int, var_dir: str, label_dir: str, result_dir: str) -> Dict:
    global LOGGER
    LOGGER = setup_logger(result_dir, name=f"n{n_coarse}")

    log("\n========================================")
    log(f"✅ Run scale n={n_coarse} (Lazy+LOG+CACHE+AMP+TILED_EVAL + Restormer/AFNO/CNO + SignFix)")
    log(f"VAR_DIR   = {var_dir}")
    log(f"LABEL_DIR = {label_dir}")
    log(f"RESULT_DIR= {result_dir}")
    log(f"DEVICE    = {DEVICE} | USE_AMP={USE_AMP}")
    log(f"MAX_DAYS  = {MAX_DAYS} | PATCH_SIZE={PATCH_SIZE}")
    log(f"EVAL_TILE = {EVAL_TILE} | EVAL_OVERLAP={EVAL_OVERLAP}")
    log("========================================\n")

    fig_dir  = os.path.join(result_dir, "figures")
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    theta_files = sorted(glob.glob(os.path.join(var_dir, "thetao_Oday_FGOALS-f3-H_omip2_r1i1p1f1_gn_*.nc")))
    if not theta_files:
        raise RuntimeError(f"找不到 thetao 文件：{var_dir}")

    items = build_day_level_index(theta_files, label_dir, max_days=MAX_DAYS)
    if not items:
        raise RuntimeError(f"❌ n={n_coarse} 没有任何可用 (day,lev) item，请检查 label_dir={label_dir}")

    file_ids = np.array([it["file_id"] for it in items], dtype=int)
    unique_files = np.sort(np.unique(file_ids))
    split_idx = int(0.7 * len(unique_files))
    train_files = set(unique_files[:split_idx])
    val_files   = set(unique_files[split_idx:])

    train_items = [it for it in items if it["file_id"] in train_files]
    val_items   = [it for it in items if it["file_id"] in val_files]

    log(f"Items: {len(items)} | Train items: {len(train_items)} | Val items: {len(val_items)}")
    log(f"Days: {len(unique_files)} | Train days: {len(train_files)} | Val days: {len(val_files)}")

    # norm
    X_mean, X_std, Y_mean, Y_std = estimate_norm_from_train(train_items, n_samples=2_000_000, seed=0)

    # patch datasets + loaders
    train_ds = LazyLevPatchDataset(train_items, X_mean, X_std, Y_mean, Y_std, patch_size=PATCH_SIZE, seed=1, patches_per_item=TRAIN_PATCHES_PER_ITEM)
    val_ds   = LazyLevPatchDataset(val_items,   X_mean, X_std, Y_mean, Y_std, patch_size=PATCH_SIZE, seed=2, patches_per_item=1)

    train_loader = _make_loader(train_ds, batch_size=BATCH_SIZE_IMG_TRAIN, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = _make_loader(val_ds, batch_size=BATCH_SIZE_IMG_TRAIN, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # full-map eval loader (single worker)
    val_full_ds = LazyLevFullEvalDataset(val_items, X_mean, X_std, Y_mean, Y_std)
    val_full_loader = DataLoader(
        val_full_ds,
        batch_size=BATCH_SIZE_IMG_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,  # 关键
        pin_memory=True,  # 关键
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    # Baseline K (sign-fixed)
    K_T, K_S = fit_scalar_K_stream(train_items, max_samples_per_item=150_000, seed=0)
    log(f"✅ baseline K fit (sign-fixed): K_T={K_T:.3e}, K_S={K_S:.3e}")

    # ============ MLP ============
    model_mlp = train_points_mlp(train_items, X_mean, X_std, Y_mean, Y_std, ckpt_dir)


    rng = np.random.default_rng(0)
    num = np.zeros(4, dtype=np.float64)
    den = np.zeros(4, dtype=np.float64)
    got = 0

    for it in train_items[:max(10, len(train_items)//6)]:
        ds = xr.open_dataset(it["label_path"], **XR_OPEN_KW)
        try:
            lev = it["lev"]
            t = get2d_4dvar(ds, "theta_lr", lev)
            s = get2d_4dvar(ds, "so_lr", lev)
            qTx = get2d_4dvar(ds, "qT_x", lev); qTy = get2d_4dvar(ds, "qT_y", lev)
            qSx = get2d_4dvar(ds, "qS_x", lev); qSy = get2d_4dvar(ds, "qS_y", lev)

            dx, dy = get_dxdy(ds)
            dTdx, dTdy = phys_grad_2d(t, dx, dy)
            dSdx, dSdy = phys_grad_2d(s, dx, dy)

            # normalized
            dTdx_n = (dTdx - X_mean[4]) / X_std[4]
            dTdy_n = (dTdy - X_mean[5]) / X_std[5]
            dSdx_n = (dSdx - X_mean[6]) / X_std[6]
            dSdy_n = (dSdy - X_mean[7]) / X_std[7]

            qTx_n  = (qTx  - Y_mean[0]) / Y_std[0]
            qTy_n  = (qTy  - Y_mean[1]) / Y_std[1]
            qSx_n  = (qSx  - Y_mean[2]) / Y_std[2]
            qSy_n  = (qSy  - Y_mean[3]) / Y_std[3]

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

    # ✅ sign-fixed: q_n ≈ -K_n * grad_n
    K_vec_n = -num / np.maximum(den, 1e-12)
    log(f"✅ ResCNN prior K_vec (normalized, sign-fixed): {K_vec_n}")

    # ============ MoE/FR gate quantiles（从 train 估计阈值） ============
    gradT_s, gradS_s = [], []
    got = 0
    for it in train_items[:max(10, len(train_items)//8)]:
        ds = xr.open_dataset(it["label_path"], **XR_OPEN_KW)
        try:
            lev = it["lev"]
            t = get2d_4dvar(ds, "theta_lr", lev)
            s = get2d_4dvar(ds, "so_lr", lev)
            dx, dy = get_dxdy(ds)
            dTdx, dTdy = phys_grad_2d(t, dx, dy)
            dSdx, dSdy = phys_grad_2d(s, dx, dy)

            dTdx_n = (dTdx - X_mean[4]) / X_std[4]
            dTdy_n = (dTdy - X_mean[5]) / X_std[5]
            dSdx_n = (dSdx - X_mean[6]) / X_std[6]
            dSdy_n = (dSdy - X_mean[7]) / X_std[7]

            gT = np.sqrt(dTdx_n*dTdx_n + dTdy_n*dTdy_n).reshape(-1)
            gS = np.sqrt(dSdx_n*dSdx_n + dSdy_n*dSdy_n).reshape(-1)
            m = np.isfinite(gT) & np.isfinite(gS)
            gT = gT[m]; gS = gS[m]
            if gT.size == 0:
                continue
            take = min(100_000, gT.size)
            sel = rng.choice(gT.size, size=take, replace=False)
            gradT_s.append(gT[sel]); gradS_s.append(gS[sel])
            got += take
            if got >= 500_000:
                break
        finally:
            ds.close()

    if gradT_s:
        gradT_all = np.concatenate(gradT_s)
        gradS_all = np.concatenate(gradS_s)
        gradT_scale = float(np.quantile(gradT_all, MOE_GATE_Q))
        gradS_scale = float(np.quantile(gradS_all, MOE_GATE_Q))
        tT = float(np.quantile(gradT_all, FR_GATE_Q))
        tS = float(np.quantile(gradS_all, FR_GATE_Q))
    else:
        gradT_scale = 1.0; gradS_scale = 1.0; tT = 1.0; tS = 1.0

    gradT_scale = max(gradT_scale, 1e-6)
    gradS_scale = max(gradS_scale, 1e-6)
    tT = max(tT, 1e-6); tS = max(tS, 1e-6)
    tempT = max(FR_GATE_TEMP * tT, 1e-6)
    tempS = max(FR_GATE_TEMP * tS, 1e-6)

    log(f"✅ MoE gate scales (Q={MOE_GATE_Q}): gradT_scale={gradT_scale:.3f}, gradS_scale={gradS_scale:.3f}")
    log(f"✅ Front gate: Q={FR_GATE_Q}  tT={tT:.3f} tS={tS:.3f}  tempT={tempT:.3f} tempS={tempS:.3f}")

    # ============ Train image models ============
    models: Dict[str, Optional[nn.Module]] = {
        "Zero": None,
        "Kgrad": None,
        "MLP": None,
    }

    # CNN
    cnn = train_img_model("CNN", CNNFlux(), train_loader, val_loader, N_EPOCHS_CNN, ckpt_dir, lr=LR, clip=CLIP_GRAD)
    models["CNN"] = cnn

    # UNet
    unet = train_img_model("UNet", UNetFlux(), train_loader, val_loader, N_EPOCHS_UNET, ckpt_dir, lr=LR, clip=CLIP_GRAD)
    models["UNet"] = unet

    # ResCNN (with normalized prior K)
    resc = train_img_model("ResCNN", ResCNNFlux(K_Tx=float(K_vec_n[0]), K_Ty=float(K_vec_n[1]), K_Sx=float(K_vec_n[2]), K_Sy=float(K_vec_n[3])),
                           train_loader, val_loader, N_EPOCHS_RESCNN, ckpt_dir, lr=LR, clip=CLIP_GRAD)
    models["ResCNN"] = resc

    # EDSR_fix
    edsr = train_img_model("EDSR_fix", EDSRFluxFix(), train_loader, val_loader, N_EPOCHS_EDSRFIX, ckpt_dir, lr=LR_EDSR_FIX, clip=CLIP_GRAD)
    models["EDSR_fix"] = edsr

    # RCAN (plain)
    rcan = train_img_model("RCAN", RCANFlux(), train_loader, val_loader, N_EPOCHS_RCAN, ckpt_dir, lr=LR_RCAN, clip=CLIP_GRAD)
    models["RCAN"] = rcan

    # RCAN_HF
    rcan_hf = train_rcan_hf("RCAN_HF", RCANFlux(), train_loader, val_loader, N_EPOCHS_RCAN, ckpt_dir, lr=LR_RCAN, clip=CLIP_GRAD)
    models["RCAN_HF"] = rcan_hf

    # RCAN_HF_MOE
    rcan_moe = RCANFluxMoE(gradT_scale=gradT_scale, gradS_scale=gradS_scale)
    rcan_hf_moe = train_rcan_hf_moe("RCAN_HF_MOE", rcan_moe, train_loader, val_loader, N_EPOCHS_RCAN, ckpt_dir, lr=LR_RCAN, clip=CLIP_GRAD)
    models["RCAN_HF_MOE"] = rcan_hf_moe

    # RCAN_HF_FR (FrontResidual)
    base_for_fr = RCANFlux()
    base_for_fr = train_img_model("RCAN_base_for_FR", base_for_fr, train_loader, val_loader,
                                  max(1, N_EPOCHS_RCAN // 2), ckpt_dir, lr=LR_RCAN, clip=CLIP_GRAD)
    base_for_fr = _unwrap_model(base_for_fr)  # 这次 unwrap 的才是可能的 DP
    base_for_fr.eval()
    fr = RCANHF_FrontResidual(base_for_fr, tT=tT, tS=tS, tempT=tempT, tempS=tempS,
                              alpha_init=FR_ALPHA_INIT, delta_groups=3, delta_rcab=4, freeze_base=True)
    rcan_fr = train_rcan_hf_fr("RCAN_HF_FR", fr, train_loader, val_loader, N_EPOCHS_RCAN, ckpt_dir, lr=LR_RCAN, clip=CLIP_GRAD)
    models["RCAN_HF_FR"] = rcan_fr

    # AttUNet
    att = train_img_model("AttUNet", AttUNetFlux(), train_loader, val_loader, N_EPOCHS_ATTUNET, ckpt_dir, lr=LR, clip=CLIP_GRAD)
    models["AttUNet"] = att

    # FNO
    fno = train_img_model("FNO", FNO2d(), train_loader, val_loader, N_EPOCHS_FNO, ckpt_dir, lr=LR, clip=CLIP_GRAD)
    models["FNO"] = fno

    # Restormer
    rest = train_img_model("Restormer", RestormerFlux(), train_loader, val_loader, N_EPOCHS_RESTORMER, ckpt_dir, lr=LR_RESTORMER, clip=CLIP_GRAD)
    models["Restormer"] = rest

    # AFNO
    afno = train_img_model("AFNO", AFNOFourCastNetFlux(), train_loader, val_loader, N_EPOCHS_AFNO, ckpt_dir, lr=LR_AFNO, clip=CLIP_GRAD)
    models["AFNO"] = afno

    # CNO
    cno = train_img_model("CNO", CNOFlux(), train_loader, val_loader, N_EPOCHS_CNO, ckpt_dir, lr=LR_CNO, clip=CLIP_GRAD)
    models["CNO"] = cno

    # =========================
    # ✅ SCHOPNet
    # =========================
    schop = SCHOPNet(
        n_coarse=n_coarse,
        X_mean=X_mean, X_std=X_std,
        Y_mean=Y_mean, Y_std=Y_std,
        width=SCHOP_WIDTH,
        n_blocks=SCHOP_NBLOCKS,
        afno_depth=SCHOP_AFNO_DEPTH
    )
    schop = train_schopnet_plus(
        "SCHOPNet", schop,
        train_loader, val_loader,
        N_EPOCHS_SCHOPNET, ckpt_dir,
        lr=LR_SCHOPNET, clip=CLIP_GRAD,
        lam_diss=SCHOP_LAM_DISS, diss_warmup=0,  # 基线不 warmup
        lam_hf_T=SCHOP_LAM_HF_T, lam_hf_S=SCHOP_LAM_HF_S, use_hf=SCHOP_USE_HF,
        lam_lf=0.0, lam_div=0.0
    )

    models["SCHOPNet"] = schop

    # =========================================================
    # SCHOP 系列专用 loaders：n=10 用更大 patch（其他模型不变）
    # =========================================================
    schop_patch = SCHOP_PATCH_N10 if (n_coarse >= 10) else PATCH_SIZE
    # --- 让 SCHOP patch 不超过实际 H/W，避免 Patch too large 崩溃 ---
    ds0 = xr.open_dataset(train_items[0]["label_path"], **XR_OPEN_KW)
    try:
        lev0 = train_items[0]["lev"]
        tmp = get2d_4dvar(ds0, "theta_lr", lev0)  # [H,W]
        H0, W0 = tmp.shape
    finally:
        ds0.close()

    schop_patch = min(schop_patch, H0, W0)

    train_ds_schop = LazyLevPatchDataset(train_items, X_mean, X_std, Y_mean, Y_std,
                                         patch_size=schop_patch, seed=101, patches_per_item=TRAIN_PATCHES_PER_ITEM)
    val_ds_schop   = LazyLevPatchDataset(val_items,   X_mean, X_std, Y_mean, Y_std,
                                         patch_size=schop_patch, seed=102, patches_per_item=1)
    train_loader_schop = _make_loader(train_ds_schop, batch_size=BATCH_SIZE_IMG_TRAIN, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader_schop   = _make_loader(val_ds_schop, batch_size=BATCH_SIZE_IMG_TRAIN, shuffle=False,
                                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # =========================================================
    # 1) SCHOPNet_BP：只改变 patch（用 train_schopnet_plus，其他 loss 关）
    # =========================================================
    schop_bp = SCHOPNet(n_coarse, X_mean, X_std, Y_mean, Y_std,
                        width=SCHOP_WIDTH, n_blocks=SCHOP_NBLOCKS, afno_depth=SCHOP_AFNO_DEPTH)
    schop_bp = train_schopnet_plus(
        "SCHOPNet_BP", schop_bp,
        train_loader_schop, val_loader_schop,
        N_EPOCHS_SCHOPNET, ckpt_dir,
        lr=LR_SCHOPNET, clip=CLIP_GRAD,
        lam_diss=SCHOP_LAM_DISS, diss_warmup=0,
        lam_hf_T=SCHOP_LAM_HF_T, lam_hf_S=SCHOP_LAM_HF_S, use_hf=SCHOP_USE_HF,
        lam_lf=0.0, lam_div=0.0
    )
    models["SCHOPNet_BP"] = schop_bp

    # =========================================================
    # 2) SCHOPNet_LF：低频谱损失（专攻 n=10）
    # =========================================================
    schop_lf = SCHOPNet(n_coarse, X_mean, X_std, Y_mean, Y_std,
                        width=SCHOP_WIDTH, n_blocks=SCHOP_NBLOCKS, afno_depth=SCHOP_AFNO_DEPTH)
    schop_lf = train_schopnet_plus(
        "SCHOPNet_LF", schop_lf,
        train_loader_schop, val_loader_schop,
        N_EPOCHS_SCHOPNET, ckpt_dir,
        lr=LR_SCHOPNET, clip=CLIP_GRAD,
        lam_diss=SCHOP_LAM_DISS, diss_warmup=SCHOP_DISS_WARMUP,
        lam_hf_T=SCHOP_LAM_HF_T, lam_hf_S=SCHOP_LAM_HF_S, use_hf=SCHOP_USE_HF,
        lam_lf=SCHOP_LAM_LF, lf_k0=SCHOP_LF_K0, lf_p=SCHOP_LF_P,
        lam_div=0.0
    )
    models["SCHOPNet_LF"] = schop_lf



    # =========================================================
    # 4) SCHOPNet_LFDiv：低频谱 + divergence（n=10 最推荐组合）
    # =========================================================
    schop_lfdiv = SCHOPNet(n_coarse, X_mean, X_std, Y_mean, Y_std,
                           width=SCHOP_WIDTH, n_blocks=SCHOP_NBLOCKS, afno_depth=SCHOP_AFNO_DEPTH)
    schop_lfdiv = train_schopnet_plus(
        "SCHOPNet_LFDiv", schop_lfdiv,
        train_loader_schop, val_loader_schop,
        N_EPOCHS_SCHOPNET, ckpt_dir,
        lr=LR_SCHOPNET, clip=CLIP_GRAD,
        lam_diss=SCHOP_LAM_DISS, diss_warmup=SCHOP_DISS_WARMUP,
        lam_hf_T=SCHOP_LAM_HF_T, lam_hf_S=SCHOP_LAM_HF_S, use_hf=SCHOP_USE_HF,
        lam_lf=SCHOP_LAM_LF, lf_k0=SCHOP_LF_K0, lf_p=SCHOP_LF_P,
        lam_div=SCHOP_LAM_DIV
    )
    models["SCHOPNet_LFDiv"] = schop_lfdiv

    # =========================
    # ✅ SCAPEXNet（Adaptive Prior Experts）
    # =========================
    if n_coarse == 3:
        alpha_s_max = SCAPEX_ALPHA_S_MAX_N3
        lam_div = SCAPEX_LAM_DIV_N3
        lam_lf  = 0.0
    elif n_coarse >= 10:
        alpha_s_max = SCAPEX_ALPHA_S_MAX_OTH
        lam_div = 0.0
        lam_lf  = SCAPEX_LAM_LF_N10
    else:
        alpha_s_max = SCAPEX_ALPHA_S_MAX_OTH
        lam_div = 0.0
        lam_lf  = 0.0

    scapex = SCAPEXNet(
        n_coarse=n_coarse,
        X_mean=X_mean, X_std=X_std,
        Y_mean=Y_mean, Y_std=Y_std,
        K_vec_n=K_vec_n,                       # ✅ 用你拟合的 normalized-space K_vec
        width=SCAPEX_WIDTH,
        n_blocks=SCAPEX_NBLOCKS,
        afno_depth=SCAPEX_AFNO_DEPTH,
        alpha_s_max=alpha_s_max,
    )

    # 直接复用 train_schopnet_plus（因为 forward(return_parts=True) 形状兼容）
    scapex = train_schopnet_plus(
        "SCAPEX", scapex,
        train_loader, val_loader,
        N_EPOCHS_SCAPEX, ckpt_dir,
        lr=LR_SCAPEX, clip=CLIP_GRAD,
        lam_diss=SCAPEX_LAM_DISS, diss_warmup=SCAPEX_DISS_WARMUP,
        lam_hf_T=SCAPEX_LAM_HF_T, lam_hf_S=SCAPEX_LAM_HF_S, use_hf=SCAPEX_USE_HF,
        lam_lf=lam_lf, lf_k0=SCHOP_LF_K0, lf_p=SCHOP_LF_P,
        lam_div=lam_div
    )
    models["SCAPEX"] = scapex


    # ============ Evaluate ============
    metrics = evaluate_all_models_stream(
        models=models,
        model_mlp=model_mlp,
        K_T=K_T, K_S=K_S,
        val_loader_full=val_full_loader,
        Y_mean=Y_mean, Y_std=Y_std,
        out_dir_fig=fig_dir,
        n_coarse=n_coarse,
    )

    # close caches best-effort
    try:
        train_ds.close_cache()
        val_ds.close_cache()
    except Exception:
        pass

    # dump json
    out_json = os.path.join(result_dir, f"metrics_n{n_coarse}.json")
    with open(out_json, "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    log(f"Saved: {out_json}")

    return metrics


# =========================
# main: multi-scale
# =========================
def resolve_label_dir_for_n(label_root: str, n: int) -> str:
    # 兼容两种布局：
    # 1) thelabel/n3/...
    # 2) thelabel/...(不分n)
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

        metrics = run_one_scale(n, VAR_DIR, label_dir, result_dir)
        all_scale_metrics[n] = metrics
        # ✅ 新增：把该 n 的最终指标扁平化加进总表
        all_rows.extend(metrics_to_rows(metrics, n=n, add_n=True))

        # ✅ 可选：每个 n 跑完就落盘一次（防崩溃丢结果）
        out_csv = os.path.join(RESULT_ROOT, "all_scales_metrics.csv")
        write_rows_csv(out_csv, all_rows, fieldnames=["n", "method", "component", "rmse", "corr", "r2"])
        print("Saved (partial):", out_csv)
    # 跨尺度汇总
    plot_skill_vs_scale(all_scale_metrics, RESULT_ROOT)

    # 最终再写一次（确保最后状态）
    out_csv = os.path.join(RESULT_ROOT, "all_scales_metrics.csv")
    write_rows_csv(out_csv, all_rows, fieldnames=["n", "method", "component", "rmse", "corr", "r2"])
    print("Saved (final):", out_csv)
    # 汇总 JSON
    out_json = os.path.join(RESULT_ROOT, "all_scales_metrics.json")
    with open(out_json, "w") as f:
        import json
        json.dump(all_scale_metrics, f, indent=2)
    log(f"Saved: {out_json}")


# ============================================================
# Parameter sensitivity experiment code for SCAPEX
# Paste this block near the end of your script (after main()).
# ============================================================

# =========================
# Sensitivity experiment config
# =========================
N_EPOCHS_SCAPEX_SENS = 15
TRAIN_PATCHES_PER_ITEM_SENS = 2
SENS_MAX_GRAD_SAMPLES = 500_000
SENS_MAX_K_SAMPLES = 600_000

SENS_ALPHA_S_MAX_LIST = [0.35, 0.45, 0.55, 0.70, 0.90, 1.00]
SENS_LAM_DISS_LIST    = [0.00, 0.01, 0.03, 0.05, 0.08, 0.12]
SENS_LAM_LF_LIST      = [0.00, 0.02, 0.04, 0.06, 0.08, 0.12]
SENS_LAM_DIV_LIST     = [0.00, 0.01, 0.02, 0.04, 0.08]


# =========================
# Helper: fit normalized prior K_vec for SCAPEX
# q_n ≈ -K_n * grad_n
# =========================
def fit_normalized_prior_Kvec(
    train_items,
    X_mean, X_std,
    Y_mean, Y_std,
    max_total_samples=SENS_MAX_K_SAMPLES,
    seed=0
):
    rng = np.random.default_rng(seed)
    num = np.zeros(4, dtype=np.float64)
    den = np.zeros(4, dtype=np.float64)
    got = 0

    take_items = train_items[:max(10, len(train_items)//6)]

    for it in take_items:
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

            valid = (
                np.isfinite(dTdx_n) & np.isfinite(qTx_n) &
                np.isfinite(dTdy_n) & np.isfinite(qTy_n) &
                np.isfinite(dSdx_n) & np.isfinite(qSx_n) &
                np.isfinite(dSdy_n) & np.isfinite(qSy_n)
            )

            idx = np.where(valid.reshape(-1))[0]
            if idx.size == 0:
                continue

            take = min(120_000, idx.size)
            sel = rng.choice(idx, size=take, replace=False)

            dTdx_s = dTdx_n.reshape(-1)[sel]
            qTx_s  = qTx_n.reshape(-1)[sel]
            dTdy_s = dTdy_n.reshape(-1)[sel]
            qTy_s  = qTy_n.reshape(-1)[sel]
            dSdx_s = dSdx_n.reshape(-1)[sel]
            qSx_s  = qSx_n.reshape(-1)[sel]
            dSdy_s = dSdy_n.reshape(-1)[sel]
            qSy_s  = qSy_n.reshape(-1)[sel]

            num[0] += float(np.sum(dTdx_s * qTx_s))
            den[0] += float(np.sum(dTdx_s * dTdx_s) + 1e-12)
            num[1] += float(np.sum(dTdy_s * qTy_s))
            den[1] += float(np.sum(dTdy_s * dTdy_s) + 1e-12)
            num[2] += float(np.sum(dSdx_s * qSx_s))
            den[2] += float(np.sum(dSdx_s * dSdx_s) + 1e-12)
            num[3] += float(np.sum(dSdy_s * qSy_s))
            den[3] += float(np.sum(dSdy_s * dSdy_s) + 1e-12)

            got += take
            if got >= max_total_samples:
                break
        finally:
            ds.close()

    K_vec_n = -num / np.maximum(den, 1e-12)
    return K_vec_n.astype(np.float32)


# =========================
# Helper: default SCAPEX params by scale
# =========================
def get_default_scapex_params(n_coarse: int):
    if n_coarse == 3:
        alpha_s_max = SCAPEX_ALPHA_S_MAX_N3
        lam_div = SCAPEX_LAM_DIV_N3
        lam_lf = 0.0
    elif n_coarse >= 10:
        alpha_s_max = SCAPEX_ALPHA_S_MAX_OTH
        lam_div = 0.0
        lam_lf = SCAPEX_LAM_LF_N10
    else:
        alpha_s_max = SCAPEX_ALPHA_S_MAX_OTH
        lam_div = 0.0
        lam_lf = 0.0

    return {
        "alpha_s_max": float(alpha_s_max),
        "lam_diss": float(SCAPEX_LAM_DISS),
        "lam_lf": float(lam_lf),
        "lam_div": float(lam_div),
        "lam_hf_T": float(SCAPEX_LAM_HF_T),
        "lam_hf_S": float(SCAPEX_LAM_HF_S),
        "use_hf": bool(SCAPEX_USE_HF),
        "diss_warmup": int(SCAPEX_DISS_WARMUP),
    }


# =========================
# Helper: prepare one scale context only once
# =========================
def prepare_scale_context_for_sensitivity(
    n_coarse: int,
    var_dir: str,
    label_dir: str,
    result_dir: str,
    patch_size: int = PATCH_SIZE,
    train_patches_per_item: int = TRAIN_PATCHES_PER_ITEM_SENS,
):
    global LOGGER
    LOGGER = setup_logger(result_dir, name=f"sensitivity_n{n_coarse}")

    log("\n========================================")
    log(f"✅ Prepare sensitivity context | n={n_coarse}")
    log(f"VAR_DIR   = {var_dir}")
    log(f"LABEL_DIR = {label_dir}")
    log(f"RESULT_DIR= {result_dir}")
    log(f"DEVICE    = {DEVICE} | USE_AMP={USE_AMP}")
    log(f"PATCH_SIZE= {patch_size} | TRAIN_PATCHES_PER_ITEM={train_patches_per_item}")
    log("========================================\n")

    theta_files = sorted(glob.glob(os.path.join(var_dir, "thetao_Oday_FGOALS-f3-H_omip2_r1i1p1f1_gn_*.nc")))
    if not theta_files:
        raise RuntimeError(f"找不到 thetao 文件：{var_dir}")

    items = build_day_level_index(theta_files, label_dir, max_days=MAX_DAYS)
    if not items:
        raise RuntimeError(f"❌ n={n_coarse} 没有任何可用 item，请检查 label_dir={label_dir}")

    file_ids = np.array([it["file_id"] for it in items], dtype=int)
    unique_files = np.sort(np.unique(file_ids))
    split_idx = int(0.7 * len(unique_files))
    train_files = set(unique_files[:split_idx])
    val_files   = set(unique_files[split_idx:])

    train_items = [it for it in items if it["file_id"] in train_files]
    val_items   = [it for it in items if it["file_id"] in val_files]

    log(f"Items: {len(items)} | Train items: {len(train_items)} | Val items: {len(val_items)}")
    log(f"Days: {len(unique_files)} | Train days: {len(train_files)} | Val days: {len(val_files)}")

    X_mean, X_std, Y_mean, Y_std = estimate_norm_from_train(train_items, n_samples=2_000_000, seed=0)

    train_ds = LazyLevPatchDataset(
        train_items, X_mean, X_std, Y_mean, Y_std,
        patch_size=patch_size, seed=101, patches_per_item=train_patches_per_item
    )
    val_ds = LazyLevPatchDataset(
        val_items, X_mean, X_std, Y_mean, Y_std,
        patch_size=patch_size, seed=102, patches_per_item=1
    )

    train_loader = _make_loader(
        train_ds,
        batch_size=BATCH_SIZE_IMG_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = _make_loader(
        val_ds,
        batch_size=BATCH_SIZE_IMG_TRAIN,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_full_ds = LazyLevFullEvalDataset(val_items, X_mean, X_std, Y_mean, Y_std)
    val_full_loader = DataLoader(
        val_full_ds,
        batch_size=BATCH_SIZE_IMG_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    K_T, K_S = fit_scalar_K_stream(train_items, max_samples_per_item=150_000, seed=0)
    log(f"✅ baseline K fit (physical): K_T={K_T:.3e}, K_S={K_S:.3e}")

    K_vec_n = fit_normalized_prior_Kvec(
        train_items=train_items,
        X_mean=X_mean, X_std=X_std,
        Y_mean=Y_mean, Y_std=Y_std,
        max_total_samples=SENS_MAX_K_SAMPLES,
        seed=0
    )
    log(f"✅ normalized prior K_vec_n: {K_vec_n}")

    return {
        "n_coarse": n_coarse,
        "train_items": train_items,
        "val_items": val_items,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "val_full_loader": val_full_loader,
        "K_T": float(K_T),
        "K_S": float(K_S),
        "K_vec_n": K_vec_n,
    }


# =========================
# Helper: close dataset cache
# =========================
def close_scale_context_for_sensitivity(ctx):
    for k in ["train_ds", "val_ds"]:
        ds = ctx.get(k, None)
        if ds is not None:
            try:
                ds.close_cache()
            except Exception:
                pass


# =========================
# Evaluation for single model only
# Adds diagnostics for sensitivity experiments
# =========================
def evaluate_single_model_stream(
    model_name: str,
    model: nn.Module,
    val_loader_full: DataLoader,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    out_dir_fig: str,
    n_coarse: int,
    qs_sample_cap: int = 300_000,
):
    os.makedirs(out_dir_fig, exist_ok=True)

    stats = [OnlineStats1D() for _ in range(4)]

    down_T_pos = 0
    down_T_n   = 0
    down_S_pos = 0
    down_S_n   = 0

    rng = np.random.default_rng(123)
    qs_samples = []

    t0 = time.perf_counter()

    for batch in val_loader_full:
        Xn, Y_phys, ocean, file_id, lev, X_phys = batch

        Xn = Xn.to(DEVICE)
        ocean2d = ocean.numpy()[0]
        Yt = Y_phys.numpy()[0]
        Xp = X_phys.numpy()[0]

        m2d = ocean2d & np.isfinite(Yt[0]) & np.isfinite(Yt[1]) & np.isfinite(Yt[2]) & np.isfinite(Yt[3])
        if m2d.sum() == 0:
            continue

        if (EVAL_TILE and EVAL_TILE > 0) and (model_name in TILED_METHODS):
            yp = predict_cnn_full_tiled(model, Xn, Y_mean, Y_std, tile=EVAL_TILE, overlap=EVAL_OVERLAP)[0]
        else:
            model.eval()
            with torch.no_grad():
                with autocast_ctx():
                    yp_n = model(Xn).detach()
            yp = (
                yp_n.float().cpu().numpy() * Y_std.reshape(1, 4, 1, 1) +
                Y_mean.reshape(1, 4, 1, 1)
            )[0].astype(np.float32, copy=False)

        yp[:, ~m2d] = np.nan

        yt_flat = Yt[:, m2d]
        yp_flat = yp[:, m2d]

        for c in range(4):
            y = yt_flat[c].reshape(-1)
            p = yp_flat[c].reshape(-1)
            good = np.isfinite(y) & np.isfinite(p)
            if np.any(good):
                stats[c].update(y[good], p[good])

        # down-gradient fraction
        dTdx = Xp[4][m2d]
        dTdy = Xp[5][m2d]
        dSdx = Xp[6][m2d]
        dSdy = Xp[7][m2d]

        qTx = yp[0][m2d]
        qTy = yp[1][m2d]
        qSx = yp[2][m2d]
        qSy = yp[3][m2d]

        prodT = -(qTx * dTdx + qTy * dTdy)
        prodS = -(qSx * dSdx + qSy * dSdy)

        prodT = prodT[np.isfinite(prodT)]
        prodS = prodS[np.isfinite(prodS)]

        if prodT.size > 0:
            down_T_pos += int(np.sum(prodT > 0.0))
            down_T_n   += int(prodT.size)
        if prodS.size > 0:
            down_S_pos += int(np.sum(prodS > 0.0))
            down_S_n   += int(prodS.size)

        # qS magnitude sample for stability diagnosis
        if len(qs_samples) < qs_sample_cap:
            qs_mag = np.sqrt(qSx*qSx + qSy*qSy)
            qs_mag = qs_mag[np.isfinite(qs_mag)]
            if qs_mag.size > 0:
                take = min(5000, qs_mag.size, qs_sample_cap - len(qs_samples))
                if take > 0:
                    sel = rng.choice(qs_mag.size, size=take, replace=False)
                    qs_samples.extend(list(qs_mag[sel]))

    mm = {}
    rmse_list, corr_list, r2_list = [], [], []

    for c, name in enumerate(flux_names):
        out = stats[c].finalize()
        mm[name] = out
        rmse_list.append(out["rmse"])
        corr_list.append(out["corr"])
        r2_list.append(out["r2"])

    mm["mean"] = {
        "rmse": float(np.nanmean(rmse_list)),
        "corr": float(np.nanmean(corr_list)),
        "r2":   float(np.nanmean(r2_list)),
    }

    mm["diagnostics"] = {
        "down_grad_frac_T": float(down_T_pos / down_T_n) if down_T_n > 0 else np.nan,
        "down_grad_frac_S": float(down_S_pos / down_S_n) if down_S_n > 0 else np.nan,
        "qS_mag_p99": float(np.quantile(np.array(qs_samples, dtype=np.float32), 0.99)) if len(qs_samples) > 10 else np.nan,
        "qS_mag_p999": float(np.quantile(np.array(qs_samples, dtype=np.float32), 0.999)) if len(qs_samples) > 10 else np.nan,
    }

    log(f"✅ Single-model eval done | {model_name} | n={n_coarse} | time={time.perf_counter()-t0:.1f}s")
    return mm


# =========================
# Single sensitivity setting: train only SCAPEX
# =========================
def run_scapex_sensitivity_one_setting_from_context(
    ctx: dict,
    result_dir: str,
    *,
    alpha_s_max=None,
    lam_diss=None,
    lam_lf=None,
    lam_div=None,
    tag_suffix="",
):
    global LOGGER
    LOGGER = setup_logger(result_dir, name=f"scapex_sens_n{ctx['n_coarse']}")

    os.makedirs(result_dir, exist_ok=True)
    fig_dir  = os.path.join(result_dir, "figures")
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    n_coarse = ctx["n_coarse"]
    defaults = get_default_scapex_params(n_coarse)

    alpha_s_max_use = float(alpha_s_max) if alpha_s_max is not None else defaults["alpha_s_max"]
    lam_diss_use    = float(lam_diss)    if lam_diss    is not None else defaults["lam_diss"]
    lam_lf_use      = float(lam_lf)      if lam_lf      is not None else defaults["lam_lf"]
    lam_div_use     = float(lam_div)     if lam_div     is not None else defaults["lam_div"]

    log("\n----------------------------------------")
    log(f"SCAPEX sensitivity setting | n={n_coarse}")
    log(f"alpha_s_max = {alpha_s_max_use}")
    log(f"lam_diss    = {lam_diss_use}")
    log(f"lam_lf      = {lam_lf_use}")
    log(f"lam_div     = {lam_div_use}")
    log("----------------------------------------\n")

    scapex = SCAPEXNet(
        n_coarse=n_coarse,
        X_mean=ctx["X_mean"], X_std=ctx["X_std"],
        Y_mean=ctx["Y_mean"], Y_std=ctx["Y_std"],
        K_vec_n=ctx["K_vec_n"],
        width=SCAPEX_WIDTH,
        n_blocks=SCAPEX_NBLOCKS,
        afno_depth=SCAPEX_AFNO_DEPTH,
        alpha_s_max=alpha_s_max_use,
    )

    tag = f"SCAPEX_SENS{tag_suffix}"

    scapex = train_schopnet_plus(
        tag=tag,
        model=scapex,
        train_loader=ctx["train_loader"],
        val_loader=ctx["val_loader"],
        n_epochs=N_EPOCHS_SCAPEX_SENS,
        ckpt_dir=ckpt_dir,
        lr=LR_SCAPEX,
        clip=CLIP_GRAD,
        lam_diss=lam_diss_use,
        diss_warmup=defaults["diss_warmup"],
        lam_hf_T=defaults["lam_hf_T"],
        lam_hf_S=defaults["lam_hf_S"],
        use_hf=defaults["use_hf"],
        lam_lf=lam_lf_use,
        lf_k0=SCHOP_LF_K0,
        lf_p=SCHOP_LF_P,
        lam_div=lam_div_use,
    )

    metrics = evaluate_single_model_stream(
        model_name="SCAPEX",
        model=scapex,
        val_loader_full=ctx["val_full_loader"],
        Y_mean=ctx["Y_mean"],
        Y_std=ctx["Y_std"],
        out_dir_fig=fig_dir,
        n_coarse=n_coarse,
    )

    out_json = os.path.join(result_dir, "metrics_scapex_sensitivity.json")
    with open(out_json, "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    log(f"Saved: {out_json}")

    return metrics


# =========================
# Plot sensitivity results
# =========================
def plot_sensitivity_results(rows, out_dir, sweep_name, param_name, n_coarse):
    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda x: x["value"])
    vals = [r["value"] for r in rows_sorted]

    r2_mean4 = [r["r2_mean4"] for r in rows_sorted]
    r2_qSx   = [r["r2_qSx"] for r in rows_sorted]
    r2_qSy   = [r["r2_qSy"] for r in rows_sorted]

    rmse_mean4 = [r["rmse_mean4"] for r in rows_sorted]
    dgT = [r["down_grad_frac_T"] for r in rows_sorted]
    dgS = [r["down_grad_frac_S"] for r in rows_sorted]
    qs99 = [r["qS_mag_p99"] for r in rows_sorted]

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    axs[0].plot(vals, r2_mean4, marker="o", label="mean(4) R2")
    axs[0].plot(vals, r2_qSx,   marker="o", label="qS_x R2")
    axs[0].plot(vals, r2_qSy,   marker="o", label="qS_y R2")
    axs[0].set_xlabel(param_name)
    axs[0].set_ylabel("R2")
    axs[0].set_title(f"{sweep_name}: skill sensitivity | n={n_coarse}")
    axs[0].legend(frameon=True)

    axs[1].plot(vals, rmse_mean4, marker="o", label="mean(4) RMSE")
    axs[1].plot(vals, dgT,        marker="o", label="down-grad frac T")
    axs[1].plot(vals, dgS,        marker="o", label="down-grad frac S")
    axs[1].plot(vals, qs99,       marker="o", label="qS |mag| p99")
    axs[1].set_xlabel(param_name)
    axs[1].set_ylabel("Value")
    axs[1].set_title(f"{sweep_name}: diagnostic sensitivity | n={n_coarse}")
    axs[1].legend(frameon=True)

    savefig_tight(fig, os.path.join(out_dir, f"{sweep_name}_{param_name}_n{n_coarse}.png"))


# =========================
# Generic sweep runner
# =========================
def run_scapex_param_sweep(
    sweep_name: str,
    n_coarse: int,
    values,
    param_name: str,
    var_dir: str,
    label_dir: str,
    result_root: str,
):
    os.makedirs(result_root, exist_ok=True)

    prep_dir = os.path.join(result_root, "_prepare")
    os.makedirs(prep_dir, exist_ok=True)

    ctx = prepare_scale_context_for_sensitivity(
        n_coarse=n_coarse,
        var_dir=var_dir,
        label_dir=label_dir,
        result_dir=prep_dir,
        patch_size=PATCH_SIZE,
        train_patches_per_item=TRAIN_PATCHES_PER_ITEM_SENS,
    )

    rows = []

    try:
        for v in values:
            safe_v = str(v).replace(".", "p")
            setting_name = f"{param_name}_{safe_v}"
            setting_dir = os.path.join(result_root, setting_name)
            os.makedirs(setting_dir, exist_ok=True)

            kwargs = {
                "alpha_s_max": None,
                "lam_diss": None,
                "lam_lf": None,
                "lam_div": None,
            }
            kwargs[param_name] = float(v)

            metrics = run_scapex_sensitivity_one_setting_from_context(
                ctx=ctx,
                result_dir=setting_dir,
                alpha_s_max=kwargs["alpha_s_max"],
                lam_diss=kwargs["lam_diss"],
                lam_lf=kwargs["lam_lf"],
                lam_div=kwargs["lam_div"],
                tag_suffix=f"_{param_name}_{safe_v}",
            )

            row = {
                "sweep": sweep_name,
                "n": int(n_coarse),
                "param": param_name,
                "value": float(v),

                "r2_mean4": metrics["mean"]["r2"],
                "rmse_mean4": metrics["mean"]["rmse"],
                "corr_mean4": metrics["mean"]["corr"],

                "r2_qTx": metrics["qT_x"]["r2"],
                "r2_qTy": metrics["qT_y"]["r2"],
                "r2_qSx": metrics["qS_x"]["r2"],
                "r2_qSy": metrics["qS_y"]["r2"],

                "rmse_qTx": metrics["qT_x"]["rmse"],
                "rmse_qTy": metrics["qT_y"]["rmse"],
                "rmse_qSx": metrics["qS_x"]["rmse"],
                "rmse_qSy": metrics["qS_y"]["rmse"],

                "down_grad_frac_T": metrics["diagnostics"]["down_grad_frac_T"],
                "down_grad_frac_S": metrics["diagnostics"]["down_grad_frac_S"],
                "qS_mag_p99": metrics["diagnostics"]["qS_mag_p99"],
                "qS_mag_p999": metrics["diagnostics"]["qS_mag_p999"],
            }
            rows.append(row)

            out_partial = os.path.join(result_root, f"{sweep_name}_{param_name}_n{n_coarse}_partial.csv")
            write_rows_csv(
                out_partial,
                rows,
                fieldnames=[
                    "sweep", "n", "param", "value",
                    "r2_mean4", "rmse_mean4", "corr_mean4",
                    "r2_qTx", "r2_qTy", "r2_qSx", "r2_qSy",
                    "rmse_qTx", "rmse_qTy", "rmse_qSx", "rmse_qSy",
                    "down_grad_frac_T", "down_grad_frac_S",
                    "qS_mag_p99", "qS_mag_p999"
                ]
            )
            log(f"Saved partial csv: {out_partial}")

    finally:
        close_scale_context_for_sensitivity(ctx)

    out_csv = os.path.join(result_root, f"{sweep_name}_{param_name}_n{n_coarse}.csv")
    write_rows_csv(
        out_csv,
        rows,
        fieldnames=[
            "sweep", "n", "param", "value",
            "r2_mean4", "rmse_mean4", "corr_mean4",
            "r2_qTx", "r2_qTy", "r2_qSx", "r2_qSy",
            "rmse_qTx", "rmse_qTy", "rmse_qSx", "rmse_qSy",
            "down_grad_frac_T", "down_grad_frac_S",
            "qS_mag_p99", "qS_mag_p999"
        ]
    )
    log(f"Saved final csv: {out_csv}")

    plot_sensitivity_results(
        rows=rows,
        out_dir=result_root,
        sweep_name=sweep_name,
        param_name=param_name,
        n_coarse=n_coarse
    )

    return rows


# =========================
# One-click runners for four sensitivity experiments
# =========================
def run_sensitivity_alpha_smax():
    n = 3
    label_dir = resolve_label_dir_for_n(LABEL_ROOT, n)
    result_root = os.path.join(RESULT_ROOT, "sensitivity_alpha_smax_n3")
    return run_scapex_param_sweep(
        sweep_name="sensitivity",
        n_coarse=n,
        values=SENS_ALPHA_S_MAX_LIST,
        param_name="alpha_s_max",
        var_dir=VAR_DIR,
        label_dir=label_dir,
        result_root=result_root,
    )


def run_sensitivity_lam_diss():
    n = 5
    label_dir = resolve_label_dir_for_n(LABEL_ROOT, n)
    result_root = os.path.join(RESULT_ROOT, "sensitivity_lam_diss_n5")
    return run_scapex_param_sweep(
        sweep_name="sensitivity",
        n_coarse=n,
        values=SENS_LAM_DISS_LIST,
        param_name="lam_diss",
        var_dir=VAR_DIR,
        label_dir=label_dir,
        result_root=result_root,
    )


def run_sensitivity_lam_lf():
    n = 10
    label_dir = resolve_label_dir_for_n(LABEL_ROOT, n)
    result_root = os.path.join(RESULT_ROOT, "sensitivity_lam_lf_n10")
    return run_scapex_param_sweep(
        sweep_name="sensitivity",
        n_coarse=n,
        values=SENS_LAM_LF_LIST,
        param_name="lam_lf",
        var_dir=VAR_DIR,
        label_dir=label_dir,
        result_root=result_root,
    )


def run_sensitivity_lam_div():
    n = 3
    label_dir = resolve_label_dir_for_n(LABEL_ROOT, n)
    result_root = os.path.join(RESULT_ROOT, "exp_6_sensitivity_lam_div_n3")
    return run_scapex_param_sweep(
        sweep_name="sensitivity",
        n_coarse=n,
        values=SENS_LAM_DIV_LIST,
        param_name="lam_div",
        var_dir=VAR_DIR,
        label_dir=label_dir,
        result_root=result_root,
    )


# =========================
# Master sensitivity main
# =========================
def main_sensitivity():
    os.makedirs(RESULT_ROOT, exist_ok=True)

    log("========================================")
    log("✅ Start SCAPEX parameter sensitivity experiments")
    log(f"DEVICE={DEVICE} | USE_AMP={USE_AMP}")
    log(f"N_EPOCHS_SCAPEX_SENS={N_EPOCHS_SCAPEX_SENS}")
    log(f"TRAIN_PATCHES_PER_ITEM_SENS={TRAIN_PATCHES_PER_ITEM_SENS}")
    log("========================================")

    # 1) alpha_s_max @ n=3
    run_sensitivity_alpha_smax()

    # 2) lam_diss @ n=5
    run_sensitivity_lam_diss()

    # 3) lam_lf @ n=10
    run_sensitivity_lam_lf()

    # 4) lam_div @ n=3
    run_sensitivity_lam_div()

    print("✅ All sensitivity experiments finished.")


# =========================
# Optional: single quick run entry
# Change SENS_RUN_MODE below if you only want one sweep
# =========================
SENS_RUN_MODE = os.environ.get("SENS_RUN_MODE", "all").lower()
# available:
#   "all"
#   "alpha_s_max"
#   "lam_diss"
#   "lam_lf"
#   "lam_div"


def main_sensitivity_dispatch():
    if SENS_RUN_MODE == "all":
        main_sensitivity()
    elif SENS_RUN_MODE == "alpha_s_max":
        run_sensitivity_alpha_smax()
    elif SENS_RUN_MODE == "lam_diss":
        run_sensitivity_lam_diss()
    elif SENS_RUN_MODE == "lam_lf":
        run_sensitivity_lam_lf()
    elif SENS_RUN_MODE == "lam_div":
        run_sensitivity_lam_div()
    else:
        raise ValueError(f"Unknown SENS_RUN_MODE={SENS_RUN_MODE}")


# ============================================================
# Replace your original __main__ block with this:
# RUN_MODE=main         -> run original benchmark
# RUN_MODE=sensitivity  -> run sensitivity experiment(s)
# ============================================================
RUN_MODE = os.environ.get("RUN_MODE", "main").lower()

if __name__ == "__main__":
    if RUN_MODE == "main":
        main()
    elif RUN_MODE == "sensitivity":
        main_sensitivity_dispatch()
    else:
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE}")
