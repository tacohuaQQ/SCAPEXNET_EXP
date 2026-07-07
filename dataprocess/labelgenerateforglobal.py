#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多尺度（不同粗化因子 n）生成 SGS label：
- 对每个 n 生成一份 label，并输出到独立文件夹：label_{n}
- label 定义：q = <uφ> - <u><φ>（只在 j/i 上 coarse-grain）
- 改进点：每层 mask、可选面积加权、输出 ocean_fraction、输出 dx/dy/area、输出 coarse 背景场

用法：
1) 修改 DATA_DIR_IN / DATA_DIR_OUT_BASE
2) 设置 N_LIST = [3, 5, 10] 或你想要的 n
3) 直接运行
"""

import os
import glob
import numpy as np
import xarray as xr

# =====================
# 路径配置
# =====================
DATA_DIR_IN = "/home/Data/zhufuhua/MyData/para2/data/global"
DATA_DIR_OUT_BASE = "/home/Data/zhufuhua/MyData/para2/data/global/thelabel"
os.makedirs(DATA_DIR_OUT_BASE, exist_ok=True)

# =====================
# 你要跑的粗化因子列表
# =====================
N_LIST = [3, 5, 10]   # 例如：只要 10 天数据也没问题，脚本按目录文件数来

# =====================
# 地球半径
# =====================
EARTH_RADIUS = 6371e3  # m

# =====================
# 选项：是否用面积加权 coarse mean
# =====================
USE_AREA_WEIGHTED = True

# 粗格点海洋有效比例阈值（建议 0.5~0.8）
MIN_OCEAN_FRAC = 0.5


def calc_dx_dy_2d(lon2d: np.ndarray, lat2d: np.ndarray):
    """
    根据 2D 经纬度计算物理网格间距 dx/dy（单位 m）
    - lon unwrap 沿 i 方向避免跨经线跳变
    - dx, dy 取绝对值（仅用于梯度尺度，不含方向）
    """
    lon_rad = np.deg2rad(lon2d)
    lon_rad = np.unwrap(lon_rad, axis=1)
    lat_rad = np.deg2rad(lat2d)

    # forward diff
    dlon = lon_rad[:, 1:] - lon_rad[:, :-1]          # (j, i-1)
    dx = EARTH_RADIUS * np.cos(lat_rad[:, :-1]) * dlon

    dlat = lat_rad[1:, :] - lat_rad[:-1, :]          # (j-1, i)
    dy = EARTH_RADIUS * dlat

    # pad to (j,i)
    dx_full = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
    dy_full = np.pad(dy, ((0, 1), (0, 0)), mode="edge")

    dx_full = np.abs(dx_full)
    dy_full = np.abs(dy_full)

    # 防止出现 0 导致后续除法 inf
    dx_full = np.maximum(dx_full, 1.0)
    dy_full = np.maximum(dy_full, 1.0)

    return dx_full, dy_full


def coarse_mean_simple(da: xr.DataArray, factor_j: int, factor_i: int) -> xr.DataArray:
    """在 j, i 上做 block 平均（非加权），跳过 NaN。"""
    return da.coarsen(j=factor_j, i=factor_i, boundary="trim").mean(skipna=True)


def coarse_mean_weighted(da: xr.DataArray, w2d: xr.DataArray, factor_j: int, factor_i: int) -> xr.DataArray:
    """
    在 j,i 上做面积加权平均：
        <X> = sum(w * X) / sum(w)
    w2d 是 (j,i) 的权重（例如 cell area），会自动 broadcast 到 da 的其它维。
    """
    m = xr.where(np.isfinite(da), 1.0, np.nan)
    w_eff = w2d * m

    num = (da * w_eff).coarsen(j=factor_j, i=factor_i, boundary="trim").sum(skipna=True)
    den = (w_eff).coarsen(j=factor_j, i=factor_i, boundary="trim").sum(skipna=True)

    return num / den


def coarse_mask_fraction(mask_bool: xr.DataArray, w2d: xr.DataArray | None,
                         factor_j: int, factor_i: int) -> xr.DataArray:
    """
    计算 coarse 后的有效比例（0~1），mask_bool 是 True/False。
    - 若提供 w2d，则做加权比例：sum(w*mask)/sum(w)
    - 否则普通比例：mean(mask)
    """
    mf = mask_bool.astype("float32")
    if w2d is None:
        return mf.coarsen(j=factor_j, i=factor_i, boundary="trim").mean(skipna=True)

    num = (mf * w2d).coarsen(j=factor_j, i=factor_i, boundary="trim").sum(skipna=True)
    den = (w2d).coarsen(j=factor_j, i=factor_i, boundary="trim").sum(skipna=True)
    return num / den


def build_flux_label_for_one_day(theta_path: str, factor: int, out_dir: str):
    """
    对单个 thetao 文件（对应同日期的 uo/vo/so）生成 label，并写入 out_dir
    """
    base = os.path.basename(theta_path)
    uo_path = theta_path.replace("thetao_", "uo_")
    vo_path = theta_path.replace("thetao_", "vo_")
    so_path = theta_path.replace("thetao_", "so_")

    if not (os.path.exists(uo_path) and os.path.exists(vo_path) and os.path.exists(so_path)):
        print(f"⚠️ 找不到对应的 uo/vo/so 文件，跳过:\n  {base}")
        return

    print(f"\n[n={factor}] 📂 处理文件组：{base}")

    ds_theta = xr.open_dataset(theta_path)
    ds_uo    = xr.open_dataset(uo_path)
    ds_vo    = xr.open_dataset(vo_path)
    ds_so    = xr.open_dataset(so_path)

    try:
        theta = ds_theta["thetao"].load()  # (time, lev, j, i)
        uo    = ds_uo["uo"].load()
        vo    = ds_vo["vo"].load()
        so    = ds_so["so"].load()

        lat = ds_theta["latitude"].load()   # (j,i)
        lon = ds_theta["longitude"].load()  # (j,i)

        # 对齐四个变量
        theta, uo, vo, so = xr.align(theta, uo, vo, so, join="inner")

        # 对齐 lat/lon 到同一 j/i
        if set(lat.dims) == {"j", "i"}:
            lat = lat.sel(j=theta["j"], i=theta["i"])
            lon = lon.sel(j=theta["j"], i=theta["i"])
        else:
            raise RuntimeError("latitude/longitude 维度不是 (j,i)，请检查数据文件。")

        time = theta["time"]
        lev  = theta["lev"]
        j_vals = theta["j"].values
        i_vals = theta["i"].values

        print(
            f"  ✅ aligned dims: time={theta.sizes.get('time')}, lev={theta.sizes.get('lev')}, "
            f"j={theta.sizes.get('j')}, i={theta.sizes.get('i')}"
        )

        # 1) 每层有效 mask
        valid_mask_4d = np.isfinite(theta) & np.isfinite(uo) & np.isfinite(vo) & np.isfinite(so)
        theta = theta.where(valid_mask_4d)
        uo    = uo.where(valid_mask_4d)
        vo    = vo.where(valid_mask_4d)
        so    = so.where(valid_mask_4d)

        # 2) HR 乘积
        fluxT_u_hr = uo * theta
        fluxT_v_hr = vo * theta
        fluxS_u_hr = uo * so
        fluxS_v_hr = vo * so

        # 3) HR 面积权重
        dx_hr, dy_hr = calc_dx_dy_2d(lon.values, lat.values)
        area_hr = xr.DataArray(
            (dx_hr * dy_hr).astype("float32"),
            coords={"j": theta["j"], "i": theta["i"]},
            dims=("j", "i"),
        )

        # 4) coarse-grain
        fj = factor
        fi = factor

        if USE_AREA_WEIGHTED:
            theta_lr   = coarse_mean_weighted(theta,      area_hr, fj, fi)
            so_lr      = coarse_mean_weighted(so,         area_hr, fj, fi)
            uo_lr      = coarse_mean_weighted(uo,         area_hr, fj, fi)
            vo_lr      = coarse_mean_weighted(vo,         area_hr, fj, fi)
            fluxT_u_lr = coarse_mean_weighted(fluxT_u_hr, area_hr, fj, fi)
            fluxT_v_lr = coarse_mean_weighted(fluxT_v_hr, area_hr, fj, fi)
            fluxS_u_lr = coarse_mean_weighted(fluxS_u_hr, area_hr, fj, fi)
            fluxS_v_lr = coarse_mean_weighted(fluxS_v_hr, area_hr, fj, fi)
            lat_lr     = coarse_mean_weighted(lat,        area_hr, fj, fi)
            lon_lr     = coarse_mean_weighted(lon,        area_hr, fj, fi)
        else:
            theta_lr   = coarse_mean_simple(theta,      fj, fi)
            so_lr      = coarse_mean_simple(so,         fj, fi)
            uo_lr      = coarse_mean_simple(uo,         fj, fi)
            vo_lr      = coarse_mean_simple(vo,         fj, fi)
            fluxT_u_lr = coarse_mean_simple(fluxT_u_hr, fj, fi)
            fluxT_v_lr = coarse_mean_simple(fluxT_v_hr, fj, fi)
            fluxS_u_lr = coarse_mean_simple(fluxS_u_hr, fj, fi)
            fluxS_v_lr = coarse_mean_simple(fluxS_v_hr, fj, fi)
            lat_lr     = lat.coarsen(j=fj, i=fi, boundary="trim").mean(skipna=True)
            lon_lr     = lon.coarsen(j=fj, i=fi, boundary="trim").mean(skipna=True)

        # 5) 粗网格坐标（取每个 block 的第一个 index）
        nj_trim = (j_vals.size // fj) * fj
        ni_trim = (i_vals.size // fi) * fi
        j_vals_trim = j_vals[:nj_trim].reshape(-1, fj)[:, 0]
        i_vals_trim = i_vals[:ni_trim].reshape(-1, fi)[:, 0]

        assert theta_lr.sizes["j"] == len(j_vals_trim)
        assert theta_lr.sizes["i"] == len(i_vals_trim)

        def _assign_ji(da: xr.DataArray) -> xr.DataArray:
            return da.assign_coords(j=("j", j_vals_trim), i=("i", i_vals_trim))

        theta_lr   = _assign_ji(theta_lr)
        so_lr      = _assign_ji(so_lr)
        uo_lr      = _assign_ji(uo_lr)
        vo_lr      = _assign_ji(vo_lr)
        fluxT_u_lr = _assign_ji(fluxT_u_lr)
        fluxT_v_lr = _assign_ji(fluxT_v_lr)
        fluxS_u_lr = _assign_ji(fluxS_u_lr)
        fluxS_v_lr = _assign_ji(fluxS_v_lr)
        lat_lr     = _assign_ji(lat_lr)
        lon_lr     = _assign_ji(lon_lr)

        # 6) SGS 通量
        qT_x = fluxT_u_lr - uo_lr * theta_lr
        qT_y = fluxT_v_lr - vo_lr * theta_lr
        qS_x = fluxS_u_lr - uo_lr * so_lr
        qS_y = fluxS_v_lr - vo_lr * so_lr

        # 7) ocean_fraction
        mask_theta = np.isfinite(theta)
        if USE_AREA_WEIGHTED:
            ocean_frac = coarse_mask_fraction(mask_theta.astype(bool), area_hr, fj, fi)
        else:
            ocean_frac = mask_theta.astype("float32").coarsen(j=fj, i=fi, boundary="trim").mean(skipna=True)

        ocean_frac = _assign_ji(ocean_frac)

        if MIN_OCEAN_FRAC is not None:
            good = ocean_frac >= float(MIN_OCEAN_FRAC)
            qT_x = qT_x.where(good)
            qT_y = qT_y.where(good)
            qS_x = qS_x.where(good)
            qS_y = qS_y.where(good)

        # 8) 粗网格 dx/dy/area
        dx_lr, dy_lr = calc_dx_dy_2d(lon_lr.values, lat_lr.values)
        area_lr = dx_lr * dy_lr

        dx_da = xr.DataArray(dx_lr.astype("float32"), coords={"j": j_vals_trim, "i": i_vals_trim}, dims=("j", "i"))
        dy_da = xr.DataArray(dy_lr.astype("float32"), coords={"j": j_vals_trim, "i": i_vals_trim}, dims=("j", "i"))
        area_da = xr.DataArray(area_lr.astype("float32"), coords={"j": j_vals_trim, "i": i_vals_trim}, dims=("j", "i"))

        dx_da.attrs.update({"long_name": "grid spacing in x-direction", "units": "m"})
        dy_da.attrs.update({"long_name": "grid spacing in y-direction", "units": "m"})
        area_da.attrs.update({"long_name": "grid cell area (approx)", "units": "m2"})

        # 9) 输出 Dataset
        ds_label = xr.Dataset(
            data_vars={
                "qT_x": qT_x.astype("float32"),
                "qT_y": qT_y.astype("float32"),
                "qS_x": qS_x.astype("float32"),
                "qS_y": qS_y.astype("float32"),

                "theta_lr": theta_lr.astype("float32"),
                "so_lr":    so_lr.astype("float32"),
                "uo_lr":    uo_lr.astype("float32"),
                "vo_lr":    vo_lr.astype("float32"),

                "ocean_fraction": ocean_frac.astype("float32"),

                "dx":   dx_da,
                "dy":   dy_da,
                "area": area_da,
            },
            coords={
                "time": time,
                "lev":  lev,
                "j": ("j", j_vals_trim),
                "i": ("i", i_vals_trim),
                "latitude":  (("j", "i"), lat_lr.astype("float32").values),
                "longitude": (("j", "i"), lon_lr.astype("float32").values),
            },
            attrs=dict(ds_theta.attrs),
        )

        ds_label.attrs["title"] = (ds_label.attrs.get("title", "") + " | SGS flux labels + grid metrics").strip()
        ds_label.attrs["sgs_definition"] = "q = <u*phi> - <u><phi>, phi in {T,S}, coarse in (j,i)"
        ds_label.attrs["coarsen_factor_j"] = int(fj)
        ds_label.attrs["coarsen_factor_i"] = int(fi)
        ds_label.attrs["use_area_weighted"] = str(bool(USE_AREA_WEIGHTED))
        ds_label.attrs["min_ocean_fraction"] = float(MIN_OCEAN_FRAC)

        # 10) 保存到对应 label_{n} 文件夹
        out_name = base.replace("thetao_", "label_flux_")
        out_path = os.path.join(out_dir, out_name)

        encoding = {var: {"zlib": True, "dtype": "float32"} for var in ds_label.data_vars}
        ds_label.to_netcdf(out_path, encoding=encoding)

        print(f"  ✅ saved: {out_path}")
        print(f"  ✅ dims: lev={ds_label.dims['lev']}, j={ds_label.dims['j']}, i={ds_label.dims['i']}")

    finally:
        ds_theta.close()
        ds_uo.close()
        ds_vo.close()
        ds_so.close()


def main():
    theta_files = sorted(glob.glob(os.path.join(DATA_DIR_IN, "thetao_Oday_FGOALS-f3-H_omip2_r1i1p1f1_gn_*.nc")))
    print(f"在输入目录找到 {len(theta_files)} 个 thetao 文件。")

    if len(theta_files) == 0:
        raise RuntimeError(f"没有找到 thetao 文件，请检查 DATA_DIR_IN={DATA_DIR_IN}")

    for n in N_LIST:
        out_dir = os.path.join(DATA_DIR_OUT_BASE, f"label_{n}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n==============================")
        print(f"开始生成 n={n} 的 label -> {out_dir}")
        print(f"==============================")

        for theta_path in theta_files:
            build_flux_label_for_one_day(theta_path, factor=n, out_dir=out_dir)

    print("\n✅ 全部完成：已按 n 输出到不同 label_n 文件夹。")


if __name__ == "__main__":
    main()
