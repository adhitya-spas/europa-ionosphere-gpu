"""
reconstruction_main_refactored.py

A tidy, step-by-step main script that mirrors your documented pipeline:
  Step 0) Load data
  Step 1) Build ionosphere
  Step 2) Generate rays (VHF nadir, HF oblique)
  Step 3) Forward model (STEC → dual-freq Δt → VTEC)
  Step 4) Build geometry D (path-length weighted)
  Step 5) Reconstruct (ART / Weighted ART / δ(Δt) branch)
  Step 6) Evaluate & plot

NOTE: This uses YOUR existing functions/types/conventions. No behavior changes.
"""

# ---------------------- Imports ----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sp 

from modify_df import load_mission_df
from ionosphere_design import (
    build_ionosphere, add_gaussian_enhancement
)
from ray_trace_passive import (
    trace_passive_nadir, trace_passive_oblique,
    compute_STEC_along_path, dualfreq_to_VTEC, reconstruct_art, reconstruct_art_sparse
)
from improved_geometry import (
    build_geometry_matrix_weighted_sparse,
    build_geometry_matrix_weighted,
    calculate_measurement_weights,
    weighted_reconstruction_art,
    weighted_reconstruction_art_sparse
)

# ---------------------- Config -----------------------
# VHF & HF radio parameters (same conventions as your current main)
F_C_VHF = 60e6        # VHF center frequency [Hz]
BW_VHF  = 10e6        # VHF bandwidth [Hz]
F_C_HF  = 9e6         # HF center frequency [Hz]
BW_HF   = 1e6         # HF bandwidth [Hz]

# Rays / geometry
NPTS_VHF = 500        # samples along each VHF ray
NPTS_HF  = 500        # samples along each HF ray (match VHF to keep ds small)
SAT_LATS = np.linspace(-6.0, 6.0, 60)  # ground-track latitudes for rays

# HF true incidence angles (from vertical, 0° = nadir)
# THETA_LIST_HF = [30, 40, 50, 60, 70, 80, 85]
THETA_LIST_HF = [50,45,40,35,30,25,20,15,10,5]

# Integration times (seconds), per your current logic
INTEG_TIMES_VHF = 0.12
# INTEG_TIMES_HF_PER_ANGLE = np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
INTEG_TIMES_HF_PER_ANGLE = np.linspace(0.05, 0.15, 20)  # 20 values from 0.05s to 0.15s

# ART settings
N_ITERS = 20
RELAX   = 0.1

# Some constants
C = 3e8  # speed of light [m/s]
K_IONO = 40.3  # ionospheric delay constant in SI: Δt = K_IONO * TEC / (c f^2)

# ---------------------- Step -1: GPU SUBSTITUTES ----------------------
USE_GPU = True
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import scipy.sparse as sp
except Exception:
    USE_GPU = False
    cp = None
    cpx_sparse = None

def build_passive_plot_inputs_GPU(
    iono, lats, alts_m,
    D_hf,                 # geometry (n_hf_rays x n_pix); dense OK, sparse preferred
    integ_hf,             # (n_hf_rays,) integration times [s]
    theta_per_ray,        # (n_hf_rays,) true incidence angles [deg]
    vtec_hf,              # (n_hf_rays,) VTEC estimates from your helper
    delta_t_hf,           # (n_hf_rays,) noisy HF dual-freq delay [s]
    delta_t_vhf,          # (n_vhf_rays,) noisy VHF dual-freq delay [s]
    sat_lats_vhf=np.linspace(-6.0, 6.0, 60),
    hf_origin_lat=None,
    F_C_HF=9e6, BW_HF=1e6, F_C_VHF=60e6,
):
    if not USE_GPU:
        raise RuntimeError("Set USE_GPU=True and ensure CuPy is available.")

    # ---- Push scalars/arrays to GPU (float32 is enough here) ----
    iono_gpu   = cp.asarray(iono, dtype=cp.float32)            # (n_alt, n_lat)
    x_gpu      = iono_gpu.ravel(order="C")                     # (n_pix,)
    integ_gpu  = cp.asarray(integ_hf, dtype=cp.float32)
    theta_gpu  = cp.asarray(theta_per_ray, dtype=cp.float32)
    vtec_hf_gp = cp.asarray(vtec_hf, dtype=cp.float32)
    dt_hf_gpu  = cp.asarray(delta_t_hf, dtype=cp.float32)
    dt_vhf_gpu = cp.asarray(delta_t_vhf, dtype=cp.float32)

    # ---- Geometry to GPU (sparse preferred) ----
    # If D_hf is dense (np.ndarray), convert to GPU sparse CSR to save memory
    # You can also build CSR on CPU (scipy.sparse.csr_matrix) then move to GPU.
    if hasattr(D_hf, "tocsr"):  # already SciPy sparse
        D_hf_gpu = cpx_sparse.csr_matrix(D_hf)
    else:
        # assume dense np.ndarray
        D_hf_gpu = cpx_sparse.csr_matrix(cp.asarray(D_hf))

    # ---- 2) True slant TEC per HF ray: tec_true = D_hf @ x ----
    tec_true_gpu = D_hf_gpu @ x_gpu     # (n_hf_rays,)

    # ---- 3A) Passive TEC estimator absolute error (m^-2) ----
    stec_est_gpu = vtec_hf_gp / cp.cos(cp.deg2rad(theta_gpu))
    delta_tec_abs_gpu = cp.abs(stec_est_gpu - tec_true_gpu)

    # ---- 3B) δ(Δt) residual → TEC → VHF range error ----
    c = cp.asarray(3e8, dtype=cp.float32)
    f1_h = cp.asarray(F_C_HF - 0.5*BW_HF, dtype=cp.float32)
    f2_h = cp.asarray(F_C_HF + 0.5*BW_HF, dtype=cp.float32)
    freq_factor_h = (f1_h**2 * f2_h**2) / (f2_h**2 - f1_h**2)

    # Match a single VHF Δt reference to all HF rays (same convention as your main)
    if hf_origin_lat is None:
        hf_origin_lat = float(sat_lats_vhf[0])
    vhf_idx = int(np.argmin(np.abs(np.asarray(sat_lats_vhf) - hf_origin_lat)))
    dt_vhf_match_gpu = cp.full_like(dt_hf_gpu, dt_vhf_gpu[vhf_idx])
    delta_dt_gpu = -(dt_vhf_match_gpu - dt_hf_gpu)

    tec_residual_gpu = (c * delta_dt_gpu / K_IONO) * freq_factor_h    # m^-2
    range_err_gpu    = (K_IONO / (cp.asarray(F_C_VHF, dtype=cp.float32)**2)) * cp.abs(tec_residual_gpu)

    # ---- Return NumPy for plotting, keep GPU speed for compute ----
    t_meas      = cp.asnumpy(integ_gpu)
    tec_for_bin = cp.asnumpy(tec_true_gpu)
    dtec_abs    = cp.asnumpy(delta_tec_abs_gpu)
    rng_err     = cp.asnumpy(range_err_gpu)
    return t_meas, tec_for_bin, dtec_abs, rng_err

def _gpu_binned_mean_2d(x_cpu, y_cpu, z_cpu, x_edges_cpu, y_edges_cpu):
    if not USE_GPU:
        raise RuntimeError("CU mode only.")
    x = cp.asarray(x_cpu, dtype=cp.float32)
    y = cp.asarray(y_cpu, dtype=cp.float32)
    z = cp.asarray(z_cpu, dtype=cp.float32)
    xb = cp.asarray(x_edges_cpu, dtype=cp.float32)
    yb = cp.asarray(y_edges_cpu, dtype=cp.float32)

    # Weighted sum + counts on GPU
    H, _, _ = cp.histogram2d(y, x, bins=[yb, xb], weights=z)  # note (y,x) order like NumPy example
    C, _, _ = cp.histogram2d(y, x, bins=[yb, xb])
    M = H / cp.maximum(C, 1)

    return cp.asnumpy(M)  # back to CPU for plotting

# ---------------------- Step 0: Load data ----------------------
def step0_load_dataframe(path="new_mission_df.pkl", mission_name="E6a Exit"):
    df = load_mission_df(path)
    row = df[df["Mission"] == mission_name].iloc[0]
    alt_km_1d, Ne_1d = row["Altitude"], row["Ne"]
    return alt_km_1d, Ne_1d

# ---------------------- Step 1: Build ionosphere ----------------
def step1_build_iono(alt_km_1d, Ne_1d,
                     lat_extent=(-10, 10), lat_res=200,
                     add_gaussian=True):
    # Build base map (extrude 1D profile to 2D)
    lats, alt_km, iono = build_ionosphere(
        pd.DataFrame({"altitude": alt_km_1d, "ne": Ne_1d}),
        lat_extent=lat_extent, lat_res=lat_res
    )

    # Optional Gaussian enhancement (same call pattern you already use)
    if add_gaussian:
        iono = add_gaussian_enhancement(
            ionosphere_map=iono,
            latitudes=lats,
            altitude=alt_km,
            lat_center=0.0,
            alt_center=150.0,
            lat_width=1.0,
            alt_width=20.0,
            amplitude=5e11
        )

    alts_m = alt_km * 1e3
    return lats, alt_km, alts_m, iono

# ---------------------- Step 2: Generate rays -------------------
def step2_generate_rays(lats, alts_m):
    # 2A) VHF nadir rays
    rays_vhf = trace_passive_nadir(SAT_LATS, alts_m, npts=NPTS_VHF)
    integ_vhf = np.full(len(rays_vhf), INTEG_TIMES_VHF)

    # 2B) HF oblique rays (true incidence angles from vertical)
    rays_hf_all = []
    theta_per_ray = []
    integ_hf_all = []

    T_PER_ANGLE = np.full(len(THETA_LIST_HF), 0.12, dtype=float)  # e.g. 0.12s


    for idx, theta in enumerate(THETA_LIST_HF):
        rays_hf = trace_passive_oblique(
            sat_lats=SAT_LATS,
            h0=alts_m.max(),
            hsat=alts_m.max(),
            theta_i_deg=theta,       # true incidence from vertical
            npts=NPTS_HF
        )
        rays_hf_all.extend(rays_hf)
        theta_per_ray.extend([theta] * len(rays_hf))
        integ_hf_all.extend([T_PER_ANGLE[idx]] * len(rays_hf))

    theta_per_ray = np.array(theta_per_ray, dtype=float)
    integ_hf_all  = np.array(integ_hf_all, dtype=float)

    return rays_vhf, integ_vhf, rays_hf_all, theta_per_ray, integ_hf_all

# ===== GPU STEP 2: Dense HF sweep (angles, tracks, and T) =====
def step2_generate_rays_gpu_sweep(
    lats, alts_m,
    *,
    # angle sweep (deg from vertical)
    theta_list_hf=None,
    # satellite ground-track latitudes (deg)
    sat_lats=None,
    # per-ray polyline resolution
    npts_hf=500,
    # integration-time sweep (seconds)
    t_sweep=None,
):
    """
    Build a dense set of HF rays by sweeping angles, tracks, and integration times.
    Geometry is independent of T, so we replicate rays for each T.
    Returns:
        rays_hf        : list[np.ndarray] of shape (n_points, 2) = [lat_deg, alt_m]
        theta_per_ray  : (N,) array of true incidence angles (deg from vertical)
        integ_hf       : (N,) array of per-ray integration times (s)
    """
    import numpy as np
    from ray_trace_passive import trace_passive_oblique

    # defaults that match your existing setup if not provided
    if theta_list_hf is None:
        theta_list_hf = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
    if sat_lats is None:
        sat_lats = np.linspace(-6.0, 6.0, 181)  # denser than 60 to broaden TEC coverage
    if t_sweep is None:
        t_sweep = np.linspace(0.05, 0.15, 21)   # dense x-axis

    # ----- Build base geometry once (independent of T) -----
    rays_hf_base = []
    theta_base   = []
    for theta in theta_list_hf:
        rays_this_theta = trace_passive_oblique(
            sat_lats=sat_lats,
            h0=alts_m.max(),
            hsat=alts_m.max(),
            theta_i_deg=theta,
            npts=npts_hf
        )
        rays_hf_base.extend(rays_this_theta)
        theta_base.extend([theta]*len(rays_this_theta))

    rays_hf = []
    theta_per_ray = []
    integ_hf = []

    # ----- Replicate geometry across the T grid -----
    for T in t_sweep:
        rays_hf.extend(rays_hf_base)
        theta_per_ray.extend(theta_base)
        integ_hf.extend([T]*len(rays_hf_base))

    return rays_hf, np.asarray(theta_per_ray, float), np.asarray(integ_hf, float)


# ---------------------- Step 3: Forward model -------------------
def step3_forward_model(iono, lats, alts_m,
                        rays_vhf, integ_vhf,
                        rays_hf,  theta_per_ray, integ_hf):
    # VHF: STEC → Δt (with noise) → VTEC (verticalized inside helper)
    stec_vhf = np.array([
        compute_STEC_along_path(iono, lats, alts_m, ray) for ray in rays_vhf
    ])
    vtec_vhf = []
    delta_t_vhf = []
    for i, ray_stec in enumerate(stec_vhf):
        vtec_i, dt_i = dualfreq_to_VTEC(
            stec_slant=np.array([ray_stec]),
            f_c=F_C_VHF,
            bw=BW_VHF,
            theta_deg=0.0,
            integration_time=integ_vhf[i],
            return_deltat=True
        )
        vtec_vhf.append(vtec_i[0])
        delta_t_vhf.append(dt_i[0])
    vtec_vhf = np.array(vtec_vhf)
    delta_t_vhf = np.array(delta_t_vhf)

    # HF: per-ray STEC → Δt → VTEC
    stec_hf = np.array([
        compute_STEC_along_path(iono, lats, alts_m, ray) for ray in rays_hf
    ])
    vtec_hf, delta_t_hf = dualfreq_to_VTEC(
        stec_slant=stec_hf,
        f_c=F_C_HF,
        bw=BW_HF,
        theta_deg=theta_per_ray,          # vectorized; helper accepts array
        integration_time=integ_hf,
        return_deltat=True
    )

    return vtec_vhf, delta_t_vhf, vtec_hf, delta_t_hf

# ===== GPU STEP 3: Forward model (TEC truth via D_hf @ x on GPU) =====
USE_GPU = True
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except Exception:
    USE_GPU = False
    cp = None
    cpx_sparse = None

def step3_forward_model_gpu(
    iono, lats, alts_m,
    D_hf,                 # geometry matrix for HF rays (SciPy CSR or dense np.ndarray)
    theta_per_ray,        # (N,) deg from vertical
    integ_hf,             # (N,) seconds
    f_c_hf=9e6, bw_hf=1e6,
):
    """
    Compute per-ray slant TEC (truth) on the GPU, then convert to VTEC & Δt
    using your existing CPU helper. Heavy part (D @ x) is GPU-accelerated.

    Returns:
        tec_true  : (N,) slant TEC truth (m^-2)
        vtec_hf   : (N,) verticalized TEC estimates (as your helper outputs)
        delta_t_hf: (N,) dual-frequency group delay with noise (s)
    """
    import numpy as np
    from ray_trace_passive import dualfreq_to_VTEC

    if not USE_GPU:
        raise RuntimeError("CuPy not available; use the CPU path or set USE_GPU=True.")

    # ---- Push ionosphere to GPU and vectorize ----
    iono_gpu = cp.asarray(iono, dtype=cp.float32)          # (n_alt, n_lat)
    x_gpu = iono_gpu.ravel(order="C")                      # (n_pix,)

    # ---- Geometry to GPU (sparse preferred) ----
    if hasattr(D_hf, "tocsr"):  # SciPy sparse
        D_hf_gpu = cpx_sparse.csr_matrix(D_hf)
    else:
        # assume dense np.ndarray
        D_hf_gpu = cpx_sparse.csr_matrix(cp.asarray(D_hf))

    # ---- TEC truth: one shot on GPU ----
    tec_true_gpu = D_hf_gpu @ x_gpu                        # (N,)
    tec_true = cp.asnumpy(tec_true_gpu)                    # back to CPU for helper

    # ---- Convert to VTEC & Δt via your existing helper (vectorized & cheap) ----
    vtec_hf, delta_t_hf = dualfreq_to_VTEC(
        stec_slant=tec_true,
        f_c=f_c_hf,
        bw=bw_hf,
        theta_deg=np.asarray(theta_per_ray, float),
        integration_time=np.asarray(integ_hf, float),
        return_deltat=True
    )
    return tec_true, vtec_hf, delta_t_hf


# ---------------------- Step 4: Build geometry ------------------
def step4_build_geometry(rays_vhf, rays_hf, lats, alts_m):
    # ALWAYS sparse + float32 to avoid OOM
    D_vhf = build_geometry_matrix_weighted_sparse(rays_vhf, lats, alts_m, dtype=np.float32)
    D_hf  = build_geometry_matrix_weighted_sparse(rays_hf,  lats, alts_m, dtype=np.float32)
    return D_vhf, D_hf

# ---------------------- Step 5: Reconstruct ---------------------
def step5_reconstruct(D_vhf, D_hf, vtec_vhf, vtec_hf,
                      delta_t_vhf, delta_t_hf,
                      lats, alts_m,
                      n_iters=N_ITERS, relax=RELAX):
    
    n_lat, n_alt = len(lats), len(alts_m)
    # 5A) ART (VHF-only, HF-only, stacked)
    # VHF-only
    if sp.issparse(D_vhf):
        Ne_rec_vhf = reconstruct_art_sparse(D_vhf, vtec_vhf, n_lat, n_alt, n_iters, relax)
    else:
        Ne_rec_vhf = reconstruct_art(D_vhf, vtec_vhf, n_lat, n_alt, n_iters, relax)

    # HF-only
    if sp.issparse(D_hf):
        Ne_rec_hf = reconstruct_art_sparse(D_hf, vtec_hf, n_lat, n_alt, n_iters, relax)
    else:
        Ne_rec_hf = reconstruct_art(D_hf, vtec_hf, n_lat, n_alt, n_iters, relax)

    # Combined
    D_all = sp.vstack([D_vhf, D_hf], format="csr") if (sp.issparse(D_vhf) or sp.issparse(D_hf)) \
            else np.vstack([D_vhf, D_hf])
    vtec_all = np.concatenate([vtec_vhf, vtec_hf])

    if sp.issparse(D_all):
        Ne_rec_all = reconstruct_art_sparse(D_all, vtec_all, n_lat, n_alt, n_iters, relax)
    else:
        Ne_rec_all = reconstruct_art(D_all, vtec_all, n_lat, n_alt, n_iters, relax)


    # 5B) Weighted ART (weights ∝ √(BW·T·SNR); here SNR assumed 1 unless you pass it)
    integ_all = np.concatenate([
        np.full_like(vtec_vhf, INTEG_TIMES_VHF, dtype=float),
        # NOTE: we don't have the per-ray HF T here yet; caller passes it:
        # We'll return a closure result instead (see wrapper below)
        # or you can pass integ_hf externally into this function.
    ])

    # We'll build weights in the wrapper where we have HF T.
    return Ne_rec_vhf, Ne_rec_hf, Ne_rec_all, D_all, vtec_all

def step5_weighted_reconstruct(D_all, vtec_all, integ_vhf, integ_hf, bw_vhf=BW_VHF, bw_hf=BW_HF,
                               n_lat=None, n_alt=None,
                               n_iters=50, relax=0.2):
    # Build bandwidth & integration arrays aligned with stacked measurements
    all_integration_times = np.concatenate([integ_vhf, integ_hf])
    all_bandwidths = np.concatenate([
        np.full(integ_vhf.shape[0], bw_vhf),
        np.full(integ_hf.shape[0],  bw_hf)
    ])
    weights = calculate_measurement_weights(
        integration_times=all_integration_times,
        bandwidths=all_bandwidths
    )
    if sp.issparse(D_all):
        Ne_rec_weighted = weighted_reconstruction_art_sparse(
            D_all, vtec_all, weights, n_lat, n_alt, n_iters=n_iters, relax=relax
        )
    else:
        Ne_rec_weighted = weighted_reconstruction_art(
            D_all, vtec_all, weights, n_lat, n_alt, n_iters=n_iters, relax=relax
        )
    return Ne_rec_weighted, weights

# ---------------------- Step 6: Evaluate & Plot -----------------
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b)**2)))

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a0, b0 = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    return float((a0 * b0).sum() / denom) if denom > 0 else 0.0

def plot_recons(lats, alts_m, recs, titles):
    fig, axs = plt.subplots(1, len(recs), figsize=(6*len(recs), 6), sharey=True)
    if len(recs) == 1: axs = [axs]
    for ax, rec, title in zip(axs, recs, titles):
        x_edges = centers_to_edges(lats)
        y_edges = centers_to_edges(alts_m/1e3)
        im = ax.pcolormesh(x_edges, y_edges, rec*1e-6, shading='auto', cmap='viridis')

        ax.set_title(title)
        ax.set_xlabel("Latitude (°)")
        if ax is axs[0]:
            ax.set_ylabel("Altitude (km)")
        fig.colorbar(im, ax=ax, label='Ne (×10⁶ cm⁻³)')
    plt.tight_layout(); plt.show(block=False); plt.savefig("img/A_gpu_reconstructions.png")

def plot_raypaths(lats, alts_m, rays_hf, theta_per_ray, rays_vhf,
                  title_hf="Modeled Raypaths (HF)",
                  title_vhf="Modeled Raypaths (VHF)",
                  max_rays_per_angle=None):
    """
    Two-panel figure:
      left  = HF oblique rays, color-coded by true incidence angle (deg from vertical)
      right = VHF nadir rays (vertical)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    # group HF rays by angle (preserve order)
    angle_to_indices = defaultdict(list)
    for i, th in enumerate(theta_per_ray):
        angle_to_indices[float(th)].append(i)

    # Visually nice, ordered by increasing angle (like your example)
    angles_sorted = sorted(angle_to_indices.keys())

    # Pick distinct colors for angles (same order every time)
    # You can tweak the list to match your house style
    palette = [
        "#0015BC",  # deep blue
        "#4527A0",  # indigo
        "#8E24AA",  # purple
        "#D81B60",  # magenta
        "#FB8C00",  # orange
        "#FFC107",  # amber
        "#FDD835",  # yellow
        "#43A047",  # green (backup if needed)
    ]
    # map angle -> color (wrap if more angles)
    angle_color = {ang: palette[i % len(palette)] for i, ang in enumerate(angles_sorted)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    ax_hf, ax_vhf = axes

    # ---- HF panel ----
    for ang in angles_sorted:
        idxs = angle_to_indices[ang]
        if max_rays_per_angle is not None:
            idxs = idxs[:max_rays_per_angle]
        col = angle_color[ang]
        for i in idxs:
            ray = rays_hf[i]
            # ray: (n_points, 2) = [lat_deg, alt_m]
            ax_hf.plot(ray[:, 0], ray[:, 1] / 1e3, lw=0.6, color=col, alpha=0.9)

    # Legend — one entry per angle
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=angle_color[ang], lw=2, label=f"{int(round(ang))}°")
        for ang in angles_sorted
    ]
    leg = ax_hf.legend(
        handles=legend_handles,
        title="HF θ (deg, from vertical)",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        framealpha=0.9,
        fontsize=9,
        title_fontsize=10,
        ncol=1
    )

    ax_hf.set_title(title_hf)
    ax_hf.set_xlabel("Latitude (Degree)")
    ax_hf.set_ylabel("Altitude (km)")
    ax_hf.set_xlim(lats.min(), lats.max())
    ax_hf.set_ylim(0, alts_m.max() / 1e3)
    ax_hf.grid(False)

    # ---- VHF panel ----
    for ray in rays_vhf:
        ax_vhf.plot(ray[:, 0], ray[:, 1] / 1e3, lw=0.6, color="#5C6BC0", alpha=0.9)

    ax_vhf.set_title(title_vhf)
    ax_vhf.set_xlabel("Latitude (Degree)")
    ax_vhf.set_xlim(lats.min(), lats.max())
    ax_vhf.set_ylim(0, alts_m.max() / 1e3)
    ax_vhf.grid(False)

    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("img/B_gpu_raypaths.png")

def build_passive_plot_inputs(
    iono, lats, alts_m,
    rays_hf,                  # list of HF polylines (n_hf_rays)
    integ_hf,                 # (n_hf_rays,) integration times [s]
    theta_per_ray,            # (n_hf_rays,) true incidence angle from vertical [deg]
    vtec_hf,                  # (n_hf_rays,) VTEC estimates returned by your dualfreq helper
    delta_t_hf,               # (n_hf_rays,) noisy HF dual-freq delay [s]
    delta_t_vhf,              # (n_vhf_rays,) noisy VHF dual-freq delay [s]
    sat_lats_vhf=np.linspace(-6.0, 6.0, 60),
    hf_origin_lat=None,   # must match how you built VHF rays
    F_C_HF=9e6, BW_HF=1e6,                 # HF center & bandwidth [Hz]
    F_C_VHF=60e6,                          # VHF center [Hz]
):
    # 1) x-axis: per-HF-ray integration time (seconds)
    t_meas = integ_hf.astype(float)

    # 2) y-axis: true slant TEC per HF ray (m^-2)
    tec_for_bin = np.array([compute_STEC_along_path(iono, lats, alts_m, ray)
                            for ray in rays_hf], dtype=float)

    # 3A) Passive TEC estimator absolute error (m^-2)
    theta_rad = np.deg2rad(theta_per_ray)
    stec_hf_est = vtec_hf / np.cos(theta_rad)           # estimated slant TEC (m^-2)
    delta_tec_err_passive = np.abs(stec_hf_est - tec_for_bin)

    # 3B) Residual TEC after passive HF correction via δ(Δt) → VHF range error (m)
    f1_h = F_C_HF - 0.5 * BW_HF
    f2_h = F_C_HF + 0.5 * BW_HF
    freq_factor_h = (f1_h**2 * f2_h**2) / (f2_h**2 - f1_h**2)

    # Match each HF ray to a single VHF reference Δt (same convention as your delta_dt_branch)
    if hf_origin_lat is None:
        hf_origin_lat = float(sat_lats_vhf[0])
    vhf_idx = int(np.argmin(np.abs(np.asarray(sat_lats_vhf) - hf_origin_lat)))
    delta_t_vhf_matched = np.full_like(delta_t_hf, delta_t_vhf[vhf_idx])

    delta_dt = -(delta_t_vhf_matched - delta_t_hf)                  # seconds
    tec_residual_after_passive = (C * delta_dt / K_IONO) * freq_factor_h   # m^-2
    range_error_after_passive = (K_IONO / (F_C_VHF**2)) * np.abs(tec_residual_after_passive)  # meters

    return t_meas, tec_for_bin, delta_tec_err_passive, range_error_after_passive


def plot_passive_tec_error_heatmap_from_arrays(
    t_meas, tec_for_bin, delta_tec_err_passive,
    t_min=0.05, t_max=0.15, tec_min=1.5e15, tec_max=5e15,
    nbins_t=50, nbins_tec=50,
    title="Average TEC Estimate Error for Passive (Simulated)",
    cmap="plasma"
):
    df = pd.DataFrame({
        "vtec": tec_for_bin,           # y-axis (slant TEC truth)
        "t": t_meas,                   # x-axis (integration time, s)
        "err": delta_tec_err_passive,  # color (|ΔTEC|, m^-2)
    })

    # Filter to match your original figure window
    df = df[(df['t'] >= t_min) & (df['t'] <= t_max)]
    df = df[(df['vtec'] >= tec_min) & (df['vtec'] <= tec_max)]

    tec_bins = np.linspace(tec_min, tec_max, nbins_tec)
    t_bins   = np.linspace(t_min,   t_max,   nbins_t)

    heatmap, yedges, xedges = np.histogram2d(
        df['vtec'], df['t'], bins=[tec_bins, t_bins], weights=df['err'])
    counts,  _,     _       = np.histogram2d(
        df['vtec'], df['t'], bins=[tec_bins, t_bins])

    avg_error = heatmap / np.maximum(counts, 1)

    plt.figure(figsize=(8,6))
    plt.pcolormesh(xedges, yedges, avg_error, shading='auto', cmap=cmap)
    plt.xlabel("Integration Time (s)")
    plt.ylabel("TEC (electrons/m$^{2}$)")
    plt.title(title)
    plt.colorbar(label="ΔTEC Error (m$^{-2}$)")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("img/C_passive_tec_error_heatmap_from_arrays.png")


def plot_vhf_range_error_heatmap_from_arrays(
    t_meas, tec_for_bin, range_error_after_passive,
    t_min=0.05, t_max=0.15, tec_min=1.5e15, tec_max=5e15,
    nbins_t=50, nbins_tec=50,
    title="VHF Range Error after Passive HF Correction",
    cmap="jet"
):
    df_range = pd.DataFrame({
        "vtec": tec_for_bin,                    # y-axis (slant TEC truth)
        "t": t_meas,                            # x-axis (integration time, s)
        "range_err": range_error_after_passive  # color (meters)
    })

    df_range = df_range[(df_range['t'] >= t_min) & (df_range['t'] <= t_max)]
    df_range = df_range[(df_range['vtec'] >= tec_min) & (df_range['vtec'] <= tec_max)]

    tec_bins = np.linspace(tec_min, tec_max, nbins_tec)
    t_bins   = np.linspace(t_min,   t_max,   nbins_t)

    heatmap_r, yedges_r, xedges_r = np.histogram2d(
        df_range['vtec'], df_range['t'],
        bins=[tec_bins, t_bins], weights=df_range['range_err'])
    counts_r, _, _ = np.histogram2d(
        df_range['vtec'], df_range['t'], bins=[tec_bins, t_bins])

    avg_range_err = heatmap_r / np.maximum(counts_r, 1)

    plt.figure(figsize=(8,6))
    plt.pcolormesh(xedges_r, yedges_r, avg_range_err, shading='auto', cmap=cmap)
    plt.xlabel("Integration Time (sec)")
    plt.ylabel("Total Electron Content (m$^{-2}$)")
    plt.title(title)
    plt.colorbar(label="Δr (meters)")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("img/D_vhf_range_error_heatmap.png")

def plot_passive_tec_error_heatmap_GPU(
    t_meas, tec_for_bin, delta_tec_err_passive,
    t_min=0.05, t_max=0.15, tec_min=1.5e15, tec_max=5e15,
    nbins_t=50, nbins_tec=50, title="Average TEC Estimate Error for Passive (Simulated)"
):
    # Filter on CPU (cheap) to keep plotting logic identical
    import pandas as pd
    df = pd.DataFrame({"t": t_meas, "vtec": tec_for_bin, "err": delta_tec_err_passive})
    df = df[(df["t"]>=t_min)&(df["t"]<=t_max)&(df["vtec"]>=tec_min)&(df["vtec"]<=tec_max)]

    t_edges   = np.linspace(t_min,   t_max,   nbins_t)
    tec_edges = np.linspace(tec_min, tec_max, nbins_tec)

    Z = _gpu_binned_mean_2d(df["t"].values, df["vtec"].values, df["err"].values, t_edges, tec_edges)

    plt.figure(figsize=(8,6))
    plt.pcolormesh(t_edges, tec_edges, Z, shading="auto", cmap="plasma")
    plt.xlabel("Integration Time (s)")
    plt.ylabel("TEC (electrons/m$^{2}$)")
    plt.title(title)
    plt.colorbar(label="ΔTEC Error (m$^{-2}$)")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("img/C_gpu_hf_passive_tec_error_heatmap.png")

def plot_vhf_range_error_heatmap_GPU(
    t_meas, tec_for_bin, range_error_after_passive,
    t_min=0.05, t_max=0.15, tec_min=1.5e15, tec_max=5e15,
    nbins_t=50, nbins_tec=50, title="VHF Range Error after Passive HF Correction"
):
    import pandas as pd
    df = pd.DataFrame({"t": t_meas, "vtec": tec_for_bin, "rng": range_error_after_passive})
    df = df[(df["t"]>=t_min)&(df["t"]<=t_max)&(df["vtec"]>=tec_min)&(df["vtec"]<=tec_max)]

    t_edges   = np.linspace(t_min,   t_max,   nbins_t)
    tec_edges = np.linspace(tec_min, tec_max, nbins_tec)

    Z = _gpu_binned_mean_2d(df["t"].values, df["vtec"].values, df["rng"].values, t_edges, tec_edges)

    plt.figure(figsize=(8,6))
    plt.pcolormesh(t_edges, tec_edges, Z, shading="auto", cmap="jet")
    plt.xlabel("Integration Time (sec)")
    plt.ylabel("Total Electron Content (m$^{-2}$)")
    plt.title(title)
    plt.colorbar(label="Δr (meters)")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("img/D_gpu_vhf_range_error_heatmap.png")



# ---------------------- δ(Δt) helper (optional) -----------------
def delta_dt_branch(delta_t_vhf, delta_t_hf, D_hf, lats, alts_m,
                    f_c_hf=F_C_HF, bw_hf=BW_HF,
                    n_iters=N_ITERS, relax=RELAX):
    # Match a single VHF reference to all HF rays (same as your current main)
    origin_lat = None  # first HF ray's starting latitude
    # For simplicity: assume first HF ray corresponds to SAT_LATS[0] origin
    # In your main, you matched by ground track; here we replicate that behavior:

    # Build a VHF array aligned with HF rays by taking the VHF Δt at the origin lat.
    # (Your current code: takes the first HF origin latitude and finds nearest VHF track.)
    # We'll mimic that exactly:

    # Use SAT_LATS[0] as origin; map to nearest VHF index
    sat_lats_vhf = SAT_LATS
    origin_lat = SAT_LATS[0]
    origin_lat_idx = int(np.argmin(np.abs(sat_lats_vhf - origin_lat)))
    delta_t_vhf_matched = np.full(len(delta_t_hf), delta_t_vhf[origin_lat_idx])

    # Convert δ(Δt) to TEC with the HF frequency pair
    c = 3e8
    f1_h = f_c_hf - 0.5*bw_hf
    f2_h = f_c_hf + 0.5*bw_hf
    freq_factor_h = (f1_h**2 * f2_h**2) / (f2_h**2 - f1_h**2)

    delta_dt_all = -(delta_t_vhf_matched - delta_t_hf)
    tec_est_ddt_all = (c * delta_dt_all / K_IONO) * freq_factor_h
    # tec_est_ddt_all = (c * delta_dt_all / 80.6) * freq_factor_h

    Ne_rec_ddt = reconstruct_art_sparse(D_hf, tec_est_ddt_all, len(lats), len(alts_m), n_iters, relax)
    return Ne_rec_ddt

def centers_to_edges(c):
    c = np.asarray(c)
    e = np.empty(c.size + 1, dtype=c.dtype)
    e[1:-1] = 0.5*(c[1:] + c[:-1])
    e[0]     = c[0] - 0.5*(c[1] - c[0])
    e[-1]    = c[-1] + 0.5*(c[-1] - c[-2])
    return e

# ---------------------- Function calls ----------------------------
def run_pipeline():
    # Step 0
    alt_km_1d, Ne_1d = step0_load_dataframe()

    # Step 1
    lats, alt_km, alts_m, iono = step1_build_iono(alt_km_1d, Ne_1d, lat_extent=(-10, 10), lat_res=200)

    # Step 2
    if not USE_GPU:
        rays_vhf, integ_vhf, rays_hf, theta_per_ray, integ_hf = step2_generate_rays(lats, alts_m)
    else:
        rays_hf, theta_per_ray, integ_hf = step2_generate_rays_gpu_sweep(
        lats, alts_m,
        theta_list_hf=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85],  # or your list
        sat_lats=np.linspace(-6.0, 6.0, 181),   # denser tracks for richer TEC
        npts_hf=500,
        t_sweep=np.linspace(0.05, 0.15, 21)     # dense T grid (x-axis)
        )
        rays_vhf = trace_passive_nadir(SAT_LATS, alts_m, npts=NPTS_VHF)
        integ_vhf = np.full(len(rays_vhf), INTEG_TIMES_VHF, dtype=float)

    # Step 3
    if not USE_GPU:
        vtec_vhf, delta_t_vhf, vtec_hf, delta_t_hf = step3_forward_model(
            iono, lats, alts_m, rays_vhf, integ_vhf, rays_hf, theta_per_ray, integ_hf
        )
    else:
        # D_hf = build_geometry_matrix_weighted_sparse(rays_hf, lats, alts_m, dtype=np.float32)
        # D_vhf = build_geometry_matrix_weighted_sparse(rays_vhf, lats, alts_m, dtype=np.float32)
        D_vhf, D_hf = step4_build_geometry(rays_vhf, rays_hf, lats, alts_m)
        print("HF geometry:", D_hf.shape, "nnz=", D_hf.nnz, "density=", D_hf.nnz/(D_hf.shape[0]*D_hf.shape[1]))
        print("VHF geometry:", D_vhf.shape, "nnz=", D_vhf.nnz, "density=", D_vhf.nnz/(D_vhf.shape[0]*D_vhf.shape[1]))
        D_all = scipy.sparse.vstack([D_vhf, D_hf], format="csr")
        # D_all_gpu = cpx_sparse.csr_matrix(D_all)

        tec_true, vtec_hf, delta_t_hf = step3_forward_model_gpu(
        iono=iono, lats=lats, alts_m=alts_m,
        D_hf=D_hf,
        theta_per_ray=theta_per_ray,
        integ_hf=integ_hf,
        f_c_hf=F_C_HF, bw_hf=BW_HF
        )
        # VHF path (CPU; cheap)
        stec_vhf = np.array([compute_STEC_along_path(iono, lats, alts_m, ray) for ray in rays_vhf])
        vtec_vhf, delta_t_vhf = dualfreq_to_VTEC(
            stec_slant=stec_vhf,
            f_c=F_C_VHF,
            bw=BW_VHF,
            theta_deg=0.0,
            integration_time=integ_vhf,
            return_deltat=True
        )

    # Step 4
    if not USE_GPU:
        D_vhf, D_hf = step4_build_geometry(rays_vhf, rays_hf, lats, alts_m)
        print("HF geometry:", D_hf.shape, "nnz=", D_hf.nnz, "density=", D_hf.nnz/(D_hf.shape[0]*D_hf.shape[1]))
        print("VHF geometry:", D_vhf.shape, "nnz=", D_vhf.nnz, "density=", D_vhf.nnz/(D_vhf.shape[0]*D_vhf.shape[1]))


    # Step 5A (ARTs) + collect for weighted
    Ne_rec_vhf, Ne_rec_hf, Ne_rec_all, D_all, vtec_all = step5_reconstruct(
        D_vhf, D_hf, vtec_vhf, vtec_hf, delta_t_vhf, delta_t_hf, lats, alts_m,
        n_iters=N_ITERS, relax=RELAX
    )

    # Step 5B (Weighted ART)
    Ne_rec_weighted, weights = step5_weighted_reconstruct(
        D_all, vtec_all, integ_vhf, integ_hf, bw_vhf=BW_VHF, bw_hf=BW_HF,
        n_lat=len(lats), n_alt=len(alts_m), n_iters=50, relax=0.2
    )

    # Optional δ(Δt) branch (same behavior as your script)
    Ne_rec_ddt = delta_dt_branch(delta_t_vhf, delta_t_hf, D_hf, lats, alts_m,
                                 f_c_hf=F_C_HF, bw_hf=BW_HF,
                                 n_iters=N_ITERS, relax=RELAX)

    # Step 6 (plots)
    plot_recons(
        lats, alts_m,
        [Ne_rec_vhf, Ne_rec_hf, Ne_rec_all, Ne_rec_weighted, Ne_rec_ddt],
        ["VHF only", "HF only", "Combined (standard)", "Combined (weighted)", "δ(Δt)-based"]
    )
    plot_raypaths(
    lats=lats,
    alts_m=alts_m,
    rays_hf=rays_hf,
    theta_per_ray=theta_per_ray,
    rays_vhf=rays_vhf,
    # optional: limit lines per angle to reduce clutter
    # max_rays_per_angle=60,
    )

    if not USE_GPU:
        t_meas, tec_for_bin, delta_tec_err_passive, range_error_after_passive = build_passive_plot_inputs(
            iono=iono, lats=lats, alts_m=alts_m,
            rays_hf=rays_hf,
            integ_hf=integ_hf,
            theta_per_ray=theta_per_ray,
            vtec_hf=vtec_hf,
            delta_t_hf=delta_t_hf,
            delta_t_vhf=delta_t_vhf,
        )
        # Then make the two figures
        plot_passive_tec_error_heatmap_from_arrays(t_meas, tec_for_bin, delta_tec_err_passive)
        plot_vhf_range_error_heatmap_from_arrays(t_meas, tec_for_bin, range_error_after_passive)

    else:
        t_meas, tec_for_bin, delta_tec_err_passive, range_error_after_passive = build_passive_plot_inputs_GPU(
            iono=iono, lats=lats, alts_m=alts_m,
            D_hf=D_hf,                      # use GPU @ for tec_true
            integ_hf=integ_hf,
            theta_per_ray=theta_per_ray,
            vtec_hf=vtec_hf,
            delta_t_hf=delta_t_hf,
            delta_t_vhf=delta_t_vhf,
            sat_lats_vhf=SAT_LATS,
            hf_origin_lat=float(SAT_LATS[0]),
        )
        plot_passive_tec_error_heatmap_GPU(t_meas, tec_for_bin, delta_tec_err_passive)
        plot_vhf_range_error_heatmap_GPU(t_meas, tec_for_bin, range_error_after_passive)

    return {
        "lats": lats,
        "alts_m": alts_m,
        "iono": iono,
        "vtec_vhf": vtec_vhf,
        "vtec_hf": vtec_hf,
        "delta_t_vhf": delta_t_vhf,
        "delta_t_hf": delta_t_hf,
        "theta_per_ray": theta_per_ray,
        "integ_vhf": integ_vhf,
        "integ_hf": integ_hf,
        "D_vhf": D_vhf,
        "D_hf": D_hf,
        "D_all": D_all,
        "Ne_rec_vhf": Ne_rec_vhf,
        "Ne_rec_hf": Ne_rec_hf,
        "Ne_rec_all": Ne_rec_all,
        "Ne_rec_weighted": Ne_rec_weighted,
        "Ne_rec_ddt": Ne_rec_ddt,
        "weights": weights,
        "vtec_all": vtec_all,
    }

# ---------------------- Entry point -----------------------------
if __name__ == "__main__":
    results = run_pipeline()
    # You can add any post-analysis you already do here (heatmaps, θ studies, etc.)