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

from modify_df import load_mission_df
from ionosphere_design import (
    build_ionosphere, add_gaussian_enhancement
)
from ray_trace_passive import (
    trace_passive_nadir, trace_passive_oblique,
    compute_STEC_along_path, dualfreq_to_VTEC, reconstruct_art
)
from improved_geometry import (
    build_geometry_matrix_weighted,
    calculate_measurement_weights,
    weighted_reconstruction_art
)

from physics_constants import C_LIGHT, C_IONO, R_EUROPA

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
THETA_LIST_HF = [30, 40, 50, 60, 70, 80, 85]

# Integration times (seconds), per your current logic
INTEG_TIMES_VHF = 0.12
INTEG_TIMES_HF_PER_ANGLE = np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])

# ART settings
N_ITERS = 20
RELAX   = 0.1

# Use physics_constants for canonical values (C_LIGHT, C_IONO)

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
        integ_hf_all.extend([INTEG_TIMES_HF_PER_ANGLE[idx]] * len(rays_hf))

    theta_per_ray = np.array(theta_per_ray, dtype=float)
    integ_hf_all  = np.array(integ_hf_all, dtype=float)

    return rays_vhf, integ_vhf, rays_hf_all, theta_per_ray, integ_hf_all

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

# ---------------------- Step 4: Build geometry ------------------
def step4_build_geometry(rays_vhf, rays_hf, lats, alts_m):
    D_vhf = build_geometry_matrix_weighted(rays_vhf, lats, alts_m)
    D_hf  = build_geometry_matrix_weighted(rays_hf,  lats, alts_m)
    return D_vhf, D_hf

# ---------------------- Step 5: Reconstruct ---------------------
def step5_reconstruct(D_vhf, D_hf, vtec_vhf, vtec_hf,
                      delta_t_vhf, delta_t_hf,
                      lats, alts_m,
                      n_iters=N_ITERS, relax=RELAX):
    # 5A) ART (VHF-only, HF-only, stacked)
    Ne_rec_vhf = reconstruct_art(D_vhf, vtec_vhf, len(lats), len(alts_m), n_iters, relax)
    Ne_rec_hf  = reconstruct_art(D_hf,  vtec_hf,  len(lats), len(alts_m), n_iters, relax)

    D_all    = np.vstack([D_vhf, D_hf])
    vtec_all = np.concatenate([vtec_vhf, vtec_hf])
    Ne_rec_all = reconstruct_art(D_all, vtec_all, len(lats), len(alts_m), n_iters, relax)

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
    plt.tight_layout(); plt.show(block=False); plt.savefig("img/A_reconstructions.png")

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
    plt.savefig("img/B_raypaths.png")

def build_passive_plot_inputs(
    iono, lats, alts_m,
    rays_hf,                  # list of HF polylines (n_hf_rays)
    integ_hf,                 # (n_hf_rays,) integration times [s]
    theta_per_ray,            # (n_hf_rays,) true incidence angle from vertical [deg]
    vtec_hf,                  # (n_hf_rays,) VTEC estimates returned by your dualfreq helper
    delta_t_hf,               # (n_hf_rays,) noisy HF dual-freq delay [s]
    delta_t_vhf,              # (n_vhf_rays,) noisy VHF dual-freq delay [s]
    SAT_LATS=np.linspace(-6.0, 6.0, 60),   # must match how you built VHF rays
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
    origin_lat = SAT_LATS[0]
    vhf_idx = int(np.argmin(np.abs(SAT_LATS - origin_lat)))
    delta_t_vhf_matched = np.full_like(delta_t_hf, delta_t_vhf[vhf_idx])

    delta_dt = -(delta_t_vhf_matched - delta_t_hf)                  # seconds
    tec_residual_after_passive = (C_LIGHT * delta_dt / C_IONO) * freq_factor_h   # m^-2
    range_error_after_passive = (C_IONO / (F_C_VHF**2)) * np.abs(tec_residual_after_passive)  # meters

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
    c = C_LIGHT
    f1_h = f_c_hf - 0.5*bw_hf
    f2_h = f_c_hf + 0.5*bw_hf
    freq_factor_h = (f1_h**2 * f2_h**2) / (f2_h**2 - f1_h**2)

    delta_dt_all = -(delta_t_vhf_matched - delta_t_hf)
    tec_est_ddt_all = (c * delta_dt_all / C_IONO) * freq_factor_h

    Ne_rec_ddt = reconstruct_art(D_hf, tec_est_ddt_all, len(lats), len(alts_m), n_iters, relax)
    return Ne_rec_ddt

def centers_to_edges(c):
    c = np.asarray(c)
    e = np.empty(c.size + 1, dtype=c.dtype)
    e[1:-1] = 0.5*(c[1:] + c[:-1])
    e[0]     = c[0] - 0.5*(c[1] - c[0])
    e[-1]    = c[-1] + 0.5*(c[-1] - c[-2])
    return e

# ---------------------- Orchestrator ----------------------------
def run_pipeline():
    # Step 0
    alt_km_1d, Ne_1d = step0_load_dataframe()

    # Step 1
    lats, alt_km, alts_m, iono = step1_build_iono(alt_km_1d, Ne_1d, lat_extent=(-10, 10), lat_res=200)

    # Step 2
    rays_vhf, integ_vhf, rays_hf, theta_per_ray, integ_hf = step2_generate_rays(lats, alts_m)

    # Step 3
    vtec_vhf, delta_t_vhf, vtec_hf, delta_t_hf = step3_forward_model(
        iono, lats, alts_m, rays_vhf, integ_vhf, rays_hf, theta_per_ray, integ_hf
    )

    # Step 4
    D_vhf, D_hf = step4_build_geometry(rays_vhf, rays_hf, lats, alts_m)

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