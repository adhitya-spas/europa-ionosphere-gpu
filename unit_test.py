# ---------------------- Imports ----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sp 
import os
from datetime import datetime

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
SAT_LATS_VHF = np.linspace(-10.0, 10.0, 60)  #np.linspace(-10.0, -5.0, 10)  # ground-track latitudes for rays
SAT_LATS_HF = np.linspace(-5.0, +5.0, 60)

# HF true incidence angles (from vertical, 0° = nadir)
# THETA_LIST_HF = [30, 40, 50, 60, 70, 80, 85]
THETA_LIST_HF = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10]

# Integration times (seconds), per your current logic
INTEG_TIMES_VHF = 0.12
# INTEG_TIMES_HF_PER_ANGLE = np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
INTEG_TIMES_HF_PER_ANGLE = np.linspace(0.05, 0.15, len(THETA_LIST_HF))  # 20 values from 0.05s to 0.15s

# ART settings
N_ITERS = 20
RELAX   = 0.1

# Use physics_constants for canonical physical values

# Create one timestamped folder per process start
IMG_TS = datetime.now().strftime("%m_%d_%H_%M_%S")
IMG_DIR = os.path.join("img", IMG_TS)
os.makedirs(IMG_DIR, exist_ok=True)

# Top-level smoothing toggle (0.0 = off)
SMOOTH_SIGMA = 1.2

USE_GPU = True
if USE_GPU:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import scipy.sparse as sp
else:
    USE_GPU = False
    cp = None
    cpx_sparse = None

## FUNCTIONS ##

# ---------------------- Step 0: Load data ----------------------
def step0_load_dataframe(path="new_mission_df.pkl", mission_name="E6a Exit"):
    df = load_mission_df(path)
    row = df[df["Mission"] == mission_name].iloc[0]
    alt_km_1d, Ne_1d = row["Altitude"], row["Ne"]
    return alt_km_1d, Ne_1d
    
# ---------------------- Step 1: Build ionosphere ----------------------
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
    rays_vhf = trace_passive_nadir(SAT_LATS_VHF, alts_m, npts=NPTS_VHF)
    integ_vhf = np.full(len(rays_vhf), INTEG_TIMES_VHF)

    # 2B) HF oblique rays (true incidence angles from vertical)
    rays_hf_all = []
    theta_per_ray = []
    integ_hf_all = []

    T_PER_ANGLE = np.full(len(THETA_LIST_HF), 0.12, dtype=float)  # e.g. 0.12s

    for idx, theta in enumerate(THETA_LIST_HF):
        rays_hf = trace_passive_oblique(
            sat_lats=lats,
            h0_m=alts_m.max(),
            hsat_m=alts_m.max(),
            theta_i_deg=theta,
            npts=NPTS_HF,
            R_E_m=1.5608e6,
            mode="left_fixed", #"sat_right", 
            lat_left_deg=0.0          # << choose your left start latitude
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
        # rays_this_theta = trace_passive_oblique(
        #     sat_lats=sat_lats,
        #     h0=alts_m.max(),
        #     hsat=alts_m.max(),
        #     theta_i_deg=theta,
        #     npts=npts_hf
        # )
        rays_this_theta = trace_passive_oblique(
            sat_lats=sat_lats,
            h0_m=alts_m.max(),
            hsat_m=alts_m.max(),
            theta_i_deg=theta,
            npts=npts_hf,
            R_E_m=1.5608e6,
            mode="left_fixed", #"sat_right", 
            lat_left_deg=0.0          # << choose your left start latitude
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
if USE_GPU:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
else:
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
    sat_lats_vhf = SAT_LATS_VHF
    origin_lat = SAT_LATS_VHF[0]
    origin_lat_idx = int(np.argmin(np.abs(sat_lats_vhf - origin_lat)))
    delta_t_vhf_matched = np.full(len(delta_t_hf), delta_t_vhf[origin_lat_idx])

    # Convert δ(Δt) to TEC with the HF frequency pair
    c = C_LIGHT
    f1_h = f_c_hf - 0.5*bw_hf
    f2_h = f_c_hf + 0.5*bw_hf
    freq_factor_h = (f1_h**2 * f2_h**2) / (f2_h**2 - f1_h**2)

    delta_dt_all = -(delta_t_vhf_matched - delta_t_hf)
    tec_est_ddt_all = (c * delta_dt_all / C_IONO) * freq_factor_h
    # (legacy) tec_est_ddt_all = (C_LIGHT * delta_dt_all / C_IONO) * freq_factor_h  # use `physics_constants` in code

    Ne_rec_ddt = reconstruct_art_sparse(D_hf, tec_est_ddt_all, len(lats), len(alts_m), n_iters, relax)
    return Ne_rec_ddt

def centers_to_edges(c):
    c = np.asarray(c)
    e = np.empty(c.size + 1, dtype=c.dtype)
    e[1:-1] = 0.5*(c[1:] + c[:-1])
    e[0]     = c[0] - 0.5*(c[1] - c[0])
    e[-1]    = c[-1] + 0.5*(c[-1] - c[-2])
    return e

# =======================
# STEP 3 MATH: UNIT TESTS
# =======================
def _make_uniform_iono(lats, alts_m, Ne_val=1.0e10):
    """Simple, spatially uniform ionosphere for analytical checks."""
    iono_uni = np.full((len(alts_m), len(lats)), Ne_val, dtype=float)
    return iono_uni

def test_eq9_dualfreq_vertical_nonoise(lats, alts_m):
    """
    Peters Eq. (7)+(9): Using δt between two freqs gives TEC exactly.
    Here we create 1 vertical path with known VTEC_true and ensure
    dualfreq_to_VTEC returns it (θ=0 so VTEC=TEC).
    """
    from ray_trace_passive import dualfreq_to_VTEC
    # build trivial grid & uniform Ne
    iono_uni = _make_uniform_iono(lats, alts_m, Ne_val=1.0e10)
    # one nadir VHF ray: (lat, alt) vertical line
    ray_vert = np.column_stack([np.full_like(alts_m, 0.0), alts_m])
    # exact STEC = ∫ Ne dz = Ne * height
    V = (alts_m.max() - alts_m.min())  # meters
    VTEC_true = 1.0e10 * V  # m^-2
    # feed directly as "slant TEC" to dualfreq_to_VTEC
    VTEC_est, dt = dualfreq_to_VTEC(
        stec_slant=np.array([VTEC_true]),
        f_c=F_C_HF, bw=BW_HF,
        theta_deg=0.0,                # verticalization factor = 1
        integration_time=1e9,         # huge => drive timing noise → 0
        return_deltat=True
    )
    assert np.allclose(VTEC_est[0], VTEC_true, rtol=1e-6, atol=0.0), \
        f"Eq.9 inversion failed: got {VTEC_est[0]:.3e} vs {VTEC_true:.3e}"

def test_verticalization_cos_theta():
    """
    Verticalization check: VTEC ≈ TEC_slant * cos(theta).
    Allow small relative error due to internal float precision and constants.
    """
    from ray_trace_passive import dualfreq_to_VTEC
    TEC_slant = np.array([1.23e16])   # arbitrary slant TEC
    th = 60.0                         # deg from vertical

    VTEC, _ = dualfreq_to_VTEC(
        stec_slant=TEC_slant,
        f_c=F_C_HF,
        bw=BW_HF,
        theta_deg=th,
        integration_time=1e9,         # drive timing noise ~ 0
        return_deltat=True
    )

    expected = TEC_slant[0] * np.cos(np.deg2rad(th))
    # use realistic tolerances (float32-ish chain)
    np.testing.assert_allclose(
        VTEC[0], expected,
        rtol=1e-6, atol=1e-2,         # tweak atol as needed if values are very large
        err_msg=f"VTEC={VTEC[0]:.6e} vs expected={expected:.6e}"
    )

def test_freq_separation_scaling():
    """
    With Eq. (7)&(9), |δt| scales with the (1/f1^2 - 1/f2^2) term.
    HF (narrow, 1 MHz around 9 MHz) should yield a much larger |δt|
    than VHF (10 MHz around 60 MHz) for the same STEC.
    """
    from ray_trace_passive import dualfreq_to_VTEC
    TEC_slant = np.array([2.0e16])

    def measure_dt(f0, bw):
        VTEC, dt = dualfreq_to_VTEC(
            stec_slant=TEC_slant, f_c=f0, bw=bw,
            theta_deg=0.0, integration_time=1e9, return_deltat=True
        )
        return np.abs(dt.mean())

    dt_hf  = measure_dt(F_C_HF,  BW_HF)   # ~9 MHz, 1 MHz
    dt_vhf = measure_dt(F_C_VHF, BW_VHF)  # ~60 MHz, 10 MHz
    assert dt_hf > dt_vhf*5, f"Expected HF |δt| ≫ VHF |δt|, got HF={dt_hf:.2e}, VHF={dt_vhf:.2e}"

def test_noise_scaling_with_T_and_BW():
    """
    Your implementation uses timing-noise σ_dt ∝ 1/(BW * sqrt(T)).
    Verify that doubling T and BW reduces σ_dt by ~factor 2 and 2, respectively.
    """
    from ray_trace_passive import dualfreq_to_VTEC
    rng = np.random.default_rng(0)
    TEC_slant = np.full(4000, 1.0e16)  # many trials for a reliable std estimate

    def sample_std(bw, T):
        # sample many to estimate std
        _, dt = dualfreq_to_VTEC(
            stec_slant=TEC_slant, f_c=F_C_HF, bw=bw,
            theta_deg=0.0, integration_time=T, return_deltat=True
        )
        return np.std(dt)

    s1 = sample_std(BW_HF,          0.05)
    s2 = sample_std(BW_HF,          0.20)  # 4x T ⇒ ~½ σ
    s3 = sample_std(BW_HF*2.0,      0.05)  # 2x BW ⇒ ~½ σ
    assert s2 < 0.6*s1, f"T scaling off: σ(4T) should be ~0.5σ(T), got {s2/s1:.2f}"
    assert s3 < 0.6*s1, f"BW scaling off: σ(2BW) should be ~0.5σ(BW), got {s3/s1:.2f}"

def test_gpu_truth_matches_path_integral(lats, alts_m):
    """
    Consistency: sparse D @ x on GPU equals explicit per-ray STEC integral
    (within a small tolerance due to grid discretization).
    """
    # small sample to keep it quick
    rays_hf_small, th_small, T_small = step2_generate_rays_gpu_sweep(
        lats, alts_m, theta_list_hf=[5,30,60], sat_lats=np.linspace(-2,2,11), npts_hf=200, t_sweep=[0.1]
    )
    rays_hf_small = rays_hf_small[:50]  # trim
    # uniform iono so STEC = Ne * path length
    iono_uni = _make_uniform_iono(lats, alts_m, Ne_val=5e9)
    Dv, Dh = step4_build_geometry(rays_vhf=[], rays_hf=rays_hf_small, lats=lats, alts_m=alts_m)

    # truth via D @ x
    x = iono_uni.astype(np.float32).ravel(order="C")
    tec_true = Dh @ x  # CPU is fine for test size

    # explicit path integral
    stec_explicit = np.array([compute_STEC_along_path(iono_uni, lats, alts_m, r) for r in rays_hf_small])

    rel_err = np.linalg.norm(tec_true - stec_explicit)/np.linalg.norm(tec_true)
    assert rel_err < 1e-2, f"D@x vs. explicit STEC mismatch: rel_err={rel_err:.3e}"


## UNIT TEST BEGIN ##

if __name__ == "__main__":
    
    print("Step 0: Load Data")
    alt_km_1d, Ne_1d = step0_load_dataframe()

    # Step 1
    print("Step 1: Build Ionosphere")
    lats, alt_km, alts_m, iono = step1_build_iono(alt_km_1d, Ne_1d, lat_extent=(-10, 10), lat_res=200)

    # Step 2
    print("Step 2: Generate Rays")
    if not USE_GPU:
        rays_vhf, integ_vhf, rays_hf, theta_per_ray, integ_hf = step2_generate_rays(lats, alts_m)
    else:
        rays_hf, theta_per_ray, integ_hf = step2_generate_rays_gpu_sweep(
        lats, alts_m,
        theta_list_hf=THETA_LIST_HF,  # or your list
        sat_lats=SAT_LATS_HF,  # np.linspace(-6.0, 6.0, 181),   # denser tracks for richer TEC
        npts_hf=500,
        t_sweep= INTEG_TIMES_HF_PER_ANGLE  # np.linspace(0.05, 0.15, 21)     # dense T grid (x-axis)
        )
        rays_vhf = trace_passive_nadir(SAT_LATS_VHF, alts_m, npts=NPTS_VHF)
        integ_vhf = np.full(len(rays_vhf), INTEG_TIMES_VHF, dtype=float)

    # ... your Step 0–2 code above ...
    print("\n[UNIT] Running Step 3 math checks...")
    test_eq9_dualfreq_vertical_nonoise(lats, alts_m)
    test_verticalization_cos_theta()
    test_freq_separation_scaling()
    test_noise_scaling_with_T_and_BW()
    test_gpu_truth_matches_path_integral(lats, alts_m)
    print("[OK] Step 3 math checks passed.")

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

    # ---------------------- Step 3: Diagnostics ----------------------
    print("\n[DIAG] Step 3 checks")

    def _assert_close(a, b, rtol=1e-2, atol=1e-6, msg=""):
        import numpy as _np
        if not _np.allclose(a, b, rtol=rtol, atol=atol):
            raise AssertionError(msg or f"Not close: {a} vs {b} (rtol={rtol}, atol={atol})")

    # Basic shapes & counts
    print(f"VHF: n_rays={len(rays_vhf)}, integ_vhf.shape={integ_vhf.shape}")
    print(f"HF : n_rays={len(rays_hf)}, theta_per_ray.shape={theta_per_ray.shape}, integ_hf.shape={integ_hf.shape}")

    # Forward-model outputs present
    print(f"vtec_vhf.shape={vtec_vhf.shape}, delta_t_vhf.shape={delta_t_vhf.shape}")
    print(f"vtec_hf.shape={vtec_hf.shape},   delta_t_hf.shape={delta_t_hf.shape}")

    # Sanity on values
    import numpy as _np
    print(f"VHF Δt stats: mean={_np.mean(delta_t_vhf):.3e} s, std={_np.std(delta_t_vhf):.3e}, "
        f"min={_np.min(delta_t_vhf):.3e}, max={_np.max(delta_t_vhf):.3e}")
    print(f"HF  Δt stats: mean={_np.mean(delta_t_hf):.3e} s, std={_np.std(delta_t_hf):.3e}, "
        f"min={_np.min(delta_t_hf):.3e}, max={_np.max(delta_t_hf):.3e}")

    # Verticalization check (HF): VTEC *should* be cos(theta) smaller than STEC.
    # If you ran GPU path, tec_true is defined. If CPU path, recompute a tiny subset for the check.
    try:
        tec_slant_check = tec_true  # from GPU path
    except NameError:
        # CPU path: build slant TEC for a small sample to keep it light
        sample = slice(0, min(500, len(rays_hf)))
        tec_slant_check = _np.array([compute_STEC_along_path(iono, lats, alts_m, ray)
                                    for ray in rays_hf[sample]])
        # align the angle subset
        theta_for_check = _np.asarray(theta_per_ray, float)[sample]
    else:
        theta_for_check = _np.asarray(theta_per_ray, float)

    theta_rad = _np.deg2rad(theta_for_check)
    stec_est = _np.asarray(vtec_hf[:theta_for_check.size]) / _np.cos(theta_rad)
    err = stec_est - _np.asarray(tec_slant_check)
    rmse = _np.sqrt(_np.mean(err**2))
    rel = rmse / (max(_np.mean(_np.asarray(tec_slant_check)), 1.0))
    print(f"[Step3] HF verticalization: RMSE={rmse:.3e} ({rel:.2%} of mean STEC)")

    # Step 4
    print("Step 4: Build Geometry Matrices")
    if not USE_GPU:
        D_vhf, D_hf = step4_build_geometry(rays_vhf, rays_hf, lats, alts_m)
        print("HF geometry:", D_hf.shape, "nnz=", D_hf.nnz, "density=", D_hf.nnz/(D_hf.shape[0]*D_hf.shape[1]))
        print("VHF geometry:", D_vhf.shape, "nnz=", D_vhf.nnz, "density=", D_vhf.nnz/(D_vhf.shape[0]*D_vhf.shape[1]))

    # Step 5A (ARTs) + collect for weighted
    print("Step 5: Reconstruct")
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

    # ---------------------- Step 5: Diagnostics ----------------------
    print("\n[DIAG] Step 5 checks")

    import numpy as _np
    from scipy.sparse import issparse

    # Flatten helper — must match how D was built (row-major ravel)
    def _flat_ne(Ne_map):  # Ne_map shape (n_lat, n_alt) or (n_alt, n_lat) — we’ll detect
        arr = _np.asarray(Ne_map)
        return arr.ravel(order="C")

    # 5A: Unweighted reconstruction re-prediction
    n_vhf_rows = D_vhf.shape[0]
    n_hf_rows  = D_hf.shape[0]
    assert D_all.shape[0] == n_vhf_rows + n_hf_rows, "D_all row count mismatch"

    vtec_all_hat = (D_all @ _flat_ne(Ne_rec_all)) if issparse(D_all) else D_all.dot(_flat_ne(Ne_rec_all))
    res_all = vtec_all_hat - vtec_all
    rmse_all = _np.sqrt(_np.mean(res_all**2))
    rel_all  = rmse_all / max(_np.mean(_np.abs(vtec_all)), 1.0)
    print(f"[5A] Unweighted: RMSE={rmse_all:.3e} ({rel_all:.2%} of mean |VTEC|); "
        f"res min/max={res_all.min():.3e}/{res_all.max():.3e}")

    # Split residuals by band
    res_vhf = res_all[:n_vhf_rows]
    res_hf  = res_all[n_vhf_rows:]
    rmse_vhf = _np.sqrt(_np.mean(res_vhf**2))
    rmse_hf  = _np.sqrt(_np.mean(res_hf**2))
    print(f"[5A] Split: RMSE_VHF={rmse_vhf:.3e}, RMSE_HF={rmse_hf:.3e}")

    # 5B: Weighted reconstruction re-prediction (optional but nice to compare)
    vtec_all_hat_w = (D_all @ _flat_ne(Ne_rec_weighted)) if issparse(D_all) else D_all.dot(_flat_ne(Ne_rec_weighted))
    res_all_w = vtec_all_hat_w - vtec_all
    rmse_all_w = _np.sqrt(_np.mean(res_all_w**2))
    rel_all_w  = rmse_all_w / max(_np.mean(_np.abs(vtec_all)), 1.0)
    print(f"[5B] Weighted: RMSE={rmse_all_w:.3e} ({rel_all_w:.2%} of mean |VTEC|); "
        f"ΔRMSE={(rmse_all_w - rmse_all):+.3e}")

    # Basic sanity on recon fields
    def _stats(name, M):
        arr = _np.asarray(M)
        print(f"[5] {name}: shape={arr.shape}, mean={_np.nanmean(arr):.3e}, "
            f"min={_np.nanmin(arr):.3e}, max={_np.nanmax(arr):.3e}")

    _stats("Ne_rec_vhf", Ne_rec_vhf)
    _stats("Ne_rec_hf", Ne_rec_hf)
    _stats("Ne_rec_all", Ne_rec_all)
    _stats("Ne_rec_weighted", Ne_rec_weighted)

    # Optional: nonnegativity check (relax/remove if your solver allows signed deviations)
    neg_v = (Ne_rec_all < 0).sum()
    neg_w = (Ne_rec_weighted < 0).sum()
    print(f"[5] negatives: Ne_rec_all={neg_v} voxels, Ne_rec_weighted={neg_w} voxels")

    # Quick weight sanity
    print(f"[5] weights: shape={weights.shape}, mean={weights.mean():.3f}, "
        f"min={weights.min():.3f}, max={weights.max():.3f}")