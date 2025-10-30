# ray_trace.py

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Tuple
import scipy.sparse as sp

R_E = 1569e3  # Europa radius in m

def pack_rays_mid_ds(ray_paths, R_E=1569e3):
    """
    Convert list of [N_i x 2] rays (lat[deg], alt[m]) into padded arrays:
      lat_mid [n_rays, max_seg], alt_mid [..], ds_m [..], seg_counts [n_rays]
    where each 'segment' is between consecutive samples of a ray.
    """
    import numpy as np
    deg2m = (np.pi / 180.0) * R_E

    n = len(ray_paths)
    seg_counts = np.array([max(0, p.shape[0] - 1) for p in ray_paths], dtype=np.int32)
    max_seg = int(seg_counts.max()) if n > 0 else 0

    lat_mid = np.zeros((n, max_seg), dtype=np.float32)
    alt_mid = np.zeros((n, max_seg), dtype=np.float32)
    ds_m    = np.zeros((n, max_seg), dtype=np.float32)

    for i, path in enumerate(ray_paths):
        m = seg_counts[i]
        if m == 0: 
            continue
        lat0 = path[:-1, 0]; lat1 = path[1:, 0]
        alt0 = path[:-1, 1]; alt1 = path[1:, 1]
        lat_mid[i, :m] = 0.5 * (lat0 + lat1)
        alt_mid[i, :m] = 0.5 * (alt0 + alt1)
        dlat_m = (lat1 - lat0) * deg2m
        dalt_m = (alt1 - alt0)
        ds_m[i, :m] = np.hypot(dlat_m, dalt_m)

    return lat_mid, alt_mid, ds_m, seg_counts


def trace_passive_nadir(
    sat_lats: np.ndarray,
    alts:     np.ndarray,
    npts:     int = 500
) -> list[np.ndarray]:
    """
    Build purely-vertical nadir rays from each sat_lat:
      (lat, top_of_grid) → (lat, 0)
    Follows Eqn A.11 in the paper.
    """
    rays = []
    top = alts.max()
    for lat in sat_lats:
        zs = np.linspace(top, 0.0, npts)
        xs = np.full_like(zs, lat)
        rays.append(np.vstack([xs, zs]).T)
    return rays

def trace_passive_oblique_old(
    sat_lats: np.ndarray,
    h0: float,
    hsat: float,
    theta_i_deg: float,  # TRUE incidence angle from vertical
    npts: int = 200,
    R_E: float = 6.371e6
) -> list[np.ndarray]:
    """
    Generate oblique two-leg ray paths for each satellite ground-track latitude:
      • Leg 1 (downward): from (lat_top, h0) to (lat_ref, 0)
      • Leg 2 (upward)  : from (lat_ref, 0) to (lat_sat, hsat)

    theta_i_deg: TRUE incidence angle from vertical (0 = nadir, 90 = horizontal)
    """
    theta = np.deg2rad(theta_i_deg)
    deg_per_m = 180.0 / (np.pi * R_E)
    paths = []

    for lat_sat in sat_lats:
        if np.isclose(theta, 0.0):
            # pure nadir
            lat_ref = lat_sat
            lat_top = lat_sat
        else:
            # cot = 1.0 / np.tan(theta)
            # lat_ref = lat_sat - hsat * cot * deg_per_m
            # lat_top = lat_ref -   h0 * cot * deg_per_m
            tan_theta = np.tan(theta)
            lat_ref = lat_sat - hsat * tan_theta * deg_per_m
            lat_top = lat_ref - h0 * tan_theta * deg_per_m

        # split points evenly between the two legs ( to accoount for odd numbers)
        n1 = npts // 2
        n2 = npts - n1

        # Leg 1: down from h0 at lat_top to surface at lat_ref
        xs1 = np.linspace(lat_top, lat_ref, n1)
        ys1 = np.linspace(h0,       0.0,     n1)

        # Leg 2: up from surface at lat_ref to hsat at lat_sat
        xs2 = np.linspace(lat_ref,  lat_sat, n2)
        ys2 = np.linspace(0.0,      hsat,    n2)

        ray = np.vstack([
            np.concatenate([xs1, xs2]),
            np.concatenate([ys1, ys2])
        ]).T  # shape (npts, 2)

        paths.append(ray)

    return paths

def trace_passive_oblique_old2(
    sat_lats: np.ndarray,
    h0: float,
    hsat: float,
    theta_i_deg: float,  # TRUE incidence angle from vertical
    npts: int = 200,
    R_E: float = 1.5608e6
) -> list[np.ndarray]:
    """
    Generate oblique two-leg ray paths for each satellite ground-track latitude:
      • Leg 1 (downward): from (lat_top, h0) to (lat_ref, 0)
      • Leg 2 (upward)  : from (lat_ref, 0) to (lat_sat, hsat)

    theta_i_deg: TRUE incidence angle from vertical (0 = nadir, 90 = horizontal)
    """
    theta = np.deg2rad(theta_i_deg)
    deg_per_m = 180.0 / (np.pi * R_E)
    paths = []

    for lat_sat in sat_lats:
        if np.isclose(theta, 0.0):
            # pure nadir
            lat_ref = lat_sat
            lat_top = lat_sat
        else:
            # cot = 1.0 / np.tan(theta)
            # lat_ref = lat_sat - hsat * cot * deg_per_m
            # lat_top = lat_ref -   h0 * cot * deg_per_m
            tan_theta = np.tan(theta)
            lat_ref = lat_sat - hsat * tan_theta * deg_per_m
            lat_top = lat_ref - h0 * tan_theta * deg_per_m

        # split points evenly between the two legs ( to accoount for odd numbers)
        n1 = npts // 2
        n2 = npts - n1

        # Leg 1: down from h0 at lat_top to surface at lat_ref
        xs1 = np.linspace(lat_top, lat_ref, n1)
        ys1 = np.linspace(h0,       0.0,     n1)

        # Leg 2: up from surface at lat_ref to hsat at lat_sat
        xs2 = np.linspace(lat_ref,  lat_sat, n2)
        ys2 = np.linspace(0.0,      hsat,    n2)

        ray = np.vstack([
            np.concatenate([xs1, xs2]),
            np.concatenate([ys1, ys2])
        ]).T  # shape (npts, 2)

        paths.append(ray)

    return paths

def trace_passive_oblique(
    sat_lats: np.ndarray,
    h0_m: float,
    hsat_m: float,
    theta_i_deg: float,     # TRUE incidence (deg from vertical)
    npts: int = 200,
    R_E_m: float = 1.5608e6,  # Europa radius in meters
    *,
    mode: str = "sat_right",  # "sat_right" (old behavior) or "left_fixed"
    lat_left_deg: float | None = None
) -> list[np.ndarray]:
    """
    Build straight two-leg oblique rays.

    Modes
    -----
    - 'sat_right' (default): for each `lat_sat` in sat_lats, make a V whose right tip is (lat_sat, hsat_m).
      Leg 1: (lat_top, h0_m) -> (lat_ref, 0)
      Leg 2: (lat_ref, 0) -> (lat_sat, hsat_m)

    - 'left_fixed': for each `lat_left` in sat_lats (or lat_left_deg if provided),
      make a V whose left start is (lat_left, h0_m), reflection at (lat_ref, 0),
      and right tip (lat_sat, hsat_m).

    Returns
    -------
    list[np.ndarray], each array shaped (npts, 2) with columns = [lat_deg, alt_m]
    """
    if mode not in ("sat_right", "left_fixed","mirror"):
        raise ValueError("mode must be 'sat_right' or 'left_fixed' or 'mirror'")

    theta = np.deg2rad(theta_i_deg)
    deg_per_m = 180.0 / (np.pi * R_E_m)  # convert horizontal meters → degrees latitude

    paths: list[np.ndarray] = []
    n1 = npts // 2
    n2 = npts - n1

    # ---- LEFT-FIXED MODE (now supports multiple sat_lats) ----
    if mode == "left_fixed":
        if (sat_lats is None or len(sat_lats) == 0) and lat_left_deg is None:
            raise ValueError("Provide sat_lats or lat_left_deg when mode='left_fixed'")

        # if user provided a single left latitude instead of array
        left_starts = np.atleast_1d(sat_lats if len(np.atleast_1d(sat_lats)) > 0 else [lat_left_deg])

        for lat_left in left_starts:
            if np.isclose(theta, 0.0):
                lat_ref = lat_left
                lat_sat = lat_left
            else:
                dlat_down = h0_m  * np.tan(theta) * deg_per_m
                dlat_up   = hsat_m * np.tan(theta) * deg_per_m
                lat_ref = lat_left + dlat_down
                lat_sat = lat_ref  + dlat_up

            # Leg 1: left → reflection
            xs1 = np.linspace(lat_left, lat_ref, n1)
            ys1 = np.linspace(h0_m,     0.0,     n1)

            # Leg 2: reflection → right
            xs2 = np.linspace(lat_ref,  lat_sat, n2)
            ys2 = np.linspace(0.0,      hsat_m,  n2)

            ray = np.column_stack([np.r_[xs1, xs2], np.r_[ys1, ys2]])
            paths.append(ray)

        return paths
    
    # ---- MIRROR MODE: single-leg rays from ground at lat0 to ±lat ends (by theta) ----
    if mode == "mirror":
        # starting ground latitude
        lat0 = float(lat_left_deg) if (lat_left_deg is not None) else 0.0

        # accept scalar or array of angles
        thetas = np.atleast_1d(theta_i_deg).astype(float)

        for th in thetas:
            th_rad = np.deg2rad(th)
            if np.isclose(th_rad, 0.0):
                lat_end = lat0
            else:
                # horizontal-to-latitude conversion via deg_per_m
                lat_end = lat0 + hsat_m * np.tan(th_rad) * deg_per_m

            # single up-leg from ground to satellite altitude
            xs = np.linspace(lat0,  lat_end, npts)
            ys = np.linspace(0.0,   hsat_m, npts)

            paths.append(np.column_stack([xs, ys]))
        return paths

    # ---- SAT-RIGHT MODE (original behavior) ----
    for lat_sat in np.asarray(sat_lats, float):
        if np.isclose(theta, 0.0):
            lat_ref = lat_sat
            lat_top = lat_sat
        else:
            tan_theta = np.tan(theta)
            lat_ref = lat_sat - hsat_m * tan_theta * deg_per_m
            lat_top = lat_ref -  h0_m  * tan_theta * deg_per_m

        # Leg 1: down from (lat_top, h0) to (lat_ref, 0)
        xs1 = np.linspace(lat_top, lat_ref, n1)
        ys1 = np.linspace(h0_m,    0.0,     n1)

        # Leg 2: up from (lat_ref, 0) to (lat_sat, hsat)
        xs2 = np.linspace(lat_ref, lat_sat, n2)
        ys2 = np.linspace(0.0,     hsat_m,  n2)

        ray = np.column_stack([np.r_[xs1, xs2], np.r_[ys1, ys2]])
        paths.append(ray)

    return paths


def compute_STEC_along_path(
    ionosphere: np.ndarray,
    lats:       np.ndarray,
    alts:       np.ndarray,
    path:       np.ndarray
) -> float:
    """
    Integrate Ne·ds along a ray path.
    ds ≈ sqrt((d_lat·deg2m)**2 + d_alt**2)
    Follows Eqn A.12 in the paper. 
    """
    deg2m = (np.pi/180.0)*R_E
    stec  = 0.0
    for (lat0, alt0), (lat1, alt1) in zip(path[:-1], path[1:]):
        i_lat = int(np.argmin(np.abs(lats - lat0)))
        i_alt = int(np.argmin(np.abs(alts - alt0)))
        dlat  = (lat1 - lat0)*deg2m
        dalt  = alt1 - alt0
        ds    = np.hypot(dlat, dalt)
        stec += ionosphere[i_alt, i_lat]*ds
    return stec

def dualfreq_to_VTEC_old(
    stec_slant: np.ndarray,  # from compute_STEC_along_path
    f_c:         float,      # center frequency [Hz]
    bw:          float,      # total bandwidth [Hz]
    theta_deg:   float       # incidence angle [deg]
) -> np.ndarray:
    """
    Dual-frequency TEC estimate (Peters et al. Eq. 9):
    1) f1 = f_c - bw/2, f2 = f_c + bw/2
    2) Compute group delay difference: Δt = C·STEC·(1/f1² − 1/f2²)
    3) Invert Eq. 9: TEC = (c·Δt/80.6)·[f1² f2²/(f2²−f1²)]
    [basically c*stec_slant = TEC because we simulate delta_t]
    4) Verticalize via cos(theta)
    """
    C = 40.3   # group-delay constant [m^3/s^2] - two-way
    c = 3e8    # speed of light [m/s]
    f1 = f_c - bw/2.0
    f2 = f_c + bw/2.0

    # Step 2: Compute group delay difference (Eq. 14, Kintner and Ledvina (2005)) [In reality, this is the data we receive from the receiver]
    delta_t = C * stec_slant * (1.0 / f1**2 - 1.0 / f2**2)
    # try using integration time here from the plot:
    # delta_t = 1
    
    # Step 3: Invert Eq. 9 to get TEC
    freq_factor = (f1**2 * f2**2) / (f2**2 - f1**2)
    TEC = (c * delta_t / 80.6) * freq_factor

    # Step 4: Verticalize
    theta = np.deg2rad(theta_deg)
    VTEC = TEC * np.cos(theta)
    return VTEC

def dualfreq_to_VTEC_old2(
    stec_slant: np.ndarray,  # from compute_STEC_along_path
    f_c:         float,      # center frequency [Hz]
    bw:          float,      # total bandwidth [Hz]
    theta_deg:   float,       # incidence angle [deg]
    integration_time: float = 1.0,  # Add this!
    return_deltat: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Dual-frequency TEC estimate (Peters et al. Eq. 9):
    1) f1 = f_c - bw/2, f2 = f_c + bw/2
    2) Compute group delay difference: Δt = C·STEC·(1/f1² − 1/f2²)
    3) Invert Eq. 9: TEC = (c·Δt/80.6)·[f1² f2²/(f2²−f1²)]
    [basically c*stec_slant = TEC because we simulate delta_t]
    4) Verticalize via cos(theta)
    """
    C = 80.6   # group-delay constant [m^3/s^2] - two-way
    c = 3e8    # speed of light [m/s]
    f1 = f_c - bw/2.0
    f2 = f_c + bw/2.0

    # Step 2: Compute group delay difference (Eq. 14, Kintner and Ledvina (2005)) [In reality, this is the data we receive from the receiver]
    inv_freq_diff = (1.0 / f1**2 - 1.0 / f2**2)
    delta_t = (C / c) * stec_slant * inv_freq_diff

    # Integration time-dependent delay noise
    base_noise_scale = 2e-7  # matches ~0.2 µs error at 0.1 s and 1 MHz
    sigma_dt = base_noise_scale / np.sqrt(bw * integration_time)
    delta_t += np.random.normal(0, sigma_dt, size=delta_t.shape)

    freq_factor = (f1**2 * f2**2) / (f2**2 - f1**2)
    TEC = (c * delta_t / 80.6) * freq_factor

    # Step 4: Verticalize
    theta = np.deg2rad(theta_deg)
    VTEC = TEC * np.cos(theta)
    if return_deltat:
        return VTEC, delta_t
    else:
        return VTEC

def dualfreq_to_VTEC(
    stec_slant: np.ndarray,  # from compute_STEC_along_path
    f_c: float,              # center frequency [Hz]
    bw: float,               # total bandwidth [Hz]
    theta_deg: float,        # incidence angle [deg]
    integration_time: float = 1.0,  # [s]
    snr_linear: float = 10.0,       # linear SNR (power ratio, not dB)
    return_deltat: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Dual-frequency TEC estimate (Peters et al. Eq. 9):
    1) f1 = f_c - bw/2, f2 = f_c + bw/2
    2) Compute group delay difference: Δt = (C / c) · STEC · (1/f1² − 1/f2²)
    3) Invert Eq. 9: TEC = (c · Δt / C) · [f1² f2² / (f2² − f1²)]
    4) Verticalize via cos(theta)
    Additive white Gaussian delay noise is simulated based on autocorrelation peak resolution.

    Parameters:
        snr_linear : power SNR for passive signal (default = 10)

    Returns:
        VTEC or (VTEC, Δt) depending on return_deltat
    """
    C = 40.3 #80.6   # group-delay constant [m^3/s^2]
    c = 3e8    # speed of light [m/s]
    f1 = f_c - bw / 2.0
    f2 = f_c + bw / 2.0

    # Step 1: group delay difference from Eq. 14 (Kintner and Ledvina)
    inv_freq_diff = (1.0 / f1**2 - 1.0 / f2**2)
    delta_t = (C / c) * stec_slant * inv_freq_diff

    # Step 2: simulate timing error from matched filter theory (Appendix B)
    sigma_dt = 1.0 / (2 * np.pi * bw * np.sqrt(integration_time) * snr_linear)
    delta_t += np.random.normal(0, sigma_dt, size=delta_t.shape)

    # Step 3: TEC inversion (Eq. 9)
    freq_factor = (f1**2 * f2**2) / (f2**2 - f1**2)
    TEC = (c * delta_t / C) * freq_factor

    # Step 4: Verticalize to get VTEC
    theta = np.deg2rad(theta_deg)
    VTEC = TEC * np.cos(theta)

    if return_deltat:
        return VTEC, delta_t
    else:
        return VTEC

def plot_ray_tracing(ray_paths, lats, alts, background, reconstruction):
    """
    Visualize ray‐paths over the true ionosphere and the ART reconstruction.

    Parameters
    ----------
    ray_paths     : list of (n_pts, 2) arrays, each [[lat, alt], …]
    lats          : 1D array of latitude grid (degrees)
    alts          : 1D array of altitude grid (meters)
    background    : 2D array [n_alt x n_lat] of the “true” Ne (or scaled) values
    reconstruction: 2D array [n_alt x n_lat] of the reconstructed Ne

    Produces a side‐by‐side figure:
      • Left:  background map + overlayed ray paths  
      • Right: reconstructed map (with faint ray overlay for reference)
    """
    # Convert altitude to km for plotting
    alts_km = alts / 1e3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # — Left: true ionosphere + rays —
    im1 = ax1.pcolormesh(lats, alts_km, background, shading='auto')
    for path in ray_paths:
        ax1.plot(path[:, 0], path[:, 1] / 1e3,
                 color='k', linewidth=0.5, alpha=0.3)
    ax1.set_title('True Ionosphere + Ray‐paths')
    ax1.set_xlabel('Latitude (°)')
    ax1.set_ylabel('Altitude (km)')
    plt.colorbar(im1, ax=ax1, label='Ne (arb. units)')

    # — Right: ART reconstruction —
    im2 = ax2.pcolormesh(lats, alts_km, reconstruction,
                         shading='auto', cmap='viridis')
    for path in ray_paths:
        ax2.plot(path[:, 0], path[:, 1] / 1e3,
                 color='k', linewidth=0.3, alpha=0.15)
    ax2.set_title('Reconstructed Ionosphere')
    ax2.set_xlabel('Latitude (°)')
    plt.colorbar(im2, ax=ax2, label='Ne (arb. units)')

    plt.tight_layout()
    plt.show()

def reconstruct_art_sparse(D, y, n_lat, n_alt, n_iters=50, relax=0.2):
    """
    Kaczmarz ART for SciPy CSR/CSC without densifying rows.
    x update uses only the row's nonzero indices.
    """
    assert sp.issparse(D)
    D = D.tocsr()
    n_rays, n_pix = D.shape
    x = np.zeros(n_pix, dtype=np.float32)

    y = np.asarray(y, dtype=np.float32)

    for _ in range(n_iters):
        for i in range(n_rays):
            row = D.getrow(i)
            idx = row.indices
            dat = row.data
            if dat.size == 0:
                continue
            proj  = (dat * x[idx]).sum()          # d_i · x
            denom = (dat * dat).sum()             # ||d_i||^2
            if denom <= 0:
                continue
            resid = y[i] - proj
            coef  = relax * (resid / denom)
            x[idx] += coef * dat                  # x += coef * d_i (sparse)
    return x.reshape((n_alt, n_lat))

def reconstruct_art(D, stec, n_lat, n_alt, n_iters=10, relax=0.1):
    """
    Simple ART (Kaczmarz) reconstruction:
      D       : geometry matrix (n_rays x n_pixels)
      stec    : measured projections
      n_lat   : number of latitude bins
      n_alt   : number of altitude bins
      n_iters : total ART iterations
      relax   : relaxation parameter (lambda)
    Returns:
      Ne_rec  : 2D reconstructed electron density [n_alt x n_lat]
    """
    n_rays, n_pix = D.shape
    Ne = np.zeros(n_pix)

    for _ in range(n_iters):
        for i in range(n_rays):
            Di = D[i]
            proj = Di @ Ne
            denom = Di @ Di
            if denom > 0:
                Ne += relax * (stec[i] - proj) / denom * Di

    return Ne.reshape((n_alt, n_lat))


def reconstruct_art_delt(D: np.ndarray, y: np.ndarray, n_lat: int, n_alt: int,
                    n_iters: int = 25, relax: float = 0.4) -> np.ndarray:
    """
    Algebraic Reconstruction Technique (ART) solver.
    Args:
        D       : geometry matrix [M x N]
        y       : VTEC measurements [M]
        n_lat   : number of latitude grid points
        n_alt   : number of altitude grid points
        n_iters : number of ART iterations
        relax   : relaxation factor for update
    Returns:
        Ne [n_alt x n_lat] estimated electron density grid
    """
    M, N = D.shape
    x = np.zeros(N)  # initial guess

    for _ in range(n_iters):
        for i in range(M):
            Di = D[i, :]
            denom = np.dot(Di, Di)
            if denom > 0:
                residual = y[i] - np.dot(Di, x)
                x += (relax * residual / denom) * Di

    return x.reshape((n_alt, n_lat))





#-----------------------------------
def trace_rays(ionosphere, lats, alts, tx_lats, sat_lats, npts=500):
    """
    Shoot straight rays through a 2D ionosphere map:
      ionosphere: 2D array [n_alt x n_lat] of Ne values
      lats       : 1D array of latitude grid (degrees)
      alts       : 1D array of altitude grid (m)
      tx_lats    : list of transmitter latitudes (degrees)
      sat_lats   : list of satellite track latitudes (degrees)
      npts       : number of samples per ray

    Returns:
      ray_paths : list of shape (npts, 2) arrays of (lat, alt)
      stec      : 1D array of accumulated Ne along each ray
      D         : geometry matrix [n_rays x (n_lat*n_alt)] of 0/1 sampling
    """
    n_lat = len(lats)
    n_alt = len(alts)
    n_rays = len(tx_lats) * len(sat_lats)

    # pre-allocate
    D = np.zeros((n_rays, n_lat * n_alt))
    stec = np.zeros(n_rays)
    ray_paths = []

    i = 0
    for tx in tx_lats:
        # map tx latitude into grid-index space
        x0 = np.interp(tx, lats, np.arange(n_lat))
        y0 = np.interp(alts[0], alts, np.arange(n_alt))  # if flight at surface; adjust if needed

        for sat in sat_lats:
            x1 = np.interp(sat, lats, np.arange(n_lat))
            y1 = np.interp(alts[-1], alts, np.arange(n_alt))

            # sample equally spaced points in index‐space
            xs = np.linspace(x0, x1, npts)
            ys = np.linspace(y0, y1, npts)

            this_path = []
            s = 0.0
            for xi, yi in zip(xs, ys):
                ix, iy = int(round(xi)), int(round(yi))
                if 0 <= ix < n_lat and 0 <= iy < n_alt:
                    s += ionosphere[iy, ix]
                    D[i, iy * n_lat + ix] = 1
                    this_path.append((lats[ix], alts[iy]))
            ray_paths.append(np.array(this_path))
            stec[i] = s
            i += 1

    return ray_paths, stec, D

def invert_delay(delta_t, theta_deg,
                 f1=8.5e6, f2=9.5e6,    # HF twin-frequencies
                 c=3e8):                # speed of light
    """
    Given a delay difference delta_t (s) at incidence angle theta_deg (°),
    returns the vertical TEC estimate (m^-2).
    """
    # 1) frequency factor from Peters Eq. 9
    freq_factor = (f1**2 * f2**2) / (f2**2 - f1**2)
    # 2) raw slant TEC from delay (Peters Eq. 6)
    tec_slant = (c * delta_t / 40.3) * freq_factor #(c * delta_t / 80.6) * freq_factor
    # 3) map to vertical TEC (Appendix A / Eq. A.13)
    tec_vertical = tec_slant * np.cos(np.radians(theta_deg))
    return tec_vertical