"""Quick sanity checks for TEC ↔ Δt consistency.

Run as: python -m tools.iono_consistency_check
"""
import numpy as np
from physics_constants import C_LIGHT, C_IONO


def slab_test(Ne_value=1e10, alt_top_m=200e3, f_low=8.5e6, f_high=9.5e6, tol_rel=0.02):
    """Build a vertical uniform slab Ne and test forward/inverse TEC↔Δt.

    Ne_value : electron density (m^-3)
    alt_top_m : top of slab altitude in meters
    Returns: True if within tolerance
    """
    # STEC (vertical) = integral Ne ds over 0..alt_top (two-way not applied)
    stec = Ne_value * alt_top_m

    # forward: delta_t = (C_IONO / C_LIGHT) * STEC * (1/f_low^2 - 1/f_high^2)
    dt = (C_IONO / C_LIGHT) * stec * (1.0 / (f_low**2) - 1.0 / (f_high**2))

    # invert: TEC = (C_LIGHT * dt / C_IONO) * freq_factor
    freq_factor = (f_low**2 * f_high**2) / (f_high**2 - f_low**2)
    tec_inv = (C_LIGHT * dt / C_IONO) * freq_factor

    # verticalization (theta=0) should match stec
    rel_err = abs(tec_inv - stec) / max(abs(stec), 1.0)
    return rel_err <= tol_rel, rel_err, stec, tec_inv


def oblique_test(theta_deg=30.0):
    # Simple geometry: VTEC = STEC * cos(theta)
    v = 5e14
    stec = v / np.cos(np.deg2rad(theta_deg))
    return stec, v


def main():
    ok, rel_err, stec, tec_inv = slab_test()
    if ok:
        print("GREEN: vertical slab test passed. rel_err=", rel_err)
    else:
        print("RED: vertical slab test FAILED. rel_err=", rel_err)

    stec_ob, v = oblique_test(30.0)
    print(f"Oblique demo: theta=30°, STEC~{stec_ob:.3e}, VTEC~{v:.3e}")


if __name__ == '__main__':
    main()
