"""Physics constants and unit guidance for the ionosphere reconstruction repo.

All values use SI units (meters, seconds, Hz, electrons per m^3 / m^-2 for TEC).

Notes:
- C_LIGHT: speed of light in vacuum in m/s.
- C_IONO: ionospheric group-delay constant used to relate TEC ↔ group delay. Use
  40.3 (SI) here; older code sometimes used 80.6 which corresponds to a two-leg
  convention or different unit convention. We intentionally expose only 40.3 and
  enforce the single-source-of-truth here to avoid the ~×2 errors seen previously.
- R_EUROPA: mean radius of Europa (meters). Use this everywhere geometry needs
  a planetary radius.

Keep this module minimal: only declare the three constants above.
"""

# Speed of light [m/s]
C_LIGHT = 299_792_458.0

# Ionospheric group-delay constant (m^3 / s^2). Use 40.3 in SI units.
# Historical variants like 80.6 are implementations of a two-way convention or
# different unit conversions — DO NOT use those literals elsewhere.
C_IONO = 40.3

# Europa mean radius [m]
R_EUROPA = 1_560_800.0
