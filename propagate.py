"""
propagate.py -- GPS constellation propagation core (Phase 1)

Hand-rolled coordinate transforms, one function per stage of the pipeline:

    SGP4  -> TEME          Stage 0: propagation model (via the sgp4 library --
                                    the model itself is NOT hand-rolled)
    TEME  -> ECEF          Stage 1: rotation by Greenwich Mean Sidereal Time
    ECEF  -> geodetic      Stage 2: WGS84 subpoint (ground track), iterative
    ECEF  -> topocentric   Stage 3: East-North-Up transform -> az / el / range

Every transform below is implemented straight from the equations so it can be
checked by hand and validated against Skyfield in validate.py. The only stage we
do not reimplement is SGP4, which is a large analytic model with no portfolio
value in rewriting.
"""

import numpy as np

# --- WGS84 ellipsoid constants ---
WGS84_A = 6378137.0                      # equatorial (semi-major) radius, metres
WGS84_F = 1.0 / 298.257223563            # flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2    # first eccentricity squared, e^2 = 2f - f^2


# ---------- Stage 0: SGP4 -> TEME ----------
def teme_position_km(satrec, jd, fr):
    """Run SGP4 for one satellite at Julian date (jd + fr).

    Returns the TEME position vector in kilometres. `satrec` is an sgp4 Satrec.
    """
    error, r, v = satrec.sgp4(jd, fr)
    if error != 0:
        raise RuntimeError(f"SGP4 error code {error}")
    return np.array(r)  # km, TEME frame


# ---------- Stage 1: TEME -> ECEF ----------
def gmst_radians(jd_ut1):
    """Greenwich Mean Sidereal Time as an angle in radians.

    Uses the days-since-J2000 formula: theta = 280.46061837 + 360.98564736629 * d.
    The 360.9856 deg/day term is Earth's sidereal rotation rate.
    """
    d = jd_ut1 - 2451545.0
    theta_deg = (280.46061837 + 360.98564736629 * d) % 360.0
    return np.radians(theta_deg)


def teme_to_ecef(r_teme, theta):
    """Rotate a TEME position vector into the Earth-fixed frame by GMST angle theta (rad).

    This is R3(theta), a rotation about the z-axis. The z-row is untouched because
    rotating about the pole does not change the polar component.
    """
    c, s = np.cos(theta), np.sin(theta)
    R3 = np.array([[ c,   s,   0.0],
                   [-s,   c,   0.0],
                   [ 0.0, 0.0, 1.0]])
    return R3 @ r_teme


# ---------- Stage 2: ECEF -> geodetic subpoint ----------
def ecef_to_geodetic(r_ecef):
    """Convert an ECEF position (metres) to WGS84 geodetic (lat_rad, lon_rad, alt_m).

    Longitude is closed-form; latitude/altitude need a short iteration because the
    Earth is an ellipsoid. Converges in 2-3 passes; 5 is plenty.
    """
    x, y, z = r_ecef
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))            # initial guess
    alt = 0.0
    for _ in range(5):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)  # prime-vertical radius
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + alt)))
    return lat, lon, alt


# ---------- Stage 3: ECEF -> topocentric az / el / range ----------
def geodetic_to_ecef(lat, lon, alt):
    """Convert geodetic (lat_rad, lon_rad, alt_m) to an ECEF position vector (metres)."""
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_enu(rho_ecef, lat0, lon0):
    """Rotate an ECEF range vector into the observer's East-North-Up frame."""
    slat, clat = np.sin(lat0), np.cos(lat0)
    slon, clon = np.sin(lon0), np.cos(lon0)
    R = np.array([[-slon,         clon,        0.0 ],
                  [-slat * clon, -slat * slon, clat],
                  [ clat * clon,  clat * slon, slat]])
    return R @ rho_ecef


def enu_to_azel(enu):
    """Convert an ENU range vector to (az_rad, el_rad, range_m).

    Azimuth is measured clockwise from north, wrapped into [0, 2*pi).
    """
    e, n, u = enu
    rng = np.linalg.norm(enu)
    el = np.arcsin(u / rng)
    az = np.arctan2(e, n) % (2 * np.pi)
    return az, el, rng


# ---------- Full chain ----------
def look_angles(satrec, obs_lat_deg, obs_lon_deg, obs_alt_m, jd, fr):
    """Full hand-rolled pipeline for one satellite from one ground site.

    Returns a dict with topocentric az/el/range and the geodetic subpoint.
    """
    # Stage 0: SGP4 -> TEME, converting km -> m
    r_teme = teme_position_km(satrec, jd, fr) * 1000.0

    # Stage 1: TEME -> ECEF. We treat (jd + fr) as UT1; the UT1-UTC offset is
    # under a second (~0.004 deg of rotation) and is intentionally ignored here.
    theta = gmst_radians(jd + fr)
    r_ecef = teme_to_ecef(r_teme, theta)

    # Stage 2: geodetic subpoint (the ground-track point)
    sub_lat, sub_lon, sub_alt = ecef_to_geodetic(r_ecef)

    # Stage 3: topocentric look-angles from the ground site
    obs_lat, obs_lon = np.radians(obs_lat_deg), np.radians(obs_lon_deg)
    r_obs = geodetic_to_ecef(obs_lat, obs_lon, obs_alt_m)
    rho = r_ecef - r_obs
    enu = ecef_to_enu(rho, obs_lat, obs_lon)
    az, el, rng = enu_to_azel(enu)

    return {
        "az_deg": np.degrees(az),
        "el_deg": np.degrees(el),
        "range_km": rng / 1000.0,
        "sub_lat_deg": np.degrees(sub_lat),
        "sub_lon_deg": np.degrees(sub_lon),
        "sub_alt_km": sub_alt / 1000.0,
    }
