"""
propagate.py -- coordinate transforms and orbital propagation core

The pipeline is unchanged from Phase 1; it is now parameterised by a Body so the
same code serves Earth/SGP4 and Moon/Kepler:

    state source  -> inertial position     Stage 0  (SGP4, or Kepler solve)
    inertial      -> body-fixed            Stage 1  (Body.inertial_to_fixed)
    body-fixed    -> geodetic subpoint     Stage 2  (ground track)
    body-fixed    -> topocentric az/el     Stage 3  (visibility)

Stages 2 and 3 are genuinely body-agnostic. Stage 2 collapses to the spherical
case automatically when the body's flattening is 0 (the iteration converges on
the first pass because e^2 = 0), so no separate sphere/ellipsoid branch exists.
Stage 3 never depended on the central body at all -- the East-North-Up rotation
only cares about the observer's latitude and longitude.

All distances are kilometres unless a name says otherwise.
"""

import math

import numpy as np

from bodies import EARTH


# ---------------- Stage 0a: SGP4 -> TEME (Earth) ----------------
def teme_position_km(satrec, jd, fr):
    """Run SGP4 for one satellite at Julian date (jd + fr). Returns TEME km."""
    error, r, v = satrec.sgp4(jd, fr)
    if error != 0:
        raise RuntimeError(f"SGP4 error code {error}")
    return np.array(r)


# ---------------- Stage 0b: Kepler elements -> inertial ----------------
def solve_kepler(M, e, tol=1e-12, itmax=60):
    """Solve M = E - e sin E for eccentric anomaly E by Newton iteration."""
    E = M if e < 0.8 else math.pi
    for _ in range(itmax):
        d = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= d
        if abs(d) < tol:
            break
    return E


def advance_true_anomaly(sma, ecc, ta0_deg, dt_s, mu):
    """True anomaly (deg) dt seconds after starting at ta0.

    true -> eccentric -> mean anomaly, advance linearly in mean anomaly at the
    mean motion n = sqrt(mu/a^3), then invert back. Mean anomaly is the only one
    of the three that is linear in time, which is the whole reason for the
    round trip.
    """
    n = math.sqrt(mu / sma ** 3)
    nu0 = math.radians(ta0_deg)
    E0 = math.atan2(math.sqrt(1 - ecc ** 2) * math.sin(nu0), ecc + math.cos(nu0))
    M = (E0 - ecc * math.sin(E0)) + n * dt_s
    E = solve_kepler(M % (2 * math.pi), ecc)
    nu = math.atan2(math.sqrt(1 - ecc ** 2) * math.sin(E), math.cos(E) - ecc)
    return math.degrees(nu)


def kepler_to_inertial(sma, ecc, inc_deg, raan_deg, aop_deg, ta_deg, mu):
    """Classical elements -> inertial position and velocity (km, km/s).

    Build the state in the perifocal frame, where the orbit is a plane conic:
        r_pqw = [r cos(nu), r sin(nu), 0],  r = a(1-e^2)/(1 + e cos nu)
        v_pqw = sqrt(mu/p) * [-sin(nu), e + cos(nu), 0]
    then rotate into the inertial frame with R3(-RAAN) R1(-i) R3(-AOP).
    """
    i, Om, w, nu = map(math.radians, (inc_deg, raan_deg, aop_deg, ta_deg))
    p = sma * (1.0 - ecc ** 2)
    r = p / (1.0 + ecc * math.cos(nu))

    r_pqw = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    v_pqw = np.array([-math.sqrt(mu / p) * math.sin(nu),
                      math.sqrt(mu / p) * (ecc + math.cos(nu)), 0.0])

    cO, sO = math.cos(Om), math.sin(Om)
    ci, si = math.cos(i), math.sin(i)
    cw, sw = math.cos(w), math.sin(w)
    Q = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci     ],
    ])
    return Q @ r_pqw, Q @ v_pqw


# ---------------- Stage 1: inertial -> body-fixed ----------------
def inertial_to_fixed(r_inertial, body, jd):
    """Rotate an inertial position into the body-fixed frame."""
    return body.inertial_to_fixed(jd) @ r_inertial


# ---------------- Stage 2: body-fixed -> geodetic subpoint ----------------
def fixed_to_geodetic(r_fixed, body):
    """Body-fixed position (km) -> (lat_rad, lon_rad, alt_km) on the body's ellipsoid.

    Longitude is closed form. Latitude needs an iteration on an ellipsoid because
    the local normal does not pass through the centre. When flattening is zero
    (spherical body) e^2 = 0, N = a, and the first pass is already exact -- the
    same code covers both cases.
    """
    x, y, z = r_fixed
    a, e2 = body.radius_km, body.e2

    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - e2))
    alt = 0.0
    for _ in range(5):
        N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)   # prime-vertical radius
        alt = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - e2 * N / (N + alt)))
    return lat, lon, alt


def geodetic_to_fixed(lat_rad, lon_rad, alt_km, body):
    """Geodetic (lat, lon, alt) -> body-fixed position vector (km)."""
    a, e2 = body.radius_km, body.e2
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    return np.array([
        (N + alt_km) * math.cos(lat_rad) * math.cos(lon_rad),
        (N + alt_km) * math.cos(lat_rad) * math.sin(lon_rad),
        (N * (1 - e2) + alt_km) * math.sin(lat_rad),
    ])


# ---------------- Stage 3: body-fixed -> topocentric az/el ----------------
def fixed_to_enu(rho_fixed, lat0_rad, lon0_rad):
    """Rotate a body-fixed range vector into the observer's East-North-Up frame.

    Body-agnostic: this only depends on where the observer stands on the body.
    """
    slat, clat = math.sin(lat0_rad), math.cos(lat0_rad)
    slon, clon = math.sin(lon0_rad), math.cos(lon0_rad)
    R = np.array([
        [-slon,         clon,        0.0 ],
        [-slat * clon, -slat * slon, clat],
        [ clat * clon,  clat * slon, slat],
    ])
    return R @ rho_fixed


def enu_to_azel(enu):
    """ENU range vector -> (az_rad, el_rad, range_km). Azimuth clockwise from north."""
    e, n, u = enu
    rng = float(np.linalg.norm(enu))
    el = math.asin(u / rng)
    az = math.atan2(e, n) % (2 * math.pi)
    return az, el, rng


# ---------------- Full chain ----------------
def look_angles_from_inertial(r_inertial, body, jd,
                              obs_lat_deg, obs_lon_deg, obs_alt_km):
    """Stages 1-3 for one inertial position. Body-agnostic."""
    r_fixed = inertial_to_fixed(r_inertial, body, jd)
    sub_lat, sub_lon, sub_alt = fixed_to_geodetic(r_fixed, body)

    obs_lat, obs_lon = math.radians(obs_lat_deg), math.radians(obs_lon_deg)
    r_obs = geodetic_to_fixed(obs_lat, obs_lon, obs_alt_km, body)
    enu = fixed_to_enu(r_fixed - r_obs, obs_lat, obs_lon)
    az, el, rng = enu_to_azel(enu)

    return {
        "az_deg": math.degrees(az),
        "el_deg": math.degrees(el),
        "range_km": rng,
        "sub_lat_deg": math.degrees(sub_lat),
        "sub_lon_deg": math.degrees(sub_lon),
        "sub_alt_km": sub_alt,
    }


def look_angles(satrec, obs_lat_deg, obs_lon_deg, obs_alt_m, jd, fr, body=EARTH):
    """Earth/SGP4 convenience wrapper, signature-compatible with Phase 1.

    Note obs_alt is in METRES here for backward compatibility with validate.py
    and the Phase 1-3 service; the body-agnostic core works in kilometres.
    """
    r_teme = teme_position_km(satrec, jd, fr)
    return look_angles_from_inertial(r_teme, body, jd + fr,
                                     obs_lat_deg, obs_lon_deg, obs_alt_m / 1000.0)
