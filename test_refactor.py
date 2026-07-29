"""
test_refactor.py -- prove the body-abstraction refactor did not change Earth results

The Phase 1 implementations are inlined below verbatim as the reference. The test
drives both the old and the new transform chain with the same synthetic inertial
positions and asserts they agree to floating-point noise.

This deliberately does NOT need network access or a TLE: SGP4 is untouched by the
refactor, so feeding synthetic inertial vectors isolates exactly the code that
changed (Stages 1-3).

Run:  python test_refactor.py
"""

import math

import numpy as np

import propagate
from bodies import EARTH, MOON

# ---------------- Phase 1 originals, inlined as the reference ----------------
OLD_A = 6378137.0
OLD_F = 1.0 / 298.257223563
OLD_E2 = 2 * OLD_F - OLD_F ** 2


def old_gmst_radians(jd_ut1):
    d = jd_ut1 - 2451545.0
    return np.radians((280.46061837 + 360.98564736629 * d) % 360.0)


def old_teme_to_ecef(r_teme, theta):
    c, s = np.cos(theta), np.sin(theta)
    R3 = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return R3 @ r_teme


def old_ecef_to_geodetic(r_ecef):
    x, y, z = r_ecef
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - OLD_E2))
    alt = 0.0
    for _ in range(5):
        N = OLD_A / np.sqrt(1 - OLD_E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - OLD_E2 * N / (N + alt)))
    return lat, lon, alt


def old_geodetic_to_ecef(lat, lon, alt):
    N = OLD_A / np.sqrt(1 - OLD_E2 * np.sin(lat) ** 2)
    return np.array([
        (N + alt) * np.cos(lat) * np.cos(lon),
        (N + alt) * np.cos(lat) * np.sin(lon),
        (N * (1 - OLD_E2) + alt) * np.sin(lat),
    ])


def old_ecef_to_enu(rho_ecef, lat0, lon0):
    slat, clat = np.sin(lat0), np.cos(lat0)
    slon, clon = np.sin(lon0), np.cos(lon0)
    R = np.array([[-slon, clon, 0.0],
                  [-slat * clon, -slat * slon, clat],
                  [clat * clon, clat * slon, slat]])
    return R @ rho_ecef


def old_enu_to_azel(enu):
    e, n, u = enu
    rng = np.linalg.norm(enu)
    return np.arctan2(e, n) % (2 * np.pi), np.arcsin(u / rng), rng


def old_chain(r_teme_m, jd, lat_deg, lon_deg, alt_m):
    """Phase 1 Stages 1-3, working in metres as the original did."""
    theta = old_gmst_radians(jd)
    r_ecef = old_teme_to_ecef(r_teme_m, theta)
    sub_lat, sub_lon, sub_alt = old_ecef_to_geodetic(r_ecef)
    lat0, lon0 = np.radians(lat_deg), np.radians(lon_deg)
    r_obs = old_geodetic_to_ecef(lat0, lon0, alt_m)
    az, el, rng = old_enu_to_azel(old_ecef_to_enu(r_ecef - r_obs, lat0, lon0))
    return {
        "az_deg": np.degrees(az),
        "el_deg": np.degrees(el),
        "range_km": rng / 1000.0,
        "sub_lat_deg": np.degrees(sub_lat),
        "sub_lon_deg": np.degrees(sub_lon),
        "sub_alt_km": sub_alt / 1000.0,
    }


# ---------------- Test data ----------------
SITE = (37.336812334419164, -121.88117116201111, 26.0)   # lat deg, lon deg, alt m

rng_ = np.random.default_rng(20260728)
CASES = []
for _ in range(400):
    # random GPS-like inertial positions, 20000-27000 km radius, any direction
    v = rng_.normal(size=3)
    v /= np.linalg.norm(v)
    r_km = v * rng_.uniform(20000.0, 27000.0)
    jd = 2451545.0 + rng_.uniform(0.0, 10000.0)
    CASES.append((r_km, jd))


def test_earth_regression():
    worst = {k: 0.0 for k in
             ("az_deg", "el_deg", "range_km", "sub_lat_deg", "sub_lon_deg", "sub_alt_km")}
    for r_km, jd in CASES:
        old = old_chain(r_km * 1000.0, jd, SITE[0], SITE[1], SITE[2])
        new = propagate.look_angles_from_inertial(
            r_km, EARTH, jd, SITE[0], SITE[1], SITE[2] / 1000.0)
        for k in worst:
            d = abs(old[k] - new[k])
            if k == "az_deg":                      # wrap azimuth difference
                d = abs((old[k] - new[k] + 180) % 360 - 180)
            worst[k] = max(worst[k], d)

    print(f"Earth regression over {len(CASES)} synthetic positions:")
    for k, v in worst.items():
        unit = "km" if k.endswith("_km") else "deg"
        print(f"  max |old - new|  {k:14s} = {v:.3e} {unit}")
    assert worst["el_deg"] < 1e-9, "elevation changed"
    assert worst["az_deg"] < 1e-9, "azimuth changed"
    assert worst["range_km"] < 1e-9, "range changed"
    assert worst["sub_lat_deg"] < 1e-9, "subpoint latitude changed"
    assert worst["sub_lon_deg"] < 1e-9, "subpoint longitude changed"
    assert worst["sub_alt_km"] < 1e-9, "subpoint altitude changed"
    print("  PASS: Earth path is numerically identical to Phase 1\n")


def test_moon_sanity():
    """The Moon path should be self-consistent, not compared against Earth."""
    # A point at a known selenographic location should round-trip through the
    # fixed<->geodetic pair exactly.
    lat, lon, alt = math.radians(-89.9), math.radians(0.0), 0.0
    r_fixed = propagate.geodetic_to_fixed(lat, lon, alt, MOON)
    lat2, lon2, alt2 = propagate.fixed_to_geodetic(r_fixed, MOON)
    print("Moon round-trip (Shackleton, on the surface):")
    print(f"  |r| = {np.linalg.norm(r_fixed):.6f} km  (expect {MOON.radius_km})")
    print(f"  d_lat = {abs(math.degrees(lat - lat2)):.3e} deg   "
          f"d_alt = {abs(alt - alt2):.3e} km")
    assert abs(np.linalg.norm(r_fixed) - MOON.radius_km) < 1e-9
    assert abs(math.degrees(lat - lat2)) < 1e-12

    # Spherical body: the geodetic iteration must equal the closed-form sphere.
    p = np.array([1200.0, -800.0, 900.0])
    lat_s = math.degrees(math.asin(p[2] / np.linalg.norm(p)))
    lat_i, _, _ = propagate.fixed_to_geodetic(p, MOON)
    print(f"  spherical check: closed-form {lat_s:.9f} vs iterative "
          f"{math.degrees(lat_i):.9f} deg")
    assert abs(lat_s - math.degrees(lat_i)) < 1e-9
    print("  PASS: spherical Moon reduces correctly, no separate code path\n")


def test_moon_rotation_is_orthonormal():
    for jd in (2451545.0, 2461000.5, 2470000.25):
        R = MOON.inertial_to_fixed(jd)
        err = np.abs(R @ R.T - np.eye(3)).max()
        det = np.linalg.det(R)
        assert err < 1e-12, f"not orthonormal at jd={jd}"
        assert abs(det - 1.0) < 1e-12, f"determinant {det} at jd={jd}"
    print("Moon rotation matrix: orthonormal, det = +1 at all sampled epochs")

    # The lunar pole should sit far from the MJ2000Eq z-axis -- this is the
    # whole reason the Moon needs three rotations where Earth needs one.
    R = MOON.inertial_to_fixed(2451545.0)
    pole_inertial = R.T @ np.array([0.0, 0.0, 1.0])
    tilt = math.degrees(math.acos(abs(pole_inertial[2])))
    print(f"  lunar pole is {tilt:.2f} deg off the MJ2000Eq z-axis "
          f"(Earth's TEME pole would be 0)\n")
    assert tilt > 10.0


def test_kepler_round_trip():
    """Elements -> state -> elements, and energy/momentum conservation."""
    a, e, i, Om, w, nu = 6541.4, 0.6, 62.94, 90.0, 90.013, 120.0
    mu = MOON.mu_km3_s2
    r, v = propagate.kepler_to_inertial(a, e, i, Om, w, nu, mu)

    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    energy = vn ** 2 / 2 - mu / rn
    a_back = -mu / (2 * energy)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / mu - r / rn
    i_back = math.degrees(math.acos(h[2] / np.linalg.norm(h)))

    print("Kepler elements -> state -> elements (HELO, e=0.6):")
    print(f"  SMA  in {a:.4f}  out {a_back:.4f} km      d = {abs(a-a_back):.3e}")
    print(f"  ECC  in {e:.4f}  out {np.linalg.norm(e_vec):.4f}       "
          f"d = {abs(e-np.linalg.norm(e_vec)):.3e}")
    print(f"  INC  in {i:.4f}  out {i_back:.4f} deg    d = {abs(i-i_back):.3e}")
    assert abs(a - a_back) < 1e-6
    assert abs(e - np.linalg.norm(e_vec)) < 1e-9
    assert abs(i - i_back) < 1e-9

    # Propagating a full period must return to the same true anomaly.
    T = 2 * math.pi * math.sqrt(a ** 3 / mu)
    nu_T = propagate.advance_true_anomaly(a, e, nu, T, mu)
    d = abs((nu_T - nu + 180) % 360 - 180)
    print(f"  true anomaly after exactly one period: d = {d:.3e} deg")
    assert d < 1e-6
    print("  PASS: Kepler solver and element conversion are self-consistent\n")


if __name__ == "__main__":
    test_earth_regression()
    test_moon_sanity()
    test_moon_rotation_is_orthonormal()
    test_kepler_round_trip()
    print("ALL TESTS PASSED")
