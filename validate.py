"""
validate.py -- Phase 1 validation harness

Runs the hand-rolled pipeline (propagate.py) and Skyfield side by side on the live
GPS constellation, from a chosen ground site, at the current instant. Prints a
per-satellite az/el table and a residual summary.

The residual between the two paths IS the artifact. Both paths share the exact same
SGP4/TEME position (we hand Skyfield's parsed Satrec to our own code), so the
difference isolates the coordinate transforms: our GMST-only rotation versus
Skyfield's full TEME->ITRS reduction (precession, nutation, polar motion, UT1).
Expect a small fraction of a degree. A larger residual is a bug worth hunting.
"""

import numpy as np
from datetime import datetime, timezone
from sgp4.api import jday
from skyfield.api import load, wgs84

import propagate

# --- CHOOSE YOUR GROUND SITE (edit these three lines) ---
SITE_LAT = 37.336812334419164     # deg, +North   (example: San Jose, CA -- change to your site)
SITE_LON = -121.88117116201111   # deg, +East
SITE_ALT = 26.0        # metres above the WGS84 ellipsoid

GPS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"
EL_MASK_DEG = 10.0     # horizon mask for "visible now"


def main():
    ts = load.timescale()

    # Download (and locally cache) the live GPS TLEs via Skyfield's loader.
    sats = load.tle_file(GPS_TLE_URL)
    print(f"Loaded {len(sats)} GPS satellites")

    # One common timestamp feeds both paths.
    now = datetime.now(timezone.utc)
    t = ts.from_datetime(now)
    jd, fr = jday(now.year, now.month, now.day,
                  now.hour, now.minute, now.second + now.microsecond * 1e-6)
    observer = wgs84.latlon(SITE_LAT, SITE_LON, SITE_ALT)
    print(f"Site: {SITE_LAT:.4f}, {SITE_LON:.4f}    Time (UTC): {now:%Y-%m-%d %H:%M:%S}\n")

    rows = []
    for sat in sats:
        try:
            # hand-rolled path -- reuse the Satrec Skyfield already parsed
            hand = propagate.look_angles(sat.model, SITE_LAT, SITE_LON, SITE_ALT, jd, fr)
            # Skyfield path -- fully independent transform chain
            alt_sky, az_sky, _ = (sat - observer).at(t).altaz()
            d_el = hand["el_deg"] - alt_sky.degrees
            d_az = (hand["az_deg"] - az_sky.degrees + 180) % 360 - 180  # wrap to [-180,180)
            rows.append((sat.name, hand, alt_sky.degrees, az_sky.degrees, d_el, d_az))
        except Exception as ex:
            print(f"  skipped {sat.name}: {ex}")

    hdr = (f"{'Satellite':20}{'el hand':>10}{'el sky':>10}{'d_el':>9}"
           f"{'az hand':>10}{'az sky':>10}{'d_az':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, hand, el_sky, az_sky, d_el, d_az in rows:
        print(f"{name:20}{hand['el_deg']:10.4f}{el_sky:10.4f}{d_el:9.4f}"
              f"{hand['az_deg']:10.4f}{az_sky:10.4f}{d_az:9.4f}")

    d_el = np.array([r[4] for r in rows])
    d_az = np.array([r[5] for r in rows])
    print("\nResidual (hand-rolled minus Skyfield):")
    print(f"  elevation:  max |d| = {np.abs(d_el).max():.4f} deg    mean |d| = {np.abs(d_el).mean():.4f} deg")
    print(f"  azimuth:    max |d| = {np.abs(d_az).max():.4f} deg    mean |d| = {np.abs(d_az).mean():.4f} deg")

    print(f"\nVisible now (elevation > {EL_MASK_DEG:.0f} deg):")
    up = [(name, hand) for (name, hand, *_rest) in rows if hand["el_deg"] > EL_MASK_DEG]
    if up:
        for name, hand in sorted(up, key=lambda x: -x[1]["el_deg"]):
            print(f"  {name:20} el {hand['el_deg']:6.2f} deg    az {hand['az_deg']:7.2f} deg")
    else:
        print("  (none above the mask right now -- try again later)")


if __name__ == "__main__":
    main()
