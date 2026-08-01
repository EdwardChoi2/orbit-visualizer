"""
analysis.py -- 28-day availability sweep, matching the study's methodology

The live dashboard shows a one-hour rolling window, which is an operational
"right now" figure. It is NOT comparable to the trade study's published
availability, because lunar geometry beats against a ~27.3-day rotation period:
the same constellation reads 100% one week and much less the next.

This module computes the comparable number -- 28 days from the study epoch at a
300 s step, 7 deg mask, counting epochs with >= 4 satellites visible -- exactly
as the run log defines it:

    "Shack/Mare/Tsiol avail %  =  4-satellite availability at each site"

GDOP and PDOP are separate requirements and are reported separately, not folded
into availability. Conflating them over-constrains the metric and under-reports.

Cost control. A naive loop is ~226k Kepler solves plus transforms per scenario.
Two things make it affordable on a Pi 5:

  * Everything is vectorised over satellites with numpy. In two-body the
    perifocal->inertial rotation is CONSTANT per satellite (elements never
    change), so those matrices are built once and reused for every epoch.
  * The sweep is chunked, so the caller can yield to the event loop between
    chunks and the 2 s live loops keep their cadence.

Only Kepler (lunar) scenarios are swept. TLE accuracy degrades badly beyond
about a week, so a 28-day SGP4 sweep would be numerically meaningless -- better
to report nothing than a confidently wrong number.
"""

import math
from datetime import datetime, timedelta, timezone

import numpy as np

import propagate

STEP_S = 300.0          # study: CCSDS-OEM output at 300 s fixed step
DURATION_DAYS = 28      # study: "reliable standard for frozen families"
MIN_SATS = 4
CHUNK_EPOCHS = 400      # epochs per chunk, so the caller can yield between them


class SweepState:
    """Incremental state for one scenario's sweep, advanced chunk by chunk."""

    def __init__(self, scn):
        self.scn = scn
        self.body = scn.body
        self.mask_rad = math.radians(self.body.default_mask_deg)
        self.sites = list(scn.sites)

        src = scn.source
        self.epoch = src.epoch
        sats = src._sats
        self.n_sats = len(sats)

        # --- per-satellite constants, built once ---
        self.a = np.array([s["a"] for s in sats])
        self.e = np.array([s["e"] for s in sats])
        mu = self.body.mu_km3_s2
        self.n_mean = np.sqrt(mu / self.a ** 3)
        self.p = self.a * (1 - self.e ** 2)
        self.sqrt_mu_p = np.sqrt(mu / self.p)

        # mean anomaly at epoch, from the catalogued true anomaly
        nu0 = np.radians([s["ta0"] for s in sats])
        E0 = np.arctan2(np.sqrt(1 - self.e ** 2) * np.sin(nu0),
                        self.e + np.cos(nu0))
        self.M0 = E0 - self.e * np.sin(E0)

        # perifocal -> inertial rotation, CONSTANT in two-body: elements are fixed
        inc = np.radians([s["i"] for s in sats])
        Om = np.radians([s["raan"] for s in sats])
        w = np.radians([s["aop"] for s in sats])
        cO, sO, ci, si, cw, sw = (np.cos(Om), np.sin(Om), np.cos(inc),
                                  np.sin(inc), np.cos(w), np.sin(w))
        self.Q = np.empty((self.n_sats, 3, 3))
        self.Q[:, 0, 0] = cO * cw - sO * sw * ci
        self.Q[:, 0, 1] = -cO * sw - sO * cw * ci
        self.Q[:, 0, 2] = sO * si
        self.Q[:, 1, 0] = sO * cw + cO * sw * ci
        self.Q[:, 1, 1] = -sO * sw + cO * cw * ci
        self.Q[:, 1, 2] = -cO * si
        self.Q[:, 2, 0] = sw * si
        self.Q[:, 2, 1] = cw * si
        self.Q[:, 2, 2] = ci

        # --- per-site constants ---
        self.obs_pos, self.enu_R = [], []
        for s in self.sites:
            lat, lon = math.radians(s.lat_deg), math.radians(s.lon_deg)
            self.obs_pos.append(
                propagate.geodetic_to_fixed(lat, lon, s.alt_km, self.body))
            slat, clat, slon, clon = (math.sin(lat), math.cos(lat),
                                      math.sin(lon), math.cos(lon))
            self.enu_R.append(np.array([
                [-slon,        clon,        0.0 ],
                [-slat * clon, -slat * slon, clat],
                [ clat * clon,  clat * slon, slat],
            ]))
        self.obs_pos = np.array(self.obs_pos)

        # --- accumulators ---
        self.total_epochs = int(DURATION_DAYS * 86400 / STEP_S)
        self.done = 0
        n_sites = len(self.sites)
        self.n_ok = np.zeros(n_sites, dtype=np.int64)      # epochs with >= 4 sats
        self.gdops = [[] for _ in range(n_sites)]
        self.cur_outage = np.zeros(n_sites)                # running gap, epochs
        self.max_outage = np.zeros(n_sites)
        self.per_day_ok = np.zeros((n_sites, DURATION_DAYS), dtype=np.int64)
        self.per_day_n = np.zeros(DURATION_DAYS, dtype=np.int64)

    # ---------- vectorised propagation ----------
    def _positions_inertial(self, dt_s):
        """All satellite positions at dt seconds past epoch. Shape (n_sats, 3)."""
        M = self.M0 + self.n_mean * dt_s
        M = np.mod(M, 2 * np.pi)

        # Newton iteration on the whole array at once
        E = np.where(self.e < 0.8, M, np.pi)
        for _ in range(30):
            d = (E - self.e * np.sin(E) - M) / (1.0 - self.e * np.cos(E))
            E -= d
            if np.max(np.abs(d)) < 1e-12:
                break

        nu = np.arctan2(np.sqrt(1 - self.e ** 2) * np.sin(E), np.cos(E) - self.e)
        r = self.p / (1.0 + self.e * np.cos(nu))
        r_pqw = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(r)], axis=1)
        return np.einsum('nij,nj->ni', self.Q, r_pqw)

    @staticmethod
    def _gdop(unit_los):
        if unit_los.shape[0] < 4:
            return None
        G = np.hstack([-unit_los, np.ones((unit_los.shape[0], 1))])
        try:
            H = np.linalg.inv(G.T @ G)
        except np.linalg.LinAlgError:
            return None
        d = np.diag(H)
        if np.any(d < 0):
            return None
        return float(math.sqrt(d.sum()))

    def run_chunk(self):
        """Advance the sweep by up to CHUNK_EPOCHS. Returns True when finished."""
        end = min(self.done + CHUNK_EPOCHS, self.total_epochs)
        for k in range(self.done, end):
            t = self.epoch + timedelta(seconds=k * STEP_S)
            R = self.body.inertial_to_fixed(_jd(t))
            r_fixed = self._positions_inertial(k * STEP_S) @ R.T   # (n_sats, 3)
            day = min(int(k * STEP_S // 86400), DURATION_DAYS - 1)
            self.per_day_n[day] += 1

            for si in range(len(self.sites)):
                rho = r_fixed - self.obs_pos[si]
                enu = rho @ self.enu_R[si].T
                rng = np.linalg.norm(enu, axis=1)
                el = np.arcsin(enu[:, 2] / rng)
                vis = el > self.mask_rad
                nvis = int(np.count_nonzero(vis))

                if nvis >= MIN_SATS:
                    self.n_ok[si] += 1
                    self.per_day_ok[si, day] += 1
                    g = self._gdop(enu[vis] / rng[vis, None])
                    if g is not None:
                        self.gdops[si].append(g)
                    if self.cur_outage[si] > self.max_outage[si]:
                        self.max_outage[si] = self.cur_outage[si]
                    self.cur_outage[si] = 0
                else:
                    self.cur_outage[si] += 1
        self.done = end

        if self.done >= self.total_epochs:
            for si in range(len(self.sites)):
                if self.cur_outage[si] > self.max_outage[si]:
                    self.max_outage[si] = self.cur_outage[si]
            return True
        return False

    @property
    def progress(self):
        return self.done / self.total_epochs

    def result(self):
        sites = {}
        for si, s in enumerate(self.sites):
            g = self.gdops[si]
            per_day = [
                round(100.0 * self.per_day_ok[si, d] / self.per_day_n[d], 1)
                if self.per_day_n[d] else None
                for d in range(DURATION_DAYS)
            ]
            sites[s.name] = {
                "availability_pct": round(100.0 * self.n_ok[si] / self.total_epochs, 2),
                "median_gdop": round(float(np.median(g)), 2) if g else None,
                "max_outage_hr": round(self.max_outage[si] * STEP_S / 3600.0, 2),
                "per_day_pct": per_day,
            }
        return {
            "type": "analysis",
            "scenario": self.scn.id,
            "epoch_utc": self.epoch.isoformat(),
            "duration_days": DURATION_DAYS,
            "step_s": STEP_S,
            "elevation_mask_deg": self.body.default_mask_deg,
            "criterion": f">= {MIN_SATS} satellites above the mask",
            "epochs": self.total_epochs,
            "sites": sites,
        }


def _jd(dt):
    y, m = dt.year, dt.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    frac = (dt.hour + dt.minute / 60.0
            + (dt.second + dt.microsecond * 1e-6) / 3600.0) / 24.0
    return (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1))
            + dt.day + frac + b - 1524.5)
