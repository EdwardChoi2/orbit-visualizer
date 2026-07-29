"""
sources.py -- satellite sources and scenario definitions

A Source answers one question: "where is every satellite, in the inertial frame,
at this instant?" How it answers is its own business:

    TLESource     TLE + SGP4                -> TEME          (Earth catalogues)
    KeplerSource  classical elements + Kepler -> MoonMJ2000Eq (Prometheus study)

Everything downstream -- the body-fixed rotation, the subpoint, the topocentric
look-angles -- is shared, because propagate.py was made body-agnostic in the
refactor. Adding a new constellation means adding a Source, not touching the
pipeline.

A Scenario pairs a Source with the observation sites that make sense for it.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import propagate
from bodies import EARTH, MOON, Body

# ---------------- Julian date ----------------
def datetime_to_jd(dt: datetime) -> float:
    """UTC datetime -> Julian date. Standard Fliegel-Van Flandern style conversion."""
    y, m = dt.year, dt.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    day_frac = (dt.hour + dt.minute / 60.0
                + (dt.second + dt.microsecond * 1e-6) / 3600.0) / 24.0
    return (math.floor(365.25 * (y + 4716))
            + math.floor(30.6001 * (m + 1))
            + dt.day + day_frac + b - 1524.5)


@dataclass(frozen=True)
class Site:
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0


# ---------------- Base ----------------
class Source:
    """Produces inertial positions for a set of satellites at a given instant."""

    id: str = ""
    label: str = ""
    body: Body = EARTH
    note: str = ""
    verified: bool = True

    def ensure_loaded(self):
        """Load or refresh whatever the source needs. Safe to call repeatedly."""
        raise NotImplementedError

    def is_ready(self) -> bool:
        raise NotImplementedError

    def positions(self, when: datetime):
        """-> list of dicts: {name, layer, r_km (np array, inertial frame)}."""
        raise NotImplementedError

    def track_half_window_s(self) -> float:
        """Half-width of the ground-track window, scaled to the orbit period."""
        raise NotImplementedError

    def layers(self):
        return []


# ---------------- Earth: TLE + SGP4 ----------------
class TLESource(Source):
    def __init__(self, id, label, url, refresh_s=6 * 3600, note=""):
        self.id, self.label, self.url = id, label, url
        self.body = EARTH
        self.note = note
        self.refresh_s = refresh_s
        self._sats = []
        self._loaded_at = None

    def ensure_loaded(self):
        from skyfield.api import load
        stale = (self._loaded_at is None or
                 (datetime.now(timezone.utc) - self._loaded_at).total_seconds()
                 > self.refresh_s)
        if stale:
            self._sats = load.tle_file(self.url)
            self._loaded_at = datetime.now(timezone.utc)
        return len(self._sats)

    def is_ready(self):
        return bool(self._sats)

    def positions(self, when: datetime):
        from sgp4.api import jday
        jd, fr = jday(when.year, when.month, when.day,
                      when.hour, when.minute,
                      when.second + when.microsecond * 1e-6)
        out = []
        for s in self._sats:
            try:
                r = propagate.teme_position_km(s.model, jd, fr)
            except Exception:
                continue
            out.append({"name": s.name.strip(), "layer": None,
                        "norad_id": int(s.model.satnum), "r_km": r})
        return out

    def track_half_window_s(self):
        return 3600.0        # +/- 1 hour reads well for MEO ground tracks

    @property
    def loaded_at(self):
        return self._loaded_at


# ---------------- Moon: classical elements + Kepler ----------------
class KeplerSource(Source):
    """Propagates a Prometheus config from its Keplerian elements.

    Two-body only. The source study used LP165P 20x20 gravity plus Earth/Sun
    third bodies, so this will drift from the GMAT ephemerides over days -- and
    notably will NOT preserve the frozen-orbit conditions (stable AOP and ECC),
    since those depend on the perturbed model. Correct for a live visualiser,
    wrong for study-accurate long propagation. Stated, not hidden.
    """

    def __init__(self, json_path, config_id, label=None, body=MOON):
        self.json_path = Path(json_path)
        self.config_id = config_id
        self.id = f"prometheus-{config_id.lower()}"
        self.label = label or f"Prometheus {config_id}"
        self.body = body
        self._sats = []
        self._epoch = None
        self._max_period_s = None
        self._layers = []
        self.note = ""
        self.verified = True

    def ensure_loaded(self):
        if self._sats:
            return len(self._sats)
        doc = json.loads(self.json_path.read_text())
        self._epoch = self._parse_epoch(doc["constants"]["epoch"])
        cfg = next(c for c in doc["configs"] if c["config_id"] == self.config_id)
        self.note = cfg.get("note", "")
        self.verified = bool(cfg.get("verified", True))

        mu = self.body.mu_km3_s2
        seen_layers = []
        for s in cfg["satellites"]:
            k = s["kepler"]
            layer = s.get("layer") or cfg.get("family")
            if layer not in seen_layers:
                seen_layers.append(layer)
            self._sats.append({
                "name": s["name"], "layer": layer, "plane": s.get("plane"),
                "a": k["SMA_km"], "e": k["ECC"], "i": k["INC_deg"],
                "raan": k["RAAN_deg"], "aop": k["AOP_deg"], "ta0": k["TA_deg"],
            })
        self._layers = seen_layers
        a_max = max(s["a"] for s in self._sats)
        self._max_period_s = 2 * math.pi * math.sqrt(a_max ** 3 / mu)
        return len(self._sats)

    @staticmethod
    def _parse_epoch(text):
        # "01 Jan 2026 00:00:00.000 UTC"
        cleaned = text.replace(" UTC", "").strip()
        for fmt in ("%d %b %Y %H:%M:%S.%f", "%d %b %Y %H:%M:%S"):
            try:
                return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"unparsed epoch: {text!r}")

    def is_ready(self):
        return bool(self._sats)

    def positions(self, when: datetime):
        mu = self.body.mu_km3_s2
        dt_s = (when - self._epoch).total_seconds()
        out = []
        for s in self._sats:
            ta = propagate.advance_true_anomaly(s["a"], s["e"], s["ta0"], dt_s, mu)
            r, _v = propagate.kepler_to_inertial(
                s["a"], s["e"], s["i"], s["raan"], s["aop"], ta, mu)
            out.append({"name": s["name"], "layer": s["layer"],
                        "plane": s.get("plane"), "r_km": r})
        return out

    def track_half_window_s(self):
        # Half an orbit either side shows the full track shape without overlap.
        return 0.5 * self._max_period_s

    def layers(self):
        return list(self._layers)

    @property
    def epoch(self):
        return self._epoch


# ---------------- Scenarios ----------------
@dataclass
class Scenario:
    id: str
    label: str
    source: Source
    sites: list
    group: str = ""
    coverage: dict = field(default_factory=dict)   # documented study results

    @property
    def body(self):
        return self.source.body


DATA_DIR = Path(__file__).parent / "data"
PROM = DATA_DIR / "prometheus_orbits.json"

EARTH_SITES = [Site("San Jose", 37.336812334419164, -121.88117116201111, 0.026)]

LUNAR_SITES = [
    Site("Shackleton", -89.90, 0.00, 0.0),
    Site("Mare Tranquillitatis", 0.68, 23.43, 0.0),
    Site("Tsiolkovskiy", -20.35, 128.97, 0.0),
]

# Documented availability from the trade study, Shackleton / Mare / Tsiolkovskiy.
# Only configs with figures I can point at are annotated; the rest stay blank
# rather than being filled in with plausible-looking numbers.
_COVERAGE = {
    "OPT-B":        {"Shackleton": 100.0, "Mare Tranquillitatis": 100.0, "Tsiolkovskiy": 99.9},
    "OPT-D":        {"Shackleton": 99.4, "Mare Tranquillitatis": 97.0, "Tsiolkovskiy": 97.8},
    "HELO-4P-16":   {"Shackleton": 100.0, "Mare Tranquillitatis": 88.4, "Tsiolkovskiy": 97.5},
    "FELO-F7d-16":  {"Shackleton": 0.0, "Mare Tranquillitatis": 32.0, "Tsiolkovskiy": 34.4},
    "LLO-4P-16-90": {"Shackleton": 0.0, "Mare Tranquillitatis": 0.0, "Tsiolkovskiy": 0.0},
}

_PROM_CONFIGS = [
    ("OPT-B",        "OPT-B  -  36 sats  -  recommended",     "Prometheus / combined"),
    ("OPT-D",        "OPT-D  -  28 sats  -  minimum viable",  "Prometheus / combined"),
    ("HELO-4P-16",   "HELO-4P-16  -  16 sats",                "Prometheus / HELO"),
    ("HELO-4P-12",   "HELO-4P-12  -  12 sats",                "Prometheus / HELO"),
    ("NS-FELO-20",   "NS-FELO-20  -  20 sats",                "Prometheus / NS-FELO"),
    ("FELO-F7d-16",  "FELO-F7d-16  -  16 sats",               "Prometheus / FELO"),
    ("FELO-F7-12",   "FELO-F7-12  -  12 sats",                "Prometheus / FELO"),
    ("LLO-3P-12-90", "LLO-3P-12-90  -  12 sats",              "Prometheus / LLO"),
    ("LLO-4P-16-90", "LLO-4P-16-90  -  16 sats  -  0% all sites", "Prometheus / LLO"),
]


def build_scenarios():
    scenarios = {}

    gps = TLESource(
        id="gps",
        label="GPS operational constellation",
        url="https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",
        note="Live TLEs from CelesTrak, propagated with SGP4.",
    )
    scenarios["gps"] = Scenario("gps", "GPS  -  live TLE", gps,
                                EARTH_SITES, group="Earth")

    for cfg_id, label, group in _PROM_CONFIGS:
        src = KeplerSource(PROM, cfg_id, label=label)
        scenarios[src.id] = Scenario(src.id, label, src, LUNAR_SITES,
                                     group=group,
                                     coverage=_COVERAGE.get(cfg_id, {}))
    return scenarios
