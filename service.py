"""
service.py -- Phase 2: the always-on propagation service

Architecture:

    background asyncio task  ->  in-memory snapshot  ->  FastAPI endpoints  ->  browser

A single background task propagates the whole catalogue on a fixed cadence and
stores the result in a module-level snapshot. HTTP requests only READ that
snapshot -- they never trigger propagation. Consequences:

  * request latency is a dict lookup, not an SGP4 run
  * N browser tabs cost the same CPU as one
  * the propagation rate is a deliberate parameter, not a side effect of traffic

Nothing here is written to disk. Satellite state at time t is worthless at t+5s,
so persisting it would burn microSD write endurance for no benefit.

Run with:   uvicorn service:app --host 0.0.0.0 --port 8000
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from sgp4.api import jday
from skyfield.api import load

import propagate

# ---------------- Configuration ----------------
SITE_LAT = 37.336812334419164   # deg, +North
SITE_LON = -121.88117116201111  # deg, +East
SITE_ALT = 26.0                 # metres above the WGS84 ellipsoid

GPS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"
PROPAGATION_INTERVAL_S = 2.0    # how often the loop recomputes the whole catalogue
TLE_REFRESH_INTERVAL_S = 6 * 3600  # re-fetch TLEs every 6 hours
EL_MASK_DEG = 10.0              # horizon mask for "visible"

# ---------------- Shared in-memory state ----------------
# Read by endpoints, written only by the background loop. Single-writer, so no
# lock is needed: each cycle rebuilds a fresh dict and rebinds this name, which
# is atomic in CPython. Readers always see a complete, self-consistent snapshot.
_snapshot = {
    "updated_utc": None,
    "site": {"lat_deg": SITE_LAT, "lon_deg": SITE_LON, "alt_m": SITE_ALT},
    "satellite_count": 0,
    "satellites": [],
    "error": "service starting, no propagation cycle has run yet",
}

_satellites = []       # parsed Skyfield EarthSatellite objects
_tles_loaded_at = None


def _load_tles():
    """Fetch (and locally cache) the GPS TLE set."""
    global _satellites, _tles_loaded_at
    _satellites = load.tle_file(GPS_TLE_URL)
    _tles_loaded_at = datetime.now(timezone.utc)
    return len(_satellites)


def _propagate_all():
    """Run the full hand-rolled pipeline for every satellite at the current instant."""
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day,
                  now.hour, now.minute, now.second + now.microsecond * 1e-6)

    results = []
    for sat in _satellites:
        try:
            r = propagate.look_angles(sat.model, SITE_LAT, SITE_LON, SITE_ALT, jd, fr)
            results.append({
                "name": sat.name,
                "norad_id": int(sat.model.satnum),
                "az_deg": round(float(r["az_deg"]), 4),
                "el_deg": round(float(r["el_deg"]), 4),
                "range_km": round(float(r["range_km"]), 3),
                "lat_deg": round(float(r["sub_lat_deg"]), 5),
                "lon_deg": round(float(r["sub_lon_deg"]), 5),
                "alt_km": round(float(r["sub_alt_km"]), 3),
                "visible": bool(r["el_deg"] > EL_MASK_DEG),
            })
        except Exception as ex:
            results.append({"name": sat.name, "error": str(ex)})

    return {
        "updated_utc": now.isoformat(),
        "site": {"lat_deg": SITE_LAT, "lon_deg": SITE_LON, "alt_m": SITE_ALT},
        "elevation_mask_deg": EL_MASK_DEG,
        "satellite_count": len(results),
        "satellites": results,
    }


async def _propagation_loop():
    """Background task: refresh TLEs periodically, propagate on a fixed cadence."""
    global _snapshot
    while True:
        try:
            stale = (_tles_loaded_at is None or
                     (datetime.now(timezone.utc) - _tles_loaded_at).total_seconds()
                     > TLE_REFRESH_INTERVAL_S)
            if stale:
                # Blocking network I/O -- push to a thread so the event loop
                # (and therefore the HTTP server) stays responsive.
                count = await asyncio.to_thread(_load_tles)
                print(f"[tle] loaded {count} satellites")

            # SGP4 for the whole catalogue is CPU-bound; offload it too so
            # requests are never blocked mid-cycle.
            _snapshot = await asyncio.to_thread(_propagate_all)

        except Exception as ex:
            print(f"[loop] error: {ex}")
            _snapshot = {**_snapshot, "error": str(ex)}

        await asyncio.sleep(PROPAGATION_INTERVAL_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the propagation loop on boot, cancel it cleanly on shutdown."""
    task = asyncio.create_task(_propagation_loop())
    print("[service] propagation loop started")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    print("[service] propagation loop stopped")


app = FastAPI(
    title="Orbit Visualiser",
    description="Live GPS constellation propagation and ground-site visibility.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Liveness check: is the service up and is its snapshot fresh?"""
    updated = _snapshot.get("updated_utc")
    age = None
    if updated:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(updated)).total_seconds()
    return {
        "status": "ok",
        "snapshot_age_s": round(age, 3) if age is not None else None,
        "satellite_count": _snapshot.get("satellite_count", 0),
        "tles_loaded_utc": _tles_loaded_at.isoformat() if _tles_loaded_at else None,
    }


@app.get("/api/satellites")
def all_satellites():
    """Current state of the entire catalogue."""
    return _snapshot


@app.get("/api/visible")
def visible_satellites():
    """Only satellites currently above the elevation mask, highest first."""
    sats = [s for s in _snapshot.get("satellites", []) if s.get("visible")]
    sats.sort(key=lambda s: -s["el_deg"])
    return {
        "updated_utc": _snapshot.get("updated_utc"),
        "elevation_mask_deg": EL_MASK_DEG,
        "count": len(sats),
        "satellites": sats,
    }


@app.get("/api/satellite/{norad_id}")
def one_satellite(norad_id: int):
    """State of a single satellite by NORAD catalogue number."""
    for s in _snapshot.get("satellites", []):
        if s.get("norad_id") == norad_id:
            return {"updated_utc": _snapshot.get("updated_utc"), **s}
    raise HTTPException(status_code=404, detail=f"NORAD ID {norad_id} not in catalogue")
