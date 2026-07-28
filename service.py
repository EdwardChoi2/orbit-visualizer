"""
service.py -- Phase 3: propagation service with WebSocket push and dashboard

Architecture:

    fast loop (2 s)    ->\
                          in-memory snapshot -> WebSocket broadcast -> browsers
    slow loop (5 min)  ->/

Two loops at different cadences because the two data products change at very
different rates:

  * live position (az/el/subpoint) changes every second        -> 2 s cadence
  * ground-track geometry is nearly identical minute to minute -> 5 min cadence

Recomputing 32 tracks x 61 points every 2 s would waste ~97% of that compute for
no visible benefit. Decoupling update rates by rate-of-change is what keeps this
comfortable on a Pi 5.

Nothing is written to disk. Satellite state at time t is worthless at t+5s.

Run with:   uvicorn service:app --host 0.0.0.0 --port 8000
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from sgp4.api import jday
from skyfield.api import load

import propagate

# ---------------- Configuration ----------------
SITE_LAT = 37.336812334419164   # deg, +North
SITE_LON = -121.88117116201111  # deg, +East
SITE_ALT = 26.0                 # metres above the WGS84 ellipsoid

GPS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"

PROPAGATION_INTERVAL_S = 2.0        # fast loop: live positions
TRACK_INTERVAL_S = 300.0            # slow loop: ground-track geometry
TLE_REFRESH_INTERVAL_S = 6 * 3600   # re-fetch TLEs every 6 hours

TRACK_HALF_WINDOW_MIN = 60          # track spans now +/- this many minutes
TRACK_STEP_MIN = 2                  # sample interval along the track

EL_MASK_DEG = 10.0                  # horizon mask for "visible"

STATIC_DIR = Path(__file__).parent / "static"

# ---------------- Shared in-memory state ----------------
# Written only by the background loops, read by endpoints and the broadcaster.
# Each cycle rebuilds a fresh dict and rebinds the name (atomic in CPython), so
# readers always see a complete, self-consistent snapshot without locking.
_snapshot = {
    "updated_utc": None,
    "site": {"lat_deg": SITE_LAT, "lon_deg": SITE_LON, "alt_m": SITE_ALT},
    "elevation_mask_deg": EL_MASK_DEG,
    "satellite_count": 0,
    "satellites": [],
}
_tracks = {"computed_utc": None, "tracks": []}

_satellites = []
_tles_loaded_at = None
_clients: set[WebSocket] = set()


# ---------------- Catalogue ----------------
def _load_tles():
    global _satellites, _tles_loaded_at
    _satellites = load.tle_file(GPS_TLE_URL)
    _tles_loaded_at = datetime.now(timezone.utc)
    return len(_satellites)


def _jday_for(dt):
    return jday(dt.year, dt.month, dt.day,
                dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)


# ---------------- Fast loop: live state ----------------
def _propagate_all():
    """Full hand-rolled pipeline for every satellite at the current instant."""
    now = datetime.now(timezone.utc)
    jd, fr = _jday_for(now)

    results = []
    for sat in _satellites:
        try:
            r = propagate.look_angles(sat.model, SITE_LAT, SITE_LON, SITE_ALT, jd, fr)
            # Convert numpy scalars to native Python types at this boundary --
            # numpy types are fine inside the compute layer but are not JSON
            # serialisable on the way out.
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


# ---------------- Slow loop: ground tracks ----------------
def _compute_tracks():
    """Sample each satellite's subpoint over a window centred on now.

    Returns a flat list of [lat, lon] pairs per satellite. The frontend splits
    the polyline wherever it crosses the antimeridian.
    """
    now = datetime.now(timezone.utc)
    offsets = range(-TRACK_HALF_WINDOW_MIN, TRACK_HALF_WINDOW_MIN + 1, TRACK_STEP_MIN)

    out = []
    for sat in _satellites:
        pts = []
        for m in offsets:
            t = now + timedelta(minutes=m)
            jd, fr = _jday_for(t)
            try:
                r = propagate.look_angles(sat.model, SITE_LAT, SITE_LON, SITE_ALT, jd, fr)
                pts.append([round(float(r["sub_lat_deg"]), 3),
                            round(float(r["sub_lon_deg"]), 3)])
            except Exception:
                continue  # drop a bad sample rather than lose the whole track
        out.append({"norad_id": int(sat.model.satnum), "name": sat.name, "points": pts})

    return {"computed_utc": now.isoformat(), "tracks": out}


# ---------------- WebSocket fan-out ----------------
async def _broadcast(message: dict):
    """Send one message to every connected client, dropping any that have died."""
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


# ---------------- Background tasks ----------------
async def _fast_loop():
    global _snapshot
    while True:
        try:
            stale = (_tles_loaded_at is None or
                     (datetime.now(timezone.utc) - _tles_loaded_at).total_seconds()
                     > TLE_REFRESH_INTERVAL_S)
            if stale:
                # Blocking network I/O -- run off the event loop so the HTTP and
                # WebSocket servers stay responsive.
                count = await asyncio.to_thread(_load_tles)
                print(f"[tle] loaded {count} satellites")

            # SGP4 across the catalogue is CPU-bound; offload it too.
            _snapshot = await asyncio.to_thread(_propagate_all)
            await _broadcast({"type": "state", **_snapshot})
        except Exception as ex:
            print(f"[fast] error: {ex}")
        await asyncio.sleep(PROPAGATION_INTERVAL_S)


async def _track_loop():
    global _tracks
    while True:
        try:
            if not _satellites:
                await asyncio.sleep(2)   # catalogue not loaded yet, retry shortly
                continue
            _tracks = await asyncio.to_thread(_compute_tracks)
            await _broadcast({"type": "tracks", **_tracks})
            print(f"[track] recomputed {len(_tracks['tracks'])} ground tracks")
        except Exception as ex:
            print(f"[track] error: {ex}")
        await asyncio.sleep(TRACK_INTERVAL_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [asyncio.create_task(_fast_loop()), asyncio.create_task(_track_loop())]
    print("[service] propagation loops started")
    yield
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
    print("[service] propagation loops stopped")


app = FastAPI(
    title="Orbit Visualiser",
    description="Live GPS constellation propagation, ground tracks and site visibility.",
    version="0.3.0",
    lifespan=lifespan,
)


# ---------------- Dashboard ----------------
@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Push state on every fast cycle, and tracks whenever they are recomputed."""
    await websocket.accept()
    _clients.add(websocket)
    try:
        # Prime the new client immediately rather than making it wait for the
        # next broadcast -- tracks especially, since those are 5 minutes apart.
        if _tracks["computed_utc"]:
            await websocket.send_json({"type": "tracks", **_tracks})
        if _snapshot["updated_utc"]:
            await websocket.send_json({"type": "state", **_snapshot})
        while True:
            await websocket.receive_text()   # blocks until the client disconnects
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _clients.discard(websocket)


# ---------------- REST endpoints (kept for debugging and tooling) ----------------
@app.get("/health")
def health():
    updated = _snapshot.get("updated_utc")
    age = None
    if updated:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated)).total_seconds()
    return {
        "status": "ok",
        "snapshot_age_s": round(age, 3) if age is not None else None,
        "satellite_count": _snapshot.get("satellite_count", 0),
        "tracks_computed_utc": _tracks.get("computed_utc"),
        "websocket_clients": len(_clients),
        "tles_loaded_utc": _tles_loaded_at.isoformat() if _tles_loaded_at else None,
    }


@app.get("/api/satellites")
def all_satellites():
    return _snapshot


@app.get("/api/visible")
def visible_satellites():
    sats = [s for s in _snapshot.get("satellites", []) if s.get("visible")]
    sats.sort(key=lambda s: -s["el_deg"])
    return {
        "updated_utc": _snapshot.get("updated_utc"),
        "elevation_mask_deg": EL_MASK_DEG,
        "count": len(sats),
        "satellites": sats,
    }


@app.get("/api/tracks")
def ground_tracks():
    return _tracks


@app.get("/api/satellite/{norad_id}")
def one_satellite(norad_id: int):
    for s in _snapshot.get("satellites", []):
        if s.get("norad_id") == norad_id:
            return {"updated_utc": _snapshot.get("updated_utc"), **s}
    raise HTTPException(status_code=404, detail=f"NORAD ID {norad_id} not in catalogue")
