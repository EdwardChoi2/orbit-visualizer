"""
service.py -- Phase 4: multi-body, multi-scenario propagation service

    per-scenario fast loop (2 s)   ->\
                                      snapshot -> WebSocket push -> subscribed clients
    per-scenario track loop        ->/

Each scenario pairs a Source (how to get inertial positions) with a Body (how to
rotate into the body-fixed frame) and a set of observation sites. Earth/GPS and
Moon/Prometheus run through exactly the same pipeline; nothing here branches on
which body it is.

Two decisions worth naming:

  * Look-angles are computed for EVERY site of a scenario, not just a selected
    one. The topocentric transform is a 3x3 multiply per satellite per site --
    negligible -- and doing all of them means switching sites in the browser is
    instant with no server round trip.

  * Clients subscribe to one scenario. A scenario with no subscribers is still
    propagated (keeps state warm and the code simple), but only subscribers are
    pushed to.

Nothing is written to disk.

Run:  uvicorn service:app --host 0.0.0.0 --port 8000
"""

import asyncio
import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

import propagate
import sources as src_mod

STATIC_DIR = Path(__file__).parent / "static"

PROPAGATION_INTERVAL_S = 2.0
TRACK_INTERVAL_S = 300.0
TRACK_STEPS = 60                      # samples across the whole track window

SCENARIOS = src_mod.build_scenarios()
DEFAULT_SCENARIO = "gps"

# scenario_id -> snapshot / tracks
_snapshots = {}
_tracks = {}
_clients = {}                         # WebSocket -> scenario_id


# ---------------- Propagation ----------------
def _site_payload(scn):
    return [{"name": s.name, "lat_deg": s.lat_deg, "lon_deg": s.lon_deg,
             "alt_km": s.alt_km} for s in scn.sites]


def _dop(unit_los):
    """Geometric and position dilution of precision from unit line-of-sight vectors.

    Build the geometry matrix G with one row per visible satellite:
        [-ex, -ey, -ez, 1]      (negated direction cosines, plus clock-bias term)
    then H = inv(G^T G). GDOP is the root of the trace of all four diagonal terms;
    PDOP uses only the three position terms. Needs >= 4 satellites and a
    non-singular G -- a near-singular G means the satellites are clustered and
    the position solution is poorly conditioned, which is exactly what DOP
    measures.
    """
    if len(unit_los) < 4:
        return None, None
    G = np.array([[-v[0], -v[1], -v[2], 1.0] for v in unit_los])
    try:
        H = np.linalg.inv(G.T @ G)
    except np.linalg.LinAlgError:
        return None, None
    d = np.diag(H)
    if np.any(d < 0):
        return None, None
    return float(math.sqrt(d.sum())), float(math.sqrt(d[:3].sum()))


def _propagate(scn, when):
    """Full pipeline for one scenario at one instant, for all of its sites."""
    body = scn.body
    mask = body.default_mask_deg
    jd = src_mod.datetime_to_jd(when)
    R = body.inertial_to_fixed(jd)         # Stage 1, computed once per cycle

    obs = {}
    for s in scn.sites:
        lat0, lon0 = math.radians(s.lat_deg), math.radians(s.lon_deg)
        obs[s.name] = (lat0, lon0,
                       propagate.geodetic_to_fixed(lat0, lon0, s.alt_km, body))

    los = {s.name: [] for s in scn.sites}
    out = []
    for p in scn.source.positions(when):
        r_fixed = R @ p["r_km"]
        lat, lon, alt = propagate.fixed_to_geodetic(r_fixed, body)   # Stage 2

        per_site = {}
        any_visible = False
        for s in scn.sites:                                          # Stage 3
            lat0, lon0, r_obs = obs[s.name]
            enu = propagate.fixed_to_enu(r_fixed - r_obs, lat0, lon0)
            az, el, rng = propagate.enu_to_azel(enu)
            vis = math.degrees(el) > mask
            if vis:
                any_visible = True
                los[s.name].append(np.asarray(enu) / rng)
            per_site[s.name] = {"az_deg": round(math.degrees(az), 4),
                                "el_deg": round(math.degrees(el), 4),
                                "range_km": round(rng, 3),
                                "visible": bool(vis)}

        out.append({
            "name": p["name"],
            "layer": p.get("layer"),
            "plane": p.get("plane"),
            "norad_id": p.get("norad_id"),
            "lat_deg": round(math.degrees(lat), 5),
            "lon_deg": round(math.degrees(lon), 5),
            "alt_km": round(alt, 3),
            "sites": per_site,
            "any_visible": bool(any_visible),
        })

    dop = {}
    for s in scn.sites:
        g, p_ = _dop(los[s.name])
        dop[s.name] = {
            "n_visible": len(los[s.name]),
            "gdop": round(g, 3) if g is not None else None,
            "pdop": round(p_, 3) if p_ is not None else None,
        }

    return {
        "type": "state",
        "scenario": scn.id,
        "updated_utc": when.isoformat(),
        "body": body.name,
        "frame": body.inertial_frame,
        "body_radius_km": body.radius_km,
        "elevation_mask_deg": mask,
        "sites": _site_payload(scn),
        "dop": dop,
        "satellite_count": len(out),
        "satellites": out,
    }


def _compute_tracks(scn, when):
    """Subpoint history/future, window scaled to the orbit period by the source."""
    body = scn.body
    half = scn.source.track_half_window_s()
    step = 2 * half / TRACK_STEPS

    series = {}
    for i in range(TRACK_STEPS + 1):
        t = when + timedelta(seconds=-half + i * step)
        R = body.inertial_to_fixed(src_mod.datetime_to_jd(t))
        for p in scn.source.positions(t):
            lat, lon, _ = propagate.fixed_to_geodetic(R @ p["r_km"], body)
            series.setdefault(p["name"], {"name": p["name"],
                                          "layer": p.get("layer"),
                                          "points": []})
            series[p["name"]]["points"].append(
                [round(math.degrees(lat), 3), round(math.degrees(lon), 3)])

    return {"type": "tracks", "scenario": scn.id,
            "computed_utc": when.isoformat(),
            "window_hours": round(2 * half / 3600.0, 2),
            "tracks": list(series.values())}


# ---------------- Broadcast ----------------
async def _broadcast(scenario_id, message):
    dead = []
    for ws, sid in list(_clients.items()):
        if sid != scenario_id:
            continue
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.pop(ws, None)


# ---------------- Loops ----------------
async def _fast_loop(scn):
    while True:
        try:
            await asyncio.to_thread(scn.source.ensure_loaded)
            if scn.source.is_ready():
                snap = await asyncio.to_thread(_propagate, scn,
                                               datetime.now(timezone.utc))
                _snapshots[scn.id] = snap
                await _broadcast(scn.id, snap)
        except Exception as ex:
            print(f"[fast:{scn.id}] {type(ex).__name__}: {ex}")
        await asyncio.sleep(PROPAGATION_INTERVAL_S)


async def _track_loop(scn):
    while True:
        try:
            if not scn.source.is_ready():
                await asyncio.sleep(2)
                continue
            tr = await asyncio.to_thread(_compute_tracks, scn,
                                         datetime.now(timezone.utc))
            _tracks[scn.id] = tr
            await _broadcast(scn.id, tr)
            print(f"[track:{scn.id}] {len(tr['tracks'])} tracks, "
                  f"{tr['window_hours']} hr window")
        except Exception as ex:
            print(f"[track:{scn.id}] {type(ex).__name__}: {ex}")
        await asyncio.sleep(TRACK_INTERVAL_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = []
    for scn in SCENARIOS.values():
        tasks.append(asyncio.create_task(_fast_loop(scn)))
        tasks.append(asyncio.create_task(_track_loop(scn)))
    print(f"[service] {len(SCENARIOS)} scenarios, {len(tasks)} loops started")
    yield
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
    print("[service] loops stopped")


app = FastAPI(
    title="Orbit Visualiser",
    description="Multi-body constellation propagation: Earth/SGP4 and Moon/Kepler.",
    version="0.4.0",
    lifespan=lifespan,
)


# ---------------- Dashboard ----------------
@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _catalog():
    items = []
    for scn in SCENARIOS.values():
        s = scn.source
        items.append({
            "id": scn.id, "label": scn.label, "group": scn.group,
            "body": scn.body.name, "frame": scn.body.inertial_frame,
            "mask_deg": scn.body.default_mask_deg,
            "verified": getattr(s, "verified", True),
            "note": getattr(s, "note", ""),
            "layers": s.layers(),
            "coverage": scn.coverage,
            "sites": _site_payload(scn),
            "ready": s.is_ready(),
            "count": _snapshots.get(scn.id, {}).get("satellite_count", 0),
        })
    return items


@app.get("/api/scenarios")
def scenarios():
    return {"default": DEFAULT_SCENARIO, "scenarios": _catalog()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients[websocket] = DEFAULT_SCENARIO
    try:
        await websocket.send_json({"type": "catalog", "default": DEFAULT_SCENARIO,
                                   "scenarios": _catalog()})
        await _prime(websocket, DEFAULT_SCENARIO)
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") == "subscribe":
                sid = msg.get("scenario")
                if sid in SCENARIOS:
                    _clients[websocket] = sid
                    await _prime(websocket, sid)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _clients.pop(websocket, None)


async def _prime(websocket, sid):
    """Send whatever state exists now so a new subscriber renders immediately."""
    if sid in _tracks:
        await websocket.send_json(_tracks[sid])
    if sid in _snapshots:
        await websocket.send_json(_snapshots[sid])


# ---------------- REST ----------------
@app.get("/health")
def health():
    per = {}
    now = datetime.now(timezone.utc)
    for sid, snap in _snapshots.items():
        age = (now - datetime.fromisoformat(snap["updated_utc"])).total_seconds()
        per[sid] = {"age_s": round(age, 3), "count": snap["satellite_count"],
                    "tracks": sid in _tracks}
    return {"status": "ok", "scenarios": len(SCENARIOS),
            "websocket_clients": len(_clients), "snapshots": per}


@app.get("/api/satellites/{scenario_id}")
def all_satellites(scenario_id: str):
    if scenario_id not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {scenario_id!r}")
    snap = _snapshots.get(scenario_id)
    if not snap:
        raise HTTPException(503, "no propagation cycle has completed yet")
    return snap


@app.get("/api/visible/{scenario_id}")
def visible(scenario_id: str, site: str | None = None):
    if scenario_id not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {scenario_id!r}")
    snap = _snapshots.get(scenario_id)
    if not snap:
        raise HTTPException(503, "no propagation cycle has completed yet")
    site = site or snap["sites"][0]["name"]
    if site not in snap["satellites"][0]["sites"]:
        raise HTTPException(404, f"unknown site {site!r} for this scenario")
    sats = [{**{k: v for k, v in s.items() if k != "sites"}, **s["sites"][site]}
            for s in snap["satellites"] if s["sites"][site]["visible"]]
    sats.sort(key=lambda s: -s["el_deg"])
    return {"scenario": scenario_id, "site": site,
            "updated_utc": snap["updated_utc"],
            "elevation_mask_deg": snap["elevation_mask_deg"],
            "count": len(sats), "satellites": sats}


@app.get("/api/tracks/{scenario_id}")
def tracks(scenario_id: str):
    if scenario_id not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {scenario_id!r}")
    tr = _tracks.get(scenario_id)
    if not tr:
        raise HTTPException(503, "tracks not computed yet")
    return tr
