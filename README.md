# Orbit Visualiser

Real-time orbital propagation and constellation-visibility service, running as an
always-on deployment on a headless Raspberry Pi 5.

The Pi continuously propagates satellite constellations, computes visibility from
ground sites, and pushes live geometry to a browser dashboard over the local
network. It handles two central bodies through one shared pipeline: the **GPS
constellation around Earth** from live TLEs, and **Project Prometheus**, a lunar
PNT constellation trade study, around the **Moon** from Keplerian elements.

Every coordinate transform is implemented from scratch and validated against
independent references. The 28-day availability analysis reproduces the source
trade study's published figures — and quantifies exactly where the simplified
force model diverges from it.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [The propagation pipeline](#the-propagation-pipeline)
- [Validation](#validation)
- [Known limitations](#known-limitations)
- [Running it](#running-it)
- [API](#api)
- [Project layout](#project-layout)

---

## What it does

**Live dashboard.** Ground tracks on an equirectangular projection, a polar sky
plot from the selected observer, a visible-satellite table, and live geometry —
all pushed over WebSocket at a 2-second cadence.

**Requirements compliance, live.** The Prometheus study defined five system
requirements. Four are evaluated continuously against current geometry and shown
pass/fail: ≥4 satellites visible, GDOP ≤ 6, PDOP ≤ 4, availability ≥ 95%. The
fifth (position accuracy ≤ 10 m RMS) is explicitly marked *not evaluated* — it
needs a UERE model the source study did not specify.

**28-day availability sweep.** Because lunar geometry beats against the Moon's
~27.3-day rotation, a live one-hour window is not comparable to the study's
published availability. The service computes the study-comparable figure —
28 days from the study epoch at a 300 s step — and displays it beside the
published number with the delta, plus a per-day strip that makes the beat cycle
visible at a glance.

**Ten scenarios.** GPS from live CelesTrak TLEs, plus nine Prometheus
configurations: both final architectures (OPT-B, OPT-D), the best performers from
each orbit family, and one documented failure case (LLO-4P-16-90) kept
deliberately as a negative result.

---

## Architecture

```
                    Raspberry Pi 5 (headless, systemd)
   ┌───────────────────────────────────────────────────────────────┐
   │                                                               │
   │   fast loop, 2 s  ─────┐                                      │
   │   (live positions)     │                                      │
   │                        ├──►  snapshot in RAM  ──►  WebSocket ─┼──► browser
   │   track loop, 5 min ───┤     (never written                   │
   │   (ground tracks)      │      to disk)                        │
   │                        │                                      │
   │   sweep task, once  ───┘                                      │
   │   (28-day analysis)                                           │
   │                                                               │
   └───────────────────────────────────────────────────────────────┘
```

**Propagation is decoupled from requests.** Background loops compute state on a
fixed cadence and write to an in-memory snapshot; HTTP and WebSocket handlers only
read it. Response latency is a dict lookup rather than an SGP4 run, ten browser
tabs cost the same CPU as one, and the propagation rate is a deliberate parameter
instead of a side effect of traffic.

**Three cadences, set by rate of change.** Positions move every second; ground
tracks are nearly identical minute to minute; the 28-day sweep is deterministic
from a fixed epoch and never needs recomputing. Recomputing tracks at 2 s would
waste ~97% of that work. Decoupling update rates by rate of change is what keeps
this comfortable on a Pi.

**Nothing is written to disk.** Satellite state at time *t* is worthless at
*t*+5 s, so persisting it would burn microSD write endurance for no benefit.
Logs go to journald, which caps and rotates itself.

**Central-body abstraction.** `bodies.py` holds everything that differs between
Earth and Moon — gravitational parameter, reference ellipsoid, inertial frame, and
the rotation into the body-fixed frame. `sources.py` holds everything that differs
between satellite sources — TLE/SGP4 versus Keplerian elements. The transform
pipeline takes a body and a source and never branches on which one it is.

---

## The propagation pipeline

Four stages. Stage 0 varies by source, Stage 1 by body; Stages 2 and 3 are shared.

### Stage 0 — state at an instant

*Earth:* TLE → SGP4 → position in **TEME**. SGP4 is the one component not
hand-rolled: TLE mean elements are fitted to that specific model and are not
osculating Keplerian elements, so feeding them to a two-body propagator is wrong
by kilometres within an orbit.

*Moon:* classical elements → position in **MoonMJ2000Eq**. Mean anomaly advances
linearly at *n* = √(μ/a³); Kepler's equation *M* = *E* − *e* sin *E* is solved by
Newton iteration; the perifocal state is rotated to inertial by
R₃(−Ω)·R₁(−*i*)·R₃(−ω).

### Stage 1 — inertial → body-fixed

*Earth:* one rotation about z by the Greenwich Mean Sidereal Time, because TEME's
z-axis is already Earth's spin axis:

```
θ = 280.46061837 + 360.98564736629 · d      (d = days since J2000)
r_ECEF = R₃(θ) · r_TEME
```

*Moon:* **three** rotations, because MoonMJ2000Eq is Moon-*centred* but uses
MJ2000 *equatorial* axes — the orientation of Earth's equator. The lunar pole sits
23.46° off that z-axis, so a single rotation is not valid:

```
r_fixed = R₃(W) · R₁(90° − δ₀) · R₃(α₀ + 90°) · r_inertial
```

with the IAU mean pole (α₀, δ₀) and prime meridian *W*.

### Stage 2 — body-fixed → geodetic subpoint

Longitude is closed form. Latitude needs an iteration on an ellipsoid because the
local normal does not pass through the centre. When flattening is zero (the
spherical Moon model), *e*² = 0 and the first pass is already exact — so one code
path serves both bodies with no sphere/ellipsoid branch.

### Stage 3 — body-fixed → topocentric az/el

Body-agnostic: this only depends on where the observer stands.

```
        ⎡ E ⎤   ⎡    −sin λ₀        cos λ₀       0   ⎤
        ⎢ N ⎥ = ⎢ −sin φ₀ cos λ₀  −sin φ₀ sin λ₀  cos φ₀⎥ · ρ_fixed
        ⎣ U ⎦   ⎣  cos φ₀ cos λ₀   cos φ₀ sin λ₀  sin φ₀⎦

elevation = asin(U / |ρ|)      azimuth = atan2(E, N)
```

### DOP

Geometry matrix **G** with one row per visible satellite, `[−eₓ, −e_y, −e_z, 1]`;
**H** = (**G**ᵀ**G**)⁻¹. GDOP = √tr(**H**), PDOP = √(sum of the three position
diagonals).

---

## Validation

Three independent layers, each testing something different.

**1. Against Skyfield — the transforms are implemented correctly.**
All 32 GPS satellites, same TLE and same instant fed to both paths, so SGP4 error
cancels and only the coordinate transforms are compared:

| | max \|Δ\| | mean \|Δ\| |
|---|---|---|
| Elevation | 0.0003° | 0.0002° |
| Azimuth | 0.0024° | 0.0003° |

**≈1 arcsecond.** The residual is consistent with the Earth-orientation terms
deliberately omitted — the UT1−UTC offset (a few tenths of a second of rotation,
≈15 arcsec per second of time) and polar motion (sub-arcsecond). Azimuth residuals
grow near the zenith because azimuth is ill-conditioned there: as E and N shrink,
a fixed position error subtends a larger angle.

**2. Against N2YO — the pipeline matches reality.**
GPS BIIR-11 (PRN 19), same site, same second:

| | Elevation | Azimuth |
|---|---|---|
| N2YO | 81.8° | 213.5° |
| This pipeline | 81.78° | 213.52° |

**≈1 arcminute**, inside N2YO's own display precision.

**3. Regression test — refactoring changed nothing.**
The central-body refactor was validated by driving the pre-refactor
implementation and the refactored one with 400 synthetic positions: **max delta
1.3 × 10⁻¹³ degrees**, i.e. floating-point noise. `test_refactor.py` also covers
the Moon path independently — surface round-trip exact, rotation matrix
orthonormal with det = +1, Kepler elements → state → elements recovering
SMA/ECC/INC to 10⁻¹³, and a full-period propagation returning to the exact
starting true anomaly.

**4. 28-day sweep against the source trade study.**
Reproducing the study's own published availability using its methodology (28 days,
300 s step, 7° mask, ≥4 satellites visible):

| Config | Shackleton | Mare Tranquillitatis | Tsiolkovskiy |
|---|---|---|---|
| OPT-D | 98.0 (study 99.4) | **83.2** (97.0) | 98.7 (97.8) |
| OPT-B | 100.0 (100.0) | **92.1** (100.0) | 100.0 (99.9) |
| HELO-4P-16 | 100.0 (100.0) | **66.7** (88.4) | 95.8 (97.5) |
| FELO-F7d-16 | 0.0 (0.0) | 29.5 (32.0) | 31.9 (34.4) |

Every site except Mare Tranquillitatis reproduces within ~2 points, including the
pure-geometry exclusions (FELO at 27° inclination cannot see the south pole:
0.0% against 0.0%). See below for why Mare diverges.

---

## Known limitations

These are stated in the dashboard as well as here.

**Two-body propagation vs. the study's perturbed model.** The source trade study
propagated with an LP165P 20×20 lunar gravity field plus Earth and Sun third
bodies, integrated with RK8(9). This service propagates two-body from Keplerian
elements. In two-body, orbit planes are fixed in inertial space — RAAN, AOP and
inclination never move. The Moon rotates beneath them on a 27.3-day cycle, so
geometry relative to a surface site beats at exactly that period. The real
perturbed dynamics precess those planes, and the study's *frozen* orbit design
depends on that precession to hold its geometry.

This shows up almost entirely at **Mare Tranquillitatis**, which is the marginal
site by construction — the one where HELO alone reached only 88.4%, and the reason
the FELO layer exists at all. Its per-day strip in the dashboard shows the
signature clearly: ~100% at both ends of the 28 days with a trough in the middle.
Pure-geometry results, which do not depend on precession, reproduce exactly.

Ingesting the study's CCSDS-OEM ephemerides directly would remove this limitation;
the `Source` abstraction is designed to accept that without pipeline changes.

**Earth-orientation terms.** The TEME→ECEF rotation uses GMST only, omitting
precession, nutation, polar motion, and the UT1−UTC offset. Quantified above at
~1 arcsecond.

**Lunar libration.** The Moon's orientation uses the IAU *mean* pole and prime
meridian; physical libration terms (E1–E13, ~0.02°) are omitted. This is the
lunar analogue of the GMST-only simplification above.

**One unverified parameter.** NS-FELO-20's per-plane argument-of-periapsis split
(90°/270°) was reconstructed rather than recovered from a verified parameter
sheet. Because OPT-B contains NS-FELO-20, OPT-B inherits the caveat; its HELO
layer is exact. Both are flagged `verified: false` in the data and badged in the
UI.

**Prometheus is a design study.** These are not real satellites. The constellation
is the author's own trade-study output, not an operational or planned system.

---

## Running it

Requires Python 3.11+ on Linux (developed on Raspberry Pi OS Lite 64-bit, Pi 5 4 GB).

```bash
git clone https://github.com/EdwardChoi2/orbit-visualizer.git
cd orbit-visualizer
python3 -m venv .venv
source .venv/bin/activate
pip install 'uvicorn[standard]' fastapi skyfield numpy
uvicorn service:app --host 0.0.0.0 --port 8000
```

Then open `http://<host>:8000/`.

### As a systemd service

```bash
sudo cp orbit-viz.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now orbit-viz
systemctl status orbit-viz
journalctl -u orbit-viz -f
```

The unit runs as an unprivileged user, restarts on failure with a crash-loop
limit, starts at boot, and logs to journald.

### Tests

```bash
python test_refactor.py
```

Needs no network and no TLEs — it drives the transform chain with synthetic
positions so it isolates exactly the code under test.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Dashboard |
| `WS /ws` | Live push; client sends `{"type":"subscribe","scenario":"<id>"}` |
| `GET /api/scenarios` | Catalogue of all scenarios, sites, and layer parameters |
| `GET /api/satellites/{scenario}` | Full current state |
| `GET /api/visible/{scenario}?site=` | Visible satellites, highest elevation first |
| `GET /api/tracks/{scenario}` | Ground-track polylines |
| `GET /api/analysis/{scenario}` | 28-day availability sweep |
| `GET /health` | Liveness, snapshot age, client count, sweep status |
| `GET /docs` | Generated OpenAPI documentation |

---

## Project layout

```
bodies.py            Central-body definitions: μ, ellipsoid, frames, rotations
propagate.py         Coordinate transforms and the Kepler solver
sources.py           TLESource / KeplerSource, scenario definitions, sites
analysis.py          Vectorised 28-day availability sweep
service.py           FastAPI app, background loops, WebSocket fan-out
validate.py          Skyfield comparison harness
test_refactor.py     Regression and self-consistency tests
static/index.html    Dashboard (no build step, no framework)
data/                Prometheus orbital element sets
orbit-viz.service    systemd unit
```

---

## Context

Built as an engineering portfolio project. The Prometheus constellation is the
author's own lunar PNT trade study (AE295A/B, San José State University), ported
here from GMAT and MATLAB into a live service.
