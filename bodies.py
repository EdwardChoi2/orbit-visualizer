"""
bodies.py -- central-body definitions

Everything that differs between propagating around the Earth and propagating
around the Moon is collected here: gravitational parameter, reference ellipsoid,
inertial frame, and the rotation from that inertial frame into the body-fixed
frame.

The rest of the pipeline (subpoint conversion, topocentric look-angles) takes a
Body and does not branch on which one it is.

One physically important asymmetry, which is why `inertial_to_fixed` is a method
rather than a shared formula:

  * Earth / TEME: the frame's z-axis IS Earth's rotation axis, so going to the
    Earth-fixed frame is a single rotation about z by the sidereal angle.

  * Moon / MoonMJ2000Eq: this frame is Moon-CENTRED but uses MJ2000 EQUATORIAL
    axes -- i.e. the orientation of *Earth's* equator. The Moon's pole is nowhere
    near that z-axis, so the body-fixed transform needs three rotations: line up
    with the lunar pole (right ascension and declination), then spin by the
    lunar prime-meridian angle.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

J2000_JD = 2451545.0


def _R1(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,   s],
                     [0.0,  -s,   c]])


def _R3(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[  c,   s, 0.0],
                     [ -s,   c, 0.0],
                     [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Body:
    name: str
    mu_km3_s2: float          # gravitational parameter
    radius_km: float          # equatorial radius of the reference ellipsoid
    flattening: float         # 0.0 for a spherical model
    inertial_frame: str       # name of the frame states are supplied in
    fixed_frame: str          # name of the body-fixed frame
    default_mask_deg: float   # elevation mask used for visibility
    _rotation: Callable[[float], np.ndarray]

    @property
    def e2(self) -> float:
        """First eccentricity squared of the reference ellipsoid, e^2 = 2f - f^2."""
        return 2.0 * self.flattening - self.flattening ** 2

    def inertial_to_fixed(self, jd) -> np.ndarray:
        """3x3 rotation matrix from the inertial frame to the body-fixed frame."""
        return self._rotation(jd)


# ---------------- Earth ----------------
def _earth_rotation(jd_ut1):
    """R3(GMST). Single z-rotation: TEME's z-axis is already Earth's spin axis.

    theta = 280.46061837 + 360.98564736629 * d, d = days since J2000.
    The 360.9856 deg/day term is Earth's sidereal rotation rate -- slightly more
    than 360 because Earth turns once relative to the stars faster than relative
    to the Sun.
    """
    d = jd_ut1 - J2000_JD
    theta = math.radians((280.46061837 + 360.98564736629 * d) % 360.0)
    return _R3(theta)


EARTH = Body(
    name="Earth",
    mu_km3_s2=398600.4418,
    radius_km=6378.137,
    flattening=1.0 / 298.257223563,      # WGS84
    inertial_frame="TEME",
    fixed_frame="ECEF (ITRF-approx)",
    default_mask_deg=10.0,
    _rotation=_earth_rotation,
)


# ---------------- Moon ----------------
def _moon_rotation(jd_tdb):
    """IAU mean lunar orientation: R3(W) . R1(90 - dec0) . R3(ra0 + 90).

    Mean pole and prime meridian from the IAU/WGCCRE report:
        ra0  = 269.9949 + 0.0031 T      (T = Julian centuries from J2000)
        dec0 =  66.5392 + 0.0130 T
        W    =  38.3213 + 13.17635815 d - 1.4e-12 d^2

    The physical libration terms (E1..E13) are deliberately omitted -- they are
    the lunar analogue of the nutation/polar-motion terms dropped from the Earth
    GMST rotation, and contribute at the ~0.02 deg level. Stating that the model
    is mean-orientation-only, and knowing what it omits, is the point.
    """
    d = jd_tdb - J2000_JD
    T = d / 36525.0
    ra0 = math.radians(269.9949 + 0.0031 * T)
    dec0 = math.radians(66.5392 + 0.0130 * T)
    W = math.radians((38.3213 + 13.17635815 * d - 1.4e-12 * d * d) % 360.0)
    return _R3(W) @ _R1(math.pi / 2.0 - dec0) @ _R3(ra0 + math.pi / 2.0)


MOON = Body(
    name="Moon",
    mu_km3_s2=4902.800,                  # matches the Prometheus study constants
    radius_km=1737.4,
    flattening=0.0,                      # spherical Moon, as used in the study
    inertial_frame="MoonMJ2000Eq",
    fixed_frame="Selenographic (IAU mean)",
    default_mask_deg=7.0,                # study elevation mask
    _rotation=_moon_rotation,
)


BODIES = {"earth": EARTH, "moon": MOON}
