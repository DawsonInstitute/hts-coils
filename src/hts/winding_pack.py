"""Self-consistent winding-pack model for HTS solenoids.

The legacy ``hts_coil_field`` places all N turns on a single filament of radius R.
That is fine as an idealised upper bound, but the conductor, hoop-stress and thermal
models need an actual winding-pack *geometry*. This module provides one geometry that is
shared by all of them:

    inner radius a1, outer radius a2 = a1 + c, half-length b, total amp-turns N*I,
    engineering current density J_e = N*I / (2 b (a2 - a1)).

Everything is exact/analytic (complete elliptic integrals) or a direct sum of those loop
solutions; no free fitting constants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk

MU0 = 4e-7 * np.pi


# --------------------------------------------------------------------------- loops
def loop_field(a: float, I: float, z, rho) -> Tuple[np.ndarray, np.ndarray]:
    """Exact (B_z, B_rho) [T] of a filamentary loop of radius ``a`` carrying ``I`` at (z, rho).

    z is measured from the loop plane. Vectorised over z and rho (broadcast together).
    ``scipy.special.ellipk/ellipe`` take the parameter m = k^2.
    """
    z, rho = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(rho, dtype=float))
    s2 = (a + rho) ** 2 + z ** 2
    d2 = (a - rho) ** 2 + z ** 2
    m = np.clip(4.0 * a * rho / s2, 0.0, 1.0 - 1e-15)
    K, E = ellipk(m), ellipe(m)
    pre = MU0 * I / (2.0 * np.pi * np.sqrt(s2))
    with np.errstate(divide="ignore", invalid="ignore"):
        Bz = pre * (K + (a * a - rho * rho - z * z) / d2 * E)
        Br = np.where(rho > 1e-12,
                      pre * z / rho * (-K + (a * a + rho * rho + z * z) / d2 * E), 0.0)
    return Bz, Br


# ------------------------------------------------------------------- winding pack
@dataclass(frozen=True)
class WindingPack:
    a1: float            # inner radius [m]
    a2: float            # outer radius [m]
    b: float             # half-length [m]
    N: int               # turns
    I: float             # current per turn [A] (shared by ``tapes_per_turn`` parallel tapes)
    tapes_per_turn: int = 1
    tape_width: float = 4e-3
    tape_thickness: float = 0.2e-3

    @property
    def radial_build(self) -> float:
        return self.a2 - self.a1

    @property
    def a_mean(self) -> float:
        return 0.5 * (self.a1 + self.a2)

    @property
    def amp_turns(self) -> float:
        return self.N * self.I

    @property
    def Je(self) -> float:
        """Engineering current density over the whole pack cross-section [A/m^2]."""
        return self.amp_turns / (2.0 * self.b * self.radial_build)

    @property
    def conductor_area(self) -> float:
        return self.N * self.tapes_per_turn * self.tape_width * self.tape_thickness


def pack_from_conductor(a1: float, N: int, I: float, tapes_per_turn: int,
                        tape_width: float = 4e-3, tape_thickness: float = 0.2e-3,
                        fill_factor: float = 1.0, aspect: float = 1.0) -> WindingPack:
    """Build the pack whose cross-section holds exactly the requested conductor.

    area = N * tapes_per_turn * w * t / fill_factor ; axial length / radial build = ``aspect``.
    ``fill_factor`` = conductor fraction of the pack (insulation, cooling, structure < 1);
    1.0 is the optimistic limit.
    """
    area = N * tapes_per_turn * tape_width * tape_thickness / fill_factor
    c = float(np.sqrt(area / aspect))
    return WindingPack(a1=a1, a2=a1 + c, b=0.5 * aspect * c, N=N, I=I,
                       tapes_per_turn=tapes_per_turn, tape_width=tape_width,
                       tape_thickness=tape_thickness)


def thick_solenoid_center_field(a1: float, a2: float, b: float, NI: float) -> Tuple[float, float]:
    """Analytic (Fabry) centre field of a uniform-J pack. Returns (B0 [T], J_e [A/m^2])."""
    Je = NI / (2.0 * b * (a2 - a1))
    al, be = a2 / a1, b / a1
    F = be * np.log((al + np.hypot(al, be)) / (1.0 + np.hypot(1.0, be)))
    return float(MU0 * Je * a1 * F), float(Je)


def _filaments(pack: WindingPack, nr: int, nz: int):
    ra = pack.a1 + (np.arange(nr) + 0.5) * pack.radial_build / nr
    za = -pack.b + (np.arange(nz) + 0.5) * 2.0 * pack.b / nz
    RA, ZA = np.meshgrid(ra, za, indexing="ij")
    return RA.ravel(), ZA.ravel(), pack.amp_turns / (nr * nz)


def pack_field(pack: WindingPack, z, rho, nr: int = 12, nz: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """(B_z, B_rho) of the pack at (z, rho), as a sum of exact loop fields (nr x nz filaments)."""
    z = np.asarray(z, dtype=float)
    rho = np.asarray(rho, dtype=float)
    shape = np.broadcast(z, rho).shape
    zf, rf = np.broadcast_to(z, shape).ravel(), np.broadcast_to(rho, shape).ravel()
    ra, za, Ik = _filaments(pack, nr, nz)
    Bz = np.zeros(zf.size)
    Br = np.zeros(zf.size)
    for a_k, z_k in zip(ra, za):
        bz, br = loop_field(a_k, Ik, zf - z_k, rf)
        Bz += bz
        Br += br
    return Bz.reshape(shape), Br.reshape(shape)


def center_field(pack: WindingPack) -> float:
    return float(pack_field(pack, 0.0, 0.0)[0])


def peak_field(pack: WindingPack, nr: int = 12, nz: int = 12, n_probe: int = 13) -> float:
    """Maximum |B| over the pack cross-section [T] (probe nodes sit between filaments)."""
    rr = np.linspace(pack.a1, pack.a2, n_probe)
    zz = np.linspace(0.0, pack.b, n_probe)   # axisymmetric & mirror-symmetric: upper half
    RR, ZZ = np.meshgrid(rr, zz, indexing="ij")
    Bz, Br = pack_field(pack, ZZ, RR, nr, nz)
    return float(np.max(np.hypot(Bz, Br)))


def field_ripple(pack: WindingPack, r_roi: float, n: int = 25, kind: str = "pp") -> float:
    """Ripple of B_z over the sphere of radius ``r_roi`` about the centre.

    kind='pp'  : (max - min) / B_z(0)
    kind='rms' : volume-weighted std / mean (axisymmetric weight ~ rho)
    """
    zz, rr = np.meshgrid(np.linspace(-r_roi, r_roi, n), np.linspace(0.0, r_roi, n), indexing="ij")
    mask = zz ** 2 + rr ** 2 <= r_roi ** 2
    Bz, _ = pack_field(pack, zz, rr)
    Bz = Bz[mask]
    if kind == "pp":
        return float((Bz.max() - Bz.min()) / center_field(pack))
    w = rr[mask] + 1e-12
    mu = np.average(Bz, weights=w)
    return float(np.sqrt(np.average((Bz - mu) ** 2, weights=w)) / mu)


# --------------------------------------------------------------------- mechanics
def hoop_stress_check(pack: WindingPack, B_peak: float, sigma_allow_hoop: float = 400e6,
                      t_reinf: float = 0.0, E_ratio: float = 1.0,
                      sigma_allow_radial: float = 35e6) -> Dict[str, float]:
    """Hoop (in-plane) and radial (transverse/delamination) checks for a winding pack.

    Bare hoop stress (Wilson estimate, upper bound with no load sharing): sigma = B_peak * J_e * a_mean.
    A reinforcement overband of radial thickness ``t_reinf`` (modulus ratio ``E_ratio`` to the
    pack) shares the load: sigma_cond = sigma_bare / (1 + E_ratio * t_reinf / c).
    Radial stress upper bound: sigma_r ~ B_peak * J_e * c / 2.

    ``sigma_allow_hoop`` is the conservative in-plane tensile allowable of Hastelloy-substrate REBCO
    (irreversible degradation is typically quoted at several hundred MPa; 400 MPa here is an
    assumption to be replaced by the cited value for the actual tape). ``sigma_allow_radial`` is a
    transverse (c-axis) delamination limit.
    """
    c = pack.radial_build
    sigma_bare = B_peak * pack.Je * pack.a_mean
    sigma_cond = sigma_bare / (1.0 + E_ratio * t_reinf / c)
    sigma_r = B_peak * pack.Je * c / 2.0
    t_req = max(0.0, (sigma_bare / sigma_allow_hoop - 1.0) * c / E_ratio)
    return {
        "sigma_hoop_bare_Pa": sigma_bare,
        "sigma_hoop_conductor_Pa": sigma_cond,
        "sigma_radial_est_Pa": sigma_r,
        "hoop_ok": bool(sigma_cond <= sigma_allow_hoop),
        "radial_ok": bool(sigma_r <= sigma_allow_radial),
        "t_reinf_required_m": t_req,
        "magnetic_pressure_Pa": B_peak ** 2 / (2.0 * MU0),
    }


# ----------------------------------------------------------------------- thermal
def current_sharing_temperature(I_tape: float, B_peak: float, A_tape: float, Tc: float = 90.0,
                                **jc_kw) -> Optional[float]:
    """T_cs [K] at which I_tape = I_c(B_peak, T_cs); None if I_c < I_tape even at 1 K."""
    from .materials import jc_vs_tb

    def f(T: float) -> float:
        return jc_vs_tb(T=T, B=B_peak, Tc=Tc, **jc_kw) * A_tape - I_tape

    if f(1.0) < 0.0:
        return None
    return float(brentq(f, 1.0, Tc - 1e-6))
