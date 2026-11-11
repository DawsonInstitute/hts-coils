#!/usr/bin/env python3
"""
High-field scaling enhancements for HTS coils (5-10 T capability).

Implements enhanced field scaling, space-relevant thermal modeling,
and COMSOL integration for high-field validation as requested.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

from .coil import hts_coil_field, field_from_loops
from .materials import jc_vs_tb, enhanced_thermal_simulation


def scale_hts_coil_field(r: np.ndarray, I: float = 1800, N: int = 1000, R: float = 0.16, T: float = 15,
                         util_target: float = 0.3, fill_factor: float = 1.0,
                         roi_fraction: float = 0.1) -> Dict[str, float]:
    """
    Field, conductor count and feasibility of an N-turn HTS solenoid, using ONE consistent
    winding-pack geometry (see :mod:`hts.winding_pack`).

    The conductor count (tapes per turn) is chosen so the tape runs at ``util_target`` of its
    critical current at the *peak field on the winding*. Because the pack grows with the tape
    count, and the peak field depends on the pack, this is solved as a fixed point.

    Parameters:
    -----------
    r : np.ndarray
        Evaluation point [x, y, z] (m); the origin gives the centre field
    I : float
        Current per turn (A)
    N : int
        Number of turns
    R : float
        Winding-pack INNER radius (m)
    T : float
        Operating temperature (K)
    util_target : float
        Target I / I_c of the tape at B_peak
    fill_factor : float
        Conductor fraction of the pack cross-section (1.0 = optimistic limit)
    roi_fraction : float
        Ripple region of interest: sphere of radius roi_fraction * R

    Returns:
    --------
    result : Dict[str, float]
        Field analysis including ``B_filament`` (the idealised thin-loop value mu0*N*I/(2R),
        kept for comparison only) and ``converged`` (fixed-point status).
    """
    from .winding_pack import pack_from_conductor, pack_field, peak_field, field_ripple, MU0

    r = np.asarray(r, dtype=float)
    tape_width = 4e-3       # m (4mm standard REBCO tape)
    tape_thickness = 0.2e-3  # m (0.2mm including substrate)

    def _state(tpt):
        pk = pack_from_conductor(R, N, I, tpt, tape_width, tape_thickness, fill_factor)
        Bp = peak_field(pk)
        Jc = jc_vs_tb(T=T, B=Bp, Tc=90.0, Jc0=300e6, B0=5.0, n=1.5)  # Kim model, A/m^2
        Imax1 = Jc * tape_width * tape_thickness
        need = int(np.ceil(I / (util_target * Imax1))) if Imax1 > 0 else 10**9
        return pk, Bp, Jc, Imax1, max(1, need)

    # More tapes -> bigger pack -> lower B_peak -> higher I_c, so the required count falls as the
    # count rises. Find the smallest tpt with need(tpt) <= tpt by bisection.
    lo, hi, cap = 1, 1, 5000
    while hi < cap and _state(hi)[4] > hi:
        lo, hi = hi, min(hi * 2, cap)
    converged = _state(hi)[4] <= hi
    while converged and hi - lo > 1:
        mid = (lo + hi) // 2
        if _state(mid)[4] <= mid:
            hi = mid
        else:
            lo = mid
    tapes_per_turn = hi
    pack, B_peak, J_c, I_max_single_tape, _ = _state(tapes_per_turn)
    I_max = I_max_single_tape * tapes_per_turn

    Bz, Brho = pack_field(pack, r[2], np.hypot(r[0], r[1]))
    B_z = float(Bz)
    B_magnitude = float(np.hypot(Bz, Brho))
    ripple = field_ripple(pack, roi_fraction * R)

    current_utilization = I / I_max if I_max > 0 else float('inf')
    thermal_feasible = T < 80  # Well below Tc
    field_feasible = bool(converged and (current_utilization <= 0.35) and (ripple < 0.01))

    return {
        'B_magnitude': B_magnitude,
        'B_z': B_z,
        'B_peak': float(B_peak),
        'B_filament': float(MU0 * N * I / (2.0 * R)),
        'ripple': ripple,
        'J_c': J_c,
        'J_e': float(pack.Je),
        'I_max': I_max,
        'current_utilization': current_utilization,
        'field_feasible': field_feasible,
        'thermal_feasible': thermal_feasible,
        'temperature': T,
        'tapes_per_turn': tapes_per_turn,
        'effective_thickness': tape_thickness * tapes_per_turn,
        'I_max_single_tape': I_max_single_tape,
        'pack_inner_radius': pack.a1,
        'pack_outer_radius': pack.a2,
        'pack_half_length': pack.b,
        'converged': converged,
    }


def compute_field_ripple(R: float, NI: float, r_roi: float, coils_z: Tuple[float, ...] = (0.0,),
                         n: int = 41, kind: str = 'pp') -> float:
    """
    Ripple of B_z over a spherical region of interest (radius ``r_roi``) for coaxial thin loops
    of radius R at axial positions ``coils_z`` carrying NI amp-turns each (exact loop fields).

    kind='pp'  : (max - min) / B_z(0)            (peak-to-peak)
    kind='rms' : weighted std / weighted mean     (volume weight ~ rho)
    For an extended winding pack use :func:`hts.winding_pack.field_ripple`.
    """
    from .winding_pack import loop_field
    zz, rr = np.meshgrid(np.linspace(-r_roi, r_roi, n), np.linspace(0.0, r_roi, n), indexing='ij')
    mask = zz ** 2 + rr ** 2 <= r_roi ** 2
    Bz = sum(loop_field(R, NI, zz - zc, rr)[0] for zc in coils_z)[mask]
    if kind == 'pp':
        B0 = sum(loop_field(R, NI, -zc, 0.0)[0] for zc in coils_z)
        return float((Bz.max() - Bz.min()) / B0)
    w = rr[mask] + 1e-12
    mu = np.average(Bz, weights=w)
    return float(np.sqrt(np.average((Bz - mu) ** 2, weights=w)) / mu)


def thermal_margin_space(coil_params: Dict[str, float], T_env: float = 4,
                        cryocooler_power: float = 150, min_margin_K: float = 20.0) -> Dict[str, Any]:
    """
    Space-relevant thermal modeling including vacuum conditions.

    Margin basis: if ``coil_params`` contains ``I_tape`` (A per tape), ``B_peak`` (T) and
    ``A_tape`` (m^2 of superconducting cross-section) the margin is measured to the
    CURRENT-SHARING temperature T_cs (I_tape = I_c(B_peak, T_cs)), which is the physically
    relevant limit under field. Otherwise it falls back to the zero-field T_c = 90 K.

    Radiation: ``T_env`` is the temperature of the surface the coil exchanges radiation with.
    Heat flows INTO the coil only when T_env > T_op; a colder surroundings never contributes to
    the load (no credit is taken for radiative cooling). For a shielded cryostat pass the shield
    temperature as ``T_env``.

    Parameters:
    -----------
    coil_params : Dict[str, float]
        Coil parameters (T, R, N, conductor_height, Q_AC, optionally I_tape/B_peak/A_tape)
    T_env : float
        Temperature of the radiating surroundings (K, default: 4 for deep space)
    cryocooler_power : float
        Cryocooler power capacity (W, default: 150)
    min_margin_K : float
        Required margin for ``space_feasible`` (K)

    Returns:
    --------
    thermal_result : Dict[str, Any]
        Comprehensive thermal analysis results
    """
    from .winding_pack import current_sharing_temperature

    T_op = coil_params.get('T', 20.0)  # Operating temperature

    # Surface area of a solenoid: sides + ends. A pack-derived height may be supplied.
    R = coil_params.get('R', 0.15)  # Coil radius (m)
    N = coil_params.get('N', 600)
    tape_height = coil_params.get('conductor_height', 0.004)  # 4mm tape height
    h = coil_params.get('height', N * tape_height)  # Total coil height
    surface_area = 2 * np.pi * R * h + 2 * np.pi * R**2

    # Stefan-Boltzmann radiative INLEAK in vacuum (zero when surroundings are colder)
    sigma_sb = 5.67e-8  # W/(m²·K⁴)
    emissivity = coil_params.get('emissivity', 0.1)
    Q_rad = sigma_sb * emissivity * surface_area * max(0.0, T_env**4 - T_op**4)

    Q_AC = coil_params.get('Q_AC', 0.92)  # W, AC losses at 1 mHz (from manuscript)
    Q_total = Q_rad + Q_AC

    T_c = 90.0  # K (zero-field, zero-current)
    T_limit, basis = T_c, 'Tc'
    if all(k in coil_params for k in ('I_tape', 'B_peak', 'A_tape')):
        T_cs = current_sharing_temperature(coil_params['I_tape'], coil_params['B_peak'],
                                           coil_params['A_tape'], Tc=T_c,
                                           Jc0=300e6, B0=5.0, n=1.5)
        T_limit, basis = (T_cs if T_cs is not None else 0.0), 'T_cs'

    if Q_total < cryocooler_power:
        # With an active cryocooler the internal thermal resistance dominates.
        # R_th is an ASSUMED lumped value (typical 0.1-1.0 K/W); override via coil_params['R_th'].
        R_th_internal = coil_params.get('R_th', 0.5)  # K/W
        T_final = T_op + Q_total * R_th_internal
        margin = T_limit - T_final
        space_feasible = margin > min_margin_K
        cryocooler_adequate = True
    else:
        T_final = T_c  # thermal runaway
        margin = 0
        space_feasible = False
        cryocooler_adequate = False

    return {
        'thermal_margin_K': max(margin, 0),
        'margin_basis': basis,
        'T_limit': T_limit,
        'margin_to_Tc_K': max(T_c - T_final, 0),
        'T_c': T_c,
        'T_final': T_final,
        'heat_load_W': Q_total,
        'Q_rad_W': Q_rad,
        'Q_AC_W': Q_AC,
        'surface_area_m2': surface_area,
        'space_feasible': space_feasible,
        'cryocooler_adequate': cryocooler_adequate,
        'cryocooler_margin_W': cryocooler_power - Q_total
    }


def validate_high_field_parameters(I: float = 1800, N: int = 1000, R: float = 0.16,
                                  T: float = 15, B_target: float = 5.0,
                                  t_reinforcement: float = 0.0,
                                  sigma_allow_hoop: float = 400e6) -> Dict[str, Any]:
    """
    Validate high-field coil parameters for 5-10 T operation with ONE consistent geometry
    (winding pack) used for field, conductor count, thermal surface and mechanics.

    Parameters:
    -----------
    I : float
        Current per turn (A)
    N : int
        Number of turns
    R : float
        Winding-pack inner radius (m)
    T : float
        Operating temperature (K)
    B_target : float
        Target centre field (T); the design must reach >= 80% of it
    t_reinforcement : float
        Radial thickness of a structural overband (m); default none
    sigma_allow_hoop : float
        In-plane tensile allowable of the conductor (Pa); an ASSUMPTION, see
        :func:`hts.winding_pack.hoop_stress_check`

    Returns:
    --------
    validation : Dict[str, Any]
        Results. ``hoop_stress_unreinforced_*`` is the bare estimate, ``hoop_stress_reinforced_*``
        the stress after the overband; the reinforcement needed to meet the allowable is reported
        as ``t_reinf_required_m`` (it is an output, not an assumed outcome).
    """
    from .winding_pack import WindingPack, hoop_stress_check

    r_center = np.array([0, 0, 0])
    field_result = scale_hts_coil_field(r_center, I=I, N=N, R=R, T=T)

    pack = WindingPack(a1=field_result['pack_inner_radius'], a2=field_result['pack_outer_radius'],
                       b=field_result['pack_half_length'], N=N, I=I,
                       tapes_per_turn=field_result['tapes_per_turn'])
    B_peak = field_result['B_peak']

    coil_params = {
        'T': T,
        'R': pack.a_mean,
        'N': N,
        'height': 2 * pack.b,
        'Q_AC': 0.92,  # W, AC losses at 1mHz
        'I_tape': I / field_result['tapes_per_turn'],
        'B_peak': B_peak,
        'A_tape': 4e-3 * 0.2e-3,
    }
    thermal_result = thermal_margin_space(coil_params, T_env=4)

    mech = hoop_stress_check(pack, B_peak, sigma_allow_hoop=sigma_allow_hoop, t_reinf=t_reinforcement)
    hoop_unreinforced = mech['sigma_hoop_bare_Pa']
    hoop_reinforced = mech['sigma_hoop_conductor_Pa']
    reinforcement_factor = max(1.0, hoop_unreinforced / sigma_allow_hoop)

    parameters_valid = bool(
        field_result.get('field_feasible', False) and
        field_result.get('thermal_feasible', False) and
        thermal_result['space_feasible'] and
        mech['hoop_ok'] and mech['radial_ok'] and
        field_result['B_magnitude'] >= B_target * 0.8  # Allow 20% tolerance
    )

    return {
        'parameters': {'I': I, 'N': N, 'R': R, 'T': T, 'B_target': B_target},
        'field_analysis': field_result,
        'thermal_analysis': thermal_result,
        'mechanics': mech,
        'hoop_stress_unreinforced_Pa': hoop_unreinforced,
        'hoop_stress_unreinforced_MPa': hoop_unreinforced / 1e6,
        'hoop_stress_reinforced_Pa': hoop_reinforced,
        'hoop_stress_reinforced_MPa': hoop_reinforced / 1e6,
        'radial_stress_est_MPa': mech['sigma_radial_est_Pa'] / 1e6,
        't_reinf_required_m': mech['t_reinf_required_m'],
        'reinforcement_factor': reinforcement_factor,
        'rebco_stress_limit_Pa': sigma_allow_hoop,
        'rebco_stress_limit_MPa': sigma_allow_hoop / 1e6,
        'parameters_valid': parameters_valid,
        'overall_feasible': parameters_valid,  # Add alias for compatibility
        'achieved_field_T': field_result['B_magnitude'],
        'thermal_margin_K': thermal_result['thermal_margin_K'],
        'current_utilization': field_result['current_utilization']
    }


def helmholtz_high_field_configuration(target_field: float = 5.0, 
                                     target_ripple: float = 0.008) -> Dict[str, Any]:
    """
    Design Helmholtz configuration for high-field operation.
    
    Parameters:
    -----------
    target_field : float
        Target magnetic field (T, default: 5.0)
    target_ripple : float
        Target field ripple (fractional, default: 0.008)
        
    Returns:
    --------
    design : Dict[str, Any]
        Optimized Helmholtz configuration
    """
    # Starting parameters for high-field design
    # Candidates chosen from a parameter scan of the winding-pack model (hoop, radial, T_cs and
    # ripple gates all satisfied). NOTE: despite the name, each candidate is a single coil.
    configurations = [
        {'I': 3500, 'N': 1000, 'R': 0.16, 'T': 10},
        {'I': 2500, 'N': 1400, 'R': 0.12, 'T': 20},
        {'I': 1800, 'N': 2000, 'R': 0.12, 'T': 20}
    ]
    
    best_config = None
    best_score = float('inf')
    
    for config in configurations:
        validation = validate_high_field_parameters(**config)
        
        field_achieved = validation['field_analysis']['B_magnitude']
        ripple_achieved = validation['field_analysis']['ripple']
        
        # Scoring function (minimize deviation from targets)
        field_error = abs(field_achieved - target_field) / target_field
        ripple_error = abs(ripple_achieved - target_ripple) / target_ripple
        
        # Penalty for infeasible designs
        feasibility_penalty = 0 if validation['overall_feasible'] else 1000
        
        score = field_error + ripple_error + feasibility_penalty
        
        if score < best_score:
            best_score = score
            best_config = {
                'parameters': config,
                'performance': validation,
                'score': score
            }
    
    return best_config


if __name__ == "__main__":
    # Demonstrate high-field scaling capabilities
    print("=== High-Field HTS Coil Scaling Analysis ===")
    
    # Test high-field configuration
    validation = validate_high_field_parameters(I=5000, N=600, R=0.15, T=10)
    
    print(f"Parameters: I={validation['parameters']['I']}A, "
          f"N={validation['parameters']['N']}, "
          f"R={validation['parameters']['R']}m, "
          f"T={validation['parameters']['T']}K")
    
    print(f"Field: {validation['field_analysis']['B_magnitude']:.2f} T")
    print(f"Ripple: {validation['field_analysis']['ripple']*100:.3f}%")
    print(f"Space thermal margin: {validation['thermal_margin_K']:.1f} K")
    print(f"Hoop stress (reinforced): {validation['hoop_stress_reinforced_MPa']:.1f} MPa")
    print(f"Overall feasible: {validation['overall_feasible']}")
    
    # Test Helmholtz optimization
    print("\n=== Helmholtz High-Field Design ===")
    helmholtz_design = helmholtz_high_field_configuration(target_field=5.2)
    
    if helmholtz_design:
        params = helmholtz_design['parameters']
        perf = helmholtz_design['performance']
        print(f"Optimized parameters: I={params['I']}A, N={params['N']}, "
              f"R={params['R']}m, T={params['T']}K")
        print(f"Achieved field: {perf['field_analysis']['B_magnitude']:.2f} T")
        print(f"Achieved ripple: {perf['field_analysis']['ripple']*100:.4f}%")
        print(f"Design score: {helmholtz_design['score']:.3f}")