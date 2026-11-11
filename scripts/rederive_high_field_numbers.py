#!/usr/bin/env python3
"""Regenerate the headline high-field numbers with the self-consistent winding-pack model.

Writes results/high_field_rederived.json. Every number quoted in the manuscript's high-field
section should come from this file.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hts.high_field_scaling import validate_high_field_parameters, thermal_margin_space  # noqa: E402
from hts.winding_pack import MU0  # noqa: E402

DESIGNS = {
    # the geometry the manuscript originally reported as "7.07 T"
    "legacy_1800A_1000t": dict(I=1800, N=1000, R=0.16, T=15, B_target=5.0),
    # best design passing hoop, radial, T_cs, utilisation and ripple gates in a parameter scan
    "feasible_1800A_2000t": dict(I=1800, N=2000, R=0.12, T=20, B_target=5.0),
    "feasible_3500A_1000t": dict(I=3500, N=1000, R=0.16, T=10, B_target=5.0),
}


def summarise(kw):
    v = validate_high_field_parameters(**kw)
    f, t, m = v["field_analysis"], v["thermal_analysis"], v["mechanics"]
    return {
        "parameters": kw,
        "B_center_T": float(f["B_magnitude"]),
        "B_peak_T": float(f["B_peak"]),
        "B_filament_idealised_T": float(f["B_filament"]),
        "tapes_per_turn": int(f["tapes_per_turn"]),
        "Jc_at_Bpeak_MA_m2": float(f["J_c"] / 1e6),
        "current_utilisation": float(f["current_utilization"]),
        "Je_MA_m2": float(f["J_e"] / 1e6),
        "pack_inner_outer_radius_m": [float(f["pack_inner_radius"]), float(f["pack_outer_radius"])],
        "pack_half_length_m": float(f["pack_half_length"]),
        "ripple_pp_fraction_ROI_0p1R": float(f["ripple"]),
        "thermal_margin_Tcs_K": float(t["thermal_margin_K"]),
        "margin_basis": t["margin_basis"],
        "T_final_K": float(t["T_final"]),
        "hoop_stress_bare_MPa": float(v["hoop_stress_unreinforced_MPa"]),
        "radial_stress_est_MPa": float(v["radial_stress_est_MPa"]),
        "t_reinf_required_m": float(v["t_reinf_required_m"]),
        "magnetic_pressure_MPa": float(m["magnetic_pressure_Pa"] / 1e6),
        "all_gates_pass": bool(v["parameters_valid"]),
    }


if __name__ == "__main__":
    out = {k: summarise(kw) for k, kw in DESIGNS.items()}
    out["_assumptions"] = {"hoop_allowable_MPa": 400, "radial_limit_MPa": 35, "R_th_K_per_W": 0.5,
                           "fill_factor": 1.0, "util_target": 0.3, "ripple_ROI": "sphere r=0.1*R_inner",
                           "Q_AC_W": 0.92, "T_env_K": 4}
    p = ROOT / "results" / "high_field_rederived.json"
    p.write_text(json.dumps(out, indent=2))
    for k, d in out.items():
        if k.startswith("_"):
            continue
        print(k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in d.items() if x != "parameters"})
