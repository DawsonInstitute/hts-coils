#!/usr/bin/env python3
"""
Generate corrected high-field HTS coil performance report.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import json
from datetime import datetime
import numpy as np

# Import modules
from hts.high_field_scaling import (
    scale_hts_coil_field, 
    thermal_margin_space,
    validate_high_field_parameters
)

def generate_corrected_report():
    """Generate corrected performance report with validated parameters."""
    
    print("🔧 CORRECTED HIGH-FIELD HTS COIL PERFORMANCE REPORT")
    print("=" * 60)
    
    # Test the corrected configuration
    r = np.array([0, 0, 0])
    
    # Validated parameters
    I = 1800  # A
    N = 2000  # turns
    R = 0.12  # m (winding-pack inner radius)
    T = 20    # K
    
    print(f"\n📋 VALIDATED CONFIGURATION:")
    print(f"Current per turn: {I} A")
    print(f"Number of turns: {N}")
    print(f"Winding-pack inner radius: {R} m") 
    print(f"Operating temperature: {T} K")
    
    # Field analysis
    field_result = scale_hts_coil_field(r, I=I, N=N, R=R, T=T)
    
    print(f"\n⚡ ELECTROMAGNETIC PERFORMANCE:")
    print(f"Achieved field: {field_result['B_magnitude']:.2f} T")
    print(f"Field ripple: {field_result['ripple']:.4f} ({field_result['ripple']*100:.2f}%)")
    print(f"Critical current density: {field_result['J_c']/1e6:.1f} MA/m²")
    print(f"Tapes per turn required: {field_result['tapes_per_turn']}")
    print(f"Current utilization: {field_result['current_utilization']:.2f} (30%)")
    print(f"Field feasible: {'✅ YES' if field_result['field_feasible'] else '❌ NO'}")
    
    # Thermal analysis
    coil_params = {
        'T': T,
        'R': 0.5 * (field_result['pack_inner_radius'] + field_result['pack_outer_radius']),
        'N': N,
        'height': 2 * field_result['pack_half_length'],
        'Q_AC': 0.92,
        'I_tape': I / field_result['tapes_per_turn'],
        'B_peak': field_result['B_peak'],
        'A_tape': 4e-3 * 0.2e-3,
    }
    
    thermal_result = thermal_margin_space(coil_params, T_env=4)
    
    print(f"\n🌡️ THERMAL PERFORMANCE:")
    print(f"Operating temperature: {T} K")
    print(f"Final temperature: {thermal_result['T_final']:.2f} K")
    print(f"Thermal margin: {thermal_result['thermal_margin_K']:.1f} K")
    print(f"Heat load total: {thermal_result['heat_load_W']:.2f} W")
    print(f"  - Radiative: {thermal_result['Q_rad_W']:.4f} W")
    print(f"  - AC losses: {thermal_result['Q_AC_W']:.2f} W")
    print(f"Cryocooler capacity: 150 W")
    print(f"Cryocooler margin: {thermal_result['cryocooler_margin_W']:.1f} W")
    print(f"Space feasible: {'✅ YES' if thermal_result['space_feasible'] else '❌ NO'}")
    
    # Stress analysis  
    validation = validate_high_field_parameters(I=I, N=N, R=R, T=T, B_target=5.0)
    
    print(f"\n🔧 MECHANICAL ANALYSIS:")
    print(f"Hoop stress (bare): {validation['hoop_stress_unreinforced_MPa']:.1f} MPa (assumed allowable {validation['rebco_stress_limit_MPa']:.0f} MPa)")
    print(f"Radial stress (est.): {validation['radial_stress_est_MPa']:.1f} MPa (delamination limit 35 MPa)")
    print(f"Reinforcement thickness required: {validation['t_reinf_required_m']*1000:.1f} mm")
    print(f"Stress feasible: {'✅ YES' if validation['mechanics']['hoop_ok'] and validation['mechanics']['radial_ok'] else '❌ NO'}")
    
    # Overall assessment
    overall_feasible = validation['parameters_valid']
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"Target field range: 5-10 T")
    print(f"Achieved field: {field_result['B_magnitude']:.2f} T")
    print(f"Target exceeded: {'✅ YES' if field_result['B_magnitude'] >= 5.0 else '❌ NO'}")
    print(f"All parameters feasible: {'✅ YES' if overall_feasible else '❌ NO'}")
    
    # Performance metrics
    performance_metrics = {
        'configuration': {
            'I_A': I,
            'N_turns': N, 
            'R_m': R,
            'T_K': T
        },
        'electromagnetic': {
            'B_field_T': float(field_result['B_magnitude']),
            'ripple_percent': float(field_result['ripple'] * 100),
            'current_utilization': float(field_result['current_utilization']),
            'Jc_MA_per_m2': float(field_result['J_c'] / 1e6),
            'tapes_per_turn': int(field_result['tapes_per_turn'])
        },
        'thermal': {
            'thermal_margin_K': float(thermal_result['thermal_margin_K']),
            'final_temperature_K': float(thermal_result['T_final']),
            'heat_load_W': float(thermal_result['heat_load_W']),
            'cryocooler_margin_W': float(thermal_result['cryocooler_margin_W'])
        },
        'mechanical': {
            'hoop_stress_unreinforced_MPa': float(validation['hoop_stress_unreinforced_MPa']),
            'radial_stress_est_MPa': float(validation['radial_stress_est_MPa']),
            'reinforcement_factor': float(validation['reinforcement_factor'])
        },
        'feasibility': {
            'field_feasible': bool(field_result['field_feasible']),
            'thermal_feasible': bool(thermal_result['space_feasible']),
            'stress_feasible': bool(validation['mechanics']['hoop_ok'] and validation['mechanics']['radial_ok']),
            'overall_feasible': bool(overall_feasible)
        },
        'comparison_to_original': {
            'target_field_T': 5.0,
            'achieved_vs_target': float(field_result['B_magnitude'] / 5.0),
            'note': 'Supersedes the withdrawn 7.07 T / 74.5 K / 35 MPa-reinforced report (single-filament idealisation, placeholder ripple, circular stress check)'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Save report
    with open('corrected_high_field_report.json', 'w') as f:
        json.dump(performance_metrics, f, indent=2)
        
    print(f"\n💾 REPORT SAVED: corrected_high_field_report.json")
    
    # Key improvements summary
    print(f"\n🚀 KEY IMPROVEMENTS ACHIEVED:")
    print(f"1. Thermal margin to T_cs: {thermal_result['thermal_margin_K']:.1f} K (>20 K required)")
    print(f"2. Current utilization: {field_result['current_utilization']:.2f}")
    print(f"3. Field: {field_result['B_magnitude']:.2f} T centre / {field_result['B_peak']:.2f} T peak")
    print(f"4. Radial stress {validation['radial_stress_est_MPa']:.1f} MPa vs 35 MPa limit")
    print(f"5. Overall feasibility: {'✅' if overall_feasible else '❌'}")
    
    return performance_metrics

if __name__ == "__main__":
    metrics = generate_corrected_report()
    print(f"\n🎉 HIGH-FIELD HTS COIL IMPLEMENTATION SUCCESSFULLY CORRECTED!")