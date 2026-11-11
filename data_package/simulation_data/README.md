# HTS Coil Optimization - Data Package

## Overview
This package contains key simulation data from:
"Optimization of REBCO High-Temperature Superconducting Coils for 
High-Field Applications in Fusion and Antimatter Trapping"

## Contents

### simulation_results.json
Core simulation results for both baseline (2.1 T) and high-field (6.43 T, winding-pack model) configurations. The high-field figures replace an earlier withdrawn 7.07 T / 74.5 K result.

Key results:
- **High-field**: 6.43 T, 0.17% ripple (p-p, r <= 0.1*a1), 38.3 K margin to the current-sharing temperature
- **Baseline**: 2.1 T, 0.01% ripple, 70 K thermal margin  

### comsol_template.java
Template for COMSOL Multiphysics validation studies.

### reproduce_figures.py
Python script to reproduce key manuscript figures.
Usage: `python reproduce_figures.py`

## Validation
All results validated against Docker-based reproduction system.
Run `docker run hts-coil-simulator python run_high_field_simulation.py --verbose`

## Generated
2026-11-11 13:24:00

Ready for Zenodo upload and DOI assignment.
