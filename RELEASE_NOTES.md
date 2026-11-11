> **SUPERSEDED NUMBERS (2026-11-11).** The high-field figures in this document (7.07 T, 0.16% ripple, 74.5 K thermal margin, 89-tape conductor, "35 MPa reinforced" stress) come from an analysis that has been withdrawn: 7.07 T was the field of all turns on a single thin loop, inconsistent with the conductor required, and the ripple, stress and margin were placeholder or circular. The self-consistent winding-pack model gives **6.43 T** (95 tapes/turn, 20 K, 0.17% ripple over r<=0.1*a1, 38 K margin to the current-sharing temperature, 31.5 MPa radial / 50.9 MPa hoop). See `README.md` and `results/high_field_rederived.json`. The text below is kept for history and has not been rewritten.

hts-coils v2.1.0 — Validated high-field REBCO HTS coil optimization

Summary:
- Validated multi-physics HTS coil optimization framework (electromagnetic, thermal, mechanical).
- Multi-backend FEA support: FEniCSx + COMSOL with analytic fallback.
- High-field configuration validated: 7.07 T (89-tape conductor), 0.16% ripple, 74.5 K thermal margin.
- Manuscript fixes, reproducibility improvements (Docker, reproduction scripts), and completed `docs/TODO.ndjson` tasks.

Assets included: manuscript TeX source and TODO list. See Zenodo data DOI: 10.5281/zenodo.17033655 for full datasets.

Notes:
- This release is created as a draft. Publish when ready.
