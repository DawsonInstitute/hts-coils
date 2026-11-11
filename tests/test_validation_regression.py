"""Regression: the winding-pack model must reproduce the high-field benchmarks.

The benchmarks in ValidationFramework are compared against values COMPUTED by the model
(not literals), so this fails if either the model or the quoted paper numbers drift.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hts.validation_framework import ValidationFramework, model_high_field_config  # noqa: E402


def test_high_field_model_matches_benchmarks():
    cfg = model_high_field_config()
    v = ValidationFramework()
    assert v.validate_high_field_config(
        field=cfg["field"], ripple=cfg["ripple"], current=cfg["current"], turns=cfg["turns"],
        radius=cfg["radius"], tapes_per_turn=cfg["tapes_per_turn"],
        temperature=cfg["temperature"], thermal_margin=cfg["thermal_margin"])
    assert v.validate_against_paper(cfg["radial_stress"], "high_field_radial_stress", "radial")
    assert v.validate_against_paper(cfg["hoop_stress"], "high_field_hoop_stress", "hoop")


def test_withdrawn_7p07_does_not_validate():
    v = ValidationFramework()
    assert not v.validate_against_paper(7.07, "high_field", "withdrawn single-filament value")
