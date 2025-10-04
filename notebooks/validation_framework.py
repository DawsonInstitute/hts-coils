# Data Validation and Reproducibility Framework
# This module provides validation functions for HTS coil design notebooks

import numpy as np
import warnings
from typing import Dict, List, Tuple, Any

class ValidationFramework:
    """Framework for validating HTS coil calculations against paper benchmarks"""
    
    def __init__(self):
        # Paper benchmark values from rebco_hts_coil_optimization_fusion_antimatter.tex
        self.paper_benchmarks = {
            # Baseline configuration (2.1T design)
            'baseline_field': {'value': 2.1, 'tolerance': 0.01, 'unit': 'T'},
            'baseline_ripple': {'value': 0.01, 'tolerance': 0.001, 'unit': '%'},
            'baseline_current': {'value': 1171, 'tolerance': 10, 'unit': 'A'},
            'baseline_turns': {'value': 400, 'tolerance': 5, 'unit': ''},
            'baseline_radius': {'value': 0.2, 'tolerance': 0.001, 'unit': 'm'},
            
            # High-field configuration (7.07T design)
            'high_field': {'value': 7.07, 'tolerance': 0.01, 'unit': 'T'},
            'high_field_ripple': {'value': 0.16, 'tolerance': 0.01, 'unit': '%'},
            'high_field_current': {'value': 1800, 'tolerance': 20, 'unit': 'A'},
            'high_field_turns': {'value': 1000, 'tolerance': 10, 'unit': ''},
            'high_field_radius': {'value': 0.16, 'tolerance': 0.001, 'unit': 'm'},
            'high_field_tapes_per_turn': {'value': 89, 'tolerance': 2, 'unit': ''},
            'high_field_temperature': {'value': 15, 'tolerance': 1, 'unit': 'K'},
            'high_field_thermal_margin': {'value': 74.5, 'tolerance': 1.5, 'unit': 'K'},
            
            # Thermal analysis benchmarks
            'thermal_margin_baseline': {'value': 74.5, 'tolerance': 2.0, 'unit': 'K'},
            'cryocooler_power': {'value': 150, 'tolerance': 10, 'unit': 'W'},
            
            # Mechanical stress benchmarks  
            'stress_baseline': {'value': 175, 'tolerance': 10, 'unit': 'MPa'},
            'stress_reinforced': {'value': 35, 'tolerance': 5, 'unit': 'MPa'},
            'stress_limit': {'value': 35, 'tolerance': 0, 'unit': 'MPa'},
            
            # Legacy benchmarks for backward compatibility
            'ripple_baseline': {'value': 0.01, 'tolerance': 0.005, 'unit': '%'},
            'ripple_high': {'value': 0.16, 'tolerance': 0.05, 'unit': '%'},
            'current_baseline': {'value': 1171, 'tolerance': 50, 'unit': 'A'},
            'current_high': {'value': 1800, 'tolerance': 100, 'unit': 'A'},
            'turns_baseline': {'value': 400, 'tolerance': 10, 'unit': ''},
            'turns_high': {'value': 1000, 'tolerance': 50, 'unit': ''}
        }
        
        # Physical constants for validation
        self.constants = {
            'mu_0': 4 * np.pi * 1e-7,  # H/m
            'k_B': 1.380649e-23,       # J/K
            'elementary_charge': 1.602176634e-19,  # C
            'T_c_REBCO': 92.0,         # K (approximate)
        }
        
        # Valid ranges for physical parameters
        self.physical_ranges = {
            'temperature': {'min': 4.2, 'max': 300, 'unit': 'K'},
            'magnetic_field': {'min': 0, 'max': 50, 'unit': 'T'},
            'current_density': {'min': 1e6, 'max': 1e9, 'unit': 'A/m²'},
            'current': {'min': 100, 'max': 5000, 'unit': 'A'},
            'radius': {'min': 0.1, 'max': 10, 'unit': 'm'},
            'height': {'min': 0.1, 'max': 20, 'unit': 'm'}
        }
        
        # Results storage for reproducibility checks
        self.validation_results = {}
        
    def validate_against_paper(self, calculated_value: float, benchmark_key: str, 
                             description: str = "") -> bool:
        """
        Validate a calculated value against paper benchmark
        
        Args:
            calculated_value: The value calculated by our notebook
            benchmark_key: Key for the benchmark in paper_benchmarks
            description: Description of what's being validated
            
        Returns:
            bool: True if validation passes
        """
        if benchmark_key not in self.paper_benchmarks:
            print(f"❌ Unknown benchmark key: {benchmark_key}")
            return False
            
        benchmark = self.paper_benchmarks[benchmark_key]
        expected = benchmark['value']
        tolerance = benchmark['tolerance']
        unit = benchmark['unit']
        
        # Check if within tolerance
        error = abs(calculated_value - expected)
        relative_error = error / expected * 100 if expected != 0 else float('inf')
        passed = error <= tolerance
        
        # Store result
        self.validation_results[benchmark_key] = {
            'calculated': calculated_value,
            'expected': expected,
            'error': error,
            'relative_error': relative_error,
            'tolerance': tolerance,
            'passed': passed,
            'description': description
        }
        
        # Print result
        status = "✅" if passed else "❌"
        print(f"{status} {description or benchmark_key}")
        print(f"   Expected: {expected} ± {tolerance} {unit}")
        print(f"   Calculated: {calculated_value:.4f} {unit}")
        print(f"   Error: {error:.4f} {unit} ({relative_error:.2f}%)")
        
        if not passed:
            print(f"   ⚠️  Validation failed - check calculation methodology")
            
        return passed
    
    def check_physical_reasonableness(self, value: float, parameter: str, 
                                    context: str = "") -> bool:
        """
        Check if a parameter value is physically reasonable
        
        Args:
            value: Value to check
            parameter: Parameter type (e.g., 'temperature', 'magnetic_field')
            context: Additional context for the check
            
        Returns:
            bool: True if physically reasonable
        """
        if parameter not in self.physical_ranges:
            print(f"❌ Unknown parameter type: {parameter}")
            return False
            
        range_info = self.physical_ranges[parameter]
        min_val = range_info['min']
        max_val = range_info['max']
        unit = range_info['unit']
        
        is_reasonable = min_val <= value <= max_val
        
        status = "✅" if is_reasonable else "❌"
        print(f"{status} {parameter.replace('_', ' ').title()} check: {value} {unit}")
        
        if not is_reasonable:
            print(f"   ⚠️  Value outside reasonable range: {min_val}-{max_val} {unit}")
            if context:
                print(f"   Context: {context}")
                
        return is_reasonable
    
    def check_unit_consistency(self, calculation_dict: Dict[str, Dict]) -> bool:
        """
        Check dimensional consistency of calculations
        
        Args:
            calculation_dict: Dictionary with 'value' and 'units' for each quantity
            
        Returns:
            bool: True if units are consistent
        """
        print("🔍 Unit Consistency Check")
        print("-" * 30)
        
        all_consistent = True
        
        for name, info in calculation_dict.items():
            value = info.get('value', 0)
            units = info.get('units', '')
            expected_units = info.get('expected_units', units)
            
            consistent = units == expected_units
            status = "✅" if consistent else "❌"
            
            print(f"{status} {name}: {value} [{units}]")
            if not consistent:
                print(f"   Expected units: [{expected_units}]")
                all_consistent = False
                
        return all_consistent
    
    def numerical_convergence_test(self, calculation_func, parameter_name: str,
                                 base_value: float, steps: List[float]) -> bool:
        """
        Test numerical convergence by varying discretization parameters
        
        Args:
            calculation_func: Function that takes parameter and returns result
            parameter_name: Name of the parameter being varied
            base_value: Base value for the parameter
            steps: List of step sizes to test
            
        Returns:
            bool: True if calculation shows convergence
        """
        print(f"🔄 Convergence Test: {parameter_name}")
        print("-" * 40)
        
        results = []
        for step in steps:
            param_value = base_value * step
            try:
                result = calculation_func(param_value)
                results.append(result)
                print(f"   {parameter_name} = {param_value:.6f}: Result = {result:.6f}")
            except Exception as e:
                print(f"   ❌ Error at {parameter_name} = {param_value}: {e}")
                return False
        
        # Check convergence (relative change should decrease)
        if len(results) < 3:
            print("   ⚠️  Need at least 3 points for convergence check")
            return False
            
        # Calculate relative changes
        rel_changes = []
        for i in range(1, len(results)):
            if results[i-1] != 0:
                rel_change = abs(results[i] - results[i-1]) / abs(results[i-1])
                rel_changes.append(rel_change)
                
        # Check if changes are decreasing (convergence)
        converging = all(rel_changes[i] < rel_changes[i-1] for i in range(1, len(rel_changes)))
        
        status = "✅" if converging else "❌"
        print(f"{status} Convergence: {converging}")
        
        if converging:
            print(f"   Final relative change: {rel_changes[-1]:.2e}")
        else:
            print(f"   ⚠️  Solution may not be converged")
            
        return converging
    
    def uncertainty_analysis(self, nominal_value: float, uncertainties: Dict[str, float],
                           sensitivity_func) -> Dict[str, float]:
        """
        Perform uncertainty propagation analysis
        
        Args:
            nominal_value: Nominal calculation result
            uncertainties: Dictionary of parameter uncertainties
            sensitivity_func: Function that calculates sensitivity to each parameter
            
        Returns:
            dict: Uncertainty analysis results
        """
        print("📊 Uncertainty Analysis")
        print("-" * 25)
        
        total_variance = 0
        uncertainty_contributions = {}
        
        for param, uncertainty in uncertainties.items():
            # Calculate sensitivity (partial derivative)
            sensitivity = sensitivity_func(param)
            
            # Calculate contribution to total uncertainty
            contribution = (sensitivity * uncertainty) ** 2
            total_variance += contribution
            
            uncertainty_contributions[param] = {
                'uncertainty': uncertainty,
                'sensitivity': sensitivity,
                'contribution': np.sqrt(contribution),
                'percent_contribution': contribution / total_variance * 100 if total_variance > 0 else 0
            }
            
            print(f"   {param}: ±{uncertainty:.4f} → ±{np.sqrt(contribution):.4f} ({np.sqrt(contribution)/nominal_value*100:.2f}%)")
        
        total_uncertainty = np.sqrt(total_variance)
        
        print(f"\n📈 Total Uncertainty: ±{total_uncertainty:.4f} ({total_uncertainty/nominal_value*100:.2f}%)")
        print(f"   Result: {nominal_value:.4f} ± {total_uncertainty:.4f}")
        
        return {
            'nominal_value': nominal_value,
            'total_uncertainty': total_uncertainty,
            'relative_uncertainty': total_uncertainty / nominal_value * 100,
            'contributions': uncertainty_contributions
        }
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report"""
        
        report = "# Validation Report\n"
        report += "=" * 50 + "\n\n"
        
        if not self.validation_results:
            report += "No validation results available.\n"
            return report
            
        # Summary statistics
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results.values() if r['passed'])
        
        report += f"## Summary\n"
        report += f"Total Tests: {total_tests}\n"
        report += f"Passed: {passed_tests}\n"
        report += f"Failed: {total_tests - passed_tests}\n"
        report += f"Success Rate: {passed_tests/total_tests*100:.1f}%\n\n"
        
        # Detailed results
        report += "## Detailed Results\n\n"
        
        for test_name, result in self.validation_results.items():
            status = "PASS" if result['passed'] else "FAIL"
            report += f"### {test_name} [{status}]\n"
            report += f"- Description: {result['description']}\n"
            report += f"- Expected: {result['expected']:.4f}\n"
            report += f"- Calculated: {result['calculated']:.4f}\n"
            report += f"- Error: {result['error']:.4f} ({result['relative_error']:.2f}%)\n"
            report += f"- Tolerance: ±{result['tolerance']:.4f}\n\n"
            
        return report
    
    def export_results(self, filename: str = "validation_results.json"):
        """Export validation results to JSON file"""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        exportable_results = {}
        for key, result in self.validation_results.items():
            exportable_results[key] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in result.items()
            }
            
        with open(filename, 'w') as f:
            json.dump({
                'validation_results': exportable_results,
                'paper_benchmarks': self.paper_benchmarks,
                'constants': self.constants
            }, f, indent=2)
            
        print(f"📁 Validation results exported to {filename}")

    def validate_baseline_config(self, field: float, ripple: float, current: float, 
                                turns: int, radius: float) -> bool:
        """
        Validate complete baseline configuration against rebco paper benchmarks
        
        Args:
            field: Magnetic field in T
            ripple: Field ripple in %
            current: Operating current in A
            turns: Number of turns
            radius: Coil radius in m
            
        Returns:
            bool: True if all parameters validate
        """
        print("🎯 REBCO Paper Baseline Configuration Validation")
        print("-" * 55)
        
        validations = [
            self.validate_against_paper(field, 'baseline_field', 'Baseline magnetic field'),
            self.validate_against_paper(ripple, 'baseline_ripple', 'Baseline field ripple'),
            self.validate_against_paper(current, 'baseline_current', 'Baseline operating current'),
            self.validate_against_paper(turns, 'baseline_turns', 'Baseline number of turns'),
            self.validate_against_paper(radius, 'baseline_radius', 'Baseline coil radius')
        ]
        
        all_passed = all(validations)
        status = "✅ ALL PASSED" if all_passed else "❌ SOME FAILED"
        print(f"\n{status} - Baseline configuration validation complete")
        
        return all_passed
    
    def validate_high_field_config(self, field: float, ripple: float, current: float,
                                  turns: int, radius: float, tapes_per_turn: int,
                                  temperature: float, thermal_margin: float) -> bool:
        """
        Validate complete high-field configuration against rebco paper benchmarks
        
        Args:
            field: Magnetic field in T
            ripple: Field ripple in %
            current: Operating current in A
            turns: Number of turns
            radius: Coil radius in m
            tapes_per_turn: Number of tapes per turn
            temperature: Operating temperature in K
            thermal_margin: Thermal margin in K
            
        Returns:
            bool: True if all parameters validate
        """
        print("🎯 REBCO Paper High-Field Configuration Validation")
        print("-" * 57)
        
        validations = [
            self.validate_against_paper(field, 'high_field', 'High-field magnetic field'),
            self.validate_against_paper(ripple, 'high_field_ripple', 'High-field ripple'),
            self.validate_against_paper(current, 'high_field_current', 'High-field current'),
            self.validate_against_paper(turns, 'high_field_turns', 'High-field turns'),
            self.validate_against_paper(radius, 'high_field_radius', 'High-field radius'),
            self.validate_against_paper(tapes_per_turn, 'high_field_tapes_per_turn', 'Tapes per turn'),
            self.validate_against_paper(temperature, 'high_field_temperature', 'Operating temperature'),
            self.validate_against_paper(thermal_margin, 'high_field_thermal_margin', 'Thermal margin')
        ]
        
        all_passed = all(validations)
        status = "✅ ALL PASSED" if all_passed else "❌ SOME FAILED"
        print(f"\n{status} - High-field configuration validation complete")
        
        return all_passed
    
    def validate_thermal_analysis(self, thermal_margin: float, cryocooler_power: float) -> bool:
        """
        Validate thermal analysis results against rebco paper benchmarks
        
        Args:
            thermal_margin: Thermal margin in K
            cryocooler_power: Cryocooler power requirement in W
            
        Returns:
            bool: True if thermal analysis validates
        """
        print("🌡️ REBCO Paper Thermal Analysis Validation")
        print("-" * 45)
        
        validations = [
            self.validate_against_paper(thermal_margin, 'thermal_margin_baseline', 'Thermal margin'),
            self.validate_against_paper(cryocooler_power, 'cryocooler_power', 'Cryocooler power requirement')
        ]
        
        all_passed = all(validations)
        status = "✅ ALL PASSED" if all_passed else "❌ SOME FAILED"
        print(f"\n{status} - Thermal analysis validation complete")
        
        return all_passed
    
    def validate_stress_analysis(self, baseline_stress: float, reinforced_stress: float) -> bool:
        """
        Validate mechanical stress analysis against rebco paper benchmarks
        
        Args:
            baseline_stress: Baseline hoop stress in MPa
            reinforced_stress: Reinforced design stress in MPa
            
        Returns:
            bool: True if stress analysis validates
        """
        print("🔧 REBCO Paper Stress Analysis Validation")
        print("-" * 42)
        
        validations = [
            self.validate_against_paper(baseline_stress, 'stress_baseline', 'Baseline hoop stress'),
            self.validate_against_paper(reinforced_stress, 'stress_reinforced', 'Reinforced stress')
        ]
        
        # Check if reinforced stress meets design limit
        stress_limit_check = reinforced_stress <= self.paper_benchmarks['stress_limit']['value']
        status_limit = "✅" if stress_limit_check else "❌"
        print(f"{status_limit} Stress limit check: {reinforced_stress:.1f} ≤ {self.paper_benchmarks['stress_limit']['value']:.1f} MPa")
        
        all_passed = all(validations) and stress_limit_check
        status = "✅ ALL PASSED" if all_passed else "❌ SOME FAILED"
        print(f"\n{status} - Stress analysis validation complete")
        
        return all_passed
    
    def comprehensive_rebco_validation(self, baseline_config: Dict, high_field_config: Dict,
                                     thermal_results: Dict, stress_results: Dict) -> bool:
        """
        Perform comprehensive validation of all rebco paper results
        
        Args:
            baseline_config: Dict with baseline configuration parameters
            high_field_config: Dict with high-field configuration parameters  
            thermal_results: Dict with thermal analysis results
            stress_results: Dict with stress analysis results
            
        Returns:
            bool: True if all validations pass
        """
        print("🎯 COMPREHENSIVE REBCO PAPER VALIDATION")
        print("=" * 50)
        
        # Baseline configuration validation
        baseline_passed = self.validate_baseline_config(
            baseline_config['field'],
            baseline_config['ripple'], 
            baseline_config['current'],
            baseline_config['turns'],
            baseline_config['radius']
        )
        
        print("\n")
        
        # High-field configuration validation
        high_field_passed = self.validate_high_field_config(
            high_field_config['field'],
            high_field_config['ripple'],
            high_field_config['current'], 
            high_field_config['turns'],
            high_field_config['radius'],
            high_field_config['tapes_per_turn'],
            high_field_config['temperature'],
            high_field_config['thermal_margin']
        )
        
        print("\n")
        
        # Thermal analysis validation
        thermal_passed = self.validate_thermal_analysis(
            thermal_results['thermal_margin'],
            thermal_results['cryocooler_power']
        )
        
        print("\n")
        
        # Stress analysis validation
        stress_passed = self.validate_stress_analysis(
            stress_results['baseline_stress'],
            stress_results['reinforced_stress']
        )
        
        # Overall result
        all_passed = baseline_passed and high_field_passed and thermal_passed and stress_passed
        
        print("\n" + "=" * 50)
        print("📊 OVERALL VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Baseline Configuration: {'✅ PASS' if baseline_passed else '❌ FAIL'}")
        print(f"High-Field Configuration: {'✅ PASS' if high_field_passed else '❌ FAIL'}")
        print(f"Thermal Analysis: {'✅ PASS' if thermal_passed else '❌ FAIL'}")
        print(f"Stress Analysis: {'✅ PASS' if stress_passed else '❌ FAIL'}")
        print("-" * 50)
        print(f"OVERALL RESULT: {'🎉 ALL VALIDATIONS PASSED' if all_passed else '⚠️ SOME VALIDATIONS FAILED'}")
        
        return all_passed

# Convenience functions for common validation tasks
def validate_magnetic_field_calculation(B_calculated: float, config_type: str = "baseline") -> bool:
    """Validate magnetic field calculation against paper results"""
    validator = ValidationFramework()
    
    if config_type == "baseline":
        benchmark_key = "baseline_field"
    elif config_type == "high_field":
        benchmark_key = "high_field"
    else:
        print(f"❌ Unknown config_type: {config_type}")
        return False
    
    description = f"Magnetic field calculation ({config_type} configuration)"
    return validator.validate_against_paper(B_calculated, benchmark_key, description)

def validate_current_calculation(I_calculated: float, config_type: str = "baseline") -> bool:
    """Validate current calculation against paper results"""
    validator = ValidationFramework()
    
    if config_type == "baseline":
        benchmark_key = "baseline_current"
    elif config_type == "high_field":
        benchmark_key = "high_field_current"
    else:
        print(f"❌ Unknown config_type: {config_type}")
        return False
    
    description = f"Operating current calculation ({config_type} configuration)"
    return validator.validate_against_paper(I_calculated, benchmark_key, description)

def validate_rebco_baseline_quick(field: float, current: float, turns: int) -> bool:
    """Quick validation of key baseline parameters"""
    validator = ValidationFramework()
    return validator.validate_baseline_config(
        field=field, 
        ripple=0.01,  # Default paper value
        current=current, 
        turns=turns, 
        radius=0.2    # Default paper value
    )

def validate_rebco_high_field_quick(field: float, current: float, turns: int) -> bool:
    """Quick validation of key high-field parameters"""
    validator = ValidationFramework()
    return validator.validate_high_field_config(
        field=field,
        ripple=0.16,      # Default paper value
        current=current,
        turns=turns,
        radius=0.16,      # Default paper value
        tapes_per_turn=89, # Default paper value
        temperature=15,   # Default paper value
        thermal_margin=74.5 # Default paper value
    )

def check_physics_constraints(temperature: float, magnetic_field: float, 
                            current_density: float) -> bool:
    """Check if parameters satisfy basic physics constraints"""
    validator = ValidationFramework()
    
    checks = [
        validator.check_physical_reasonableness(temperature, 'temperature'),
        validator.check_physical_reasonableness(magnetic_field, 'magnetic_field'),
        validator.check_physical_reasonableness(current_density, 'current_density')
    ]
    
    return all(checks)

def create_rebco_validation_example():
    """Create example validation data matching rebco paper results"""
    
    # Example baseline configuration
    baseline_config = {
        'field': 2.1,      # T
        'ripple': 0.01,    # %
        'current': 1171,   # A
        'turns': 400,      # count
        'radius': 0.2      # m
    }
    
    # Example high-field configuration  
    high_field_config = {
        'field': 7.07,           # T
        'ripple': 0.16,          # %
        'current': 1800,         # A
        'turns': 1000,           # count
        'radius': 0.16,          # m
        'tapes_per_turn': 89,    # count
        'temperature': 15,       # K
        'thermal_margin': 74.5   # K
    }
    
    # Example thermal results
    thermal_results = {
        'thermal_margin': 74.5,  # K
        'cryocooler_power': 150  # W
    }
    
    # Example stress results
    stress_results = {
        'baseline_stress': 175,   # MPa
        'reinforced_stress': 35   # MPa
    }
    
    return {
        'baseline_config': baseline_config,
        'high_field_config': high_field_config,
        'thermal_results': thermal_results,
        'stress_results': stress_results
    }

# Example usage demonstration
if __name__ == "__main__":
    print("🧪 HTS Coil Validation Framework - REBCO Paper Edition")
    print("=" * 60)
    
    # Initialize validator
    validator = ValidationFramework()
    
    # Example rebco paper validations
    print("\n📋 Testing Individual Parameter Validations:")
    validator.validate_against_paper(2.1, 'baseline_field', 'REBCO baseline field')
    validator.validate_against_paper(7.07, 'high_field', 'REBCO high-field')
    validator.validate_against_paper(1171, 'baseline_current', 'REBCO baseline current')
    validator.validate_against_paper(1800, 'high_field_current', 'REBCO high-field current')
    
    print("\n📋 Testing Configuration Validations:")
    # Test baseline configuration
    validator.validate_baseline_config(
        field=2.1, ripple=0.01, current=1171, turns=400, radius=0.2
    )
    
    print("\n")
    
    # Test high-field configuration  
    validator.validate_high_field_config(
        field=7.07, ripple=0.16, current=1800, turns=1000, radius=0.16,
        tapes_per_turn=89, temperature=15, thermal_margin=74.5
    )
    
    print("\n📋 Testing Comprehensive Validation:")
    # Get example data
    example_data = create_rebco_validation_example()
    
    # Run comprehensive validation
    validator.comprehensive_rebco_validation(
        example_data['baseline_config'],
        example_data['high_field_config'], 
        example_data['thermal_results'],
        example_data['stress_results']
    )
    
    # Physical reasonableness checks
    print("\n📋 Physics Constraint Checks:")
    validator.check_physical_reasonableness(15, 'temperature', 'REBCO operating temperature')
    validator.check_physical_reasonableness(7.07, 'magnetic_field', 'High-field design')
    
    # Generate report
    print("\n📄 Validation Report:")
    print(validator.generate_validation_report())