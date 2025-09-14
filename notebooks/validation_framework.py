# Data Validation and Reproducibility Framework
# This module provides validation functions for HTS coil design notebooks

import numpy as np
import warnings
from typing import Dict, List, Tuple, Any

class ValidationFramework:
    """Framework for validating HTS coil calculations against paper benchmarks"""
    
    def __init__(self):
        # Paper benchmark values
        self.paper_benchmarks = {
            'baseline_field': {'value': 2.1, 'tolerance': 0.01, 'unit': 'T'},
            'high_field': {'value': 7.07, 'tolerance': 0.15, 'unit': 'T'},
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
        converging = all(rel_changes[i] <= rel_changes[i-1] * 2 for i in range(1, len(rel_changes)))
        
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

# Convenience functions for common validation tasks
def validate_magnetic_field_calculation(B_calculated: float, config_type: str = "baseline") -> bool:
    """Validate magnetic field calculation against paper results"""
    validator = ValidationFramework()
    
    benchmark_key = f"{config_type}_field"
    description = f"Magnetic field calculation ({config_type} configuration)"
    
    return validator.validate_against_paper(B_calculated, benchmark_key, description)

def validate_current_calculation(I_calculated: float, config_type: str = "baseline") -> bool:
    """Validate current calculation against paper results"""
    validator = ValidationFramework()
    
    benchmark_key = f"current_{config_type}"
    description = f"Operating current calculation ({config_type} configuration)"
    
    return validator.validate_against_paper(I_calculated, benchmark_key, description)

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

# Example usage demonstration
if __name__ == "__main__":
    print("🧪 HTS Coil Validation Framework")
    print("=" * 40)
    
    # Initialize validator
    validator = ValidationFramework()
    
    # Example validations
    validator.validate_against_paper(2.08, 'baseline_field', 'Example baseline field calculation')
    validator.validate_against_paper(7.15, 'high_field', 'Example high-field calculation')
    
    # Physical reasonableness checks
    validator.check_physical_reasonableness(77, 'temperature', 'Liquid nitrogen cooling')
    validator.check_physical_reasonableness(15, 'magnetic_field', 'High-field magnet design')
    
    # Generate report
    print("\n" + validator.generate_validation_report())