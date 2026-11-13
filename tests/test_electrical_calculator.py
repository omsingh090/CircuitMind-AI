"""Unit tests for CircuitMind AI Deterministic Electrical Calculator & Safety Validation Tool.

Tests:
1. Ohm's Law (V, I, R calculations).
2. Ohm's Law input validations (zero current, negative resistance, invalid param count, NaN/inf).
3. Power dissipation calculations (P=VI, P=I^2*R, P=V^2/R).
4. Resistive voltage divider calculations and boundary validations.
5. Operating voltage compatibility and headroom validation.
6. Under-voltage, over-voltage, and absolute maximum ratings violations.
7. Thermal power rating and derating calculations.
"""

import math
import pytest

from app.tools.electrical_calculator import ElectricalCalculator


class TestElectricalCalculator:
    """Test suite for deterministic electrical equations and safety checks."""

    def test_ohms_law_voltage_calculation(self) -> None:
        """Verify V = I * R calculation."""
        res = ElectricalCalculator.ohms_law(current=0.02, resistance=250.0)
        assert res["voltage_v"] == 5.0
        assert res["calculated"] == "voltage"
        assert res["formula"] == "V = I * R"

    def test_ohms_law_current_calculation(self) -> None:
        """Verify I = V / R calculation."""
        res = ElectricalCalculator.ohms_law(voltage=12.0, resistance=1000.0)
        assert res["current_a"] == 0.012
        assert res["calculated"] == "current"
        assert res["formula"] == "I = V / R"

    def test_ohms_law_resistance_calculation(self) -> None:
        """Verify R = V / I calculation."""
        res = ElectricalCalculator.ohms_law(voltage=3.3, current=0.01)
        assert res["resistance_ohm"] == 330.0
        assert res["calculated"] == "resistance"

    def test_ohms_law_invalid_param_count(self) -> None:
        """Verify ValueError when fewer or more than 2 parameters are passed."""
        with pytest.raises(ValueError, match="requires exactly 2 known parameters"):
            ElectricalCalculator.ohms_law(voltage=5.0)

        with pytest.raises(ValueError, match="requires exactly 2 known parameters"):
            ElectricalCalculator.ohms_law(voltage=5.0, current=0.01, resistance=500.0)

    def test_ohms_law_negative_resistance_validation(self) -> None:
        """Verify strictly positive resistance validation."""
        with pytest.raises(ValueError, match="Resistance must be strictly positive"):
            ElectricalCalculator.ohms_law(voltage=5.0, resistance=-100.0)

        with pytest.raises(ValueError, match="Resistance must be strictly positive"):
            ElectricalCalculator.ohms_law(voltage=5.0, resistance=0.0)

    def test_ohms_law_division_by_zero(self) -> None:
        """Verify division by zero prevention when current is 0."""
        with pytest.raises(ValueError, match="Current cannot be zero"):
            ElectricalCalculator.ohms_law(voltage=5.0, current=0.0)

    def test_ohms_law_nan_and_inf(self) -> None:
        """Verify non-finite parameters are rejected."""
        with pytest.raises(ValueError, match="finite numerical value"):
            ElectricalCalculator.ohms_law(voltage=float("nan"), current=0.01)

        with pytest.raises(ValueError, match="finite numerical value"):
            ElectricalCalculator.ohms_law(voltage=5.0, current=float("inf"))

    def test_power_dissipation_calculations(self) -> None:
        """Verify power dissipation formulas P = VI, P = I^2*R, and P = V^2/R."""
        # P = V * I
        p1 = ElectricalCalculator.power_dissipation(voltage=5.0, current=0.02)
        assert p1["power_w"] == 0.1
        assert p1["power_mw"] == 100.0
        assert p1["formula"] == "P = |V * I|"

        # P = I^2 * R
        p2 = ElectricalCalculator.power_dissipation(current=0.01, resistance=1000.0)
        assert p2["power_w"] == 0.1
        assert p2["power_mw"] == 100.0
        assert p2["formula"] == "P = I^2 * R"

        # P = V^2 / R
        p3 = ElectricalCalculator.power_dissipation(voltage=10.0, resistance=100.0)
        assert p3["power_w"] == 1.0
        assert p3["power_mw"] == 1000.0
        assert p3["formula"] == "P = V^2 / R"

    def test_voltage_divider_calculation(self) -> None:
        """Verify unloaded voltage divider V_out = V_in * (R2 / (R1 + R2))."""
        # 10V input into 1k and 1k divider -> 5V output
        div = ElectricalCalculator.voltage_divider(v_in=10.0, r1=1000.0, r2=1000.0)
        assert div["v_out"] == 5.0
        assert div["divider_ratio"] == 0.5
        assert div["quiescent_current_ma"] == 5.0  # 10V / 2k = 5mA

    def test_voltage_divider_invalid_resistors(self) -> None:
        """Verify negative and zero resistor validation."""
        with pytest.raises(ValueError, match="strictly positive"):
            ElectricalCalculator.voltage_divider(v_in=5.0, r1=-100.0, r2=100.0)

        with pytest.raises(ValueError, match="strictly positive"):
            ElectricalCalculator.voltage_divider(v_in=5.0, r1=0.0, r2=100.0)

    def test_voltage_compatibility_normal_operation(self) -> None:
        """Verify normal operating voltage passes validation."""
        # LM35 4V to 30V recommended, operating at 5V
        check = ElectricalCalculator.check_voltage_compatibility(
            target_voltage=5.0,
            min_voltage=4.0,
            max_voltage=30.0,
            abs_max_voltage=35.0,
        )
        assert check["is_compatible"] is True
        assert len(check["violations"]) == 0
        assert check["headroom_v"] == 25.0

    def test_voltage_compatibility_undervoltage(self) -> None:
        """Verify undervoltage condition is flagged as violation."""
        check = ElectricalCalculator.check_voltage_compatibility(
            target_voltage=3.3,
            min_voltage=4.0,
            max_voltage=30.0,
        )
        assert check["is_compatible"] is False
        assert any("UNDER-VOLTAGE" in v for v in check["violations"])

    def test_voltage_compatibility_overvoltage(self) -> None:
        """Verify exceeding recommended max voltage is flagged as violation."""
        check = ElectricalCalculator.check_voltage_compatibility(
            target_voltage=32.0,
            min_voltage=4.0,
            max_voltage=30.0,
            abs_max_voltage=35.0,
        )
        assert check["is_compatible"] is False
        assert any("VOLTAGE EXCEEDED" in v for v in check["violations"])

    def test_voltage_compatibility_absolute_maximum_violation(self) -> None:
        """Verify exceeding absolute maximum rating produces critical violation."""
        check = ElectricalCalculator.check_voltage_compatibility(
            target_voltage=36.0,
            min_voltage=4.0,
            max_voltage=30.0,
            abs_max_voltage=35.0,
        )
        assert check["is_compatible"] is False
        assert any("CRITICAL VIOLATION" in v for v in check["violations"])

    def test_voltage_compatibility_low_headroom_warning(self) -> None:
        """Verify operating near maximum rating produces an engineering headroom warning."""
        # 29.5V is within 5% of 30V max (30 - 29.5 = 0.5V < 1.5V)
        check = ElectricalCalculator.check_voltage_compatibility(
            target_voltage=29.5,
            min_voltage=4.0,
            max_voltage=30.0,
        )
        assert check["is_compatible"] is True
        assert len(check["warnings"]) > 0
        assert any("Low voltage headroom" in w for w in check["warnings"])

    def test_power_derating_checks(self) -> None:
        """Verify component power dissipation derating checks."""
        # Safe within 80% derated limit
        safe_check = ElectricalCalculator.check_power_rating(
            actual_power_w=0.5,
            rated_power_w=1.0,
            derating_factor=0.8,
        )
        assert safe_check["is_within_derated_limit"] is True
        assert len(safe_check["violations"]) == 0
        assert len(safe_check["warnings"]) == 0

        # Between 80% and 100%: Warning
        warn_check = ElectricalCalculator.check_power_rating(
            actual_power_w=0.85,
            rated_power_w=1.0,
            derating_factor=0.8,
        )
        assert warn_check["is_within_derated_limit"] is False
        assert len(warn_check["warnings"]) > 0

        # Above 100%: Violation
        viol_check = ElectricalCalculator.check_power_rating(
            actual_power_w=1.2,
            rated_power_w=1.0,
            derating_factor=0.8,
        )
        assert len(viol_check["violations"]) > 0

    def test_sensor_transfer_output_nominal_and_tolerance(self) -> None:
        """Verify nominal calculations at 0°C, 25°C, 100°C and guaranteed tolerance bounds."""
        # 0°C
        t0 = ElectricalCalculator.sensor_transfer_output(temperature_c=0.0)
        assert t0["output_voltage_mv"] == 0.0
        assert t0["output_voltage_v"] == 0.0
        assert t0["formula"] == "V_out = (T * scale_factor) + offset"

        # 25°C nominal
        t25 = ElectricalCalculator.sensor_transfer_output(temperature_c=25.0)
        assert t25["output_voltage_mv"] == 250.0
        assert t25["output_voltage_v"] == 0.25

        # 100°C nominal
        t100 = ElectricalCalculator.sensor_transfer_output(temperature_c=100.0)
        assert t100["output_voltage_mv"] == 1000.0
        assert t100["output_voltage_v"] == 1.0

        # With accuracy tolerance (e.g. ±0.5°C)
        t25_tol = ElectricalCalculator.sensor_transfer_output(
            temperature_c=25.0,
            scale_factor_mv_per_c=10.0,
            accuracy_tolerance_c=0.5,
        )
        assert t25_tol["output_voltage_mv"] == 250.0
        assert t25_tol["min_output_voltage_mv"] == 245.0
        assert t25_tol["max_output_voltage_mv"] == 255.0
        assert t25_tol["min_output_voltage_v"] == 0.245
        assert t25_tol["max_output_voltage_v"] == 0.255
        assert t25_tol["tolerance_band_mv"] == 10.0

    def test_adc_compatibility_saturation_failure(self) -> None:
        """Verify ADC saturation detection when sensor output exceeds ADC full-scale reference."""
        # LM35 at 100°C produces 1.0V; ADC reference is 0.8V
        eval_sat = ElectricalCalculator.adc_compatibility_check(
            max_sensor_voltage_v=1.0,
            adc_reference_v=0.8,
            scale_factor_mv_per_c=10.0,
        )
        assert eval_sat["is_compatible"] is False
        assert eval_sat["status"] == "FAIL"
        assert eval_sat["saturation_temp_c"] == 80.0
        assert any("ADC SATURATION VIOLATION" in v for v in eval_sat["violations"])
        assert any("clips at 80.0°C" in v for v in eval_sat["violations"])

    def test_adc_compatibility_passed_with_12bit_quantization_and_caveats(self) -> None:
        """Verify ADC compatibility pass with 12-bit quantization metrics and analog caveats."""
        eval_pass = ElectricalCalculator.adc_compatibility_check(
            max_sensor_voltage_v=1.0,
            adc_reference_v=3.3,
            adc_resolution_bits=12,
            scale_factor_mv_per_c=10.0,
        )
        assert eval_pass["is_compatible"] is True
        assert eval_pass["status"] == "PASS"
        assert eval_pass["margin_v"] == 2.3

        # Quantization checks for 12-bit (4096 codes) with 3.3V Vref
        q = eval_pass["quantization"]
        assert q is not None
        assert q["total_codes"] == 4096
        assert abs(q["v_lsb_mv"] - (3300.0 / 4096.0)) < 0.001
        # Temperature resolution per count: 0.8057 mV / 10 mV/°C = 0.0806 °C/count
        assert abs(q["temp_resolution_c_per_count"] - 0.0806) < 0.001

        # Analog interface caveats
        assert len(eval_pass["interface_caveats"]) >= 4
        assert any("Capacitive Loading" in c for c in eval_pass["interface_caveats"])
        assert any("Acquisition Time" in c for c in eval_pass["interface_caveats"])
        assert any("Ground Separation" in c for c in eval_pass["interface_caveats"])

    def test_lm35_variant_temperature_check_limits(self) -> None:
        """Verify LM35 variant temperature limits against official TI specifications."""
        # LM35 / LM35A: -55°C to 150°C
        v_a = ElectricalCalculator.lm35_variant_temperature_check("LM35A", min_temp_c=-50.0, max_temp_c=140.0)
        assert v_a["is_compatible"] is True
        assert v_a["status"] == "PASS"

        # LM35C / LM35CA: -40°C to 110°C
        v_c = ElectricalCalculator.lm35_variant_temperature_check("LM35C", min_temp_c=-40.0, max_temp_c=110.0)
        assert v_c["is_compatible"] is True
        assert v_c["status"] == "PASS"

        # LM35D: 0°C to 100°C -> Target 120°C violates max rating
        v_d_over = ElectricalCalculator.lm35_variant_temperature_check("LM35D", min_temp_c=0.0, max_temp_c=120.0)
        assert v_d_over["is_compatible"] is False
        assert v_d_over["status"] == "FAIL"
        assert any("TEMPERATURE OVERFLOW" in v for v in v_d_over["violations"])

        # LM35D: Target -10°C violates min rating
        v_d_under = ElectricalCalculator.lm35_variant_temperature_check("LM35D", min_temp_c=-10.0, max_temp_c=90.0)
        assert v_d_under["is_compatible"] is False
        assert v_d_under["status"] == "FAIL"
        assert any("TEMPERATURE UNDERFLOW" in v for v in v_d_under["violations"])

        # Generic LM35 produces uncertainty warning
        v_gen = ElectricalCalculator.lm35_variant_temperature_check("LM35", min_temp_c=10.0, max_temp_c=80.0)
        assert v_gen["is_compatible"] is True
        assert v_gen["is_generic_family"] is True
        assert any("GENERIC PART NUMBER UNCERTAINTY" in w for w in v_gen["warnings"])

