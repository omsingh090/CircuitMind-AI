"""Deterministic Electrical Calculation & Safety Validation Tool for CircuitMind AI.

Implements pure Python mathematical validation for fundamental electronics calculations:
- Ohm's Law (V = I * R)
- Power Dissipation (P = V * I, P = I^2 * R, P = V^2 / R)
- Resistive Voltage Dividers
- Voltage Operating Range and Absolute Maximum Compatibility Checks
- Thermal & Power Derating Compliance Checks

CRITICAL SAFETY DIRECTIVE:
Never approximate or allow an LLM to guess electrical safety ratings.
All calculations are strictly deterministic with rigorous input validation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


class ElectricalCalculator:
    """Deterministic mathematical tool for circuit calculations and electrical rating checks."""

    @staticmethod
    def ohms_law(
        voltage: Optional[float] = None,
        current: Optional[float] = None,
        resistance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute the unknown electrical quantity from two known parameters using Ohm's Law.

        Args:
            voltage: Potential difference in Volts (V).
            current: Electric current in Amperes (A).
            resistance: Resistance in Ohms (Ω).

        Returns:
            Dictionary containing computed and input parameters, units, formula, and status.

        Raises:
            ValueError: If not exactly two parameters are provided, or if non-finite/negative resistance is given.
        """
        provided = [p for p in (voltage, current, resistance) if p is not None]
        if len(provided) != 2:
            raise ValueError(
                f"Ohm's Law requires exactly 2 known parameters, received {len(provided)}."
            )

        # Validate finite numbers
        for val, name in [(voltage, "voltage"), (current, "current"), (resistance, "resistance")]:
            if val is not None and (math.isnan(val) or math.isinf(val)):
                raise ValueError(f"Parameter '{name}' must be a finite numerical value.")

        if resistance is not None and resistance <= 0:
            raise ValueError("Resistance must be strictly positive (> 0 Ohms).")

        if voltage is None:
            assert current is not None and resistance is not None
            v = current * resistance
            return {
                "voltage_v": round(v, 6),
                "current_a": current,
                "resistance_ohm": resistance,
                "calculated": "voltage",
                "formula": "V = I * R",
                "status": "success",
            }
        elif current is None:
            assert voltage is not None and resistance is not None
            i = voltage / resistance
            return {
                "voltage_v": voltage,
                "current_a": round(i, 6),
                "resistance_ohm": resistance,
                "calculated": "current",
                "formula": "I = V / R",
                "status": "success",
            }
        else:  # resistance is None
            assert voltage is not None and current is not None
            if current == 0:
                raise ValueError("Current cannot be zero when calculating resistance (division by zero).")
            r = abs(voltage / current)
            return {
                "voltage_v": voltage,
                "current_a": current,
                "resistance_ohm": round(r, 6),
                "calculated": "resistance",
                "formula": "R = |V / I|",
                "status": "success",
            }

    @staticmethod
    def power_dissipation(
        voltage: Optional[float] = None,
        current: Optional[float] = None,
        resistance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute electrical power dissipation in Watts from any two valid parameters.

        Args:
            voltage: Voltage across component in Volts (V).
            current: Current through component in Amperes (A).
            resistance: Component resistance in Ohms (Ω).

        Returns:
            Dictionary containing power in Watts and milliwatts, formula, and input values.
        """
        provided = [p for p in (voltage, current, resistance) if p is not None]
        if len(provided) < 2:
            raise ValueError("Power calculation requires at least two known parameters (V, I, or R).")

        for val, name in [(voltage, "voltage"), (current, "current"), (resistance, "resistance")]:
            if val is not None and (math.isnan(val) or math.isinf(val)):
                raise ValueError(f"Parameter '{name}' must be finite.")

        if resistance is not None and resistance <= 0:
            raise ValueError("Resistance must be strictly positive.")

        if voltage is not None and current is not None:
            p = abs(voltage * current)
            formula = "P = |V * I|"
        elif current is not None and resistance is not None:
            p = (current ** 2) * resistance
            formula = "P = I^2 * R"
        elif voltage is not None and resistance is not None:
            p = (voltage ** 2) / resistance
            formula = "P = V^2 / R"
        else:
            raise ValueError("Invalid combination of parameters for power dissipation.")

        return {
            "power_w": round(p, 6),
            "power_mw": round(p * 1000.0, 3),
            "formula": formula,
            "status": "success",
        }

    @staticmethod
    def voltage_divider(
        v_in: float,
        r1: float,
        r2: float,
    ) -> Dict[str, Any]:
        """Calculate unloaded resistive voltage divider output voltage and current.

        V_out = V_in * (R2 / (R1 + R2))

        Args:
            v_in: Input supply voltage in Volts.
            r1: High-side resistor in Ohms.
            r2: Low-side resistor in Ohms.

        Returns:
            Dictionary with V_out, divider current, and individual resistor power.
        """
        for val, name in [(v_in, "v_in"), (r1, "r1"), (r2, "r2")]:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Parameter '{name}' must be finite.")

        if r1 <= 0 or r2 <= 0:
            raise ValueError("Resistor values R1 and R2 must both be strictly positive (> 0 Ohms).")

        r_total = r1 + r2
        v_out = v_in * (r2 / r_total)
        divider_current = v_in / r_total
        p_r1 = (divider_current ** 2) * r1
        p_r2 = (divider_current ** 2) * r2

        return {
            "v_in": v_in,
            "r1_ohm": r1,
            "r2_ohm": r2,
            "v_out": round(v_out, 6),
            "divider_ratio": round(r2 / r_total, 6),
            "quiescent_current_ma": round(divider_current * 1000.0, 4),
            "p_r1_mw": round(p_r1 * 1000.0, 3),
            "p_r2_mw": round(p_r2 * 1000.0, 3),
            "status": "success",
        }

    @staticmethod
    def check_voltage_compatibility(
        target_voltage: float,
        min_voltage: float,
        max_voltage: float,
        abs_max_voltage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Deterministically validate if an operating voltage is safe and within component limits.

        Args:
            target_voltage: Operating voltage requested by the design (V).
            min_voltage: Minimum recommended operating voltage from datasheet (V).
            max_voltage: Maximum recommended operating voltage from datasheet (V).
            abs_max_voltage: Absolute maximum voltage limit before device damage (V).

        Returns:
            Validation assessment with status, violations, warnings, and safety margin.
        """
        for val, name in [
            (target_voltage, "target_voltage"),
            (min_voltage, "min_voltage"),
            (max_voltage, "max_voltage"),
        ]:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Parameter '{name}' must be finite.")

        if min_voltage > max_voltage:
            raise ValueError(f"min_voltage ({min_voltage}V) cannot exceed max_voltage ({max_voltage}V).")

        violations = []
        warnings = []
        is_safe = True

        # Check Absolute Maximum Rating first (Catastrophic Failure Risk)
        if abs_max_voltage is not None:
            if target_voltage >= abs_max_voltage:
                is_safe = False
                violations.append(
                    f"CRITICAL VIOLATION: Target voltage ({target_voltage}V) reaches or exceeds "
                    f"Absolute Maximum Rating ({abs_max_voltage}V). High risk of permanent hardware damage."
                )
            elif target_voltage > max_voltage:
                is_safe = False
                violations.append(
                    f"VOLTAGE EXCEEDED: Target voltage ({target_voltage}V) exceeds maximum recommended "
                    f"operating limit ({max_voltage}V)."
                )
        elif target_voltage > max_voltage:
            is_safe = False
            violations.append(
                f"VOLTAGE EXCEEDED: Target voltage ({target_voltage}V) exceeds maximum recommended "
                f"operating limit ({max_voltage}V)."
            )

        # Check Under-voltage
        if target_voltage < min_voltage:
            is_safe = False
            violations.append(
                f"UNDER-VOLTAGE: Target voltage ({target_voltage}V) is below minimum required "
                f"operating voltage ({min_voltage}V). Device will fail to operate."
            )

        # Safety headroom warning (operating within top 5% of max rating)
        headroom = max_voltage - target_voltage
        if is_safe and headroom < (0.05 * max_voltage):
            warnings.append(
                f"Low voltage headroom: Operating voltage ({target_voltage}V) is within 5% of "
                f"maximum limit ({max_voltage}V). Consider supply transient protection."
            )

        return {
            "target_voltage_v": target_voltage,
            "min_voltage_v": min_voltage,
            "max_voltage_v": max_voltage,
            "abs_max_voltage_v": abs_max_voltage,
            "is_compatible": is_safe,
            "headroom_v": round(headroom, 4) if is_safe else 0.0,
            "violations": violations,
            "warnings": warnings,
        }

    @staticmethod
    def check_power_rating(
        actual_power_w: float,
        rated_power_w: float,
        derating_factor: float = 0.8,
    ) -> Dict[str, Any]:
        """Validate component power dissipation against maximum rating and safety derating.

        Args:
            actual_power_w: Calculated operational power dissipation in Watts.
            rated_power_w: Maximum package power rating from datasheet in Watts.
            derating_factor: Conservative engineering safety factor (default: 0.8 = 80%).

        Returns:
            Validation report indicating whether thermal dissipation is acceptable.
        """
        if actual_power_w < 0 or rated_power_w <= 0:
            raise ValueError("Power values must be non-negative, and rated power must be positive.")

        safe_limit_w = rated_power_w * derating_factor
        is_safe = actual_power_w <= safe_limit_w
        violations = []
        warnings = []

        if actual_power_w > rated_power_w:
            violations.append(
                f"POWER OVERSTRESS: Dissipation ({actual_power_w * 1000.0:.1f}mW) exceeds "
                f"absolute maximum package rating ({rated_power_w * 1000.0:.1f}mW)."
            )
        elif actual_power_w > safe_limit_w:
            warnings.append(
                f"DERATING WARNING: Dissipation ({actual_power_w * 1000.0:.1f}mW) exceeds "
                f"recommended {int(derating_factor*100)}% derating threshold ({safe_limit_w * 1000.0:.1f}mW)."
            )

        return {
            "actual_power_w": round(actual_power_w, 6),
            "rated_power_w": round(rated_power_w, 6),
            "derated_limit_w": round(safe_limit_w, 6),
            "is_within_derated_limit": is_safe,
            "violations": violations,
            "warnings": warnings,
        }

    @staticmethod
    def sensor_transfer_output(
        temperature_c: float,
        scale_factor_mv_per_c: float = 10.0,
        offset_v: float = 0.0,
    ) -> Dict[str, Any]:
        """Deterministically calculate linear sensor output voltage from temperature.

        Formula:
            V_out(mV) = (temperature_c * scale_factor_mv_per_c) + (offset_v * 1000)
            V_out(V)  = V_out(mV) / 1000.0

        Args:
            temperature_c: Temperature in degrees Celsius (°C).
            scale_factor_mv_per_c: Sensor sensitivity in mV/°C (default: 10.0 for LM35).
            offset_v: Output voltage offset at 0°C in Volts (default: 0.0V for LM35).

        Returns:
            Dictionary with computed output in millivolts and Volts, formula, and input values.
        """
        for val, name in [
            (temperature_c, "temperature_c"),
            (scale_factor_mv_per_c, "scale_factor_mv_per_c"),
            (offset_v, "offset_v"),
        ]:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Parameter '{name}' must be finite.")

        if scale_factor_mv_per_c <= 0:
            raise ValueError("Scale factor must be strictly positive (> 0 mV/°C).")

        v_out_mv = (temperature_c * scale_factor_mv_per_c) + (offset_v * 1000.0)
        v_out_v = v_out_mv / 1000.0

        return {
            "temperature_c": temperature_c,
            "scale_factor_mv_per_c": scale_factor_mv_per_c,
            "offset_v": offset_v,
            "output_voltage_mv": round(v_out_mv, 4),
            "output_voltage_v": round(v_out_v, 6),
            "formula": "V_out = (T * scale_factor) + offset",
            "status": "success",
        }


# Module-level convenience alias
sensor_transfer_output = ElectricalCalculator.sensor_transfer_output


