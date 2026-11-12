"""Validation Agent for CircuitMind AI.

Performs deterministic electrical safety checks and compatibility validation.

MANDATORY SAFETY DIRECTIVE (PROJECT_RULES.md):
Never allow an LLM to independently approve electrical safety or compliance.
All electrical parameters (voltage operating limits, absolute maximums, power dissipation,
and current headroom) MUST be validated using deterministic Python calculation tools.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.graph.state import CircuitMindState, ValidationResults
from app.tools.electrical_calculator import ElectricalCalculator

logger = logging.getLogger(__name__)


class ValidationAgent:
    """Agent that performs deterministic electrical validation on circuit parameters."""

    def __init__(self) -> None:
        """Initialize the deterministic ValidationAgent."""
        self.calc = ElectricalCalculator()

    def validate(
        self,
        requirements: Dict[str, Any],
        component_info: Dict[str, Any],
    ) -> ValidationResults:
        """Validate engineering requirements against verified component datasheet limits.

        Args:
            requirements: Extracted design parameters from user query.
            component_info: Verified component specifications from datasheet evidence.

        Returns:
            ValidationResults with is_valid flag, detailed checks, violations, and warnings.
        """
        checks: List[Dict[str, Any]] = []
        violations: List[str] = []
        warnings: List[str] = []
        calculations: Dict[str, Any] = {}
        all_checks_passed = True

        target_voltage = requirements.get("target_supply_voltage_v")
        verified_specs = component_info.get("verified_specs", {})

        # Check 1: Supply Voltage Compatibility Check
        supply_spec = verified_specs.get("supply_voltage_recommended")
        abs_max_spec = verified_specs.get("supply_voltage_abs_max")

        if target_voltage is not None:
            if supply_spec:
                v_min = supply_spec.get("min_v", 4.0)
                v_max = supply_spec.get("max_v", 30.0)
                abs_v_max = abs_max_spec.get("max_v") if abs_max_spec else None

                try:
                    v_check = self.calc.check_voltage_compatibility(
                        target_voltage=target_voltage,
                        min_voltage=v_min,
                        max_voltage=v_max,
                        abs_max_voltage=abs_v_max,
                    )
                    v_details = dict(v_check)
                    v_details["min_recommended_v"] = v_min
                    v_details["max_recommended_v"] = v_max
                    checks.append({
                        "check_name": "Operating Supply Voltage Compatibility",
                        "status": "PASS" if v_check["is_compatible"] else "FAIL",
                        "details": v_details,
                    })

                    if not v_check["is_compatible"]:
                        all_checks_passed = False
                        violations.extend(v_check["violations"])

                    warnings.extend(v_check["warnings"])
                except Exception as exc:
                    logger.error("Voltage calculation error: %s", exc)
                    all_checks_passed = False
                    violations.append(f"Calculation error during voltage check: {exc}")
            else:
                warnings.append(
                    f"Target supply voltage ({target_voltage}V) specified, but datasheet recommended "
                    "operating voltage could not be verified from retrieved passages."
                )
        else:
            warnings.append("Design requirement did not specify a target supply voltage for validation.")

        # Check 2: Quiescent Power Dissipation Calculation
        quiescent_spec = verified_specs.get("quiescent_current")
        if target_voltage is not None and quiescent_spec:
            i_ma = quiescent_spec.get("value_ma", 0.131)
            i_a = i_ma / 1000.0
            try:
                p_calc = self.calc.power_dissipation(voltage=target_voltage, current=i_a)
                calculations["quiescent_power"] = p_calc
                checks.append({
                    "check_name": "Quiescent Power Dissipation",
                    "status": "PASS",
                    "details": {
                        "power_mw": p_calc["power_mw"],
                        "formula": p_calc["formula"],
                        "current_ma": i_ma,
                        "voltage_v": target_voltage,
                    },
                })
            except Exception as exc:
                logger.error("Power dissipation calculation error: %s", exc)
                warnings.append(f"Could not calculate quiescent power: {exc}")

        # Check 3: Current Requirement Compatibility
        target_current_ma = requirements.get("target_max_current_ma")
        if target_current_ma is not None and quiescent_spec:
            iq_ma = quiescent_spec.get("value_ma", 0.06)
            if target_current_ma < iq_ma:
                all_checks_passed = False
                violations.append(
                    f"CURRENT INSUFFICIENT: Target current budget ({target_current_ma}mA) is less than "
                    f"device quiescent current ({iq_ma}mA). Component will exceed power budget."
                )
                checks.append({
                    "check_name": "Current Budget Feasibility",
                    "status": "FAIL",
                    "details": {"target_current_ma": target_current_ma, "quiescent_current_ma": iq_ma},
                })
            else:
                headroom_ma = target_current_ma - iq_ma
                checks.append({
                    "check_name": "Current Budget Feasibility",
                    "status": "PASS",
                    "details": {
                        "target_current_ma": target_current_ma,
                        "quiescent_current_ma": iq_ma,
                        "headroom_ma": round(headroom_ma, 4),
                    },
                })

        # Check 4: Deterministic Sensor Transfer Function Calculations (e.g. 25°C and 100°C output)
        scale_spec = verified_specs.get("scale_factor")
        scale_val = scale_spec.get("value", 10.0) if scale_spec else 10.0

        # Always deterministically calculate 25°C output (datasheet room temperature standard)
        calc_25c = self.calc.sensor_transfer_output(temperature_c=25.0, scale_factor_mv_per_c=scale_val)
        calculations["output_at_25c"] = calc_25c

        # Deterministically calculate 100°C output (standard full-scale benchmark)
        calc_100c = self.calc.sensor_transfer_output(temperature_c=100.0, scale_factor_mv_per_c=scale_val)
        calculations["output_at_100c"] = calc_100c

        # Optional user min temperature transfer output
        t_min = requirements.get("min_temp_c")
        if t_min is not None and t_min not in (25.0, 100.0):
            calculations[f"output_at_{int(t_min)}c"] = self.calc.sensor_transfer_output(
                temperature_c=t_min, scale_factor_mv_per_c=scale_val
            )

        # Optional user max temperature transfer output
        t_max = requirements.get("max_temp_c")
        if t_max is not None and t_max not in (25.0, 100.0):
            calculations[f"output_at_{int(t_max)}c"] = self.calc.sensor_transfer_output(
                temperature_c=t_max, scale_factor_mv_per_c=scale_val
            )

        # Determine full-scale sensor output voltage for interface validation
        v_out_full_scale_v = calc_100c["output_voltage_v"]
        if t_max is not None and t_max > 100.0:
            calc_tmax = self.calc.sensor_transfer_output(temperature_c=t_max, scale_factor_mv_per_c=scale_val)
            v_out_full_scale_v = calc_tmax["output_voltage_v"]

        # Check 5 & 6: ADC Interface & Output Load Feasibility
        # Evaluated when primary circuit checks are active or interface parameters are provided
        has_supply_check = any(c.get("check_name") == "Operating Supply Voltage Compatibility" for c in checks)
        adc_ref_v = requirements.get("adc_reference_voltage_v")
        load_r_ohms = requirements.get("load_impedance_ohms")

        if has_supply_check or adc_ref_v is not None or load_r_ohms is not None:
            # Check 5: ADC Interface & Signal Headroom Check
            if adc_ref_v is not None:
                if v_out_full_scale_v > adc_ref_v:
                    all_checks_passed = False
                    violations.append(
                        f"ADC SATURATION VIOLATION: Maximum sensor output voltage ({v_out_full_scale_v}V) "
                        f"exceeds ADC reference/full-scale range ({adc_ref_v}V). Signal will clip at high temperatures."
                    )
                    checks.append({
                        "check_name": "ADC Interface & Signal Headroom",
                        "status": "FAIL",
                        "details": {
                            "max_sensor_output_v": v_out_full_scale_v,
                            "adc_reference_v": adc_ref_v,
                        },
                    })
                else:
                    checks.append({
                        "check_name": "ADC Interface & Signal Headroom",
                        "status": "PASS",
                        "details": {
                            "max_sensor_output_v": v_out_full_scale_v,
                            "adc_reference_v": adc_ref_v,
                            "signal_margin_v": round(adc_ref_v - v_out_full_scale_v, 4),
                        },
                    })
            else:
                checks.append({
                    "check_name": "ADC Interface & Signal Headroom",
                    "status": "UNVERIFIED",
                    "details": {
                        "reason": "ADC reference voltage / input range not specified by design requirements."
                    },
                })

            # Check 6: Output Load Impedance & Drive Current Check
            if load_r_ohms is not None:
                if load_r_ohms <= 0:
                    all_checks_passed = False
                    violations.append(f"INVALID LOAD: Load resistance ({load_r_ohms}Ω) must be positive.")
                    checks.append({
                        "check_name": "Output Load Impedance Feasibility",
                        "status": "FAIL",
                        "details": {"load_impedance_ohms": load_r_ohms},
                    })
                else:
                    # Max load current I = V_out_max / R_load
                    i_load_ma = (v_out_full_scale_v / load_r_ohms) * 1000.0
                    # LM35 max output sourcing current rating is 10 mA
                    max_rated_load_ma = 10.0
                    if i_load_ma > max_rated_load_ma:
                        all_checks_passed = False
                        violations.append(
                            f"OUTPUT OVERLOAD: Required load current ({round(i_load_ma, 2)}mA) exceeds "
                            f"LM35 maximum output drive rating ({max_rated_load_ma}mA)."
                        )
                        checks.append({
                            "check_name": "Output Load Impedance Feasibility",
                            "status": "FAIL",
                            "details": {
                                "load_impedance_ohms": load_r_ohms,
                                "load_current_ma": round(i_load_ma, 4),
                                "max_rated_load_ma": max_rated_load_ma,
                            },
                        })
                    else:
                        checks.append({
                            "check_name": "Output Load Impedance Feasibility",
                            "status": "PASS",
                            "details": {
                                "load_impedance_ohms": load_r_ohms,
                                "load_current_ma": round(i_load_ma, 4),
                                "max_rated_load_ma": max_rated_load_ma,
                            },
                        })
            else:
                checks.append({
                    "check_name": "Output Load Impedance Feasibility",
                    "status": "UNVERIFIED",
                    "details": {
                        "reason": "Load impedance / resistance not specified by design requirements."
                    },
                })

        # Determine explicit validation status: VALID, INVALID, or INCONCLUSIVE
        # Check if any executed check failed or violations exist
        has_failed_check = any(c.get("status") == "FAIL" for c in checks)
        has_unverified_check = any(c.get("status") == "UNVERIFIED" for c in checks)
        has_passed_checks = any(c.get("status") == "PASS" for c in checks)

        if violations or has_failed_check or not all_checks_passed:
            val_status = "INVALID"
            is_valid = False
        elif not has_passed_checks or not checks:
            val_status = "INCONCLUSIVE"
            is_valid = False
            warnings.append(
                "INCONCLUSIVE: No deterministic safety checks could be executed due to missing parameters or unverified evidence."
            )
        elif has_unverified_check or target_voltage is None or not supply_spec:
            # A supply compatibility PASS coexists with an overall INCONCLUSIVE status
            val_status = "INCONCLUSIVE"
            is_valid = False
            warnings.append(
                "INCONCLUSIVE: Supply voltage compatibility check passed, but complete circuit validation "
                "cannot be confirmed because required interface constraints (ADC reference voltage or "
                "load impedance) remain unverified or unspecified."
            )
        else:
            val_status = "VALID"
            is_valid = True

        return ValidationResults(
            status=val_status,
            is_valid=is_valid,
            checks=checks,
            violations=violations,
            warnings=warnings,
            calculations=calculations,
        )


def validation_agent_node(state: CircuitMindState) -> Dict[str, Any]:
    """LangGraph node executing deterministic electrical validation."""
    requirements = state.get("requirements", {})
    component_info = state.get("component_information", {})

    agent = ValidationAgent()
    try:
        results = agent.validate(requirements=requirements, component_info=component_info)
        return {"validation_results": results.model_dump()}
    except Exception as exc:
        logger.error("Error in validation_agent_node: %s", exc)
        return {
            "validation_results": ValidationResults(
                status="INVALID",
                is_valid=False,
                violations=[f"Deterministic validation crashed: {exc}"],
            ).model_dump(),
            "errors": [f"ValidationAgent error: {exc}"],
        }

