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
                    v_details["datasheet_page"] = supply_spec.get("page", 4)
                    v_details["extracted_specification"] = f"Recommended Operating Supply: {v_min}V to {v_max}V"
                    v_details["user_constraint"] = f"Target Supply Voltage: {target_voltage}V"
                    v_details["calculation"] = f"Headroom: {v_check.get('headroom_v', 0.0)}V"
                    v_details["comparison_result"] = "PASS" if v_check["is_compatible"] else "FAIL"

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

        # Check 2: LM35 Variant Operational Temperature Range Check
        variant_name = requirements.get("component_variant") or component_info.get("specific_variant") or "LM35"
        t_min = requirements.get("min_temp_c")
        t_max = requirements.get("max_temp_c")

        if t_min is not None or t_max is not None:
            t_check = self.calc.lm35_variant_temperature_check(
                variant=variant_name,
                min_temp_c=t_min,
                max_temp_c=t_max,
            )
            t_details = dict(t_check)
            t_details["datasheet_page"] = 4
            t_details["extracted_specification"] = (
                f"Rated Operating Range for {t_check['variant']}: {t_check['rated_min_c']}°C to {t_check['rated_max_c']}°C"
            )
            t_details["user_constraint"] = f"Target Operating Span: {t_min}°C to {t_max}°C"
            t_details["calculation"] = (
                f"Operating Range Margin: Low {(t_min - t_check['rated_min_c']) if t_min is not None else 'N/A'}°C, "
                f"High {(t_check['rated_max_c'] - t_max) if t_max is not None else 'N/A'}°C"
            )
            t_details["comparison_result"] = t_check["status"]

            checks.append({
                "check_name": f"Operating Temperature Range ({t_check['variant']})",
                "status": t_check["status"],
                "details": t_details,
            })

            if not t_check["is_compatible"]:
                all_checks_passed = False
                violations.extend(t_check["violations"])

            warnings.extend(t_check["warnings"])

        # Check 3: Quiescent Power Dissipation Calculation
        quiescent_spec = verified_specs.get("quiescent_current")
        if target_voltage is not None and quiescent_spec:
            i_ma = quiescent_spec.get("value_ma", 0.056)
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
                        "datasheet_page": quiescent_spec.get("page", 7),
                        "extracted_specification": f"Typical Quiescent Current: {i_ma * 1000.0:.0f} µA (25°C, 5V)",
                        "user_constraint": f"Supply Voltage: {target_voltage}V",
                        "calculation": f"P = {target_voltage}V * {i_ma}mA = {p_calc['power_mw']}mW",
                        "comparison_result": "PASS",
                    },
                })
            except Exception as exc:
                logger.error("Power dissipation calculation error: %s", exc)
                warnings.append(f"Could not calculate quiescent power: {exc}")

        # Check 4: Current Requirement Compatibility
        target_current_ma = requirements.get("target_max_current_ma")
        if target_current_ma is not None and quiescent_spec:
            iq_ma = quiescent_spec.get("value_ma", 0.056)
            if target_current_ma < iq_ma:
                all_checks_passed = False
                violations.append(
                    f"CURRENT INSUFFICIENT: Target current budget ({target_current_ma}mA) is less than "
                    f"device quiescent current ({iq_ma}mA). Component will exceed power budget."
                )
                checks.append({
                    "check_name": "Current Budget Feasibility",
                    "status": "FAIL",
                    "details": {
                        "target_current_ma": target_current_ma,
                        "quiescent_current_ma": iq_ma,
                        "datasheet_page": quiescent_spec.get("page", 7),
                        "extracted_specification": f"Quiescent Current: {iq_ma}mA",
                        "user_constraint": f"Target Current Budget: {target_current_ma}mA",
                        "calculation": f"Budget Deficit: {target_current_ma - iq_ma}mA",
                        "comparison_result": "FAIL",
                    },
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
                        "datasheet_page": quiescent_spec.get("page", 7),
                        "extracted_specification": f"Quiescent Current: {iq_ma}mA",
                        "user_constraint": f"Target Current Budget: {target_current_ma}mA",
                        "calculation": f"Budget Headroom: {round(headroom_ma, 4)}mA",
                        "comparison_result": "PASS",
                    },
                })

        # Check 5: Deterministic Sensor Transfer Function Calculations (0°C, 25°C, 100°C)
        scale_spec = verified_specs.get("scale_factor")
        scale_val = scale_spec.get("value", 10.0) if scale_spec else 10.0
        acc_spec = verified_specs.get("accuracy_tolerance")
        acc_val = acc_spec.get("value_c") if acc_spec else None

        # Always calculate 0°C, 25°C, and 100°C outputs
        calc_0c = self.calc.sensor_transfer_output(
            temperature_c=0.0,
            scale_factor_mv_per_c=scale_val,
            accuracy_tolerance_c=acc_val,
        )
        calculations["output_at_0c"] = calc_0c

        calc_25c = self.calc.sensor_transfer_output(
            temperature_c=25.0,
            scale_factor_mv_per_c=scale_val,
            accuracy_tolerance_c=acc_val,
        )
        calculations["output_at_25c"] = calc_25c

        calc_100c = self.calc.sensor_transfer_output(
            temperature_c=100.0,
            scale_factor_mv_per_c=scale_val,
            accuracy_tolerance_c=acc_val,
        )
        calculations["output_at_100c"] = calc_100c

        # Optional user min temperature transfer output
        if t_min is not None and t_min not in (0.0, 25.0, 100.0):
            calculations[f"output_at_{int(t_min)}c"] = self.calc.sensor_transfer_output(
                temperature_c=t_min,
                scale_factor_mv_per_c=scale_val,
                accuracy_tolerance_c=acc_val,
            )

        # Optional user max temperature transfer output
        if t_max is not None and t_max not in (0.0, 25.0, 100.0):
            calculations[f"output_at_{int(t_max)}c"] = self.calc.sensor_transfer_output(
                temperature_c=t_max,
                scale_factor_mv_per_c=scale_val,
                accuracy_tolerance_c=acc_val,
            )

        # Full-scale sensor output voltage for interface validation
        v_out_full_scale_v = calc_100c["output_voltage_v"]
        if t_max is not None and t_max > 100.0:
            calc_tmax = self.calc.sensor_transfer_output(
                temperature_c=t_max,
                scale_factor_mv_per_c=scale_val,
                accuracy_tolerance_c=acc_val,
            )
            v_out_full_scale_v = calc_tmax["output_voltage_v"]

        # Check 6 & 7: ADC Interface & Output Load Feasibility
        has_supply_check = any(c.get("check_name") == "Operating Supply Voltage Compatibility" for c in checks)
        adc_ref_v = requirements.get("adc_reference_voltage_v")
        adc_bits = requirements.get("adc_resolution_bits")
        load_r_ohms = requirements.get("load_impedance_ohms")

        if has_supply_check or adc_ref_v is not None or load_r_ohms is not None:
            # Check 6: ADC Interface & Signal Headroom Check
            if adc_ref_v is not None:
                adc_eval = self.calc.adc_compatibility_check(
                    max_sensor_voltage_v=v_out_full_scale_v,
                    adc_reference_v=adc_ref_v,
                    adc_resolution_bits=adc_bits,
                    scale_factor_mv_per_c=scale_val,
                )
                adc_details = dict(adc_eval)
                adc_details["datasheet_page"] = 1
                adc_details["extracted_specification"] = f"Scale Factor: {scale_val} mV/°C (Max Sensor Output: {v_out_full_scale_v}V)"
                adc_details["user_constraint"] = f"ADC Reference: {adc_ref_v}V" + (f", {adc_bits}-bit" if adc_bits else "")
                adc_details["calculation"] = (
                    f"Headroom: {adc_eval['margin_v']}V" if adc_eval["is_compatible"]
                    else f"Saturation at {adc_eval['saturation_temp_c']}°C (Deficit: {abs(adc_eval['margin_v'])}V)"
                )
                adc_details["comparison_result"] = adc_eval["status"]

                checks.append({
                    "check_name": "ADC Interface & Signal Headroom",
                    "status": adc_eval["status"],
                    "details": adc_details,
                })

                if not adc_eval["is_compatible"]:
                    all_checks_passed = False
                    violations.extend(adc_eval["violations"])

                warnings.extend(adc_eval["warnings"])
            else:
                checks.append({
                    "check_name": "ADC Interface & Signal Headroom",
                    "status": "UNVERIFIED",
                    "details": {
                        "reason": "ADC reference voltage / input range not specified by design requirements.",
                        "datasheet_page": 1,
                        "extracted_specification": f"Scale Factor: {scale_val} mV/°C",
                        "user_constraint": "Unspecified ADC Reference",
                        "calculation": "None",
                        "comparison_result": "UNVERIFIED",
                    },
                })

            # Check 7: Output Load Impedance & Drive Current Check
            if load_r_ohms is not None:
                if load_r_ohms <= 0:
                    all_checks_passed = False
                    violations.append(f"INVALID LOAD: Load resistance ({load_r_ohms}Ω) must be positive.")
                    checks.append({
                        "check_name": "Output Load Impedance Feasibility",
                        "status": "FAIL",
                        "details": {
                            "load_impedance_ohms": load_r_ohms,
                            "datasheet_page": 4,
                            "extracted_specification": "Maximum Output Sourcing Current: 10 mA",
                            "user_constraint": f"Load Impedance: {load_r_ohms}Ω",
                            "calculation": "Negative or Zero Resistance",
                            "comparison_result": "FAIL",
                        },
                    })
                else:
                    i_load_ma = (v_out_full_scale_v / load_r_ohms) * 1000.0
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
                                "datasheet_page": 4,
                                "extracted_specification": f"Maximum Output Sourcing Current: {max_rated_load_ma} mA",
                                "user_constraint": f"Load Impedance: {load_r_ohms}Ω",
                                "calculation": f"I_load = {v_out_full_scale_v}V / {load_r_ohms}Ω = {round(i_load_ma, 4)}mA",
                                "comparison_result": "FAIL",
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
                                "datasheet_page": 4,
                                "extracted_specification": f"Maximum Output Sourcing Current: {max_rated_load_ma} mA",
                                "user_constraint": f"Load Impedance: {load_r_ohms}Ω",
                                "calculation": f"I_load = {v_out_full_scale_v}V / {load_r_ohms}Ω = {round(i_load_ma, 4)}mA",
                                "comparison_result": "PASS",
                            },
                        })
            else:
                checks.append({
                    "check_name": "Output Load Impedance Feasibility",
                    "status": "UNVERIFIED",
                    "details": {
                        "reason": "Load impedance / resistance not specified by design requirements.",
                        "datasheet_page": 4,
                        "extracted_specification": "Maximum Output Sourcing Current: 10 mA",
                        "user_constraint": "Unspecified Load Impedance",
                        "calculation": "None",
                        "comparison_result": "UNVERIFIED",
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

