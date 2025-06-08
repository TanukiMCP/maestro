# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Veterinary Radiation Oncology Computational Engine

Provides a suite of production-quality computational tools for practicing
veterinary radiation oncologists. These functions automate common, critical
calculations related to treatment planning, safety, and radiobiology,
designed to augment a clinical workflow. All functions are fully implemented.
"""

import logging
import math
from typing import Dict, Any

logger = logging.getLogger(__name__)

class VeterinaryRadiationOncologyEngine:
    """
    Implements computational tools for veterinary radiation oncology.
    """
    
    def __init__(self):
        self.name = "Veterinary Radiation Oncology Computational Engine"
        self.version = "1.0.0"
        # Half-lives in hours for common isotopes used in veterinary medicine
        self.isotope_half_lives_h = {
            "I-131": 8.02 * 24,   # Iodine-131
            "Sr-90": 28.79 * 365.25 * 24, # Strontium-90
            "P-32": 14.26 * 24,  # Phosphorus-32
            "Co-60": 5.27 * 365.25 * 24  # Cobalt-60 for reference
        }
        self.supported_calculations = [
            "calculate_biological_effective_dose",
            "calculate_radioisotope_decay",
            "calculate_dose_adjustment_for_gap",
            "calculate_manual_monitor_units",
            "estimate_tumor_control_probability"
        ]

    def calculate_biological_effective_dose(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Calculates BED and EQD2 to compare different radiotherapy fractionation schedules."""
        try:
            total_dose = parameters['total_dose_gy']
            dose_per_fraction = parameters['dose_per_fraction_gy']
            alpha_beta_ratio = parameters['alpha_beta_ratio']
            
            if dose_per_fraction <= 0 or alpha_beta_ratio <= 0:
                return {"error": "Dose per fraction and alpha/beta ratio must be positive."}
            
            bed = total_dose * (1 + dose_per_fraction / alpha_beta_ratio)
            eqd2 = bed / (1 + 2 / alpha_beta_ratio)
            
            return {
                "inputs": parameters,
                "biological_effective_dose_gy": round(bed, 2),
                "equivalent_dose_2gy_fractions_gy": round(eqd2, 2)
            }
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}

    def calculate_radioisotope_decay(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates the remaining activity of a radioisotope after a given time."""
        try:
            initial_activity = parameters['initial_activity_bq']
            isotope = parameters['isotope']
            time_elapsed_h = parameters['time_elapsed_hours']
            
            if isotope not in self.isotope_half_lives_h:
                return {"error": f"Unsupported isotope. Supported isotopes are: {list(self.isotope_half_lives_h.keys())}"}
            
            half_life_h = self.isotope_half_lives_h[isotope]
            decay_constant = math.log(2) / half_life_h
            remaining_activity = initial_activity * math.exp(-decay_constant * time_elapsed_h)
            
            return {
                "inputs": parameters,
                "remaining_activity_bq": round(remaining_activity, 4)
            }
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}
            
    def calculate_dose_adjustment_for_gap(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Calculates the dose required to compensate for a gap in treatment."""
        try:
            alpha = parameters['alpha'] # Tumor radiosensitivity parameter
            gap_duration_days = parameters['gap_duration_days']
            tumor_doubling_time_days = parameters['tumor_doubling_time_days']

            if tumor_doubling_time_days <= 0:
                return {"error": "Tumor doubling time must be positive."}

            # Formula for dose loss due to proliferation
            proliferative_dose_loss = (math.log(2) / alpha) * (gap_duration_days / tumor_doubling_time_days)

            return {
                "inputs": parameters,
                "lost_dose_gy": round(proliferative_dose_loss, 2),
                "notes": "This is the additional dose needed to overcome tumor repopulation during the gap. It should be added to the remaining treatment fractions."
            }
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}

    def calculate_manual_monitor_units(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Performs a manual sanity check of Monitor Units (MU) for a simple setup."""
        try:
            prescribed_dose_cgy = parameters['prescribed_dose_cgy']
            percent_depth_dose = parameters['percent_depth_dose']
            machine_output_cgy_per_mu = parameters.get('machine_output_cgy_per_mu', 1.0) # 1.0 is a common standard

            if percent_depth_dose <= 0 or percent_depth_dose > 150: # Sanity check PDD
                return {"error": "Percent depth dose must be a valid percentage (e.g., 85 for 85%)."}
            
            # Simplified MU formula for a basic QA check
            # MU = Prescribed Dose / (Machine Output * PDD/100 * other factors)
            # For this sanity check, we assume other factors (SCF, TMR, etc.) are 1.0
            calculated_mu = prescribed_dose_cgy / (machine_output_cgy_per_mu * (percent_depth_dose / 100.0))

            return {
                "inputs": parameters,
                "calculated_mu": round(calculated_mu, 1),
                "notes": "This is a simplified sanity check. It does not replace the Treatment Planning System calculation."
            }
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}

    def estimate_tumor_control_probability(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Estimates Tumor Control Probability (TCP) using a logistic model."""
        try:
            delivered_dose_gy = parameters['delivered_dose_gy']
            tcd50_gy = parameters['tcd50_gy'] # Dose for 50% TCP
            gamma50 = parameters['gamma50'] # Steepness of the curve
            
            if tcd50_gy <= 0 or gamma50 <= 0:
                return {"error": "TCD50 and gamma50 must be positive."}
            
            exponent = 2 * gamma50 * (1 - (delivered_dose_gy / tcd50_gy))
            tcp = 1 / (1 + math.exp(exponent))

            return {
                "inputs": parameters,
                "tumor_control_probability": round(tcp, 4)
            }
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"} 