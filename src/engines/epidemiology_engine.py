# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Epidemiology and Public Health Engine

Provides production-quality computational tools for epidemiological modeling and
public health statistics. This engine uses standard models and metrics to
provide accurate, real-time calculations with no placeholders.
"""

import logging
from typing import Dict, List, Any

import numpy as np
from scipy.integrate import odeint

logger = logging.getLogger(__name__)

class EpidemiologyEngine:
    """
    Implements computational tools for the epidemiology and public health domain.
    """
    
    def __init__(self):
        self.name = "Epidemiology and Public Health Engine"
        self.version = "1.0.0"
        self.supported_calculations = [
            "sir_model_simulation",
            "calculate_prevalence",
            "calculate_incidence_rate",
            "calculate_basic_reproduction_number"
        ]

    def _sir_model(self, y: List[float], t: np.ndarray, N: int, beta: float, gamma: float) -> List[float]:
        """The SIR differential equations."""
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    def sir_model_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs a full Susceptible-Infected-Recovered (SIR) model simulation.
        """
        try:
            N = parameters['total_population']
            I0 = parameters['initial_infected']
            R0 = parameters.get('initial_recovered', 0)
            S0 = N - I0 - R0
            beta = parameters['contact_rate'] # Transmission rate
            gamma = parameters['recovery_rate'] # 1 / recovery_period
            days = parameters['simulation_days']
            
            t = np.linspace(0, days, days)
            y0 = [S0, I0, R0]
            
            ret = odeint(self._sir_model, y0, t, args=(N, beta, gamma))
            S, I, R = ret.T

            # Return results for key days (e.g., weekly) to avoid excessive data
            report_interval = max(1, days // 10)
            result_points = {}
            for i in range(0, days, report_interval):
                day = int(t[i])
                result_points[f"day_{day}"] = {
                    "susceptible": int(S[i]),
                    "infected": int(I[i]),
                    "recovered": int(R[i])
                }

            return {
                "simulation_summary": f"SIR model for {N} people over {days} days.",
                "peak_infection": {"day": int(np.argmax(I)), "count": int(np.max(I))},
                "final_state": {"susceptible": int(S[-1]), "infected": int(I[-1]), "recovered": int(R[-1])},
                "data_points": result_points
            }
        except KeyError as e:
            return {"error": f"Missing required parameter for SIR model: {e}"}
        except Exception as e:
            logger.error(f"Error in SIR model simulation: {e}")
            return {"error": f"An unexpected error occurred during the simulation: {e}"}

    def calculate_prevalence(self, parameters: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculates prevalence of a disease.
        Prevalence = (Total cases / Total population) * 100,000 (or other multiplier)
        """
        try:
            total_cases = parameters['total_cases']
            total_population = parameters['total_population']
            multiplier = parameters.get('multiplier', 100_000)

            if total_population == 0:
                return {"error": "Total population cannot be zero."}

            prevalence = (total_cases / total_population) * multiplier
            
            return {
                "prevalence": round(prevalence, 2),
                "interpretation": f"~{int(prevalence)} cases per {multiplier:,} people."
            }
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            return {"error": f"An error occurred: {e}"}

    def calculate_incidence_rate(self, parameters: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculates the incidence rate of a disease.
        Incidence Rate = (New cases in period / Population at risk) * 100,000
        """
        try:
            new_cases = parameters['new_cases']
            population_at_risk = parameters['population_at_risk']
            multiplier = parameters.get('multiplier', 100_000)

            if population_at_risk == 0:
                return {"error": "Population at risk cannot be zero."}

            incidence_rate = (new_cases / population_at_risk) * multiplier

            return {
                "incidence_rate": round(incidence_rate, 2),
                "interpretation": f"~{int(incidence_rate)} new cases per {multiplier:,} at-risk people in the period."
            }
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            return {"error": f"An error occurred: {e}"}

    def calculate_basic_reproduction_number(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculates the basic reproduction number (R0) from model parameters.
        R0 = beta / gamma
        """
        try:
            beta = parameters['contact_rate']
            gamma = parameters['recovery_rate']

            if gamma == 0:
                return {"error": "Recovery rate (gamma) cannot be zero."}
                
            r0 = beta / gamma
            
            interpretation = "The number of secondary infections expected from a single infected individual in a fully susceptible population."
            if r0 > 1:
                interpretation += " An R0 > 1 suggests the epidemic will grow."
            elif r0 < 1:
                interpretation += " An R0 < 1 suggests the epidemic will decline."
            else:
                interpretation += " An R0 = 1 suggests the epidemic is stable (endemic)."
                
            return {
                "r0_value": round(r0, 4),
                "interpretation": interpretation
            }
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            return {"error": f"An error occurred: {e}"} 