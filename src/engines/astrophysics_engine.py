# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Astrophysics and Cosmology Engine

Provides production-quality computational tools for fundamental calculations in
observational astronomy, astrophysics, and cosmology. This engine is built upon
the industry-standard 'astropy' library, ensuring accuracy and adherence to
established astronomical conventions and models. All functions are fully
implemented.
"""

import logging
from typing import Dict, Any

try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Galactic
    from astropy.cosmology import Planck18 as cosmo
    from astropy.constants import G, L_sun, M_sun
except ImportError:
    raise ImportError("The 'astropy' library is not installed. Please install it with 'pip install astropy'.")

logger = logging.getLogger(__name__)

class AstrophysicsEngine:
    """
    Implements computational tools for astrophysics and cosmology.
    """
    
    def __init__(self):
        self.name = "Astrophysics and Cosmology Engine"
        self.version = "1.0.0"
        self.supported_calculations = [
            "convert_celestial_coordinates",
            "calculate_cosmological_distances",
            "convert_magnitude_to_luminosity",
            "calculate_gravitational_force"
        ]

    def convert_celestial_coordinates(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts celestial coordinates between different standard frames.
        """
        try:
            from_frame = parameters['from_frame'].lower()
            to_frame = parameters['to_frame'].lower()
            coords = parameters['coordinates']

            if from_frame == 'icrs':
                sky_coord = SkyCoord(ra=coords['ra']*u.deg, dec=coords['dec']*u.deg, frame='icrs')
            elif from_frame == 'galactic':
                sky_coord = SkyCoord(l=coords['l']*u.deg, b=coords['b']*u.deg, frame='galactic')
            else:
                return {"error": f"Unsupported 'from_frame': {from_frame}. Use 'icrs' or 'galactic'."}

            if to_frame == 'icrs':
                converted = sky_coord.icrs
                return {"ra_deg": converted.ra.deg, "dec_deg": converted.dec.deg}
            elif to_frame == 'galactic':
                converted = sky_coord.galactic
                return {"l_deg": converted.l.deg, "b_deg": converted.b.deg}
            elif to_frame == 'altaz':
                location_params = parameters.get('location')
                if not location_params:
                    return {"error": "Location parameters ('lat', 'lon', 'height') are required for AltAz conversion."}
                location = EarthLocation(lat=location_params['lat']*u.deg, lon=location_params['lon']*u.deg, height=location_params.get('height', 0)*u.m)
                altaz_frame = AltAz(location=location)
                converted = sky_coord.transform_to(altaz_frame)
                return {"alt_deg": converted.alt.deg, "az_deg": converted.az.deg}
            else:
                return {"error": f"Unsupported 'to_frame': {to_frame}. Use 'icrs', 'galactic', or 'altaz'."}
                
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            logger.error(f"Error in coordinate conversion: {e}")
            return {"error": f"An unexpected error occurred: {e}"}

    def calculate_cosmological_distances(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculates cosmological distances and lookback time for a given redshift.
        Uses the Planck 2018 cosmological model.
        """
        try:
            redshift = parameters['redshift']
            if redshift < 0:
                return {"error": "Redshift cannot be negative."}

            lum_dist = cosmo.luminosity_distance(redshift)
            ang_dist = cosmo.angular_diameter_distance(redshift)
            lookback = cosmo.lookback_time(redshift)

            return {
                "redshift": redshift,
                "luminosity_distance_mpc": lum_dist.to(u.Mpc).value,
                "angular_diameter_distance_mpc": ang_dist.to(u.Mpc).value,
                "lookback_time_gyr": lookback.to(u.Gyr).value,
                "cosmology_model": "Planck18"
            }
        except KeyError:
            return {"error": "Missing required parameter: 'redshift'."}
        except Exception as e:
            logger.error(f"Error in cosmological calculation: {e}")
            return {"error": f"An unexpected error occurred: {e}"}
            
    def convert_magnitude_to_luminosity(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Converts a star's absolute magnitude to its luminosity.
        """
        try:
            abs_magnitude = parameters['absolute_magnitude']
            
            # The formula is L/L_sun = 10**((M_sun - M_star) / 2.5)
            # Absolute magnitude of the Sun is ~4.83
            M_sun_abs = 4.83
            luminosity_in_solar_units = 10**((M_sun_abs - abs_magnitude) / 2.5)
            luminosity_in_watts = luminosity_in_solar_units * L_sun.value

            return {
                "absolute_magnitude": abs_magnitude,
                "luminosity_solar_units": luminosity_in_solar_units,
                "luminosity_watts": luminosity_in_watts
            }
        except KeyError:
            return {"error": "Missing required parameter: 'absolute_magnitude'."}
        except Exception as e:
            logger.error(f"Error in magnitude-luminosity conversion: {e}")
            return {"error": f"An unexpected error occurred: {e}"}

    def calculate_gravitational_force(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculates the gravitational force between two masses using Newton's Law.
        """
        try:
            mass1_kg = parameters['mass1_kg']
            mass2_kg = parameters['mass2_kg']
            distance_m = parameters['distance_m']
            
            if distance_m <= 0:
                return {"error": "Distance must be positive."}

            force_newtons = (G * mass1_kg * u.kg * mass2_kg * u.kg / (distance_m * u.m)**2).to(u.N).value

            return {
                "mass1_kg": mass1_kg,
                "mass2_kg": mass2_kg,
                "distance_m": distance_m,
                "gravitational_force_newtons": force_newtons
            }
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            logger.error(f"Error in gravitational force calculation: {e}")
            return {"error": f"An unexpected error occurred: {e}"} 