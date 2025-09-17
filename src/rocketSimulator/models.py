# Licensed under the PolyForm Noncommercial License 1.0.0
"""Data models and constants for the rocket simulator."""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

# Physical constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
R_earth = 6.371e6  # Earth's radius (m)
M_earth = 5.972e24  # Earth's mass (kg)
g0 = 9.80665  # Standard gravity (m/s^2)
mu = M_earth * G  # Earth's gravitational parameter (m^3 s^-2)

@dataclass
class RocketStage:
    """Class representing a single rocket stage.
    
    Attributes:
        dry_mass: Mass of the stage without propellant (kg)
        propellant_mass: Mass of the propellant (kg)
        thrust: Function that returns thrust in Newtons given time and state
        isp: Function that returns specific impulse in seconds given time and state
        stage_criteria: Function that returns a value that decreases past 0 when stage separation should occur
        drag_coeff: Function that returns drag coefficient given angle of attack and Mach number
        lift_coeff: Function that returns lift coefficient given angle of attack and Mach number
        reference_area: Reference area for aerodynamic calculations (m²)
    """
    dry_mass: float
    propellant_mass: float
    thrust: Callable[[float, np.ndarray], float]
    isp: Callable[[float, np.ndarray], float]
    stage_criteria: Callable[[float, np.ndarray], float]
    drag_coeff: Callable[[float, float], float]
    lift_coeff: Callable[[float, float], float]
    reference_area: float
