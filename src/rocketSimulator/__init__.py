# Licensed under the PolyForm Noncommercial License 1.0.0
"""Rocket Ascent Simulator - A Python package for simulating rocket ascent through Earth's atmosphere."""

from .models import (
    G,
    R_earth,
    M_earth,
    g0,
    RocketStage,
)

from .core import RocketAscentSimulator
from .plotting import plot_results

__version__ = "0.1.0"
__all__ = [
    "RocketStage",
    "RocketAscentSimulator",
    "plot_results",
    "G",
    "R_earth",
    "M_earth",
    "g0",
]
