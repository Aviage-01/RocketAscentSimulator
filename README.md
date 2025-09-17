# Rocket Ascent Simulator

A Python package for simulating rocket ascent through Earth's atmosphere with configurable parameters and ascent profiles.

## Features

- Multi-stage rocket simulation
- Realistic atmospheric modeling
- Configurable thrust and ISP curves
- Visualization tools for analysis
- Extensible architecture
- Completely unreliable readme

## Installation

```bash
pip install -e .
```

## Usage

```python
from rocket_simulator import RocketStage, RocketAscentSimulator
import numpy as np

# Define stage parameters
def constant_thrust(t, state):
    return 2.5e6  # 2.5 MN thrust

def constant_isp(t, state):
    return 282.0  # seconds

def simple_drag(alpha, mach):
    return 0.1  # Simple constant drag coefficient

# Create rocket stage
stage = RocketStage(
    dry_mass=20000,  # kg
    propellant_mass=180000,  # kg
    thrust=constant_thrust,
    isp=constant_isp,
    drag_coeff=simple_drag,
    lift_coeff=lambda a, m: 0.0,  # No lift
    reference_area=3.14  # m²
)

# Create simulator
simulator = RocketAscentSimulator(
    [stage],
    ascent_profile=lambda t, state: (np.pi/4, 1.0)  # 45° pitch, full throttle
)

# Run simulation
results = simulator.simulate(t_span=(0, 500), max_step=1.0)
```

## Development

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests:
   ```bash
   pytest
   ```

## License

Polyform Noncommercial - See [LICENSE](LICENSE) for details.
