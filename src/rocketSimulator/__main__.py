# Licensed under the PolyForm Noncommercial License 1.0.0
"""
Command-line interface for the rocket simulator.
"""

def main():
    """Run the rocket simulator with example parameters."""
    import numpy as np
    try:
        from core import RocketAscentSimulator
        from models import RocketStage
        from plotting import plot_results

    except:
        from . import *

    print("Rocket Ascent Simulator")
    print("======================")
    
    # Example stage configuration
    def constant_thrust(t, state):
        return 2.5e6  # 2.5 MN thrust
    
    def constant_isp(t, state):
        return 282.0  # seconds
    
    def simple_drag(state):
        return 0.3  # Simple constant drag coefficient

    def simple_lift(state):
        return 0.1  # Simple constant lift coefficient
    
    # Create stages
    stage1 = RocketStage(
        dry_mass=20000,  # 20 tons dry mass
        propellant_mass=180000,  # 180 tons propellant
        thrust=constant_thrust,
        isp=constant_isp,
        stage_criteria=lambda t, state: 300 - t,  # Stage separation at t=300s
        drag_coeff=simple_drag,
        lift_coeff=simple_lift,  # No lift
        reference_area=3.14  # m²
    )

    payload_stage = RocketStage(
        dry_mass=10000,
        propellant_mass=0,
        thrust=lambda t, state: 1e3,
        isp=lambda t, state: 2000,
        drag_coeff=lambda state: 3,
        lift_coeff=lambda state: 1,
        reference_area=10,
        stage_criteria=lambda t, state: 1
    )

    # Create simulator
    def ascent_profile(t, state):
        # Simple gravity turn profile
        t = np.atleast_1d(t)
        pitch = np.empty_like(t)
        throttle = np.empty_like(t)

        mask_vertical_ascent = t < 10
        mask_tilt = ~mask_vertical_ascent

        pitch[mask_vertical_ascent] = np.pi/2
        throttle[mask_vertical_ascent] = 1.0

        pitch[mask_tilt] = np.maximum(np.pi/6, np.pi/2 - t[mask_tilt]/100)
        throttle[mask_tilt] = 1.0

        if np.ndim(state) == 1:
            pitch = pitch[0]
            throttle = throttle[0]

        return pitch, throttle
    
    simulator = RocketAscentSimulator([stage1, payload_stage], ascent_profile)
    
    # Run simulation
    print("Running simulation...")
    results = simulator.simulate(t_span=(0, 600), max_step=1.0)
    
    # Plot results
    print("Plotting results...")
    plot_results(results, show=True)
    
    # Print some key results
    max_alt = np.max(results['y']) / 1000  # km
    max_speed = np.max(results['velocity']) / 1000  # km/s
    final_mass = results['m'][-1] / 1000  # tons
    
    print(f"\nSimulation Complete!")
    print(f"Peak altitude: {max_alt:.1f} km")
    print(f"Maximum speed: {max_speed:.1f} km/s")
    print(f"Final mass: {final_mass:.1f} tons")

if __name__ == "__main__":
    main()
