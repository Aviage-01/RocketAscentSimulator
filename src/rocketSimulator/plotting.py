# Licensed under the PolyForm Noncommercial License 1.0.0
"""Plotting functions for rocket simulation results."""

from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    from models import R_earth
except:
    from . import R_earth

def plot_results(results: Dict, show: bool = True, save_path: Optional[str] = None) -> None:
    """
    Plot simulation results.

    Args:
        results: Dictionary containing simulation results
        show: Whether to display the plot
        save_path: If provided, save the plot to this path
    """

    states = [results['x'], results['y'], results['vx'], results['vy'], results['m']]
    stages = results['stage']

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Define colours for stages
    unique_stages = np.unique(stages)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_stages)))

    # 1. Trajectory around Earth, coloured by stage
    for stage_num, color in zip(unique_stages, colors):
        mask = stages == stage_num
        axes[0, 0].plot(results['x'][mask], results['y'][mask], color=color, label=f"Stage {int(stage_num) + 1}")

    circle = plt.Circle((0, 0), R_earth, color='blue', alpha=0.3)
    axes[0, 0].add_artist(circle)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_title("Trajectory Around Earth")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].legend()

    # 2. Altitude vs Time
    for stage_num, color in zip(unique_stages, colors):
        mask = stages == stage_num
        axes[0, 1].plot(results['t'][mask], results['altitude'][mask] / 1000, color=color,
                        label=f"Stage {int(stage_num)+1}")
    axes[0, 1].set_title("Altitude vs Time")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Altitude [km]")
    axes[0, 1].legend()

    # 3. Velocity vs Time
    for stage_num, color in zip(unique_stages, colors):
        mask = stages == stage_num
        axes[1, 0].plot(results['t'][mask], results['velocity'][mask], color=color, label=f"Stage {int(stage_num)+1}")
    axes[1, 0].set_title("Speed vs Time")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Speed [m/s]")
    axes[1, 0].legend()

    # 4. Drag vs Time
    for stage_num, color in zip(unique_stages, colors):
        mask = stages == stage_num
        axes[1, 1].plot(results['t'][mask],
                        np.linalg.norm(results['rocket']._get_drag(states, stage_num=stages)[mask], axis=1),
                        color=color, label=f"Stage {int(stage_num)+1}")

    axes[1, 1].set_title("Drag vs Time")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Drag [N]")
    axes[1, 1].legend()

    # 5. Δv vs time (cumulative)
    cumulative_dv = np.zeros_like(results['t'])
    dv_per_stage = {}
    dv_offset = 0.0  # keeps track of total Δv from previous stages

    for stage_num, color in zip(unique_stages, colors):
        mask = stages == stage_num
        t_stage = results['t'][mask]
        m_stage = results['m'][mask]

        # Get thrust for this stage over the segment
        states_stage = [results['x'][mask], results['y'][mask], results['vx'][mask], results['vy'][mask],
                        results['m'][mask]]

        pitch, throttle = results['rocket'].ascent_profile(results['t'][mask], states_stage)
        thrust_stage = throttle * results['rocket']._get_thrust_and_mass_flow(t_stage, states_stage, stage_num=stage_num,
                                                                   thrust_only=True)
        # Δv = ∫(thrust / m) dt, trapezoid integration
        dv_stage = np.cumsum(np.gradient(t_stage) * (thrust_stage / m_stage)) + dv_offset

        dv_per_stage[stage_num] = dv_stage[-1] - dv_offset
        cumulative_dv[mask] = dv_stage

        # Plot with nicer label formatting
        final_dv_kms = dv_stage[-1] / 1000  # convert to km/s
        axes[2, 0].plot(t_stage, dv_stage / 1000, color=color,
                        label=f"Stage {int(stage_num) + 1} – Δv: {dv_per_stage[stage_num] / 1000:.2f} km/s")

        # Update offset for next stage
        dv_offset = dv_stage[-1]

    axes[2, 0].set_title("Cumulative Δv over time")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].set_ylabel("Δv [km/s]")
    axes[2, 0].legend()

    # 6. Mass vs Time
    for stage_num, color in zip(unique_stages, colors):
        mask = stages == stage_num
        states_stage = [results['x'][mask], results['y'][mask], results['vx'][mask], results['vy'][mask],
                        results['m'][mask]]

        perigees, apogees = np.array(results['rocket']._perigee_apogee(states_stage)) - R_earth

        axes[2, 1].plot(results['t'][mask],
                        perigees,
                        color=color,
                        label=f"Stage {int(stage_num) + 1} perigee",
                        ls='--')
        axes[2, 1].plot(results['t'][mask],
                        apogees,
                        color=color,
                        label=f"Stage {int(stage_num) + 1} apogee",
                        ls='-')

    axes[2, 1].set_title("Time vs Apogee and Perigee")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].set_ylabel("Altitude [m]")
    axes[2, 1].legend()

    for stage_num, color in zip(unique_stages, colors):
        mask = stages == stage_num
        results['rocket']._current_stage = stage_num
        states_stage = [results['x'][mask], results['y'][mask], results['vx'][mask], results['vy'][mask],
                        results['m'][mask]]

        pitch, throttle = results['rocket'].ascent_profile(results['t'][mask], states_stage)
        #print("Stage number", stage_num, states_stage)
        #print("Pitch", pitch)

        axes[0, 2].plot(results['t'][mask],
                        pitch,
                        color=color,
                        label=f"Stage {int(stage_num) + 1} pitch",
                        ls='--')
        axes[0, 2].plot(results['t'][mask],
                        throttle,
                        color=color,
                        label=f"Stage {int(stage_num) + 1} throttle",
                        ls='-')

    axes[0, 2].set_title("Time vs commanded pitch and throttle")
    axes[0, 2].set_xlabel("Time [s]")
    axes[0, 2].set_ylabel("Commanded angle [rad]")
    axes[0, 2].legend()

    # 7. Speed vs Altitude
    for stage_num, color in zip(unique_stages, colors):
        mask = stages == stage_num
        axes[1, 2].plot(results['velocity'][mask],
                        results['altitude'][mask] / 1000,
                        color=color,
                        label=f"Stage {int(stage_num) + 1}")

    axes[1, 2].set_title("Speed vs Altitude")
    axes[1, 2].set_xlabel("Speed [m/s]")
    axes[1, 2].set_ylabel("Altitude [km]")
    axes[1, 2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()