# Licensed under the PolyForm Noncommercial License 1.0.0
"""Core simulation logic for the rocket ascent simulator."""

from typing import Callable, Dict, List, Optional, Tuple, Union
from types import MethodType
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp

try:
    from models import *
    from models import mu

except:
    from . import *
    from .models import mu

class RocketAscentSimulator:
    """
    Simulates the ascent of a multi-stage rocket through Earth's atmosphere.
    """

    def __init__(self, stages: List[RocketStage], ascent_profile: Callable[[float, np.ndarray], np.ndarray]):
        """
        Initialize the rocket simulator.

        Args:
            stages: List of RocketStage objects in ascending order (first to fire first)
            ascent_profile: Function returning (pitch, throttle) for given time and state
        """
        self.stages = stages
        self.ascent_profile = MethodType(ascent_profile, self)
        self._current_stage = 0
        self._current_stage_start_time = 0.0

    def _atmosphere_model(self, altitude: Union[float, int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple exponential atmosphere model.

        Args:
            altitude: Altitude above sea level (m). Can be float/int or np.ndarray.

        Returns:
            Tuple of (density in kg/m^3, pressure in Pa)
        """
        altitude = np.atleast_1d(altitude)  # ensure array for vectorised ops

        T = np.empty_like(altitude, dtype=float)
        p = np.empty_like(altitude, dtype=float)

        # Upper atmosphere
        mask = altitude > 25000
        T[mask] = -131.21 + 0.00299 * altitude[mask]
        p[mask] = 2.488 * ((T[mask] + 273.1) / 216.6) ** (-11.388)

        # Lower stratosphere
        mask = (altitude > 11000) & (altitude <= 25000)
        T[mask] = -56.46
        p[mask] = 22.65 * np.exp(1.73 - 0.000157 * altitude[mask])

        # Troposphere
        mask = altitude <= 11000
        T[mask] = 15.04 - 0.00649 * altitude[mask]
        p[mask] = 101.29 * ((T[mask] + 273.1) / 288.08) ** 5.256

        rho = p / (0.2869 * (T + 273.1))  # Density in kg/m^3
        p = p * 1000  # Convert pressure to Pa

        # Return scalar if input was scalar
        if np.ndim(altitude) == 0 or altitude.shape == ():
            return float(rho[0]), float(p[0])
        return rho, p

    def _gravity(self, altitude: float) -> float:
        """Calculate gravitational acceleration at given altitude."""
        return G * M_earth / (R_earth + altitude) ** 2

    def _get_thrust_and_mass_flow(self, t: float, state: np.ndarray,
                                  stage_num: Optional[np.ndarray] = None,
                                  thrust_only: bool = False) -> np.ndarray:
        """
        Vectorised computation of thrust and mass flow for multiple states and stages.

        Args:
            t: Current time (s)
            state: array of shape (5, N) representing [x, y, vx, vy, m] for N states
            stage_num: array of stage indices, length N, or None to use self._current_stage
            thrust_only: if True, return only thrust

        Returns:
            array of shape (2, N) for [thrust, mass_flow] if thrust_only=False,
            or shape (N,) if thrust_only=True
        """

        state = np.array(state)
        if state.ndim == 1:
            state = state.reshape(-1, 1)

        m = np.atleast_1d(state[4]) # rocket masses
        N = len(m)

        # Broadcast stage_num
        if stage_num is None:
            stage_num = np.full(N, self._current_stage, dtype=int)
        elif isinstance(stage_num, (int, float, np.integer, np.floating)):  # scalar
            stage_num = np.full(N, int(stage_num), dtype=int)
        elif isinstance(stage_num, (list, np.ndarray)):
            stage_num = np.array(stage_num, dtype=int)
            if stage_num.size != N:
                raise ValueError(f"stage_num must have length {N}, got {stage_num.size}")

        else:
            raise TypeError(f"stage_num must be None, scalar, list, or ndarray, got {type(stage_num)}")

        thrust_arr = np.zeros(N, dtype=float)
        mass_flow_arr = np.zeros(N, dtype=float)

        for sn in np.unique(stage_num):
            mask = stage_num == sn
            if sn >= len(self.stages):
                continue

            stage = self.stages[sn]

            # Upper stages mass
            upper_stages_mass = sum([s.dry_mass + s.propellant_mass for s in self.stages[sn + 1:]])

            # Current stage propellant
            stage_current_prop = m[mask] - upper_stages_mass - stage.dry_mass
            stage_current_prop = np.maximum(stage_current_prop, 0.0)

            # Vectorised thrust and ISP
            current_thrust = stage.thrust(t, state[:, mask])
            current_isp = stage.isp(t, state[:, mask])
            epsilon = 1e-12
            mass_flow = np.where(current_isp > 0, current_thrust / (np.maximum(current_isp, epsilon) * g0), 0.0)

            # Zero out exhausted stages
            current_thrust = np.where(stage_current_prop <= 0, 0.0, current_thrust)
            mass_flow = np.where(stage_current_prop <= 0, 0.0, mass_flow)

            thrust_arr[mask] = current_thrust
            mass_flow_arr[mask] = mass_flow

        if thrust_only:
            return thrust_arr

        return np.vstack([thrust_arr, mass_flow_arr])

    def _get_drag(self, state: np.ndarray, stage_num=None) -> float:
        """
        Calculate drag force.

        Args:
            state: Current state vector [x, y, vx, vy, m]

        Returns:
            Drag force in N
        """
        vx, vy = state[2], state[3]
        v = np.sqrt(vx ** 2 + vy ** 2)
        altitude = np.sqrt(state[0] ** 2 + state[1] ** 2) - R_earth
        rho, _ = self._atmosphere_model(altitude)

        vx_arr = np.atleast_1d(vx)
        vy_arr = np.atleast_1d(vy)
        v_arr = np.atleast_1d(v)
        rho_arr = np.atleast_1d(rho)

        N = len(v_arr)

        # Broadcast stage_num
        if stage_num is None:
            stage_num = np.full(N, self._current_stage, dtype=int)
        elif isinstance(stage_num, (int, float, np.integer, np.floating)):  # scalar
            stage_num = np.full(N, int(stage_num), dtype=int)
        elif isinstance(stage_num, (list, np.ndarray)):
            stage_num = np.array(stage_num, dtype=int)
            if stage_num.size != N:
                raise ValueError(f"stage_num must have length {N}, got {stage_num.size}")

        else:
            raise TypeError(f"stage_num must be None, scalar, list, or ndarray, got {type(stage_num)}")

        stage_num_arr = np.atleast_1d(stage_num)

        drag_force = np.zeros_like(v_arr, dtype=float)
        lift_force = np.zeros_like(v_arr, dtype=float)

        for sn in np.unique(stage_num_arr):
            mask = stage_num_arr == sn
            stage = self.stages[sn]
            drag_force[mask] = 0.5 * rho_arr[mask] * v_arr[mask] ** 2 * stage.drag_coeff(state) * stage.reference_area
            lift_force[mask] = 0.5 * rho_arr[mask] * v_arr[mask] ** 2 * stage.lift_coeff(state) * stage.reference_area

        # initialise drag_vector array
        drag_vector = np.zeros((len(vx_arr), 2))

        # Only apply where velocity is non-zero
        nonzero_mask = v_arr > 0
        vel_unit = np.zeros((len(v_arr), 2))  # 2D array for x and y components
        vel_unit[nonzero_mask] = np.stack([vx_arr, vy_arr], axis=1)[nonzero_mask] / v_arr[nonzero_mask, None]

        # Drag along velocity
        drag_vector[nonzero_mask] = -drag_force[nonzero_mask, None] * vel_unit[nonzero_mask]

        # Lift perpendicular (+90° rotation: [-vy, vx])
        lift_vector = np.zeros_like(drag_vector)
        lift_vector[nonzero_mask] = lift_force[nonzero_mask, None] * np.stack([-vy_arr, vx_arr], axis=1)[nonzero_mask] / \
                                    v_arr[nonzero_mask, None]

        # Total aerodynamic vector
        drag_vector += lift_vector

        # If original vx, vy were scalars, return 1x2 array
        if np.isscalar(vx) and np.isscalar(vy):
            drag_vector = drag_vector[0]

        return drag_vector.T

    def _equations_of_motion(self, t: float, state: np.ndarray, control = None, stage_num = None) -> np.ndarray:
        """
        Calculate time derivatives of the state vector.

        State vector: [x, y, vx, vy, m]
        Derivatives: [dx/dt, dy/dt, dvx/dt, dvy/dt, dm/dt]
        """

        if stage_num is None:
            stage_num = self._current_stage

        x, y, vx, vy, m = state
        r = np.array([x, y])
        r_mag = np.linalg.norm(r)
        altitude = r_mag - R_earth

        # Unit vectors
        radial_unit = r / r_mag  # points away from Earth's centre
        tangent_unit = np.array([-radial_unit[1], radial_unit[0]])  # 90° CCW from radial

        # Get pitch and throttle from ascent profile
        if control:
            pitch, throttle = control
        else:
            pitch, throttle = self.ascent_profile(t, state)

        # Convert pitch relative to tangent into Cartesian coordinates
        thrust_dir = -np.cos(pitch) * tangent_unit + np.sin(pitch) * radial_unit

        # Get thrust and mass flow rate
        thrust, m_dot = throttle * self._get_thrust_and_mass_flow(t, state, stage_num=stage_num)

        # Calculate gravity
        g = self._gravity(altitude)
        g_vector = -g * radial_unit

        drag_vector = self._get_drag(state, stage_num = stage_num)  # modify _get_drag to accept arrays if needed

        # Total acceleration


        try:
            a = (thrust * thrust_dir + drag_vector) / m + g_vector

        except:
            print("thrust, direction, drag", thrust, thrust_dir, drag_vector)
            raise

        vx, vy = np.squeeze(vx), np.squeeze(vy)
        ax, ay = np.squeeze(a[0]), np.squeeze(a[1])
        m_dot_val = np.squeeze(m_dot)

        dstate_dt = np.array([vx, vy, ax, ay, -m_dot_val])

        return dstate_dt

    def simulate(self, *args, causal = True, **kwargs):
        if causal:
            return self.causal_simulate(*args, **kwargs)

        else:
            return self.hamiltonian_simulate(*args, **kwargs)

    def causal_simulate(self, t_span: Tuple[float, float],
                 max_step: float = 1.0, **solver_kwargs) -> Dict:
        """
        Simulate the rocket's ascent.

        Args:
            t_span: Time span for simulation (start, end) in seconds
            max_step: Maximum step size for the solver (s)
            **solver_kwargs: Additional arguments to pass to solve_ivp

        Returns:
            Dictionary containing simulation results
        """

        self._current_stage = 0
        self._current_stage_start_time = 0.0

        t0, tf = t_span
        times = []
        states = []
        stage_numbers = []

        initial_state = np.array([
            0.0,  # x (m)
            R_earth,  # y (m), starting at Earth's surface
            0.0,  # vx (m/s)
            0.0,  # vy (m/s)
            sum([stage.dry_mass + stage.propellant_mass for stage in self.stages])  # Initial mass (kg)
        ])

        current_y0 = initial_state
        current_t0 = t0

        # Define event functions
        def crash_event(t, y):
            x, y_pos, vx, vy, m = y
            r = np.sqrt(x ** 2 + y_pos ** 2)
            altitude = r - R_earth
            return altitude

        crash_event.terminal = True
        crash_event.direction = -1

        while current_t0 < tf:
            stage_sep_event = self.stages[self._current_stage].stage_criteria
            stage_sep_event.terminal = True

            sol = solve_ivp(
                self._equations_of_motion,
                (current_t0, tf),
                current_y0,
                method='DOP853',
                max_step=max_step,
                events=[stage_sep_event, crash_event],
                **solver_kwargs
            )

            # Append results
            times.append(sol.t)
            states.append(sol.y)
            stage_numbers.append(np.full(sol.t.shape, self._current_stage))

            # Check which event triggered
            stage_triggered = len(sol.t_events[0]) > 0
            crash_triggered = len(sol.t_events[1]) > 0

            if crash_triggered:
                # Stop simulation completely
                break

            elif stage_triggered:
                # Increment stage and restart
                self._current_stage += 1
                self._current_stage_start_time = sol.t[-1]
                current_y0 = sol.y[:, -1]
                current_y0[4] = sum(
                    [stage.dry_mass + stage.propellant_mass for stage in self.stages[self._current_stage:]])
                current_t0 = sol.t[-1]

            else:
                # No events; normal completion
                break

        # Concatenate results
        times = np.concatenate(times)
        states = np.hstack(states)
        stage_numbers = np.concatenate(stage_numbers)

        results = {
            't': times,
            'x': states[0],
            'y': states[1],
            'vx': states[2],
            'vy': states[3],
            'm': states[4],
            'stage': stage_numbers,
            'rocket': self,
            'success': True,
            'message': 'Simulation completed'
        }

        # Calculate additional quantities
        r = np.sqrt(results['x'] ** 2 + results['y'] ** 2)
        v = np.sqrt(results['vx'] ** 2 + results['vy'] ** 2)
        altitude = r - R_earth

        # Angle of position vector relative to +x axis
        theta = np.arctan2(results['y'], results['x'])

        # Starting angle (launch point: x=0, y=R_earth)
        theta0 = np.arctan2(R_earth, 0)  # = pi/2
        angular_displacement = theta - theta0

        # Unwrap to keep it continuous
        angular_displacement = np.unwrap(angular_displacement)

        # Arc length = angular displacement * planet radius
        horizontal_distance = R_earth * angular_displacement

        # Flight path angle relative to local horizontal (tangent at r)
        flight_path_angle = np.arctan2(results['vx'] * results['y'] - results['vy'] * results['x'],
                                       results['vx'] * results['x'] + results['vy'] * results['y'])

        results.update({
            'altitude': altitude,
            'velocity': v,
            'horizontal_distance': horizontal_distance,
            'flight_path_angle': flight_path_angle,
        })

        return results


    def _altitude(self, state):
        """Return altitude above Earth's surface in metres."""
        R_EARTH = 6.371e6
        x, y = state[0], state[1]
        r = np.sqrt(x**2 + y**2)
        return r - R_EARTH

    def _radius_vector(self, state):
        """Return distance from Earth's centre."""
        x, y = state[0], state[1]
        return np.sqrt(x**2 + y**2)

    def _speed(self, state):
        """Return scalar velocity magnitude."""
        vx, vy = state[2], state[3]
        return np.sqrt(vx**2 + vy**2)

    def _specific_energy(self, state):
        """Return specific orbital energy (J/kg)."""
        r = self._radius_vector(state)
        v = self._speed(state)
        return 0.5 * v**2 - mu / r

    def _semi_major_axis(self, state):
        """Return orbital semi-major axis in metres."""
        energy = self._specific_energy(state)
        energy = np.where(energy==0, 1e-12, energy)
        SMA = -mu / (2 * energy)
        SMA = np.atleast_1d(SMA)
        SMA[energy==0] = np.inf
        if SMA.size == 1:
            SMA = SMA[0]
        return SMA

    def _angular_momentum(self, state, verbose = False):
        """Return scalar orbital angular momentum per unit mass (m^2/s)."""
        x, y, vx, vy = state[0], state[1], state[2], state[3]

        return x * vy - y * vx

    def _eccentricity(self, state, verbose = False):
        """Return orbital eccentricity (0=circle, 1=parabolic)."""
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        h = self._angular_momentum(state, verbose = verbose)

        r = np.sqrt(x ** 2 + y ** 2)
        v2 = vx ** 2 + vy ** 2
        rv = x * vx + y * vy

        # Eccentricity vector components
        ex = (v2 - mu / r) * x / mu - rv * vx / mu
        ey = (v2 - mu / r) * y / mu - rv * vy / mu

        e = np.sqrt(ex ** 2 + ey ** 2)

        if verbose:
            print(f"Eccentricity: {e}")
            print(f"Eccentricity vector: {ex}, {ey}")
            print(f"Angular momentum: {h}")
            print(f"Radius: {r}")
            print(f"Velocity: {v2}")
            print(f"Radial velocity: {rv}")
            print(f"Position vector: {x}, {y}")
            print(f"Velocity vector: {vx}, {vy}")


        return e

    def _orbital_period(self, state):
        """Return orbital period in seconds (assuming closed orbit)."""
        a = self._semi_major_axis(state)
        a = np.atleast_1d(a)
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        period[a <= 0] = np.inf
        if period.size == 1:
            period = period[0]
        return period

    def _perigee_apogee(self, state):
        """Return perigee and apogee altitudes above surface (m)."""
        a = self._semi_major_axis(state)
        e = self._eccentricity(state)
        rp = a * (1 - e)
        ra = a * (1 + e)
        return rp, ra

    def get_current_stage(self, state: np.ndarray) -> np.ndarray:
        """
        Determine current stage index from rocket mass, vectorised.

        Args:
            mass: scalar or 1D array of rocket masses
            stages: list of RocketStage objects in launch order

        Returns:
            np.ndarray of stage indices (or int if scalar input)
        """
        mass = np.atleast_1d(state[4])
        n_stages = len(self.stages)

        # Precompute stage mass limits using a running total of upper stages
        stage_mass_limits = np.zeros(n_stages)
        upper_mass = 0.0
        for i in reversed(range(n_stages)):
            stage_mass_limits[i] = self.stages[i].dry_mass + upper_mass
            upper_mass += self.stages[i].dry_mass + self.stages[i].propellant_mass

        # Vectorised assignment: find first stage whose limit >= current mass
        stage_num = np.zeros_like(mass, dtype=int)
        for i in range(n_stages):
            mask = mass <= stage_mass_limits[i]
            stage_num[mask] = i

        if stage_num.size == 1:
            return int(stage_num[0])

        return stage_num

    def _jacobian_eom(self, states: np.ndarray, t: float, eps: float = 1e-6) -> np.ndarray:
        """
        Numerically compute the Jacobian of _equations_of_motion w.r.t state.

        Supports vectorised state input: shape (5, N)
        Returns Jacobian of shape (5, 5, N)
        """
        n = states.shape[0]  # number of state variables (5)
        batch_size = states.shape[1] if states.ndim > 1 else 1
        jac = np.zeros((n, n, batch_size))

        # Ensure 2D for consistent broadcasting
        if states.ndim == 1:
            states = states[:, None]

        f0 = self._equations_of_motion(t, states)  # shape (5, N)

        perturbed_states = states.copy()
        for i in range(n):
            perturbed_states[i, :] += eps
            fi = self._equations_of_motion(t, perturbed_states)
            jac[:, i, :] = (fi - f0) / eps
            perturbed_states[i, :] -= eps

            # If original input was 1D, squeeze the batch dimension
        if batch_size == 1:
            return jac[:, :, 0]
        return jac

    def _costates_ode(self, t: float, lam: np.ndarray, states: np.ndarray, stage_num=None) -> np.ndarray:
        """
        Compute costate derivatives analytically (vectorised).

        lam: shape (5, N)
        states: shape (5, N)
        Returns dlam/dt of shape (5, N)
        """
        J = self._jacobian_eom(states, t)  # shape (5,5,N) if N>1
        if J.ndim == 2:
            # single state
            dlam_dt = -J.T @ lam
        else:
            # batch: multiply each 5x5 slice by corresponding lam column
            dlam_dt = -np.einsum('ijk,jk->ik', J, lam)
        return dlam_dt

    def _hamiltonian(self, t: float, y: np.ndarray, stage_num=None) -> np.ndarray:
        """
        y: 10xN array where first 5 are states, last 5 are costates
        Returns dy/dt for BVP solver
        """
        states = y[:5, :]
        lam = y[5:, :]

        # Compute optimal controls vectorised
        pitch, throttle = self._optimal_control_from_costates(states, lam, stage_num = stage_num)

        # Compute state derivatives vectorised
        dstate_dt = self._equations_of_motion(t, states, control=(pitch, throttle), stage_num = stage_num)

        # Compute costate derivatives vectorised
        dlam_dt = self._costates_ode(t, lam, states, stage_num)

        dydt = np.vstack([dstate_dt, dlam_dt])

        return dydt

    def _optimal_control_from_costates(self, states: np.ndarray, costates: np.ndarray, stage_num=None):
        """
        Compute optimal pitch and throttle for each column of costates.
        Returns pitches, throttles of shape (N,)
        """
        lam_vx = costates[2, :]
        lam_vy = costates[3, :]
        lam_m = costates[4, :]

        pitches = np.arctan2(lam_vy, lam_vx)
        throttles = np.where(lam_m < 0, 1.0, 0.0)  # bang-bang
        pitches = np.squeeze(pitches)
        throttles = np.squeeze(throttles)
        return pitches, throttles

    def hamiltonian_simulate(self, t_span: Tuple[float, float], max_step: float = 1.0, **solver_kwargs) -> Dict:
        """Full BVP ascent simulation using Hamiltonian optimisation."""
        # 1. Generate initial guess from casual simulation
        causal_results = self.causal_simulate(t_span, max_step=max_step)

        print("causal step finished. Moving to hamiltonian")

        t_guess = causal_results['t']

        t_guess = t_guess.copy()
        for i in range(1, len(t_guess)):
            if t_guess[i] <= t_guess[i - 1]:
                t_guess[i] = t_guess[i - 1] + 1e-12  # tiny increment

        state_guess = np.vstack([causal_results['x'],
                                 causal_results['y'],
                                 causal_results['vx'],
                                 causal_results['vy'],
                                 causal_results['m']])

        # 2. Initialize costates guess (zeros)
        lam_guess = np.zeros_like(state_guess) + 1e-6
        y_guess = np.vstack([state_guess, lam_guess])

        # 3. Define BCs
        def bc(ya, yb):
            # Initial state constraints
            bcs = [
                ya[0],  # x0 = 0
                ya[1] - R_earth,  # y0 = R_earth
                ya[2],  # vx0 = 0
                ya[3],  # vy0 = 0
                ya[4] - sum([s.dry_mass + s.propellant_mass for s in self.stages])  # m0
            ]

            # Final orbit target (fixed)
            target_alt = 200e3
            r_final = R_earth + target_alt
            # Ideally, constrain final radius and circular orbit velocity
            v_circ = np.sqrt(mu / r_final)
            bcs.append(yb[0] ** 2 + yb[1] ** 2 - r_final ** 2)  # final radius squared
            bcs.append(np.sqrt(yb[2] ** 2 + yb[3] ** 2) - v_circ)  # circular orbit speed

            # Costate BCs
            bcs.append(yb[5 + 4] + 1.0)  # λ_m(t_f) = -1  (maximize remaining mass)
            # other costates can be free or zero
            bcs.extend([0] * 2)  # λ_x, λ_y, λ_vx, λ_vy free

            return np.array(bcs)

        # 4. Solve BVP
        sol = solve_bvp(lambda t, y: self._hamiltonian(t, y), bc, t_guess, y_guess, **solver_kwargs)

        if not sol.success:
            print("BVP solver failed:", sol.message)

        # 5. Construct results dictionary same as casual_simulate
        states = sol.y[:5]
        results = {
            't': sol.x,
            'x': states[0],
            'y': states[1],
            'vx': states[2],
            'vy': states[3],
            'm': states[4],
            'stage': self.get_current_stage(states),  # multi-stage logic can modify this
            'rocket': self,
            'success': sol.success,
            'message': sol.message
        }

        r = np.sqrt(results['x'] ** 2 + results['y'] ** 2)
        v = np.sqrt(results['vx'] ** 2 + results['vy'] ** 2)
        altitude = r - R_earth
        theta = np.arctan2(results['y'], results['x'])
        theta0 = np.arctan2(R_earth, 0)
        angular_displacement = np.unwrap(theta - theta0)
        horizontal_distance = R_earth * angular_displacement
        flight_path_angle = np.arctan2(results['vx'] * results['y'] - results['vy'] * results['x'],
                                       results['vx'] * results['x'] + results['vy'] * results['y'])

        results.update({
            'altitude': altitude,
            'velocity': v,
            'horizontal_distance': horizontal_distance,
            'flight_path_angle': flight_path_angle
        })
        return results
