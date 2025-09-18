""""Unit my_testing for the rocket simulator core functionality."""

import numpy as np
#import pytest
from rocketSimulator import RocketStage, RocketAscentSimulator, plot_results

def test_rocket_stage_initialization():
    """Test that a RocketStage can be initialized with the correct attributes."""
    def thrust_func(t, state):
        return 1000.0
    
    def isp_func(t, state):
        return 300.0
    
    def criteria_func(t, state):
        return 1.0
    
    def drag_func(alpha, mach):
        return 0.1
    
    stage = RocketStage(
        dry_mass=1000.0,
        propellant_mass=9000.0,
        thrust=thrust_func,
        isp=isp_func,
        stage_criteria=criteria_func,
        drag_coeff=drag_func,
        lift_coeff=drag_func,  # Same function for simplicity
        reference_area=1.0
    )
    
    assert stage.dry_mass == 1000.0
    assert stage.propellant_mass == 9000.0
    assert stage.thrust(0, None) == 1000.0
    assert stage.isp(0, None) == 300.0
    assert stage.stage_criteria(0, None) == 1.0
    assert stage.drag_coeff(0, 0) == 0.1
    assert stage.reference_area == 1.0

def test_simulator_initialization():
    """Test that the simulator can be initialized with stages."""
    def dummy_func(*args):
        return 1.0
    
    stage = RocketStage(
        dry_mass=1000.0,
        propellant_mass=9000.0,
        thrust=dummy_func,
        isp=dummy_func,
        stage_criteria=dummy_func,
        drag_coeff=dummy_func,
        lift_coeff=dummy_func,
        reference_area=1.0
    )
    
    def ascent_profile(self, t, state):
        return np.pi/4, 1.0  # 45° pitch, full throttle
    
    simulator = RocketAscentSimulator([stage], ascent_profile)
    
    assert len(simulator.stages) == 1
    assert simulator._current_stage == 0
    assert simulator._current_stage_start_time == 0.0

def test_gravity_calculation():
    """Test that gravity is calculated correctly at different altitudes."""
    from rocketSimulator.core import RocketAscentSimulator
    
    # Create a dummy simulator with no stages (we won't actually run it)
    simulator = RocketAscentSimulator([], lambda t, s: (0, 0))
    
    # Test at sea level
    g_sea = simulator._gravity(0)
    assert np.isclose(g_sea, 9.8, rtol=0.1)  # Should be close to 9.8 m/s²
    
    # Test at 100km altitude
    g_100km = simulator._gravity(100000)  # 100 km
    assert g_100km < g_sea  # Should be less than sea level
    assert np.isclose(g_100km, 9.5, rtol=0.1)  # Should be about 9.5 m/s² at 100km


def test_falcon9_ascent():
    """Vectorised Falcon 9-like ascent test for reasonable final values."""

    R_EARTH = 6.371e6  # m
    g0 = 9.80665       # m/s²

    # Stage thrust/ISP depending on pressure
    def thrust_func(t, state):
        x, y = state[0], state[1]
        alt = np.sqrt(x**2 + y**2) - R_EARTH
        _, p = simulator._atmosphere_model(alt)

        T_vac = 7.6e6
        T_sl = 7.6e6 * 0.9
        thrust = T_vac + (T_sl - T_vac) * (p / 101325)
        return thrust if thrust.size > 1 else thrust[0]

    def isp_func(t, state):
        x, y = state[0], state[1]
        alt = np.sqrt(x**2 + y**2) - R_EARTH
        _, p = simulator._atmosphere_model(alt)

        Isp_vac = 340
        Isp_sl = 311
        isp = Isp_vac + (Isp_sl - Isp_vac) * (p / 101325)
        return isp if isp.size > 1 else isp[0]

    def stage_criteria(t, state):
        return 1.0

    def drag_coeff(state):
        state = np.atleast_2d(state)
        return 0.3

    stage1 = RocketStage(
        dry_mass=25600,
        propellant_mass=395700,
        thrust=thrust_func,
        isp=isp_func,
        stage_criteria=stage_criteria,
        drag_coeff=drag_coeff,
        lift_coeff=lambda state: 0.0,
        reference_area=10.0
    )

    def ascent_profile(self, t, state):
        x, y = state[0], state[1]
        alt = np.sqrt(x**2 + y**2) - R_EARTH

        pitch = np.where(
            alt < 1000,
            np.pi/2,
            np.where(
                alt < 30000,
                np.pi/2 - (alt - 1000)/(30000 - 1000)*(np.pi/2 - np.radians(80)),
                np.radians(80)
            )
        )
        return np.squeeze(pitch), np.squeeze(np.ones_like(pitch))

    global simulator
    simulator = RocketAscentSimulator([stage1], ascent_profile)

    results = simulator.simulate(t_span=(0, 180), max_step=1.0)

    assert results['altitude'][-1] > 50000
    assert results['velocity'][-1] > 1500

    plot_results(results)

def test_falcon9_orbital_ascent_casual():
    """Vectorised Falcon 9 with second stage and payload deployment."""

    R_EARTH = 6.371e6
    g0 = 9.80665

    # First stage
    def thrust1(t, state):
        alt = np.sqrt(state[0]**2 + state[1]**2) - R_EARTH
        _, p = simulator._atmosphere_model(alt)
        T_vac = 7.6e6
        T_sl = 7.6e6 * 0.9
        thrust = T_vac + (T_sl - T_vac)*(p/101325)
        return np.squeeze(thrust)

    def isp1(t, state):
        alt = np.sqrt(state[0]**2 + state[1]**2) - R_EARTH
        _, p = simulator._atmosphere_model(alt)
        Isp_vac = 340
        Isp_sl = 311
        isp = Isp_vac + (Isp_sl - Isp_vac)*(p/101325)
        return np.squeeze(isp)

    def stage1_criteria(t, state):
        # Total mass when first stage is empty: dry mass of first stage + remaining upper stages + payload
        empty_mass_stage1 = stage1.dry_mass + stage2.dry_mass + stage2.propellant_mass + payload.dry_mass
        return state[4] - empty_mass_stage1  # triggers when mass <= empty_mass_stage1

    stage1 = RocketStage(
        dry_mass=25600,
        propellant_mass=395700,
        thrust=thrust1,
        isp=isp1,
        stage_criteria=stage1_criteria,  # burn until empty
        drag_coeff=lambda s: 0.3,
        lift_coeff=lambda s: 0.0,
        reference_area=10.0
    )

    # Second stage
    def thrust2(t, state):
        alt = np.sqrt(state[0]**2 + state[1]**2) - R_EARTH
        _, p = simulator._atmosphere_model(alt)
        T_vac = 9.0e5
        T_sl = 9.0e5 * 0.95
        thrust = T_vac + (T_sl - T_vac)*(p/101325)
        return np.squeeze(thrust)

    def isp2(t, state):
        alt = np.sqrt(state[0]**2 + state[1]**2) - R_EARTH
        _, p = simulator._atmosphere_model(alt)
        Isp_vac = 348
        Isp_sl = 320
        isp = Isp_vac + (Isp_sl - Isp_vac)*(p/101325)
        return np.squeeze(isp)

    def stage2_criteria(t, state):
        # Total mass when first stage is empty: dry mass of first stage + remaining upper stages + payload
        empty_mass_stage2 = stage2.dry_mass + payload.dry_mass
        return state[4] - empty_mass_stage2  # triggers when mass <= empty_mass_stage1

    stage2 = RocketStage(
        dry_mass=4000,
        propellant_mass=107500,
        thrust=thrust2,
        isp=isp2,
        stage_criteria=stage2_criteria,
        drag_coeff=lambda s: 0.25,
        lift_coeff=lambda s: 0.0,
        reference_area=3.66
    )

    # Payload stage (coast only)
    payload = RocketStage(
        dry_mass=21000,
        propellant_mass=0,
        thrust=lambda t, s: 0.0,
        isp=lambda t, s: 0.0,
        stage_criteria=lambda t, s: 1,
        drag_coeff=lambda s: 0.0,
        lift_coeff=lambda s: 0.0,
        reference_area=1.0
    )
    """
    # Ascent profile with pitch program
    def ascent_profile(self, t, state):
        x, y, vx, vy = state[0], state[1], state[2], state[3]

        # Altitude above Earth's surface
        r = np.sqrt(x ** 2 + y ** 2)
        alt = r - R_EARTH


        # Target circular orbit
        target_alt = 400e3  # 400 km
        target_r = R_EARTH + target_alt

        # Current speed
        v = self._speed(state)

        # Current perigee/apogee
        _, r_ap = np.array(self._perigee_apogee(state)) #+ R_EARTH

        # Simple circularisation: start pitching horizontal as apogee approaches target
        pitch = np.where(
            alt < 1000,
            np.pi/2,  # vertical launch
            np.where(
                r_ap < target_r,
                (1-(r_ap / target_r)) * np.pi / 2,  # linear gravity turn
                0,  # horizontal once at target orbit
            )
        )

        pitch = np.clip(pitch, 0, np.pi / 2)
        throttle = np.ones_like(pitch)  # full throttle

        return np.squeeze(pitch), np.squeeze(throttle)
    """

    def ascent_profile1(self, t, state):
        """
        Vectorised ascent profile that attempts a robust circularisation at target_alt
        by commanding pitch so thrust produces the radial acceleration needed to:
          - damp any downward radial velocity (prevent descent after apo)
          - correct apogee error (bring r_ap toward target_r)

        Returns (pitch, throttle) where pitch is angle relative to local tangent:
          0 = thrust along local prograde (tangent),
          +pi/2 = thrust fully radial-out.
        """

        # PARAMETERS you can tune
        target_alt = 100e3  # desired circular orbit altitude above surface (m)
        target_r = R_EARTH + target_alt  # radius from Earth's centre
        tau_pos = 100.0  # characteristic time (s) to correct apogee error
        kd_radial = 1  # damping gain on radial velocity (s^-1)
        max_upwards_pitch = np.radians(90)  # don't pitch more than this
        min_upwards_pitch = np.radians(0)
        vertical_until = 100
        troposphere_ascent = 10e3
        level_asymptote = 100e3

        # Make sure state is array-shaped: shape (N, 5)
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        m = state[4]  # total mass

        # geometry & kinematics
        r = np.sqrt(x * x + y * y)  # radius from Earth's centre
        alt = r - R_EARTH
        # radial velocity = (r_vec . v_vec) / r
        radial_v = (x * vx + y * vy) / np.maximum(r, 1e-12)

        # current orbital apogee/perigee (helpers may accept vector inputs)
        # _perigee_apogee should return perigee_radius, apogee_radius (both absolute radii)
        r_pe, r_ap = self._perigee_apogee(state)
        # some helpers might return altitudes above surface; adjust if necessary.
        # we assume _perigee_apogee returns absolute radii (same convention as _radius_vector).

        # compute required radial accel a_req (m/s^2)
        #  - position correction term: accelerate proportional to apogee error over tau_pos^2
        #    (simple acceleration needed to move error in ~tau_pos seconds: a ~ 2*err/t^2)
        #  - damping term: reduce current inward velocity (make radial_v -> 0)
        err_r_ap = (target_r - r_ap)  # positive if apogee below target
        a_pos = 2.0 * err_r_ap / (tau_pos ** 2)  # m/s^2
        a_pos = np.maximum(a_pos, 0)
        a_damp = -kd_radial * radial_v  # if radial_v < 0 (falling) -> positive accel
        a_damp = np.where(err_r_ap < 0, a_damp, np.maximum(a_damp, 0))
        a_req = a_pos + a_damp

        # gravity (radial, positive outward)
        g = self._gravity(alt)

        # available thrust (call existing helper). Returns array [thrust, m_dot] or thrust-only if thrust_only True.
        # Use thrust_only to get thrust values; if function returns scalar, np.atleast_1d will unify shape.
        thrust = self._get_thrust_and_mass_flow(t, state, thrust_only=True)
        thrust = np.atleast_1d(thrust).astype(float)

        # Avoid divide by zero: mass must be >0; clamp
        mass = np.maximum(m, 1e-6)

        # a_thrust is magnitude of acceleration we *could* produce if we pointed fully along thrust direction:
        a_thrust = thrust / mass  # m/s^2 (vector)

        # avoid division by zero for r
        safe_r = np.maximum(r, 1e-12)

        h = x * vy - y * vx  # specific angular momentum (m^2/s)
        a_cent = (h * h) / (safe_r ** 3)  # h^2 / r^3

        # Compute the required sine(pitch) from: net_radial_acc = sin(pitch)*a_thrust - g  => sin(pitch) = (a_req + g)/a_thrust
        # When a_thrust is very small, fall back to small pitch (can't produce radial accel).
        # We'll guard with np.clip and safe division.
        safe_a_thrust = np.maximum(a_thrust, 1e-12)
        sin_pitch = (a_req) / safe_a_thrust

        # clamp to [-1, 1] for asin domain; if sin_pitch>1 means even full radial thrust isn't enough
        sin_pitch_clamped = np.clip(sin_pitch, -1.0, 1.0)

        # pitch relative to tangent; asin returns [-pi/2, pi/2]
        pitch_cmd = np.arcsin(sin_pitch_clamped)

        # We only want upward (positive) pitches in this controller (don't command negative pitch here)
        # and bound it to [min_upwards_pitch, max_upwards_pitch]
        pitch_cmd = np.clip(pitch_cmd, min_upwards_pitch, max_upwards_pitch)

        # Extra safety: if target apogee not yet reached and we have enough thrust, favour prograde (small pitch)
        # If err_r_ap >> 0 (apogee way under target), allow pitch to smoothly reduce toward prograde to increase tangential accel.
        # Blend between gravity turn behaviour and this controller:
        # compute a 'prograde bias' factor (0..1); when apogee small relative to target, bias towards prograde:
        frac = np.clip(np.abs(err_r_ap) / np.maximum(target_r, 1.0), 0.0, 1.0)
        # if err_r_ap positive (need more apogee), reduce pitch a bit: scale pitch by (1-frac)

        pitch_cmd = np.where(
            alt < vertical_until, np.pi / 2,
            np.where(
                alt < troposphere_ascent,
                np.pi / 2 - (alt - vertical_until) / (troposphere_ascent - vertical_until) * (np.pi / 2 - np.radians(80)),
                pitch_cmd
            )
        )

        # Make throttle full unless a_thrust is extremely small; can be tuned later
        throttle = np.ones_like(pitch_cmd)

        # If vector had shape (1,5) and user expects scalars, squeeze
        pitch_out = np.squeeze(pitch_cmd)
        #print(np.degrees(pitch_out), alt)
        throttle_out = np.squeeze(throttle)

        return pitch_out, throttle_out

    def ascent_profile2(self, t, state):
        """
        Attempt at a prograde ascent profile. It sucks.

        Returns (pitch, throttle) where pitch is angle relative to local tangent:
          0 = thrust along local prograde (tangent),
          +pi/2 = thrust fully radial-out.
        """

        #Starting turn parameters
        turn_altitude = 100
        turn_angle = np.radians(89.992)

        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        m = state[4]  # total mass

        # Calculate angle of vx, vy direction
        v_mag = np.linalg.norm([vx, vy])
        v_dir = np.arctan2(vy, vx)

        # geometry & kinematics
        r = np.sqrt(x * x + y * y)  # radius from Earth's centre
        alt = r - R_EARTH

        pitch_cmd = np.where(alt < turn_altitude, turn_angle, v_dir)
        pitch_cmd = np.clip(pitch_cmd, 0, np.pi/2)

        throttle_cmd = np.ones_like(pitch_cmd)

        # If vector had shape (1,5) and user expects scalars, squeeze
        pitch_out = np.squeeze(pitch_cmd)
        # print(np.degrees(pitch_out), alt)
        throttle_out = np.squeeze(throttle_cmd)

        return pitch_out, throttle_out

    def ascent_profile3(self, t, state):
        """
        Parameterised ascent profile with respect to altitude

        Returns (pitch, throttle) where pitch is angle relative to local tangent:
          0 = thrust along local prograde (tangent),
          +pi/2 = thrust fully radial-out.
        """

        vertical_until = 0
        troposphere_ascent = 100
        level_asymptote = 90e3

        # Make sure state is array-shaped: shape (N, 5)
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        m = state[4]  # total mass

        # geometry & kinematics
        r = np.sqrt(x * x + y * y)  # radius from Earth's centre
        alt = r - R_EARTH

        pitch_cmd = np.where(
            alt < vertical_until, np.pi / 2,
            np.where(
                alt < troposphere_ascent,
                np.pi / 2 - (alt - vertical_until) / (troposphere_ascent - vertical_until) * (
                            np.pi / 2 - np.radians(80)),
                0
            )
        )

        pitch_cmd = np.where(
            alt < level_asymptote,
            np.where(alt > troposphere_ascent,
                     np.radians(80) - (alt - troposphere_ascent) / (level_asymptote - troposphere_ascent) * np.radians(
                         80), pitch_cmd),
            np.pi / 16  # pitch_cmd
        )

        # Make throttle full unless a_thrust is extremely small; can be tuned later
        throttle = np.ones_like(pitch_cmd)

        # If vector had shape (1,5) and user expects scalars, squeeze
        pitch_out = np.squeeze(pitch_cmd)
        # print(np.degrees(pitch_out), alt)
        throttle_out = np.squeeze(throttle)

        return pitch_out, throttle_out

    global simulator
    simulator = RocketAscentSimulator([stage1, stage2, payload], ascent_profile3)

    results = simulator.simulate(t_span=(0, 8000), max_step=1.0)

    plot_results(results)

    # Check orbital altitude (~200 km) and velocity (~7800 m/s)
    assert results['altitude'][-1] > 180000
    assert results['velocity'][-1] > 7500

def test_falcon9_orbital_ascent_hamiltonian():
    """Vectorised Falcon 9 with second stage and payload deployment."""

    R_EARTH = 6.371e6
    g0 = 9.80665

    # First stage
    def thrust1(t, state):
        alt = np.sqrt(state[0]**2 + state[1]**2) - R_EARTH
        _, p = simulator._atmosphere_model(alt)
        T_vac = 7.6e6
        T_sl = 7.6e6 * 0.9
        thrust = T_vac + (T_sl - T_vac)*(p/101325)
        return np.squeeze(thrust)

    def isp1(t, state):
        alt = np.sqrt(state[0]**2 + state[1]**2) - R_EARTH
        _, p = simulator._atmosphere_model(alt)
        Isp_vac = 340
        Isp_sl = 311
        isp = Isp_vac + (Isp_sl - Isp_vac)*(p/101325)
        return np.squeeze(isp)

    def stage1_criteria(t, state):
        # Total mass when first stage is empty: dry mass of first stage + remaining upper stages + payload
        empty_mass_stage1 = stage1.dry_mass + stage2.dry_mass + stage2.propellant_mass + payload.dry_mass
        return state[4] - empty_mass_stage1  # triggers when mass <= empty_mass_stage1

    stage1 = RocketStage(
        dry_mass=25600,
        propellant_mass=395700,
        thrust=thrust1,
        isp=isp1,
        stage_criteria=stage1_criteria,  # burn until empty
        drag_coeff=lambda s: 0.3,
        lift_coeff=lambda s: 0.0,
        reference_area=10.0
    )

    # Second stage
    def thrust2(t, state):
        alt = np.sqrt(state[0]**2 + state[1]**2) - R_EARTH
        _, p = simulator._atmosphere_model(alt)
        T_vac = 9.0e5
        T_sl = 9.0e5 * 0.95
        thrust = T_vac + (T_sl - T_vac)*(p/101325)
        return np.squeeze(thrust)

    def isp2(t, state):
        alt = np.sqrt(state[0]**2 + state[1]**2) - R_EARTH
        _, p = simulator._atmosphere_model(alt)
        Isp_vac = 348
        Isp_sl = 320
        isp = Isp_vac + (Isp_sl - Isp_vac)*(p/101325)
        return np.squeeze(isp)

    def stage2_criteria(t, state):
        # Total mass when first stage is empty: dry mass of first stage + remaining upper stages + payload
        empty_mass_stage2 = stage2.dry_mass + payload.dry_mass
        return state[4] - empty_mass_stage2  # triggers when mass <= empty_mass_stage1

    stage2 = RocketStage(
        dry_mass=4000,
        propellant_mass=107500,
        thrust=thrust2,
        isp=isp2,
        stage_criteria=stage2_criteria,
        drag_coeff=lambda s: 0.25,
        lift_coeff=lambda s: 0.0,
        reference_area=3.66
    )

    # Payload stage (coast only)
    payload = RocketStage(
        dry_mass=21000,
        propellant_mass=0,
        thrust=lambda t, s: 0.0,
        isp=lambda t, s: 0.0,
        stage_criteria=lambda t, s: 1,
        drag_coeff=lambda s: 0.0,
        lift_coeff=lambda s: 0.0,
        reference_area=1.0
    )
    """
    # Ascent profile with pitch program
    def ascent_profile(self, t, state):
        x, y, vx, vy = state[0], state[1], state[2], state[3]

        # Altitude above Earth's surface
        r = np.sqrt(x ** 2 + y ** 2)
        alt = r - R_EARTH


        # Target circular orbit
        target_alt = 400e3  # 400 km
        target_r = R_EARTH + target_alt

        # Current speed
        v = self._speed(state)

        # Current perigee/apogee
        _, r_ap = np.array(self._perigee_apogee(state)) #+ R_EARTH

        # Simple circularisation: start pitching horizontal as apogee approaches target
        pitch = np.where(
            alt < 1000,
            np.pi/2,  # vertical launch
            np.where(
                r_ap < target_r,
                (1-(r_ap / target_r)) * np.pi / 2,  # linear gravity turn
                0,  # horizontal once at target orbit
            )
        )

        pitch = np.clip(pitch, 0, np.pi / 2)
        throttle = np.ones_like(pitch)  # full throttle

        return np.squeeze(pitch), np.squeeze(throttle)
    """

    def ascent_profile1(self, t, state):
        """
        Vectorised ascent profile that attempts a robust circularisation at target_alt
        by commanding pitch so thrust produces the radial acceleration needed to:
          - damp any downward radial velocity (prevent descent after apo)
          - correct apogee error (bring r_ap toward target_r)

        Returns (pitch, throttle) where pitch is angle relative to local tangent:
          0 = thrust along local prograde (tangent),
          +pi/2 = thrust fully radial-out.
        """

        # PARAMETERS you can tune
        target_alt = 100e3  # desired circular orbit altitude above surface (m)
        target_r = R_EARTH + target_alt  # radius from Earth's centre
        tau_pos = 100.0  # characteristic time (s) to correct apogee error
        kd_radial = 1  # damping gain on radial velocity (s^-1)
        max_upwards_pitch = np.radians(90)  # don't pitch more than this
        min_upwards_pitch = np.radians(0)
        vertical_until = 100
        troposphere_ascent = 10e3
        level_asymptote = 100e3

        # Make sure state is array-shaped: shape (N, 5)
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        m = state[4]  # total mass

        # geometry & kinematics
        r = np.sqrt(x * x + y * y)  # radius from Earth's centre
        alt = r - R_EARTH
        # radial velocity = (r_vec . v_vec) / r
        radial_v = (x * vx + y * vy) / np.maximum(r, 1e-12)

        # current orbital apogee/perigee (helpers may accept vector inputs)
        # _perigee_apogee should return perigee_radius, apogee_radius (both absolute radii)
        r_pe, r_ap = self._perigee_apogee(state)
        # some helpers might return altitudes above surface; adjust if necessary.
        # we assume _perigee_apogee returns absolute radii (same convention as _radius_vector).

        # compute required radial accel a_req (m/s^2)
        #  - position correction term: accelerate proportional to apogee error over tau_pos^2
        #    (simple acceleration needed to move error in ~tau_pos seconds: a ~ 2*err/t^2)
        #  - damping term: reduce current inward velocity (make radial_v -> 0)
        err_r_ap = (target_r - r_ap)  # positive if apogee below target
        a_pos = 2.0 * err_r_ap / (tau_pos ** 2)  # m/s^2
        a_pos = np.maximum(a_pos, 0)
        a_damp = -kd_radial * radial_v  # if radial_v < 0 (falling) -> positive accel
        a_damp = np.where(err_r_ap < 0, a_damp, np.maximum(a_damp, 0))
        a_req = a_pos + a_damp

        # gravity (radial, positive outward)
        g = self._gravity(alt)

        # available thrust (call existing helper). Returns array [thrust, m_dot] or thrust-only if thrust_only True.
        # Use thrust_only to get thrust values; if function returns scalar, np.atleast_1d will unify shape.
        thrust = self._get_thrust_and_mass_flow(t, state, thrust_only=True)
        thrust = np.atleast_1d(thrust).astype(float)

        # Avoid divide by zero: mass must be >0; clamp
        mass = np.maximum(m, 1e-6)

        # a_thrust is magnitude of acceleration we *could* produce if we pointed fully along thrust direction:
        a_thrust = thrust / mass  # m/s^2 (vector)

        # avoid division by zero for r
        safe_r = np.maximum(r, 1e-12)

        h = x * vy - y * vx  # specific angular momentum (m^2/s)
        a_cent = (h * h) / (safe_r ** 3)  # h^2 / r^3

        # Compute the required sine(pitch) from: net_radial_acc = sin(pitch)*a_thrust - g  => sin(pitch) = (a_req + g)/a_thrust
        # When a_thrust is very small, fall back to small pitch (can't produce radial accel).
        # We'll guard with np.clip and safe division.
        safe_a_thrust = np.maximum(a_thrust, 1e-12)
        sin_pitch = (a_req) / safe_a_thrust

        # clamp to [-1, 1] for asin domain; if sin_pitch>1 means even full radial thrust isn't enough
        sin_pitch_clamped = np.clip(sin_pitch, -1.0, 1.0)

        # pitch relative to tangent; asin returns [-pi/2, pi/2]
        pitch_cmd = np.arcsin(sin_pitch_clamped)

        # We only want upward (positive) pitches in this controller (don't command negative pitch here)
        # and bound it to [min_upwards_pitch, max_upwards_pitch]
        pitch_cmd = np.clip(pitch_cmd, min_upwards_pitch, max_upwards_pitch)

        # Extra safety: if target apogee not yet reached and we have enough thrust, favour prograde (small pitch)
        # If err_r_ap >> 0 (apogee way under target), allow pitch to smoothly reduce toward prograde to increase tangential accel.
        # Blend between gravity turn behaviour and this controller:
        # compute a 'prograde bias' factor (0..1); when apogee small relative to target, bias towards prograde:
        frac = np.clip(np.abs(err_r_ap) / np.maximum(target_r, 1.0), 0.0, 1.0)
        # if err_r_ap positive (need more apogee), reduce pitch a bit: scale pitch by (1-frac)

        pitch_cmd = np.where(
            alt < vertical_until, np.pi / 2,
            np.where(
                alt < troposphere_ascent,
                np.pi / 2 - (alt - vertical_until) / (troposphere_ascent - vertical_until) * (np.pi / 2 - np.radians(80)),
                pitch_cmd
            )
        )

        # Make throttle full unless a_thrust is extremely small; can be tuned later
        throttle = np.ones_like(pitch_cmd)

        # If vector had shape (1,5) and user expects scalars, squeeze
        pitch_out = np.squeeze(pitch_cmd)
        #print(np.degrees(pitch_out), alt)
        throttle_out = np.squeeze(throttle)

        return pitch_out, throttle_out

    def ascent_profile2(self, t, state):
        """
        Attempt at a prograde ascent profile. It sucks.

        Returns (pitch, throttle) where pitch is angle relative to local tangent:
          0 = thrust along local prograde (tangent),
          +pi/2 = thrust fully radial-out.
        """

        #Starting turn parameters
        turn_altitude = 100
        turn_angle = np.radians(89.992)

        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        m = state[4]  # total mass

        # Calculate angle of vx, vy direction
        v_mag = np.linalg.norm([vx, vy])
        v_dir = np.arctan2(vy, vx)

        # geometry & kinematics
        r = np.sqrt(x * x + y * y)  # radius from Earth's centre
        alt = r - R_EARTH

        pitch_cmd = np.where(alt < turn_altitude, turn_angle, v_dir)
        pitch_cmd = np.clip(pitch_cmd, 0, np.pi/2)

        throttle_cmd = np.ones_like(pitch_cmd)

        # If vector had shape (1,5) and user expects scalars, squeeze
        pitch_out = np.squeeze(pitch_cmd)
        # print(np.degrees(pitch_out), alt)
        throttle_out = np.squeeze(throttle_cmd)

        return pitch_out, throttle_out

    def ascent_profile3(self, t, state):
        """
        Parameterised ascent profile with respect to altitude

        Returns (pitch, throttle) where pitch is angle relative to local tangent:
          0 = thrust along local prograde (tangent),
          +pi/2 = thrust fully radial-out.
        """

        vertical_until = 0
        troposphere_ascent = 100
        level_asymptote = 90e3

        # Make sure state is array-shaped: shape (N, 5)
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        m = state[4]  # total mass

        # geometry & kinematics
        r = np.sqrt(x * x + y * y)  # radius from Earth's centre
        alt = r - R_EARTH

        pitch_cmd = np.where(
            alt < vertical_until, np.pi / 2,
            np.where(
                alt < troposphere_ascent,
                np.pi / 2 - (alt - vertical_until) / (troposphere_ascent - vertical_until) * (
                            np.pi / 2 - np.radians(80)),
                0
            )
        )

        pitch_cmd = np.where(
            alt < level_asymptote,
            np.where(alt > troposphere_ascent,
                     np.radians(80) - (alt - troposphere_ascent) / (level_asymptote - troposphere_ascent) * np.radians(
                         80), pitch_cmd),
            np.pi / 16  # pitch_cmd
        )

        # Make throttle full unless a_thrust is extremely small; can be tuned later
        throttle = np.ones_like(pitch_cmd)

        # If vector had shape (1,5) and user expects scalars, squeeze
        pitch_out = np.squeeze(pitch_cmd)
        # print(np.degrees(pitch_out), alt)
        throttle_out = np.squeeze(throttle)

        return pitch_out, throttle_out

    global simulator
    simulator = RocketAscentSimulator([stage1, stage2, payload], ascent_profile3)

    results = simulator.simulate(t_span=(0, 8000), max_step=1.0, causal = False)

    plot_results(results)

    # Check orbital altitude (~200 km) and velocity (~7800 m/s)
    assert results['altitude'][-1] > 180000
    assert results['velocity'][-1] > 7500

def test_simple_rocket_hamiltonian():


    R_EARTH = 6.371e6
    g0 = 9.80665

    stage1 = RocketStage(
        dry_mass=25000,
        propellant_mass=100000,
        thrust=lambda t, y: 1e6,
        isp=lambda t,y: 1e6,
        stage_criteria=lambda t, y: 1,  # burn until empty
        drag_coeff=lambda s: 0.3,
        lift_coeff=lambda s: 0.0,
        reference_area=10.0
    )

    def ascent_profile3(self, t, state):
        """
        Parameterised ascent profile with respect to altitude

        Returns (pitch, throttle) where pitch is angle relative to local tangent:
          0 = thrust along local prograde (tangent),
          +pi/2 = thrust fully radial-out.
        """

        vertical_until = 0
        troposphere_ascent = 100
        level_asymptote = 90e3

        # Make sure state is array-shaped: shape (N, 5)
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        m = state[4]  # total mass

        # geometry & kinematics
        r = np.sqrt(x * x + y * y)  # radius from Earth's centre
        alt = r - R_EARTH

        pitch_cmd = np.where(
            alt < vertical_until, np.pi / 2,
            np.where(
                alt < troposphere_ascent,
                np.pi / 2 - (alt - vertical_until) / (troposphere_ascent - vertical_until) * (
                            np.pi / 2 - np.radians(80)),
                0
            )
        )

        pitch_cmd = np.where(
            alt < level_asymptote,
            np.where(alt > troposphere_ascent,
                     np.radians(80) - (alt - troposphere_ascent) / (level_asymptote - troposphere_ascent) * np.radians(
                         80), pitch_cmd),
            np.pi / 16  # pitch_cmd
        )

        # Make throttle full unless a_thrust is extremely small; can be tuned later
        throttle = np.ones_like(pitch_cmd)

        # If vector had shape (1,5) and user expects scalars, squeeze
        pitch_out = np.squeeze(pitch_cmd)
        # print(np.degrees(pitch_out), alt)
        throttle_out = np.squeeze(throttle)

        return pitch_out, throttle_out

    simulator = RocketAscentSimulator([stage1], ascent_profile3)

    results = simulator.simulate(t_span=(0, 8000), max_step=1.0, causal=False)

    plot_results(results)



# Add more my_testing as needed
#test_rocket_stage_initialization()
#test_simulator_initialization()
#test_gravity_calculation()
#test_falcon9_orbital_ascent_casual()
test_falcon9_orbital_ascent_hamiltonian()