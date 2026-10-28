"""AeroWhip: aerial cable whipping with a quadrotor.

Subpackages, from the physical model upward:
    physics       coupled drone-cable plant (C++ engine bindings, model files, states)
    control       the flight pipeline for the simulators (host law, onboard firmware) and gamepad input
    trajectories  reference trajectory formats and generators
    envs          batched training environment and the Task interface
    tasks         reward definitions (registered by name)
    learning      PPO, run snapshots and the training worker
    evaluation    whip measurements and policy tests on saved motion
    takes         recorded calibration flights: loading, frame clock, cable-marker labels
    fitting       physical-parameter fitting on short segments of the calibration flights
    deployment    ROS 2 / Crazyswarm adapters and OptiTrack cable tracking
    viewer        3D rendering of simulated, replayed and live states
    ui            desktop applications (training, policy tester, model fit, cable labels)
"""
