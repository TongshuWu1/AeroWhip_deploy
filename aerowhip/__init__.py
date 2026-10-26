"""AeroWhip: aerial cable whipping with a quadrotor.

Subpackages, from the physical model upward:
    physics       coupled drone-cable plant (C++ engine bindings, model files, states)
    control       geometric flight controller and gamepad input
    trajectories  reference trajectory formats and generators
    envs          batched training environment and the Task interface
    tasks         reward definitions (registered by name)
    learning      PPO, run snapshots and the training worker
    evaluation    whip measurements and policy tests on saved motion
    takes         recorded flight takes: loading, checks, clock alignment
    fitting       physical-parameter and residual fitting from takes
    deployment    ROS 2 / Crazyswarm adapters and OptiTrack cable tracking
    viewer        3D rendering of simulated, replayed and live states
    ui            desktop applications (training, policy tester, take processing)
"""
