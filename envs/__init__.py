"""Batched training environments over the native coupled drone-cable plant.

An environment owns physics stepping, command delay, observation history and
numerical/physical validity. Rewards come from a separate Task (envs.task).
"""
