#!/bin/bash
cd /home/tony-lab/Documents/PHD/AeroWhip_deploy
D=runs/calibration_2026-10-27
.venv/bin/python scripts/compare_cable_v1.py --engine $D/engine_v1/_native.cpython-312-x86_64-linux-gnu.so --config $D/engine_v1/drone_cable_v1.json --output $D/cable_v1.json > runs/cable_v1.log 2>&1
.venv/bin/python scripts/fit_cable_flights.py --output $D/cable_fit_v2.json --start -7.1092 -3.4901 0.7259 -1.4115 --step .4 .4 .4 .4 --iterations 35 > runs/cable_fit_v2.log 2>&1
.venv/bin/python scripts/validate_coupled.py --drone-fit $D/drone_fit.json --cable-fit $D/cable_fit_v2.json --output $D/coupled_validation.json > runs/coupled_validation.log 2>&1
.venv/bin/python scripts/benchmark_batch.py --lanes 24 96 --seconds 2 > runs/benchmark_batch.log 2>&1
echo done > runs/overnight.done
