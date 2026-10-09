"""Run in PyCharm to compile/rebuild the C++ engine for this interpreter."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    try:
        import pybind11  # noqa: F401
    except ImportError:
        raise SystemExit('Install the project build dependencies (pybind11) in this Python environment first.')
    env = os.environ.copy()
    if importlib.util.find_spec('torch') is not None:
        env['AEROWHIP_BUILD_TENSOR'] = '1'
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=root, env=env, check=True)
    print('C++ engine built'+(' (including differentiable tensor physics)' if env.get('AEROWHIP_BUILD_TENSOR') == '1' else
                             ' (install fitting dependencies to build tensor physics)')+'. Restart Python to load it.', flush=True)


if __name__ == '__main__':
    main()
