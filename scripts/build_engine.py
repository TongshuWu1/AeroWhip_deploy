"""Run in PyCharm to compile/rebuild the C++ engine for this interpreter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Repository root: runs without installing.
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    try:
        import pybind11  # noqa: F401
    except ImportError:
        raise SystemExit('Install the project build dependencies (pybind11) in this Python environment first.')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=root, check=True)
    print('C++ engine built. Restart Python to load it.', flush=True)


if __name__ == '__main__':
    main()
