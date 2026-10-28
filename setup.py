"""Build the native plant as an ordinary Python extension."""
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

extensions = [Pybind11Extension(
    'aerowhip.physics._native', ['native/bindings.cpp', 'native/engine.cpp', 'native/runtime.cpp', 'native/actuator.cpp',
                                 'native/estimator.cpp', 'native/estimator_bindings.cpp'],
    depends=['native/estimator.hpp', 'native/material_frame.inc', 'native/prescribed_material.inc', 'native/engine.hpp', 'native/runtime.hpp', 'native/actuator.hpp', 'native/flight_controller.hpp', 'native/firmware.hpp'], cxx_std=17,
    extra_compile_args=(['/O2', '/fp:strict', '/openmp'] if sys.platform == 'win32'
                        else ['-O3', '-fno-fast-math', '-ffp-contract=off', '-fopenmp']),
    extra_link_args=([] if sys.platform == 'win32' else ['-fopenmp']),
)]
setup(
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
)
