"""Build the native plant as an ordinary Python extension."""
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import shutil
from pathlib import Path

extensions = [Pybind11Extension(
    'aerowhip.physics._native', ['native/bindings.cpp', 'native/engine.cpp', 'native/runtime.cpp', 'native/actuator.cpp'],
    depends=['native/material_frame.inc', 'native/prescribed_material.inc', 'native/engine.hpp', 'native/runtime.hpp', 'native/actuator.hpp'], cxx_std=17,
    extra_compile_args=(['/O2', '/fp:strict'] if sys.platform == 'win32'
                        else ['-O3', '-fno-fast-math', '-ffp-contract=off']),
)]
# The lightweight simulator can be built without LibTorch. Fitting explicitly
# opts into an extension linked to the installed Torch, not a second download.
if os.environ.get('AEROWHIP_BUILD_TENSOR') == '1':
    if sys.platform == 'win32' and shutil.which('cl') is None:
        # Match setuptools' own compiler discovery so Torch can inspect the
        # actual compiler too, including from an ordinary PyCharm process.
        from setuptools._distutils.ccompiler import new_compiler
        compiler = new_compiler(compiler='msvc')
        compiler.initialize()
        os.environ['PATH'] = str(Path(compiler.cc).parent)+os.pathsep+os.environ['PATH']
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    extensions.append(CppExtension(
        'aerowhip.physics._tensor_native', ['native/tensor_bindings.cpp', 'native/tensor_physics.cpp'],
        depends=['native/tensor_material.inc'],
        extra_compile_args=(['/O2', '/fp:strict', '/std:c++17'] if sys.platform == 'win32'
                            else ['-O3', '-std=c++17', '-fno-fast-math', '-ffp-contract=off']),
    ))
    build_ext = BuildExtension.with_options(use_ninja=False)

setup(
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
)
