from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("fastops", ["src/fastops.cpp"], cxx_std=17),
]

setup(
    name="fastops",
    version="0.1.0",
    description="Fast C++ ops for robustness/uncertainty/offset diagnostics",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
