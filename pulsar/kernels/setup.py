from __future__ import annotations

import os
import subprocess

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

class BuildCUDA(build_ext):
    
    def run(self):
        package_name = "tk_mla"
        package_path = os.path.join(os.getcwd(), package_name)

        if os.path.exists(os.path.join(package_path, "Makefile")):
            print(f"Running `make` in {package_path}...")
            subprocess.check_call(["make"], cwd=package_path)
        else:
            print(f"No Makefile found in {package_path}, skipping CUDA build.")

        build_ext.run(self)

setup(
    name="tk_mla",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    cmdclass={"build_ext": BuildCUDA},
    ext_modules=[Extension("tk_mla.mla_decode", sources=[])],
    package_data={"tk_mla": ["*.so"]},
    install_requires=[],
    zip_safe=True,
)
