# -*- coding: utf-8 -*-
from setuptools import setup, Extension
import pybind11
import sys

#  python setup.py build_ext --inplace

#  pybind11 
pybind_include = pybind11.get_include()

# 
extra_compile_args = []
extra_link_args = []  # 

if sys.platform == "win32":
    # Windows  MSVC compiler
    extra_compile_args = ["/std:c++17", "/openmp", "/O2"]  # 
else:
    # Linux/Unix or macOS  GCC / Clang
    extra_compile_args = ["-std=c++17", "-fopenmp", "-O3"]  # 
    extra_link_args = ["-fopenmp"]

# C++ 
ext_modules = [
    Extension(
        "represent_data",  #
        ["src/select_represent_data.cpp"],  # 
        include_dirs=[pybind_include, "include","/home/LAB/xxx/source/include"],  # 
        language="c++",
        extra_compile_args=extra_compile_args,  # 
        extra_link_args=extra_link_args,  # 
    ),
    Extension(
        "new_dataframe_module",  # 
        ["src/new_dataframe.cpp"],  # 
        include_dirs=[pybind_include, "include", "/home/LAB/zhangll24/source/include"], # 
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

# 
setup(
    name="represent_data_and_features",
    description="A Pybind11 wrapper for data sampling and feature generation",
    ext_modules=ext_modules,
    install_requires=["pybind11"],  #
    zip_safe=False,
)
