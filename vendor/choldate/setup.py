import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                "choldate._choldate",
                ["choldate/_choldate.pyx"],
                include_dirs=[numpy.get_include()],
            )
        ],
        language_level=3,
    ),
)
