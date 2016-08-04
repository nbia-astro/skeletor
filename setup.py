from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
from distutils.extension import Extension
from numpy import get_include

Options.annotate = False
Options.directive_defaults["boundscheck"] = True
Options.directive_defaults["cdivision"] = False
cflags = ["-Wno-unused-function", "-Wno-#warnings"]

extensions = [Extension(
    "*", ["skeletor/cython/*.pyx"],
    include_dirs=[get_include()], extra_compile_args=cflags)]
setup(
    name='skeletor',
    version='0.0.1',
    packages=['skeletor'],
    ext_modules=cythonize(extensions))
