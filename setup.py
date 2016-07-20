from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
from distutils.extension import Extension
from numpy import get_include

Options.annotate = False
Options.directive_defaults["boundscheck"] = True
Options.directive_defaults["cdivision"] = False

extensions = [Extension("*", ["*.pyx"], include_dirs=[get_include()])]
setup(ext_modules=cythonize(extensions))
