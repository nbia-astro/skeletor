from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
from distutils.extension import Extension
from numpy import get_include

extensions = [Extension("*", ["*.pyx"], include_dirs=[get_include()])]
Options.annotate = True
setup(ext_modules=cythonize(extensions))
