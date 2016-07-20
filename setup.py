from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
from distutils.extension import Extension
from numpy import get_include
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
        "--annotate", action="store_true",
        help="Annotate Cython code")
parser.add_argument(
        "--no-boundscheck", action="store_true",
        help="Instruct Cython not to check array bounds")
parser.add_argument(
        "--cdivision", action="store_true",
        help="Use C-division in Cython code")
known_args, unknown_args = parser.parse_known_args()

Options.annotate = known_args.annotate
Options.directive_defaults["boundscheck"] = not known_args.no_boundscheck
Options.directive_defaults["cdivision"] = known_args.cdivision

extensions = [Extension("*", ["*.pyx"], include_dirs=[get_include()])]
setup(ext_modules=cythonize(extensions))
