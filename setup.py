from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
from distutils.extension import Extension
from numpy import get_include

Options.annotate = False
compiler_directives = {"boundscheck": False, "cdivision": True,
                       "wraparound": False}
cflags = ["-Wno-unused-function", "-Wno-#warnings", "-fopenmp"]

extensions = [Extension(
    "*", ["skeletor/cython/*.pyx"],
    include_dirs=[get_include()], extra_compile_args=cflags,
    extra_link_args=["-fopenmp"],
    depends=["picksc/ppic2/precision.h"])]

setup(
    name='skeletor',
    version='0.0.1',
    packages=['skeletor', 'skeletor.manifolds'],
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives))
