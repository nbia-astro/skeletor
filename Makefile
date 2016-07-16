python:
	CC=mpicc python setup.py build_ext --inplace

clean:
	CC=mpicc python setup.py clean
	rm -rf C.2 __pycache__

cython_sources=$(patsubst %.pyx, %.c, $(wildcard *.pyx))
cython_modules=$(patsubst %.pyx, %.*.so, $(wildcard *.pyx))
cython_clean:
	rm -f $(cython_sources) $(cython_modules)
