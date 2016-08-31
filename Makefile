SHELL=bash

python:
	CC=mpicc python setup.py build_ext --inplace

clean:
	CC=mpicc python setup.py clean
	rm -rf C.2 **/__pycache__ skeletor/cython/*.{c,so,html}
