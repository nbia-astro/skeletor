SHELL=bash

python:
	CC=mpicc python setup.py build_ext --inplace

clean:
	CC=mpicc python setup.py clean
	rm -rf C.2 skeletor/cython/*.{c,so,html}
	rm -rf skeletor/__pycache__ skeletor/cython/__pycache__
	rm -rf skeletor/manifolds/__pycache__ dispersion_solvers/__pycache__
