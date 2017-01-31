SHELL=bash
LDFLAGS="-L/usr/local/opt/llvm/lib"
python:
	CC=mpicc LDFLAGS=$(LDFLAGS) python setup.py build_ext --inplace

clean:
	CC=mpicc python setup.py clean
	rm -rf C.2 skeletor/cython/*.{c,so,html}
	rm -rf skeletor/__pycache__ skeletor/cython/__pycache__
