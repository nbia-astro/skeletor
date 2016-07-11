python:
	CC=mpicc python setup.py build_ext --inplace

cython_c=$(patsubst %.pyx, %.c, $(wildcard *.pyx))
cython_so=$(patsubst %.pyx, %.*.so, $(wildcard *.pyx))
clean:
	CC=mpicc python setup.py clean
	rm -f C.2 $(cython_c) $(cython_so)
