from ctypes cimport float_t, complex_t
from numpy import dtype

Int = dtype("i{}".format(sizeof(int)))
Float = dtype("f{}".format(sizeof(float_t)))
Complex = dtype("c{}".format(sizeof(complex_t)))

Float2 = [('x', Float), ('y', Float)]
Complex2 = [('x', Complex), ('y', Complex)]

Particle = dtype(
        [('x', Float), ('y', Float), ('vx', Float), ('vy', Float)],
        align=True)
