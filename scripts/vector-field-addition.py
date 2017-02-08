from skeletor import Float, Float2
from numpy import ndarray, all as np_all


class MyField(ndarray):

    def __new__(cls, nx, ny, **kwds):

        obj = super().__new__(cls, shape=(nx, ny), **kwds)

        return obj

    def __add__(self, arr):
        """
        Overload addition operator so that it works for vector field in the
        form of structured arrays. The functionality is currently very limited.
        It only works if two structured arrays are added together. Adding a
        scalar array to a structured array does not work. Also, we have to
        implement the logic below for every arithmetic operation we need. That
        is not very desirable IMO.
        """

        assert self.dtype == arr.dtype

        # If the two arrays are scalars (i.e. Numpy arrays with a floating
        # point or integer dtype), then just delegate to ndarray's __add__
        # method.
        if arr.dtype.names is None:
            return super().__add__(arr)

        # If we are dealing with structured arrays (otherwise we would have
        # already returned), then create a structured array with the same shape
        # and dtype that can hold the result
        res = MyField(*arr.shape, dtype=arr.dtype)
        # Add the two arrays together component by component and return the
        # result
        for name in arr.dtype.names:
            res[name] = ndarray.__add__(self[name], arr[name])
        return res


nx, ny = 32, 64

A = MyField(nx, ny, dtype=Float)
B = MyField(nx, ny, dtype=Float)

A[...] = 1.
B[...] = 2.

C = A + B
assert np_all(C == 3.)

D = MyField(nx, ny, dtype=Float2)
E = MyField(nx, ny, dtype=Float2)
D['x'][...] = 3.
D['y'][...] = 4.
E['x'][...] = 5.
E['y'][...] = 6.

F = D + E
assert np_all(F['x'] == 8.)
assert np_all(F['y'] == 10.)
