from ppic2_wrapper import cppdsortp2yl
from dtypes import Int
import numpy


def particle_sort(particles, grid):
    """Sort particles according to their y-coordinate, out of place."""
    particles2 = numpy.empty_like(particles)
    npic = numpy.empty(grid.nypmx, Int)

    cppdsortp2yl(particles, particles2, npic, particles.np, grid)

    return particles2
