from .cython.ppic2_wrapper import cppdsortp2yl
import numpy as np


class ParticleSort:

    def __init__(self, grid):

        from .cython.types import Int

        self.grid = grid
        self.npic = np.empty(grid.nypmx, Int)

    def __call__(self, particles, particles2):

        cppdsortp2yl(
                particles, particles2, self.npic, particles.N, self.grid)

        particles2.N = particles.N
