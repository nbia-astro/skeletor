from .cython.ppic2_wrapper import cppdsortp2yl


class ParticleSort:

    def __init__(self, grid):

        from .cython.types import Int
        from numpy import empty

        self.grid = grid
        self.npic = empty(grid.nypmx, Int)

    def __call__(self, particles, particles2):

        cppdsortp2yl(
                particles, particles2, self.npic, particles.np, self.grid)

        particles2.np = particles.np
