from .cython.ppic2_wrapper import grid_t
from .domain import SubDomain


class Grid(SubDomain, grid_t):
    """
    This class defines a differentiable (sub-) manifold.
    """

    def __init__(self, nx, ny, comm, nlbx=0, nubx=2, nlby=0, nuby=1):

        # Initialize SubDomain class
        super().__init__(nx, ny, comm)

        # Ghost zone setup
        # nlbx, nlby = number of ghost zones at lower boundary in x, y
        # nubx, nuby = number of ghost zones at upper boundary in x, y
        self.nlbx = nlbx
        self.nlby = nlby
        self.nubx = nubx
        self.nuby = nuby

        # lbx, lby = first active index in x, y
        # ubx, uby = index of first ghost upper zone in x, y
        self.lbx = nlbx
        self.ubx = self.nx + self.nlbx
        self.lby = nlby
        self.uby = self.nyp + self.nlby

        # nypmx = size of particle partition, including guard cells, in y
        # nxpmx = size of particle partition, including guard cells, in x
        # nypmn = value of nyp
        self.nypmx = self.nyp + self.nlby + self.nuby
        self.nypmn = self.nyp
        self.nxpmx = self.nx + self.nlbx + self.nubx
