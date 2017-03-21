from mpi4py.MPI import COMM_WORLD as comm, SUM


class Diagnostics():

    def __init__(self, custom_averages=None):

        # Initialize dictionary
        self.ts = {}

        # Time
        self.ts['t'] = []

        # Kinetic energy of particles
        self.ts['Ekin'] = []

        # Field averages
        self.averages = {'Bx2': lambda state: state.B['x']**2,
                         'By2': lambda state: state.B['y']**2,
                         'Bz2': lambda state: state.B['z']**2}

        if custom_averages is not None:
            self.averages.update(custom_averages)

        for key in self.averages.keys():
            self.ts[key] = []

    def __call__(self, state):

        # Time
        self.ts['t'].append(state.t)

        # Kinetic energy of particles
        self.ts['Ekin'].append(self.kinetic_energy(state))

        # Volume averages
        for key in self.averages.keys():
            self.ts[key].append(self.average(self.averages[key](state)))

    def kinetic_energy(self, state):
        """Total kinetic energy of all ions"""
        ekin = state.ions.kinetic_energy()
        return comm.allreduce(ekin, op=SUM)

    def sum(self, f):
        """Sum a field over all threads"""
        return comm.allreduce(f.trim().sum(), op=SUM)

    def average(self, f):
        """Take the average of a field over all threads"""
        return self.sum(f)/comm.size/f.active.size
