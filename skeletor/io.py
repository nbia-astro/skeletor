class IO:
    def __init__(self, data_folder, local_vars, experiment, tag=''):
        """
        Initialisation creates the output directory and saves the path to the
        io object.
        It also copies the experiment script to the data directory and
        saves a dictionary with important information.
        """
        from mpi4py.MPI import COMM_WORLD as comm

        if comm.rank == 0:
            import subprocess
            from os import path
            import pickle
            from datetime import datetime
            from numpy import float64

            # Create datafolder
            if data_folder[-1] != '/': data_folder += '/'
            subprocess.call('mkdir ' + data_folder, shell=True)

            # Copy the experiment to the data folder
            experiment = path.basename(experiment)
            subprocess.call('cp ' + experiment +' '+ data_folder+experiment, \
                             shell=True)
            info = {'experiment' : experiment}

            # Save git commit number
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            git_commit = git_commit.strip().decode('utf-8')
            info.update({'git_commit' : git_commit})

            # Save the hostname
            hostname = subprocess.check_output(['hostname'])
            hostname = hostname.strip().decode('utf-8')
            info.update({'hostname' : hostname})

            # Save the time at the start of simulation
            i = datetime.now()
            simulation_start = i.strftime('%d/%m/%Y at %H:%M:%S')
            info.update({'simulation_start' : simulation_start})

            # Add tag which can be used to group simulations together
            info.update({'tag' : tag})

            # Collect all variables in local namespace that are int, float or
            # float64. These will in general be the values set by us.
            for key in local_vars.keys():
                if type(local_vars[key]) in (float, float64, int):
                    info.update({key : local_vars[key]})

            # Save the number of MPI processes used
            info.update({'MPI' : comm.size})

            # Delete idproc which is not useful information
            del info['idproc']

            # Save the dictionary info to the file info.p
            pickle.dump(info, open(data_folder+'info.p', 'wb'))

        self.data_folder = data_folder
        self.snap = 0

    def set_outputrate(self, dt):
        self.dt = dt

    def concatenate(self, arr):
        """
        Concatenate local arrays to obtain global arrays.
        The result is available on all processors.
        """
        from numpy import concatenate
        from mpi4py.MPI import COMM_WORLD as comm
        return concatenate(comm.allgather(arr))

    def output_fields(self, sources, E, grid, t):
        """Output charge density and electric field"""

        from mpi4py.MPI import COMM_WORLD as comm
        from numpy import savez

        # Combine data from all processors
        global_rho = self.concatenate(sources.rho.trim())
        global_E   = self.concatenate(E.trim())

        # Let processor 0 write to file
        if comm.rank == 0:
            savez(self.data_folder + 'fields.{:04d}.npz'.format(self.snap),\
                  rho=global_rho, Ex=global_E['x'], Ey=global_E['y'], \
                  x=grid.x, y=grid.y, t = t)

        # Update snap shot number on all processors
        self.snap += 1
