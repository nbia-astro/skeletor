class IO:
    def __init__(self, data_folder, local_vars, experiment, tag=''):
        """
        Initialisation creates the output directory and saves the path to the
        io object.
        It also copies the experiment script to the data directory and
        saves a dictionary with important information.
        """
        from mpi4py.MPI import COMM_WORLD as comm
        from mpi4py.MPI import Wtime

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

            # Start a log file
            f = open('skeletor.log', 'w')
            # Write start date and time
            f.write('Simulation started on ' + simulation_start +'\n\n')
            # Write the contents of info.p for convenience
            f.write('Contents of info.p is printed below \n')
            for key in info.keys():
                f.write(key + ' = {} \n'.format(info[key]))
            f.write('\n\nEntering main simulation loop \n')
            f.close()

        self.data_folder = data_folder
        # Snap shot number
        self.snap = 0
        # Used for computing total runtime
        self.wt = Wtime()

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

    def log(self, it, t, dt):
        from mpi4py.MPI import COMM_WORLD as comm
        if comm.rank == 0:
            f = open('skeletor.log', 'a')
            f.write('step {0}\ttime {1}\tdt {2}\n'.format(it, t, dt))
            f.close()

    def finished(self):
        """Write elapsed time to log file and move the log file to the data
        directory"""
        from mpi4py.MPI import COMM_WORLD as comm
        from mpi4py.MPI import Wtime
        seconds = Wtime() - self.wt
        if comm.rank == 0:
            import subprocess
            from datetime import datetime
            import pickle

            # Add run time to info.p
            info = pickle.load(open(self.data_folder+'info.p', 'rb'))
            info.update({'seconds' : seconds})
            pickle.dump(info, open(self.data_folder+'info.p', 'wb'))

            # Time at end of simulation
            i = datetime.now()
            endtime = i.strftime('%d/%m/%Y at %H:%M:%S')

            f = open('skeletor.log', 'a')
            f.write('Simulation ended on '+endtime+'\n')
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)

            f.write('Time elapsed was {} days {} hours {} minutes {} seconds'\
                     .format(d, h, m, s))
            f.close()

            subprocess.call('mv skeletor.log '+ self.data_folder, shell=True)

