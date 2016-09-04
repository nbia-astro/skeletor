import subprocess

# Number of processors on machine
N = 2

def scaling(N, strong=False, mpifft=False):
    # Write header to file
    f = open('scaling.txt','w')
    f.write('#procs\tny\tTotal\tPush\tDeposit\tC_guard\tA_guard\tGauss\n')
    f.close()

    kmax = int(np.log2(N))

    for k in range(kmax+1):
        # Number of processors for this run
        j = 2**k

        # Grid points along y
        if strong:
            indy = 9
        else:
            indy = 9+k

        # Execution command
        command = "mpirun -np {} python -O ppic2.py {}".format(j, indy)

        # Add flag if mpiFFT is used
        if mpifft: command + ' -mpifft'

        # Run the code
        p = subprocess.call(command, shell=True)

    data = np.loadtxt('scaling.txt')
    n_procs = data[:,0]
    time    = data[:,2]
    #subprocess.call('rm scaling.txt', shell=True)

    return (n_procs, time)


import numpy as np
import matplotlib.pyplot as plt

# Weak scaling test
#plt.figure(1)
(n_procs, time) = scaling(N, strong=False, mpifft=False)
#plt.loglog(n_procs, time/n_procs, 'x-', label='PPIC2')
#(n_procs, time) = scaling(N, strong=False, mpifft=True)
#plt.loglog(n_procs, time/n_procs, 'x-', label='mpiFFT4py')
#plt.loglog(n_procs, time[0]/n_procs, 'k', label='Ideal')
#plt.legend(frameon=False)
#plt.xlabel('# processors')
#plt.ylabel('time/(# processors)')
#plt.savefig('weak_scaling.pdf')
