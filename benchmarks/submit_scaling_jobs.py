import subprocess

# Number of processors on machine
N = 1024

def replace(file_path, pattern, subst):
   from tempfile import mkstemp
   from os import remove, close
   from shutil import move
   #Create temp file
   fh, abs_path = mkstemp()
   new_file = open(abs_path,'w')
   old_file = open(file_path)
   for line in old_file:
      new_file.write(line.replace(pattern, subst))
   #close temp file
   new_file.close()
   close(fh)
   old_file.close()
   #Remove original file
   remove(file_path)
   #Move new file
   move(abs_path, file_path)

def scaling(N, strong=False, mpifft=False):
    # Write header to file
    f = open('scaling.txt','w')
    f.write('#procs\ttime\tny\n')
    f.close()

    kmax = int(np.log2(N))

    for k in range(kmax+1):
        # Number of processors for this run
        j = 2**k

        # Grid points along y
        if strong:
            ny = 64*N
        else:
            ny = 64*j
        submitfile = 'submit'
        subprocess.call('cp submit_template.sh ' + submitfile, shell=True)
        # Execution command
        command = "mpirun python twostream.py {}".format(ny)
        # Add flag if mpiFFT is used
        if mpifft: command + ' -mpifft'

        replace(submitfile, 'COMMAND', command)
        replace(submitfile, 'NTASKS',  str(j))

        # Run the code
        subprocess.call('sbatch ' + submitfile, shell=True)

import numpy as np
scaling(N)
