import subprocess

# Number of processors
N = 2

for j in range(1, N+1):
    ny = 8*j
    command = "mpirun -np {} python twostream.py {}".format(j, ny)
    p = subprocess.call(command, shell=True)