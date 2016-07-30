![alt text](https://github.com/nbia-astro/skeletor/raw/master/images/logo.png)

# Introduction

The skeleton codes are available
[here](http://picksc.idre.ucla.edu/software/skeleton-code/). At the moment we
are only using the 2D MPI-parallelized code `ppic2`.

# Installation

I assume you intend to run the code on OS X and that you're using Homebrew.

1. Make sure that you have Python 2 and 3 as well as Open MPI installed.

   ```
   $ brew install python python3 open-mpi
   ```

2. Install tool for managing virtual Python environments.

   ```
   $ pip install virtualenvwrapper
   ```

3. Create a Python 3 virtual environment.

   ```
   $ mkvirtualenv -p python3.5 skeletor
   ```

4. Go the directory where you've cloned this repository. Then install the
   required Python packages.

   ```
   $ pip install -r requirements.txt
   ```

5. Now you should have everything to build and run the code.

   ```
   $ make
   $ mpirun -np 4 python skeletor.py
   ```

# Running interactively on multiple processors

For details see `ipyparallel`'s
[documentation](https://ipyparallel.readthedocs.io/).

### First time

Make sure [`ipyparallel`](https://pypi.python.org/pypi/ipyparallel) and
[`jupyter`](https://pypi.python.org/pypi/jupyter) are installed on your
system/in your virtual environment. Then create a parallel `iPython` profile:

```
$ ipython profile create --parallel --profile=mpi
```

Edit the file `~/.ipython/profile_mpi/ipcluster_config.py` and add the line

```
c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'
```

### Every time

Start the cluser:

```
$ ipcluster start -n 4 --profile=mpi
```

Open another terminal and start the `Jupyter` notebook:

```
$ jupyter notebook
```
