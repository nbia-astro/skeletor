# Introduction

The skeleton codes are available
[here](http://picksc.idre.ucla.edu/software/skeleton-code/). As a warm up
excercise we are currently only concerned with the serial 2D electrostatic code
`pic2` and its MPI-parallized version `ppic2`.

# Installation

I assume you intend to run the code on OS X and that you're using Homebrew.

1. Make sure that you have Python 2 and 3 as well as Open MPI installed.

   ```
   brew install python python3 open-mpi
   ```

2. Install a tool that helps manage Python virtual environments.

   ```
   pip install virtualenvwrapper
   ```

3. Create a Python 3 virtual environment ldhs

   ```
   mkvirtualenv -p python3.5 skeletor
   ```

4. Go the directory where you've cloned this repository. Then install the
   required Python packages.

   ```
   pip install -r requirements.txt
   ```

5. Now you should have everything to build and run the code.

   ```
   make
   mpirun -np 4 python skeletor.py
   ```
