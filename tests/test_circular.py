from skeletor import Float, Float3, Field, Particles
from skeletor import Ohm, InitialCondition, State
from skeletor.manifolds.second_order import Manifold
from skeletor.time_steppers.horowitz import TimeStepper
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from dispersion_solvers import HallDispersion


def test_circular(plot=False):

    # Quiet start
    quiet = True
    # Number of grid points in x- and y-direction
    nx, ny = 32, 8
    # Grid size in x- and y-direction (square cells!)
    Lx = 16
    Ly = 1
    dx = Lx/nx
    # Average number of particles per cell
    npc = 1
    # Particle charge and mass
    charge = 0.001
    mass = 1.0
    # Electron temperature
    Te = 0.0
    # Dimensionless amplitude of perturbation
    A = 0.005
    # Wavenumbers
    ikx = 1
    iky = 0
    # Number of periods to run for
    nperiods = 1.0

    # Sound speed
    cs = np.sqrt(Te/mass)

    # Total number of particles in simulation
    N = npc*nx*ny

    # Wave vector and its modulus
    kx = 2*np.pi*ikx/Lx
    ky = 2*np.pi*iky/Ly
    k = np.sqrt(kx*kx + ky*ky)

    # Angle of k-vector with respect to x-axis
    theta = np.arctan(iky/ikx) if ikx != 0 else np.pi/2

    # Magnetic field strength
    B0 = 1

    (Bx, By, Bz) = (B0*np.cos(theta), B0*np.sin(theta), 0)

    rho0 = 1.0

    va = B0

    # Ohmic resistivity
    eta = 0

    # Cyclotron frequency
    oc = charge*B0/mass

    # Hall parameter
    etaH = va**2/oc

    di = HallDispersion(kperp=0, kpar=k, va=va, cs=cs, etaH=etaH, eta=eta,
                        along_x=True, theta=theta)

    # Mode number
    m = 0

    omega = di.omega[m].real

    def frequency(kzva):
        hel = 1
        return kzva*(np.sqrt(1.0 + (0.5*kzva/oc)**2) + 0.5*kzva/(hel*oc))

    def get_dt(kzva):
        dt = 1/frequency(kzva)
        dt = 2.0**(np.floor(np.log2(dt)))
        return dt

    kmax = np.pi/dx

    # Simulation time
    tend = 2*np.pi*nperiods/omega

    # Phase factor
    def phase(x, y, t):
        return A*np.exp(1j*(di.omega[m]*t - kx*x - ky*y))

    # Linear solutions in real space
    def rho_an(x, y, t):
        return rho0 + rho0*(di.vec[m]['drho']*phase(x, y, t)).real

    def Bx_an(x, y, t):
        return Bx + B0*(di.vec[m]['bx']*phase(x, y, t)).real

    def By_an(x, y, t):
        return By + B0*(di.vec[m]['by']*phase(x, y, t)).real

    def Bz_an(x, y, t):
        return Bz + B0*(di.vec[m]['bz']*phase(x, y, t)).real

    def Ux_an(x, y, t):
        return (di.vec[m]['vx']*phase(x, y, t)).real

    def Uy_an(x, y, t):
        return (di.vec[m]['vy']*phase(x, y, t)).real

    def Uz_an(x, y, t):
        return (di.vec[m]['vz']*phase(x, y, t)).real

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    manifold = Manifold(nx, ny, comm, Lx=Lx, Ly=Ly)

    # Time step
    # dt = cfl*manifold.dx/vph
    dt = get_dt(kmax*va)

    # Number of time steps
    nt = int(tend/dt)

    # x- and y-grid
    xg, yg = np.meshgrid(manifold.x, manifold.y)

    # Maximum number of electrons in each partition
    Nmax = int(1.5*N/comm.size)

    # Create particle array
    ions = Particles(manifold, Nmax, charge=charge, mass=mass)

    # Create a uniform density field
    init = InitialCondition(npc, quiet=quiet)
    init(manifold, ions)

    # Particle position in physical units
    x = ions['x']*manifold.dx
    y = ions['y']*manifold.dy

    # Perturbation to particle velocities
    ions['vx'] = Ux_an(x, y, t=0)
    ions['vy'] = Uy_an(x, y, t=0)
    ions['vz'] = Uz_an(x, y, t=0)

    def B_an(t):
        B_an = Field(manifold, dtype=Float3)
        B_an['x'].active = Bx_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t)
        B_an['y'].active = By_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t)
        B_an['z'].active = Bz_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t)
        return B_an

    # Create vector potential
    A_an = Field(manifold, dtype=Float3)
    A_an['x'].active = 0
    A_an['y'].active = -((Bz + B0*(di.vec[m]['bz']*phase(xg, yg, 0))) /
                         (1j*kx)).real
    A_an['z'].active = ((By + B0*(di.vec[m]['by']*phase(xg, yg, 0))) /
                        (1j*kx)).real
    A_an.copy_guards()

    # Set initial magnetic field perturbation using the vector potential
    B = Field(manifold, dtype=Float3)
    manifold.curl(A_an, B, down=False)
    # Add background magnetic field
    B['x'].active += Bx
    B['y'].active += By
    B['z'].active += Bz
    B.copy_guards()

    div = Field(manifold, dtype=Float)
    div.fill(0)
    manifold.divergence(B, div)

    # Initialize Ohm's law solver
    ohm = Ohm(manifold, temperature=Te, charge=charge, eta=eta)

    # Initialize state
    state = State(ions, B)

    # Initialize timestepper
    e = TimeStepper(state, ohm, manifold)

    # Deposit charges and calculate initial electric field
    e.prepare(dt)

    # Concatenate local arrays to obtain global arrays
    # The result is available on all processors.
    def concatenate(arr):
        return np.concatenate(comm.allgather(arr))

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.cbook import mplDeprecation
        import warnings

        global_B = concatenate(e.B.trim())
        global_B_an = concatenate((B_an(t=0)).trim())
        global_div = concatenate(div.trim())

        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest')
            plt.figure(1)
            plt.clf()
            fig, axes = plt.subplots(num=1, ncols=3, nrows=2)
            vmin, vmax = charge*(1 - A), charge*(1 + A)
            im1 = axes[0, 0].imshow(global_div, vmin=vmin, vmax=vmax)
            im2 = axes[0, 1].plot(xg[0, :], global_B['z'][0, :], 'b',
                                  xg[0, :], global_B_an['z'][0, :], 'k--')
            im3 = axes[0, 2].plot(xg[0, :], global_B['y'][0, :], 'b',
                                  xg[0, :], global_B_an['y'][0, :], 'k--')
            im4 = axes[1, 0].imshow(global_B['x'], vmin=vmin, vmax=vmax)
            im5 = axes[1, 1].imshow(global_B['y'], vmin=-A, vmax=A)
            im6 = axes[1, 2].imshow(global_B['z'], vmin=-A, vmax=A)
            axes[0, 0].set_title(r'$\nabla \cdot \mathbf{B}$')
            axes[0, 1].set_title(r'$B_z$')
            axes[0, 2].set_title(r'$B_y$')
            axes[1, 0].set_title(r'$B_x$')
            axes[1, 1].set_title(r'$B_y$')
            axes[1, 2].set_title(r'$B_z$')
            axes[0, 2].set_ylim(-A, A)
            axes[0, 1].set_ylim(-A, A)
            axes[0, 2].set_xlim(0, Lx)

    diff2 = 0
    ##########################################################################
    # Main loop over time                                                    #
    ##########################################################################
    for it in range(nt):

        # The update is handled by the experiment class
        e.iterate(dt)
        manifold.divergence(e.B, div)
        divBmean = np.sqrt((div.trim()**2).mean())
        comm.allreduce(divBmean, op=MPI.SUM)

        # Difference between numerical and analytic solution
        local_B = e.B.trim()
        local_B_an = B_an(e.t).trim()
        for dim in ('x', 'y', 'z'):
            diff2 += ((local_B_an['y'] - local_B['y'])**2).mean()

        # Make figures
        if plot:
            if (it % 100 == 0):
                global_B = concatenate(e.B.trim())
                global_B_an = concatenate((B_an(e.t)).trim())
                global_div = concatenate(div.trim())
                if comm.rank == 0:
                    print("div B", divBmean)
                    im1.set_data(global_div)
                    im2[0].set_ydata(global_B['z'][0, :])
                    im2[1].set_ydata(global_B_an['z'][0, :])
                    im3[0].set_ydata(global_B['y'][0, :])
                    im3[1].set_ydata(global_B_an['y'][0, :])
                    im4.set_data(global_B['x'])
                    im5.set_data(global_B['y'])
                    im6.set_data(global_B['z'])
                    im1.autoscale()

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                        plt.pause(1e-7)

    val = np.sqrt(comm.allreduce(diff2, op=MPI.SUM)/nt)
    tol = 5e-4
    # Check if test has passed
    assert (val < tol), (val, tol)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_circular(plot=args.plot)
