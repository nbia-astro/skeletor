def uniform_density(nx, ny, npc, quiet):
    """Return Uniform distribution of particle positions"""

    if quiet:
        # Quiet start
        from numpy import sqrt, arange, meshgrid
        sqrt_npc = int(sqrt(npc))
        assert (sqrt_npc**2 == npc), 'npc need to be the square of an integer'
        dx = dy = 1/sqrt_npc
        x, y = meshgrid(arange(0, nx, dx), arange(0, ny, dy))
        x = x.flatten()
        y = y.flatten()
    else:
        # Random positions
        np = nx*ny*npc
        x = nx*numpy.random.uniform(size=np).astype(Float)
        y = ny*numpy.random.uniform(size=np).astype(Float)

    return (x, y)

def velocity_perturbation(x, y, kx, ky, ampl_vx, ampl_vy, vtx, vty):
    from numpy import random, sin
    from skeletor import Float

    # Perturbation to particle velocities
    vx = ampl_vx*sin(kx*x+ky*y)
    vy = ampl_vy*sin(kx*x+ky*y)

    # Number of particles
    np = x.shape[0]

    # Add thermal velocity
    vx += vtx*random.normal(size=np).astype(Float)
    vy += vty*random.normal(size=np).astype(Float)

    return (vx, vy)
