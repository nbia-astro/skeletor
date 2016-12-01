def log(f):
    """Custom log function that works on the
        active cells of skeletor fields"""
    from numpy import log as numpy_log
    g = f.copy()
    g[f.grid.lby:f.grid.uby, f.grid.lbx:f.grid.ubx] = numpy_log(f.trim())
    return g
