def log(f):
    """Custom log function that works on the
        active cells of skeletor fields"""
    from numpy import log as numpy_log
    g = f.copy()
    g[:-1, :-2] = numpy_log(f.trim())
    return g
