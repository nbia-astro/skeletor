def find_info_files(topdir):
    """
    Returns a list directories in which info.p files were found.
    """
    import os
    from os.path import join

    sims = []

    for root, dirs, files in os.walk(topdir):
        for name in files:
            if name == 'info.p':
                folder = root
                sims.append(folder)
    return sims

def read_info(folder):
    """
    Return dictionary from info.p in the input folder.
    """
    import pickle
    info = pickle.load(open(folder+'/info.p', 'rb'))
    info.update({'folder' : folder})
    return info

def create_database(topdir):
    """
    Returns a list of info.p dictionaries from simulations found in topdir.
    """
    infofiles = find_info_files(topdir)
    database = [read_info(file) for file in infofiles]

    return database

def find(lst, key, value):
    """
    Inputs: List of dictionaries from simulations (lst) and a key and a value.
    Output: The indices of the dictionary list for which lst[key] == value
    """
    from numpy import isclose
    index = []
    for i, dic in enumerate(lst):
        try:
            if type(value) == str:
                if dic[key] == value:
                    index.append(i)
            else:
                if isclose(dic[key], value):
                    index.append(i)
        except:
            # Key not found in all simulations
            pass
    assert(len(index) != 0), 'The simulation(s) requested were not found'
    return index

def get_subset(lst, key, value):
    """
    Inputs: List of dictionaries from simulations (lst) and a key and a value.
    Output: The dictionaries in which lst[key] == value
    """
    index = find(lst, key, value)
    subset = [lst[j] for j in index]
    if len(subset) == 0: return None
    return subset

def get_small_subset(lst, keys, values):
    """
    Input: List of dictionaries from simulations, a list of keys and a list of
    values.
    Output: The dictionaries in which lst[key] == value for all keys and
    values in the input lists.
    """
    for (key, value) in zip(keys, values):
        if len(keys) == 0:
            return lst
        lst = get_subset(lst, key, value)
    return lst

def get_sim(keys, values, topdir='.'):
    # Create a list of simulations dictionaries
    database = create_database(topdir)
    # Find simulation that fulfills requirements
    sim = get_small_subset(database, keys, values)
    # Check that there is only one such simulation
    assert(len(sim) == 1), 'More than one simulation found! Narrow the \
    requirements. Use get_small_subset manually to figure out if this is an \
    issue'
    return sim[0]

def load_fields(sim, snap):
    """
    Load snapshot number given the folder or the simulation dictionary.
    """
    from numpy import load

    if type(sim) == str: folder = sim
    if type(sim) == dict: folder = sim['folder']

    f = load(folder+'/fields.{:04d}.npz'.format(snap))

    return f

if __name__ == '__main__':
    # Create a list of simulation folders
    sims = find_info_files('.')

    # Create a list of simulations dictionaries
    database = create_database('.')

    # Consider a single dictionary and load the fields
    sim = read_info(sims[0])
    f = load_fields(sim, 20)

    # Consider a single folder and load the fields
    folder = sims[0]
    f = load_fields(folder, 20)

    # Show how a subset of a database can be selected
    # This will return a list of all simulations with the following properties
    # A tag can be a fast way of getting the simulations you want without
    # having to specity all the parameters
    keys = ('tag', 'ny', 'npc')
    values = ('test_oct1', 32, 256)
    subset = get_small_subset(database, keys, values)

    # Imagine that you run a parameter study where you vary nx and ny and keep
    # everything else fixed. You would then add the tag='param_study' in the
    # simulations
    keys = ('tag', 'ny', 'nx')
    sim = get_sim(keys, ('test_oct1', 32, 32))
    f = load_fields(sim, 20)
    sim = get_sim(keys, ('test_oct1', 64, 32))
    f = load_fields(sim, 20)

    # This way one can do things like
    ny_list = (32, 64)

    # List of simulations that we want to analyze
    sims = [get_sim(keys, ('test_oct1', ny, 32)) for ny in ny_list]

    import matplotlib.pyplot as plt

    # Loop over the simulations.
    for sim in sims:
        rho_max = []
        t   = []
        for i in range(20):
            f = load_fields(sim, i)
            rho_max.append(f['rho'].max())
            t.append(f['t'])
        plt.plot(t, rho_max, label=r'$n_y = {}$'.format(sim['ny']))
        plt.legend(frameon=False)
    plt.show()
