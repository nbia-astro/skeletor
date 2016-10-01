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

