class State:

    def __init__(self, species, B, time=0.0):
        """
        State class.
        Inputs:
        species: tuple of particle objects (ions1, ions2, ions3) or
        a single particle object. In the latter case the particle object is
        still stored as a tuple.
        """
        if type(species) == tuple:
            self.species = species
        else:
            # Convert single particle array to a tuple with one element.
            self.species = (species,)
        self.B = B
        self.t = time
