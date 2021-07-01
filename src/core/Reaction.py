import numpy as np
from .Specie import Specie

class Reaction:
    """Core object to hold information and methods on a single reaction.
    ARGS:
        - reactants (list of species): a list with reactant species
        - products (list of species): a list with product species"""

    def __init__(self, reactants=None, products=None, properties={}):
        self.reactants = reactants
        self.products = products
        self.properties = properties

    @staticmethod
    def from_ac_matrices(reactants, products):
        ajr = Reaction(None, None, {}) # must instanciate with explicit values (unclear why, probably some memory managment bug)
        ajr.properties['r_ac_det'] = np.linalg.det(reactants.matrix)
        ajr.properties['p_ac_det'] = np.linalg.det(products.matrix)
        reactants_acs = reactants.get_compoenents()
        ajr.properties['r_num'] = len(reactants_acs)
        ajr.reactants = [Specie.from_ac_matrix(ac) for ac in reactants_acs]
        products_acs = products.get_compoenents()
        ajr.properties['p_num'] = len(products_acs)
        ajr.products = [Specie.from_ac_matrix(ac) for ac in products_acs]
        return ajr

    def __eq__(self, x):
        # if not equal check properties
        conditions = []
        keys = set(list(self.properties.keys()) + list(x.properties.keys()))
        for k in keys:
            if k in self.properties.keys() and k in x.properties.keys():
                conditions.append(self.properties[k] == x.properties[k])
        return all(conditions)