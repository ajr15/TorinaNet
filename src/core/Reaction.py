import numpy as np
from .Specie import Specie

class Reaction:
    """Core object to hold information and methods on a single reaction.
    ARGS:
        - reactants (list of species): a list with reactant species
        - products (list of species): a list with product species"""

    def __init__(self, reactants=None, products=None, properties=None):
        self.reactants = reactants
        self.products = products
        if properties is None:
            self.properties = {
                "r_ac_det": round(np.prod([np.linalg.det(r.ac_matrix.matrix) for r in reactants])),
                "p_ac_det": round(np.prod([np.linalg.det(r.ac_matrix.matrix) for r in products])),
                "r_num": len(reactants),
                "p_num": len(products)
            }
        else:
            self.properties = properties

    @staticmethod
    def from_ac_matrices(reactants, products):
        ajr = Reaction(None, None, {}) # must instanciate with explicit values (unclear why, probably some memory managment bug)
        ajr.properties['r_ac_det'] = round(np.linalg.det(reactants.matrix))
        ajr.properties['p_ac_det'] = round(np.linalg.det(products.matrix))
        reactants_acs = reactants.get_compoenents()
        ajr.properties['r_num'] = len(reactants_acs)
        ajr.reactants = [Specie.from_ac_matrix(ac) for ac in reactants_acs]
        products_acs = products.get_compoenents()
        ajr.properties['p_num'] = len(products_acs)
        ajr.products = [Specie.from_ac_matrix(ac) for ac in products_acs]
        return ajr

    def _get_id_str(self):
        s = ''
        for k in ['r_ac_det', 'p_ac_det', 'r_num', 'p_num']:
            if k in self.properties.keys():
                s += "_" + k + "_" + str(self.properties[k])
            else:
                s += k + "_NONE"
        return s

    def __eq__(self, x):
        # if not equal check properties
        conditions = []
        keys = set(list(self.properties.keys()) + list(x.properties.keys()))
        for k in keys:
            if k in self.properties.keys() and k in x.properties.keys():
                conditions.append(self.properties[k] == x.properties[k])
        return all(conditions)