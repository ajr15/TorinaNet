from click import Option
import numpy as np
from typing import List, Optional
from .Specie import Specie

class Reaction:
    """Core object to hold information and methods on a single reaction.
    ARGS:
        - reactants (List[Specie]): a list with reactant species
        - products (List[Specie]): a list with product species
        - properties (dict): a dictionary with property names (keys) and values (values"""

    def __init__(self, reactants: Optional[List[Specie]]=None, products: Optional[List[Specie]]=None, properties: Optional[dict]=None):
        if not reactants is None:
            self.reactants = reactants
        else:
            self.reactants = []
        if not products is None:
            self.products = products
        else:
            self.products = []
        if properties is None:
            self.properties = {}
        else:
            self.properties = properties
        self._id_properties = {
                "r_ac_det": round(np.prod([np.linalg.det(r.ac_matrix.matrix) for r in self.reactants])),
                "p_ac_det": round(np.prod([np.linalg.det(r.ac_matrix.matrix) for r in self.products])),
                "r_num": len(self.reactants),
                "p_num": len(self.products)
            }

    @staticmethod
    def from_ac_matrices(reactants, products):
        ajr = Reaction(None, None, {}) # must instanciate with explicit values (unclear why, probably some memory managment bug)
        # first reading products from joint ac matrix
        products_acs = products.get_compoenents()
        ajr.products = [Specie.from_ac_matrix(ac) for ac in products_acs]
        # adding reactants and making sure that all are consumed in the reaction (no reactions like A + B -> A + C)
        # this is done by checking if a reactants appears in the products
        prod_id_dict = {s._get_id_str(): i for i, s in enumerate(ajr.products)}
        reactants_acs = reactants.get_compoenents()
        ajr.reactants = []
        for ac in reactants_acs:
            s = Specie.from_ac_matrix(ac)
            sid = s._get_id_str()
            # if reactant doesn't appear in the products, add it to reactants
            if not sid in prod_id_dict:
                ajr.reactants.append(s)
            # if reactant appears in the products, don't add reactant and remove product
            else:
                idx = prod_id_dict[sid]
                ajr.products.pop(idx)
                # correcting idxs in the dictionary
                for k, v in prod_id_dict.items():
                    if v > idx:
                        prod_id_dict[k] = v - 1
        # calculating reaction's properties
        ajr._id_properties['r_num'] = len(ajr.reactants)
        ajr._id_properties['p_num'] = len(ajr.products)
        ajr._id_properties['r_ac_det'] = round(np.prod([s._id_properties["determinant"] for s in ajr.reactants]))
        ajr._id_properties['p_ac_det'] = round(np.prod([s._id_properties["determinant"] for s in ajr.products]))
        return ajr


    def _get_id_str(self):
        s = ''
        for k in ['r_ac_det', 'p_ac_det', 'r_num', 'p_num']:
            if k in self._id_properties.keys():
                s += "_" + k + "_" + str(self._id_properties[k])
            else:
                s += k + "_NONE"
        return s


    def _get_charged_id_str(self):
        s = self._get_id_str()
        return s + "_rcharges_{}_pcharges_{}".format(";".join([str(s.charge) for s in self.reactants]),
                                                        ";".join([str(s.charge) for s in self.products]))


    def has_total_charge(self):
        """Method to check if a reaction has a defined total charge"""
        return all([s.has_charge() for s in self.reactants])

    def calc_total_charge(self):
        """Method to calculate the total charge in a reaction given its species"""
        if self.has_total_charge():
            return sum([s.charge for s in self.reactants])
        else:
            raise RuntimeError("Cannot set reaction charge if species do not have charges")


    def __eq__(self, x):
        # if not equal check properties
        conditions = []
        keys = set(list(self._id_properties.keys()) + list(x._id_properties.keys()))
        for k in keys:
            if k in self._id_properties.keys() and k in x._id_properties.keys():
                conditions.append(self._id_properties[k] == x._id_properties[k])
        return all(conditions)