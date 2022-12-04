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
        self._to_convension()

    def _to_convension(self):
        """Method to transform the reaction to a conventional one (makes _id_properties and sorts reactants/products)"""
        self.reactants = sorted(self.reactants, key=lambda s: s._get_charged_id_str())
        self.products = sorted(self.products, key=lambda s: s._get_charged_id_str())
        self._id_properties = {
            "r_dets": [s._id_properties["determinant"] for s in self.reactants],
            "r_atoms":[s._id_properties["number_of_atoms"] for s in self.reactants],
            "p_dets": [s._id_properties["determinant"] for s in self.products],
            "p_atoms": [s._id_properties["number_of_atoms"] for s in self.products]
        }

    @staticmethod
    def from_ac_matrices(reactants, products):
        ajr = Reaction(None, None, {}) # must instanciate with explicit values (unclear why, probably some memory managment bug)
        # first reading products from joint ac matrix
        products_acs = products.get_compoenents()
        products = [Specie.from_ac_matrix(ac) for ac in products_acs]
        prod_sids = [s._get_charged_id_str() for s in products]
        # adding reactants and making sure that all are consumed in the reaction (no reactions like A + B -> A + C)
        # this is done by checking if a reactants appears in the products
        reactants_acs = reactants.get_compoenents()
        reactants = [Specie.from_ac_matrix(ac) for ac in reactants_acs]
        # getting mutual species & adding reactants
        ajr.reactants = []
        mutual = set()
        for s in reactants:
            sid = s._get_charged_id_str()
            # if reactant appears in the products, add it to joined list - DO NOT ADD TO REACTION
            if sid in prod_sids:
                mutual.add(sid)
            # if reactant appears in the products, don't add reactant and remove product
            else:
                ajr.reactants.append(s)
        # adding products
        for s in products:
            sid = s._get_charged_id_str()
            # if reactant is not mutual with reactants - add to reaction
            if not sid in mutual:
                ajr.products.append(s)
        # making reaction conventional
        ajr._to_convension()
        return ajr

    def _get_id_str(self):
        r_strs = [s._get_id_str() for s in self.reactants]
        p_strs = [s._get_id_str() for s in self.products]
        return "{}={}".format("+".join(r_strs), "+".join(p_strs))

    def _get_charged_id_str(self):
        r_strs = [s._get_charged_id_str() for s in self.reactants]
        p_strs = [s._get_charged_id_str() for s in self.products]
        return "{}={}".format("+".join(r_strs), "+".join(p_strs))

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