import numpy as np
from typing import List, Optional
from .Specie import Specie
from ..utils.TimeFunc import TimeFunc

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
        # self._to_convension()

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
    @TimeFunc
    def from_ac_matrices(reactants, products):
        ajr = Reaction(None, None, {}) # must instanciate with explicit values (unclear why, probably some memory managment bug)
        # first reading products from joint ac matrix
        products_acs = products.get_compoenents()
        products = [ac.to_specie() for ac in products_acs]
        # adding reactants and making sure that all are consumed in the reaction (no reactions like A + B -> A + C)
        # this is done by checking if a reactants appears in the products
        reactants_acs = reactants.get_compoenents()
        reactants = [ac.to_specie() for ac in reactants_acs]
        # getting the "net" reactants and products
        # transforming reactions of type A + B -> C + B to A -> C 
        ajr.reactants = []
        mutual = list()
        for s in reactants:
            # if a reactant appears in products, save it to a dedicated list
            # make sure to "count" right, i.e. in case of A + B + B -> C + B add B only once
            if products.count(s) > mutual.count(s):
                mutual.append(s)
            else:
                ajr.reactants.append(s)
        # adding products
        for s in products:
            # if product is in mututal list dot add to products
            if not s in mutual:
                ajr.products.append(s)
            # if it does, remove from list (to deal with cases like A + B -> C + B + B)
            else:
                mutual.remove(s)
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
        
    def energy(self):
        if not all("energy" in s.properties for s in self.reactants + self.products):
            raise RuntimeError("Cannot calculate reaction energy if some species do not have calcualted energies")
        return sum([s.properties["energy"] for s in self.products]) - sum([s.properties["energy"] for s in self.reactants])


    def pretty_string(self) -> str:
        """Write reaction in a 'pretty' conventional format"""
        return "{} = {}".format(" + ".join([str(s.identifier) for s in self.reactants]), " + ".join([str(s.identifier) for s in self.products]))

    def __eq__(self, x):
        if not type(x) is Reaction:
            # in case x is None, it is not equal to the reaction (do not have the same type)
            if x is None:
                return False
            else:
                raise ValueError("Cannot compare reactions to object with type {}".format(type(x)))
        if len(self.reactants) == len(x.reactants) and len(self.products) == len(x.products):
            return all([s in self.reactants for s in x.reactants]) and all([s in self.products for s in x.products]) and \
                    all([s in x.reactants for s in self.reactants]) and all([s in x.products for s in self.products])
        else:
            return False
