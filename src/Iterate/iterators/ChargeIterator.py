from copy import copy
from typing import List
from src.core.Reaction import Reaction
from src.core.Specie import Specie
from src.iterate.filters.charge_filters import MaxAbsCharge
from ...core.RxnGraph.BaseRxnGraph import BaseRxnGraph
from itertools import product


class ChargeIterator:

    """Iterator object to add charged specie reactions to a reaction graph"""

    def __init__(self, uncharged_rxn_graph: BaseRxnGraph, charged_rxn_grph_type: BaseRxnGraph):
        self.uncharged_rxn_graph = uncharged_rxn_graph
        self.charged_rxn_grph_type = charged_rxn_grph_type


    @staticmethod
    def possible_specie_charges_iterator(total_charge: int, n_products: int, max_abs_charge: int) -> tuple:
        """Generator for all possible reaction product charges given the total reaction charge, number of products
        and the maximal specie charge"""
        # TODO: currently running with brute force - find a better algorithm !
        # making the possible ranges of charges for each specie
        charge_ranges = [range(- max_abs_charge, max_abs_charge + 1) for _ in range(n_products)]
        for comb in product(*charge_ranges):
            if sum(comb) == total_charge:
                yield comb


    def enumerate_over_reaction(self, reaction: Reaction, reactant_charges: List[List[int]], charge_filters, max_abs_charge) -> Reaction:
        """Method to yeild all the charged reactions acheived from a single uncharged reaction.
        ARGS:
            - reaction (Reaction): the reaction to iterate on
            - reactant_charges (List[List[int]]): the possible charges of the reactants in the reaction 
        RETURNS:
            (Reaction) Generator of charged reactions"""
        reactant_charges = product(*reactant_charges)
        reactants = [copy(s) for s in reaction.reactants]
        products = [copy(s) for s in reaction.products]
        for charges in reactant_charges:
            # setting charges on reactants
            for c, s in zip(charges, reactants):
                s.charge = c
            # finding total charge of reaction
            total_charge = sum(charges)
            # iterates over all possible product charges
            for charges in self.possible_specie_charges_iterator(total_charge, len(products), max_abs_charge):
                charged_products = []
                good_reaction = True
                for p, charge in zip(products, charges):
                    charged = Specie.from_ac_matrix(p.ac_matrix)
                    charged.properties = p.properties
                    charged.charge = charge
                    if all([charge_filter.check(charged) for charge_filter in charge_filters]):
                        charged_products.append(charged)
                    else:
                        good_reaction = False
                        break
                if good_reaction:
                    yield Reaction(reactants=reactants, products=charged_products, properties=reaction.properties)
            

    def enumerate_over_redox_reactions(self, specie: Specie, possible_charges: List[int], 
                                        max_reduction: int, max_oxidation: int, charge_filters) -> Reaction:
        """Method to enumerate all the redox reactions of a given specie with list of possible charges.
        iterates reductions until max_reduction reductions (single electron) and max_oxidation oxidations (single electron).
        RETURNS an iterator of possible reactions"""
        # checking if specie has a charge
        for charge in possible_charges:
            # copying specie
            _specie = copy(specie)
            _specie.charge = charge
            for red_charge in range(1, max_reduction + 1):
                charged_specie = Specie.from_ac_matrix(_specie.ac_matrix)
                charged_specie.properties = _specie.properties
                charged_specie.charge = charge - red_charge
                if all([charge_filter.check(charged_specie) for charge_filter in charge_filters]):
                    yield Reaction([_specie], [charged_specie])
            for oxi_charge in range(1, max_oxidation + 1):
                charged_specie = Specie.from_ac_matrix(_specie.ac_matrix)
                charged_specie.properties = _specie.properties
                charged_specie.charge = charge + oxi_charge
                if all([charge_filter.check(charged_specie) for charge_filter in charge_filters]):
                    yield Reaction([_specie], [charged_specie])


    def enumerate_charges(self, max_reduction: int, max_oxidation: int, charge_filters, verbose=1) -> BaseRxnGraph:
        """Method to add charge information to uncharged reaction graph."""
        # initializing run
        charged_rxn_graph = self.charged_rxn_grph_type(use_charge=True)
        species_charges_dict = {self.uncharged_rxn_graph.make_unique_id(s): [s.charge] for s in self.uncharged_rxn_graph.reactant_species} # dict of specie_id -> list of possible charges
        visited_reactions = set()
        uncharged_G = self.uncharged_rxn_graph.to_networkx_graph(use_internal_id=True)
        new_species = [s for s in species_charges_dict.keys()]
        max_abs_charge = None
        for charge_filter in charge_filters:
            if type(charge_filter) is MaxAbsCharge:
                max_abs_charge = charge_filter.max_abs_charge
        if max_abs_charge is None:
            raise ValueError("Must include MaxAbsCharge in charge filter")
        ajr = []
        while True:
            # iterating over reactions with new species
            for scanned_specie_id in new_species:
                # adding all possible redox reactions with the species
                possible_charges = set(species_charges_dict[scanned_specie_id])
                specie = self.uncharged_rxn_graph.get_specie_from_id(scanned_specie_id)
                for redox_rxn in self.enumerate_over_redox_reactions(specie, species_charges_dict[scanned_specie_id], 
                                                                        max_reduction, max_oxidation, charge_filters):
                    charged_rxn_graph.add_reaction(redox_rxn)
                    charge = redox_rxn.products[0].charge
                    possible_charges.add(charge)
                species_charges_dict[scanned_specie_id] = list(possible_charges)
                # iterating over all new reactions envolving visited species (species with charges)
                for scanned_reaction_id in uncharged_G.successors(scanned_specie_id):
                    if not scanned_reaction_id in visited_reactions \
                        and all([s in species_charges_dict for s in uncharged_G.predecessors(scanned_reaction_id)]):
                        # making new reaction visited
                        visited_reactions.add(scanned_reaction_id)
                        # reading reaction from uncharged graph
                        reaction = self.uncharged_rxn_graph.get_reaction_from_id(scanned_reaction_id)
                        reactant_charges = [species_charges_dict[self.uncharged_rxn_graph.make_unique_id(s)] for s in reaction.reactants]
                        # iterating over all charge combinations of reactants 
                        for rxn in self.enumerate_over_reaction(reaction, reactant_charges, charge_filters, max_abs_charge):
                            # adding reaction
                            charged_rxn_graph.add_reaction(rxn)
                            # adding products to new species
                            for s in rxn.products:
                                sid = self.uncharged_rxn_graph.make_unique_id(s)
                                if not sid in species_charges_dict:
                                    species_charges_dict[sid] = [s.charge]
                                    ajr.append(sid)
                                else:
                                    if not s.charge in species_charges_dict[sid]:
                                        species_charges_dict[sid].append(s.charge)
                # setting new species for next iteration
                new_species = ajr
            # checking stopping condition - all reactions are visited
            if verbose >= 1:
                print("Visited {} out of {} ({:.2f}%)".format(len(visited_reactions), 
                                                                self.uncharged_rxn_graph.get_n_reactions(), 
                                                                len(visited_reactions) / self.uncharged_rxn_graph.get_n_reactions() * 100))
            if len(visited_reactions) == self.uncharged_rxn_graph.get_n_reactions():
                return charged_rxn_graph