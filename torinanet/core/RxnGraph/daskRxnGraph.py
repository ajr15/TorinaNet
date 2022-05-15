from enum import unique
from dask import dataframe as dd
import dask as da
import pandas as pd
from typing import List
from .BaseRxnGraph import BaseRxnGraph
from ..AcMatrix.BinaryAcMatrix import BinaryAcMatrix
from ..Reaction import Reaction
from ..Specie import Specie

class daskRxnGraph (BaseRxnGraph):

    def __init__(self, species_npartitions=1, reactions_npartitions=1) -> None:
        super().__init__(BinaryAcMatrix)
        self._specie_nparts = species_npartitions
        self._rxn_nparts = reactions_npartitions
        self._species_df = dd.from_pandas(pd.DataFrame(columns=["idx", "ac_str", "visited"]), npartitions=species_npartitions)
        self._reactions_df = dd.from_pandas(pd.DataFrame(columns=["r_indicis", "p_indicis"]), npartitions=reactions_npartitions)


    @property
    def reactions(self) -> List[Reaction]:
        return getattr(self, "reactions")


    @property
    def species(self) -> List[Specie]:
        return getattr(self, "species")


    @reactions.getter
    def reactions(self) -> List[Reaction]:
        return da.compute([self.get_reaction_from_id(id_str) for id_str in self._reactions_df.index])[0]


    @species.getter
    def species(self) -> List[Specie]:
        return da.compute([self.get_specie_from_id(id_str) for id_str in self._species_df.index])[0]


    def add_specie(self, specie: Specie) -> Specie:
        s = self._read_specie_with_ac_matrix(specie)
        specie_id = s._get_id_str()
        if (self._species_df.index == specie_id).any():
            return self.get_specie_from_id(specie_id)
        else:
            visited = False if not "visited" in s.properties else s.properties["visited"]
            specie_dd = dd.from_pandas(pd.DataFrame({"idx": da.compute(len(self._species_df)),
                                                        "ac_str": s.ac_matrix._to_str(),
                                                        "visited": visited},
                                                        index=[specie_id]), npartitions=1)
            self._species_df = dd.concat([self._species_df, specie_dd])
            return s


    def add_reaction(self, reaction: Reaction) -> None:
        rxn_id = reaction._get_id_str()
        if (self._reactions_df.index != rxn_id).all():
            r_idxs = []
            for sp in reaction.reactants:
                self.add_specie(sp)
                r_idxs.append(self._species_df.loc[sp._get_id_str(), "idx"].values)
            p_idxs = []
            for sp in reaction.products:
                self.add_specie(sp)
                p_idxs.append(self._species_df.loc[sp._get_id_str(), "idx"].values)
            r_idxs = [i[0] for i in da.compute(r_idxs)[0]]
            p_idxs = [i[0] for i in da.compute(p_idxs)[0]]
            reaction_dd = dd.from_pandas(pd.DataFrame({"r_indicis": ",".join([str(i) for i in r_idxs]), 
                                                        "p_indicis": ",".join([str(i) for i in p_idxs])},
                                                        index=[rxn_id]), npartitions=1)
            self._reactions_df = dd.concat([self._reactions_df, reaction_dd])


    def compare_species(self, rxn_graph) -> List[Specie]:
        return da.compute(self._species_df.index.difference(rxn_graph._species_df))


    def has_specie(self, specie: Specie) -> bool:
        return da.compute(specie._get_id_str() in self._species_df.index)


    def has_reaction(self, reaction: Reaction) -> bool:
        return da.compute(reaction._get_id_str() in self._reaction_df.index)


    def get_specie_from_id(self, specie_id: str) -> Specie:
        ac_str = da.compute(self._species_df.loc[specie_id, "ac_str"].values)[0][0]
        m = self.ac_matrix_type(None)
        m._from_str(ac_str)
        s = m.to_specie()
        s.properties["visited"] = da.compute(self._species_df.loc[specie_id, "visited"].values)[0][0]
        return s


    def get_specie_from_index(self, idx: int) -> Specie:
        specie_id = da.compute(self._species_df[self._species_df["idx"] == idx].index.values)[0][0]
        return self.get_specie_from_id(specie_id)


    def get_reaction_from_id(self, reaction_id: str) -> Reaction:
        r_idxs = [int(x) for x in da.compute(self._reactions_df.loc[reaction_id, "r_indicis"].values)[0][0].split(",")]
        reactants = [self.get_specie_from_index(i) for i in r_idxs]
        p_idxs = [int(x) for x in da.compute(self._reactions_df.loc[reaction_id, "p_indicis"].values)[0][0].split(",")]
        products = [self.get_specie_from_index(i) for i in p_idxs]
        return Reaction(reactants, products)

    
    def to_dict(self) -> dict:
        return {"species_npartitions": self._specie_nparts, "reactions_npartitions": self._rxn_nparts}


    def make_specie_visited(self, specie: Specie):
        s_id = specie._get_id_str()
        def assign_func(df):
            df.loc[s_id, "visited"] = True

        self._species_df.map_partitions(assign_func)


    def get_unvisited_species(self) -> List[Specie]:
        """Method to get all unvisited species in reaction graph"""
        s_ids = self._species_df[self._species_df["visited"] == False].index
        return da.compute([self.get_specie_from_id(s_id) for s_id in s_ids])[0]


    def get_visited_species(self) -> List[Specie]:
        """Method to get all unvisited species in reaction graph"""
        s_ids = self._species_df[self._species_df["visited"] == True].index
        return da.compute([self.get_specie_from_id(s_id) for s_id in s_ids])[0]


    def get_n_species(self) -> int:
        """Method to get the number of species in graph"""
        return da.compute(len(self._species_df))[0]


    def get_n_reactions(self) -> int:
        """Method to get the number of reactions in graph"""
        return da.compute(len(self._reactions_df))[0]
