import shutil
from sqlalchemy import Column, Integer, String, Boolean, Float
from sqlalchemy.sql import exists, select
from typing import List
import os
from urllib.parse import urlparse
import dask as da
import openbabel as ob
from copy import copy
from typing import Optional, Callable
from torinax.pipelines.computations import Computation, SqlBase, DaskComputation, SlurmComputation, model_lookup_by_table_name, comp_sql_model_creator
from torinax.utils.openbabel import ob_read_file_to_molecule, molecule_to_obmol, atomic_numer_to_symbol
import torinanet as tn

def safe_smiles_str(smiles: str) -> str:
    """Method to encode a smiles string in a safe way for using in ORCA I/O files"""
    return smiles.replace("[", "!").replace("]", "!").replace("(", ".").replace(")", ".")


class ReadSpeciesFromGraph (Computation):

    """Computation to create main specie table in Database and read it from uncharged ReactionGraph object.
    ARGS:
        - specie_hash_func (Callable[[tn.core.Specie], int]): hash function for a specie"""

    base_specie_hash_func = tn.core.HashedCollection.generator_to_hash(tn.core.HashedCollection.FingerprintGenerators.RDKIT, fp_size=256, n_bits_per_feature=4)
    tablename = "species"
    name = "read_species_from_graph"
    __results_columns__ = {
        "id": Column(Integer, primary_key=True, autoincrement=True),
        "hash_key": Column(String),
        "smiles": Column(String),
        "charge": Column(Integer, default=0),
        "relevant": Column(Boolean, default=True)
    }

    @classmethod
    def specie_hash_func(cls, specie):
        return str(hex(cls.base_specie_hash_func(specie)))

    def specie_in_db(self, db_session, specie: tn.core.Specie) -> bool:
        """Check if specie is in the specie table in the db"""
        hash_key = self.specie_hash_func(specie)
        # checks by hash_key if specie exists in the specie table
        smiles = db_session.query(self.sql_model.smiles).filter_by(hash_key=hash_key).all()
        smiles = [s[0] for s in smiles] # artifact of sqlalchemy query results (list of tuples instead of list of strings)
        # if smiles list is empty -> specie does not exist
        if len(smiles) == 0:
            return False
        # else if specie's smiles is in smiles list -> specie exists
        elif specie.identifier in smiles:
            return True
        # now dealing with non-trivial cases (like if smiles of specie is equivalent, but not equal, to smiles in list)
        else:
            specie_ac = tn.core.BinaryAcMatrix.from_specie(specie)
            for smile in smiles:
                ac = tn.core.BinaryAcMatrix.from_specie(tn.core.Specie(smile))
                if ac == specie_ac:
                    return True
            return False


    def execute(self, db_session, rxn_graph=None) -> List[SqlBase]:
        if not rxn_graph:
            # if not provided rxn_graph, tries to read one from dist
            # basing on file path from config table in the SQL database
            rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="rxn_graph_path").one()[0]
            rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        entries = []
        # updating relevance of existing species
        for specie in db_session.query(self.sql_model).all():
            s = tn.core.Specie(specie.smiles)
            if rxn_graph.has_specie(s):
                specie.relevant = True
            else:
                # updating relevance
                specie.relevant = False
        db_session.commit()
        # now add new species
        for specie in rxn_graph.species:
            if not self.specie_in_db(db_session, specie):
                entry = self.sql_model(
                                        hash_key=self.specie_hash_func(specie),
                                        smiles=specie.ac_matrix.to_specie().identifier,
                                        charge=0, relevant=True
                                        )
                entries.append(entry)
        return entries


class OpenbabelFfError (Exception):
    pass

class OpenbabelBuildError (Exception):
    pass

class BuildMolecules (Computation):

    tablename = "species"
    name = "build_molecules"
    __results_columns__ = {
        "xyz": Column(String, default=None),
    }

    def __init__(self):
        super().__init__()
    
    @staticmethod
    @da.delayed(pure=False)
    def run(smiles: str, xyz_path: str) -> dict:
        ac_mat = tn.core.BinaryAcMatrix.from_specie(tn.core.Specie(smiles))
        try:
            molecule = ac_mat.build_geometry({})
            molecule.save_to_file(xyz_path)
            return xyz_path
        except OpenbabelBuildError or OpenbabelFfError:
            return None


    def execute(self, db_session) -> List[SqlBase]:
        # making appropriate directory for xyz files
        parent_res_dir = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="results_dir").one()[0]
        iter_no = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="macro_iteration").one()[0]
        xyz_dir = os.path.join(parent_res_dir, iter_no, "xyz")
        if not os.path.isdir(xyz_dir):
            os.mkdir(xyz_dir)
        # collecting all species without the "XYZ" attribute
        species = db_session.query(self.sql_model).filter(self.sql_model.xyz == None).all()
        # building futures for computation
        futures = []
        for specie in species:
            future = self.run(
                specie.smiles,
                os.path.join(xyz_dir, safe_smiles_str(specie.smiles) + ".xyz"),
            )
            futures.append(future)
        # computing futures and updating db with results
        results = da.compute(futures)[0]
        for specie, res in zip(species, results):
            specie.xyz = res
        db_session.commit()
        return []


class ExternalCalculation (SlurmComputation):

    name = "calculate_energies"
    tablename = "species"
    __results_columns__ = {
        "comp_input": Column(String(100)),
        "comp_output": Column(String(100)),
    }
    
    def __init__(self, slurm_client, program, input_type, comp_kwdict, output_extension, comp_source: str="normal"):
        self.program = program
        self.input_type = input_type
        self.comp_kwdict = comp_kwdict
        self.output_ext = output_extension
        self.comp_source = comp_source
        # creating relevance table for computation results
        self.comp_log_sql = comp_sql_model_creator("log", {"id": Column(String, primary_key=True), "iteration": Column(Integer), "source": Column(String)})
        super().__init__(slurm_client)

    def single_calc(self, xyz_path, comp_dir, smiles, charge):
        """Method to make an sql entry and command line string for single SLURM run"""
        molecule = ob_read_file_to_molecule(xyz_path)
        in_file_path = os.path.join(comp_dir, "inputs", safe_smiles_str(smiles) + "." + self.input_type.extension)
        # calculating number of electrons
        n_elec = 0
        for atom in molecule.atoms:
            n_elec += ob.OBElementTable().GetAtomicNum(atom.symbol)
        infile = self.input_type(in_file_path)
        kwds = copy(self.comp_kwdict)
        kwds["charge"] = charge
        # correcting for unique case of oxygen molecule - triplet ground state
        if all([atom.symbol == "O" for atom in molecule.atoms]) and len(molecule.atoms) == 2:
            kwds["mult"] = 3
        else:
            obmol = molecule_to_obmol(molecule)
            obmol.AssignSpinMultiplicity(True)
            # use openbabel to estimate multiplicities
            kwds["mult"] = obmol.GetTotalSpinMultiplicity()
            # kwds["mult"] = (n_elec + kwds["charge"]) % 2 + 1
        infile.write_file(molecule, kwds)
        return in_file_path, self.program.run_command(in_file_path)

    def make_cmd_list(self, db_session):
        # setting up appropriate directory for computation results
        iter_count = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="macro_iteration").one()[0]
        res_dir = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="results_dir").one()[0]
        comp_dir = os.path.join(res_dir, iter_count, self.tablename)
        if not os.path.isdir(comp_dir):
            os.mkdir(comp_dir)
        input_dir = os.path.join(comp_dir, "inputs")
        if not os.path.isdir(input_dir):
            os.mkdir(input_dir)
        # now continuing to main computation
        # finding all species for calculation - relevant species that were not calculated before
        species = db_session.query(self.sql_model).filter((self.sql_model.relevant==True) & ~(self.sql_model.comp_output.isnot(None) & self.sql_model.comp_input.isnot(None))).all()
        cmds = []
        entries = []
        for specie in species:
            # add log of computation
            # making run command & writing input file
            comp_input, cmd_str = self.single_calc(specie.xyz, comp_dir, specie.smiles, specie.charge)
            # update the specie table with i/o of computation
            specie.comp_output = os.path.join(comp_dir, safe_smiles_str(specie.smiles) + "_out", safe_smiles_str(specie.smiles) + "." + self.output_ext)
            specie.comp_input = comp_input
            cmds.append(cmd_str)
        # adding log computations to properly add calculated species to log 
        # ALL relevant species should be in the log
        smiles = [s[0] for s in db_session.execute("SELECT smiles FROM species WHERE relevant == 1 AND NOT smiles IN (SELECT id FROM log)")]
        for smile in smiles:
            entries.append(self.comp_log_sql(id=smile, iteration=iter_count, source=self.comp_source))
        db_session.commit()
        return entries, cmds


class ReadCompOutput (Computation):

    """Method to parse data generated by external programs"""

    name = "read_energies"
    tablename = "species"
    __results_columns__ = {
        "energy": Column(Float),
        "good_geometry": Column(Boolean),
        "successful": Column(Boolean),
        "__table_args__": {'extend_existing': True}
    }

    def __init__(self, output_type):
        self.output_type = output_type
        super().__init__()


    @staticmethod
    def compare_species(specie1, specie2) -> bool:
        # reading to openbal & guessing bonds
        ob1 = molecule_to_obmol(specie1)
        ob1.ConnectTheDots()
        ob1.PerceiveBondOrders()
        # making ac matrix
        ac1 = tn.core.BinaryAcMatrix.from_obmol(ob1)
        ob2 = molecule_to_obmol(specie2)
        ob2.ConnectTheDots()
        ob2.PerceiveBondOrders()
        ac2 = tn.core.BinaryAcMatrix.from_obmol(ob2)
        # if ac matrices are equal -> structures are equal
        return ac1 == ac2

    @classmethod
    @da.delayed(pure=True)
    def run(cls, output_type, output_path: str, xyz_path: str) -> dict:
        # reading molecule from xyz file
        original_mol = ob_read_file_to_molecule(xyz_path)
        # reading output
        try:
            output = output_type(output_path)
            out_mol = output.read_specie()
            out_d = output.read_scalar_data()
        # if errors encountered in output parsing - computation is not successful
        except:
            print("parsing error in", output_path)
            return {  
                      "successful": False,
                  }
        return {
            "energy": out_d["final_energy"],
            "successful": out_d["finished_normally"],
            "good_geometry": cls.compare_species(original_mol, out_mol)
        }

    def execute(self, db_session):
        # getting all species for reading
        species = db_session.query(self.sql_model).filter(self.sql_model.comp_output.isnot(None) & self.sql_model.energy.is_(None)).all()
        # making list of futures for calculation
        futures = []
        for specie in species:
            futures.append(self.run(
                self.output_type,
                specie.comp_output,
                specie.xyz
            ))
        dicts = da.compute(futures)[0]
        # updating table according to results
        for specie, d in zip(species, dicts):
            for k, v in d.items():
                setattr(specie, k, v)
        db_session.commit()
        return []


class ReduceGraphByCriterion (Computation):

    """Method to filter a reaction graph by a criterion function.
    ARGS:
        - target_graph (str): string for type of graph (charged or uncharged)
        - sid_query_func (callable): function that takes the db_session and returns the specie IDs to be removed"""

    tablename = None
    name = "remove_bad_species"
    
    def __init__(self, specie_query_func: callable, local_file_name: Optional[str]=None) -> None:
        self.specie_query_func = specie_query_func
        self.local_file_name = local_file_name
        self.update_species_comp = ReadSpeciesFromGraph()            
        super().__init__()

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="rxn_graph_path").one()[0]
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        print("BEFORE REDUCTION")
        print("n species =", rxn_graph.get_n_species())
        print("n reactions =", rxn_graph.get_n_reactions())
        # finding specie IDs to remove
        smiles_strs = self.specie_query_func(db_session)
        # removing IDs from graph
        for smiles in smiles_strs:
            print("removing", smiles)
            s = tn.core.Specie(smiles)
            if rxn_graph.has_specie(s):
                rxn_graph = rxn_graph.remove_specie(s)
        # saving reduced graph
        rxn_graph.save(rxn_graph_path)
        print("AFTER REDUCTION")
        print("n species =", rxn_graph.get_n_species())
        print("n reactions =", rxn_graph.get_n_reactions())
        # if a local copy is desired, making one
        if self.local_file_name:
            iter_count = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="macro_iteration").one()[0]
            res_dir = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="results_dir").one()[0]
            local_file = os.path.join(res_dir, iter_count, self.local_file_name)
            shutil.copyfile(local_file, rxn_graph_path)
        # return updated relevance terms for the species
        return self.update_species_comp.execute(db_session, rxn_graph)


class ReduceGraphByEnergyReducer (Computation):

    """Method to reduce a reaction graph by an energy reducer (in torinanet.analyze.network_reduction)."""

    name = "graph_energy_reduction"
    tablename = None
    
    def __init__(self, reducer, local_file_name: Optional[str]=None, use_atomization_energies: bool=False) -> None:
        self.reducer = reducer
        self.local_file_name = local_file_name
        self.use_atomization_energies = use_atomization_energies
        self.update_species_comp = ReadSpeciesFromGraph()
        super().__init__()

    def update_specie_energies(self, db_session, rxn_graph) -> tn.core.RxnGraph:
        """Method to update the species energies in the reaction graph from the computation"""
        # if we need to base the calculation on atomization energy, collect atom information 
        atoms_energy_dict = {}
        if self.use_atomization_energies:
            for sp in rxn_graph.source_species:
                for atom in sp.ac_matrix.get_atoms():
                    energy = db_session.execute("SELECT energy FROM species WHERE smiles == '[{}]'".format(atomic_numer_to_symbol(atom))).fetchone()[0]
                    atoms_energy_dict[atom] = energy
        # we use the "log" table to make sure that we use only species with "known energy" for the reduction
        # this is done to allow repr re-using the same DB file for multiple runs.
        # we use raw SQL because the "model_lookup_by_table_name" is unstable
        smiles_energies = db_session.execute("SELECT smiles, energy FROM species WHERE good_geometry == 1 AND successful == 1 AND smiles IN (SELECT id FROM log)")
        for smiles, energy in smiles_energies:
            s = tn.core.Specie(smiles)
            if rxn_graph.has_specie(s):
                # workaround to get the specie from the graph and update it. this method returns the "correct" specie.
                # using other methods such as specie_collection.get require proper reading with AC matrix
                s = rxn_graph.add_specie(s)
                if self.use_atomization_energies:
                    atoms_e = sum([atoms_energy_dict[atom] for atom in s.ac_matrix.get_atoms()])
                    s.properties["energy"] = float(energy) - atoms_e
                else:
                    s.properties["energy"] = float(energy)
        #         print(s, s.identifier, s.properties["energy"])
        # print("*****")
        # for s in rxn_graph.species:
        #     print(s)
        #     print(s.identifier, s.properties)
        return rxn_graph

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="rxn_graph_path").one()[0]
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        print("BEFORE REDUCTION")
        print("n species =", rxn_graph.get_n_species())
        print("n reactions =", rxn_graph.get_n_reactions())
        # removing species with bad geometry or failed computation - LEGACY NOW IT IS DONE BY DEDICATED ReduceGraphByCriterion COMPUTATION !
        # rxn_graph = self.reduce_bad_geometries(db_session, rxn_graph)
        # updating energy values for species
        rxn_graph = self.update_specie_energies(db_session, rxn_graph)
        # applying reducer on graph
        rxn_graph = self.reducer.apply(rxn_graph)
        rxn_graph.save(rxn_graph_path)
        print("AFTER REDUCTION")
        print("n species =", rxn_graph.get_n_species())
        print("n reactions =", rxn_graph.get_n_reactions())
        if self.local_file_name:
            iter_count = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="macro_iteration").one()[0]
            res_dir = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="results_dir").one()[0]
            local_file = os.path.join(res_dir, iter_count, self.local_file_name)
            shutil.copyfile(rxn_graph_path, local_file)
        return self.update_species_comp.execute(db_session, rxn_graph)


class ElementaryReactionEnumeration (Computation):

    """Method to enumerate elementary reactions"""

    name = "elemntary_reaction_enumeration"
    tablename = None
    
    def __init__(self, 
                    conversion_filters: List[tn.iterate.conversion_matrix_filters.ConvFilter],
                    ac_filters: List[tn.iterate.ac_matrix_filters.AcMatrixFilter]) -> None:
        self.conversion_filters = conversion_filters
        self.ac_filters = ac_filters
        self.update_species_comp = ReadSpeciesFromGraph()
        super().__init__()

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="rxn_graph_path").one()[0]
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        # setting up & enumerating
        stop_cond = tn.iterate.stop_conditions.MaxIterNumber(1)
        iterator = tn.iterate.Iterator(rxn_graph, dask_scheduler="synchronous")
        rxn_graph = iterator.enumerate_reactions(self.conversion_filters, 
                                                    self.ac_filters, 
                                                    stop_cond, 
                                                    verbose=1)
        # showing timing data
        tn.utils.show_time_data()
        # updating SQL
        entries = self.update_species_comp.execute(db_session, rxn_graph)
        # saving graph to disk
        iter_count = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="macro_iteration").one()[0]
        res_dir = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="results_dir").one()[0]
        local_file = os.path.join(res_dir, iter_count, "crud_graph.rxn")
        rxn_graph.save(local_file)
        shutil.copyfile(local_file, rxn_graph_path)
        return entries


class FindMvc (Computation):

    """Computation to find MVC species in graph"""

    name = "find_mvc"
    tablename = None

    def __init__(self, n_trails: int=10, max_samples: int=300, metric: str="degree"):
        self.n_trails = n_trails
        self.max_samples = max_samples
        self.metric = metric
        self.greedy_finder = tn.analyze.algorithms.vertex_cover.GreedyMvcFinder(metric)
        self.stochastic_finder = tn.analyze.algorithms.vertex_cover.StochasticMvcFinder(metric, max_samples, max_samples)
        super().__init__()

    def execute(self, db_session) -> List[SqlBase]:
        # reading graph
        rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="rxn_graph_path").one()[0]
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        # running MVC finder for n trails
        mvc = []
        min_mvc_l = len(list(rxn_graph.species))
        for _ in range(self.n_trails):
            candidate = self.stochastic_finder.find_mvc(rxn_graph)
            if candidate is not None:
                if len(candidate) < min_mvc_l and len(candidate) > 0:
                    mvc = candidate
                    min_mvc_l = len(mvc)
        # if no MVC is found stochastically, find MVC deterministically 
        if len(mvc) == 0:
            mvc = self.greedy_finder.find_mvc(rxn_graph)
        # inserting MVC data to database
        print("Found MVC with {} species".format(len(mvc)))
        mvc_smiles = set([s.identifier for s in mvc])
        species = db_session.query(model_lookup_by_table_name("species")).all()
        for specie in species:
            specie.relevant = (specie.smiles in mvc_smiles)
        db_session.commit()                
        return []
