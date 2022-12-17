import shutil
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.sql import exists, select
from typing import List
import os
import dask as da
import openbabel as ob
from copy import copy
from typing import Optional
from torinax.pipelines.computations import Computation, SqlBase, DaskComputation, SlurmComputation, model_lookup_by_table_name, comp_sql_model_creator
from torinax.utils.openbabel import ob_read_file_to_molecule, molecule_to_obmol
import torinanet as tn

class ReadSpeciesFromUnchargedGraph (Computation):

    """Computation to create main specie table in Database and read it from uncharged ReactionGraph object"""

    tablename = "uncharged_species"
    name = "read_species_from_uncharged_graph"
    __results_columns__ = {
        "smiles": Column(String(100)),
        "ac_matrix_str": Column(String(500)),
    }

    def execute(self, db_session, rxn_graph=None) -> List[SqlBase]:
        if not rxn_graph:
            # if not provided rxn_graph, tries to read one from dist
            # basing on file path from config table in the SQL database
            rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="uncharged_rxn_graph_path").one()[0]
            rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        entries = []
        for specie in rxn_graph.species:
            sid = rxn_graph.make_unique_id(specie)
            if not db_session.query(exists().where(self.sql_model.id == sid)).one()[0]:
                entry = self.sql_model(
                                        id=sid,
                                        ac_matrix_str=specie.ac_matrix._to_str(),
                                        smiles=specie.ac_matrix.to_specie().identifier,
                                        )
                entries.append(entry)
        return entries

class ReadSpeciesFromChargedGraph (Computation):

    """Computation to create main specie table in Database and read it from uncharged ReactionGraph object"""

    name = "read_species_from_charged_graph"
    tablename = "charged_species"
    __results_columns__ = {
        "uncharged_sid": Column(String(100), ForeignKey("uncharged_species.id")),
        "charge": Column(Integer),
        "relevant": Column(Boolean, default=True)
    }

    def execute(self, db_session, rxn_graph=None) -> List[SqlBase]:
        if not rxn_graph:
            # if not provided rxn_graph, tries to read one from dist
            # basing on file path from config table in the SQL database
            rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="charged_rxn_graph_path").one()[0]
            rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        entries = []
        # updating relevance of existing species
        existing_sids = db_session.query(self.sql_model.id).all()
        for sid in existing_sids:
            specie = db_session.execute(select(self.sql_model).filter_by(id=sid[0])).scalar_one()
            if rxn_graph.has_specie_id(sid[0]):
                specie.relevant = True
            else:
                # updating relevance
                specie.relevant = False
        db_session.commit()
        # adding new species to table - ALL relevant by definition
        for specie in rxn_graph.species:
            sid = rxn_graph.make_unique_id(specie)
            if not db_session.query(exists().where(self.sql_model.id == sid)).one()[0]:
                entry = self.sql_model(
                                        id=sid,
                                        uncharged_sid=specie._get_id_str(),
                                        charge=specie.charge,
                                        relevant=True
                                        ) 
                entries.append(entry)
        return entries

class OpenbabelFfError (Exception):
    pass

class OpenbabelBuildError (Exception):
    pass

class BuildMolecules (DaskComputation):

    tablename = "specie_xyz"
    name = "build_molecules"
    __results_columns__ = {
        "xyz_path": Column(String(100)),
        "successful": Column(Boolean)
    }

    def __init__(self, dask_client, connected_molecules):
        self.connected_molecules = connected_molecules
        super().__init__(dask_client)
    
    @staticmethod
    @da.delayed(pure=False)
    def run(ac_string: str, xyz_path: str, sid: str, connected_molecules) -> dict:
        ac_mat = tn.core.BinaryAcMatrix()
        ac_mat._from_str(ac_string)
        try:
            molecule = ac_mat.build_geometry(connected_molecules)
            molecule.save_to_file(xyz_path)
            m = {"id": sid,
                    "xyz_path": xyz_path,
                    "successful": True}
        except OpenbabelBuildError or OpenbabelFfError:
            m = {"id": sid,
                 "successful": False}
        
        return m

    def make_futures(self, db_session):
        specie_table = model_lookup_by_table_name(ReadSpeciesFromUnchargedGraph.tablename)
        # getting all sids for building
        # these sids are for species who were not built yet (id doesnt exist in specie_xyz table)
        sids = db_session.query(specie_table.id).except_(db_session.query(self.sql_model.id)).all()
        parent_res_dir = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="results_dir").one()[0]
        iter_no = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="macro_iteration").one()[0]
        xyz_dir = os.path.join(parent_res_dir, iter_no, "xyz")
        if not os.path.isdir(xyz_dir):
            os.mkdir(xyz_dir)
        # making list of futures for calculation
        futures = []
        for sid in sids:
            sid = sid[0] # artifact of SQL query - returns a tuple that must be reduced
            ac_str = db_session.query(specie_table.ac_matrix_str).filter_by(id=sid).one()[0]
            future = self.run(
                ac_str,
                os.path.join(xyz_dir, str(sid) + ".xyz"),
                sid,
                self.connected_molecules
            )
            futures.append(future)
        return futures


class ExternalCalculation (SlurmComputation):

    name = "calculate_energies"
    __results_columns__ = {
        "input_path": Column(String(100)),
        "output_path": Column(String(100)),
        "__table_args__": {'extend_existing': True}

    }
    
    def __init__(self, slurm_client, program, input_type, comp_kwdict, output_extension, name: str="comp_outputs",
                 specie_tablename: Optional[str]=None):
        self.tablename = name
        self.program = program
        self.input_type = input_type
        self.comp_kwdict = comp_kwdict
        self.output_ext = output_extension
        self.specie_tablename = specie_tablename
        # creating relevance table for computation results
        self.relevance_model = comp_sql_model_creator("{}_relevance".format(name), {"iteration": Column(Integer)})
        super().__init__(slurm_client)

    def single_calc(self, xyz_path, comp_dir, sid, charge):
        """Method to make an sql entry and command line string for single SLURM run"""
        molecule = ob_read_file_to_molecule(xyz_path)
        in_file_path = os.path.join(comp_dir, "inputs", sid + "." + self.input_type.extension)
        entry = self.sql_model(id=sid,
                                input_path=in_file_path,
                                output_path=os.path.join(
                                        comp_dir,
                                        sid + "_out", sid + "." + self.output_ext)
                                        )
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
        return entry, self.program.run_command(in_file_path)

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
        specie_table = model_lookup_by_table_name(BuildMolecules.tablename)
        charge_table = model_lookup_by_table_name(ReadSpeciesFromChargedGraph.tablename)
        if not self.specie_tablename:
            sids = db_session.query(charge_table.id).filter_by(relevant=True).except_(db_session.query(self.sql_model.id)).all()
            relevent_sids = db_session.query(charge_table.id).filter_by(relevant=True).except_(db_session.query(self.relevance_model.id)).all()
        else:
            custom_table = model_lookup_by_table_name(self.specie_tablename)
            sids = db_session.query(custom_table.id).except_(db_session.query(self.sql_model.id)).all()
            relevent_sids = db_session.query(custom_table.id).except_(db_session.query(self.relevance_model.id)).all()
        entries = [self.relevance_model(id=sid[0], iteration=iter_count) for sid in relevent_sids]
        cmds = []
        for sid in sids:
            sid = sid[0] # artifact of SQL query results - returns tuples
            uncharged_sid = db_session.query(charge_table.uncharged_sid).filter_by(id=sid).one()[0]
            xyz = db_session.query(specie_table.xyz_path).filter_by(id=uncharged_sid).one()[0]
            charge = db_session.query(charge_table).filter_by(id=sid).first().charge
            entry, cmd_str = self.single_calc(xyz, comp_dir, sid, charge)
            entries.append(entry)
            cmds.append(cmd_str)
        return entries, cmds

class EstimateCharges (Computation):

    """Method to estimate charges of species in RxnGraph"""

    pass

class ReadCompOutput (DaskComputation):

    """Method to parse data generated by external programs"""

    name = "read_energies"
    tablename = "energy_outputs"
    __results_columns__ = {
        "energy": Column(String(100)),
        "good_geometry": Column(Boolean),
        "successful": Column(Boolean),
        "__table_args__": {'extend_existing': True}
    }

    def __init__(self, dask_client, output_type, comp_output_table_name: str="comp_outputs"):
        self.output_type = output_type
        self.tablename = "{}_outputs".format(comp_output_table_name)
        self.comp_output_table_name = comp_output_table_name
        super().__init__(dask_client)


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
    def run(cls, output_type, sid: str, output_path: str, xyz_path: str) -> dict:
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
                      "id": sid,
                      "successful": False,
                  }
        # returning SQL entry with results
        return {
            "id": sid,
            "energy": out_d["final_energy"],
            "successful": out_d["finished_normally"],
            "good_geometry": cls.compare_species(original_mol, out_mol)
        }


    def make_futures(self, db_session):
        comp_table = model_lookup_by_table_name(self.comp_output_table_name)
        build_table = model_lookup_by_table_name(BuildMolecules.tablename)
        # getting all sids for building
        # these sids are for species who were not built yet (id doesnt exist in specie_xyz table)
        sids = db_session.query(comp_table.id).except_(db_session.query(self.sql_model.id)).all()
        # making list of futures for calculation
        futures = []
        for sid in sids:
            sid = sid[0] # artifact of SQL query - returns tuple
            uncharged_sid = sid.split("#")[0] # MUST "uncharge" the ID, for querying the build table
                                                        # TODO: find a better way to do it
            xyz_path = db_session.query(build_table.xyz_path).filter_by(id=uncharged_sid).one()[0]
            output_path = db_session.query(comp_table.output_path).filter_by(id=sid).one()[0]
            future = self.run(
                self.output_type,
                sid,
                output_path,
                xyz_path
            )
            futures.append(future)
        return futures

class ReduceGraphByCriterion (Computation):

    """Method to filter a reaction graph by a criterion function.
    ARGS:
        - target_graph (str): string for type of graph (charged or uncharged)
        - sid_query_func (callable): function that takes the db_session and returns the specie IDs to be removed"""

    tablename = None
    name = "remove_bad_species"
    
    def __init__(self, target_graph: str, sid_query_func: callable, local_file_name: Optional[str]=None) -> None:
        if not target_graph.lower() in ["charged", "uncharged"]:
            raise ValueError("invalid target graph '{}'. allowed values 'charged' and 'uncharged'".format(target_graph))
        self.target_graph = target_graph.lower()
        self.sid_query_func = sid_query_func
        self.local_file_name = local_file_name
        if self.target_graph == "charged":
            self.update_species_comp = ReadSpeciesFromChargedGraph()
        else:
            self.update_species_comp = ReadSpeciesFromUnchargedGraph()
            
        super().__init__()

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="{}_rxn_graph_path".format(self.target_graph)).one()[0]
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        print("BEFORE REDUCTION")
        print("n species =", len(rxn_graph.species))
        print("n reactions =", len(rxn_graph.reactions))
        # finding specie IDs to remove
        sids = self.sid_query_func(db_session)
        # removing IDs from graph
        for sid in sids:
            sid = sid[0] # artifact of SQL query - returns tuple
            if rxn_graph.has_specie_id(sid):
                specie = rxn_graph.get_specie_from_id(sid)
                rxn_graph = rxn_graph.remove_specie(specie)
        # saving reduced graph
        rxn_graph.save(rxn_graph_path)
        print("AFTER REDUCTION")
        print("n species =", len(rxn_graph.species))
        print("n reactions =", len(rxn_graph.reactions))
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
    
    def __init__(self, reducer, target_graph: str, local_file_name: Optional[str]=None, energy_comp_table_name: str="comp_outputs") -> None:
        if not target_graph.lower() in ["charged", "uncharged"]:
            raise ValueError("invalid target graph '{}'. allowed values 'charged' and 'uncharged'".format(target_graph))
        self.target_graph = target_graph.lower()
        self.reducer = reducer
        self.local_file_name = local_file_name
        self.energy_output_tablename = "{}_outputs".format(energy_comp_table_name)
        self.energy_relevance_tablename = "{}_relevance".format(energy_comp_table_name)
        if self.target_graph == "charged":
            self.update_species_comp = ReadSpeciesFromChargedGraph()
        else:
            self.update_species_comp = ReadSpeciesFromUnchargedGraph()
        super().__init__()

    def update_specie_energies(self, db_session, rxn_graph) -> tn.core.RxnGraph:
        """Method to update the species energies in the reaction graph from the computation"""
        comp_out = model_lookup_by_table_name(self.energy_output_tablename)
        relevance_model = model_lookup_by_table_name(self.energy_relevance_tablename)
        relevant_ids = db_session.query(relevance_model.id)
        sids_energies = db_session.query(comp_out.id, comp_out.energy).filter(
                        comp_out.good_geometry & comp_out.successful & comp_out.id.in_(relevant_ids)).all()
        for sid, energy in sids_energies:
            if rxn_graph.has_specie_id(sid):
                specie = rxn_graph.get_specie_from_id(sid)
                specie.properties["energy"] = float(energy)
        return rxn_graph

    def reduce_bad_geometries(self, db_session, rxn_graph) -> tn.core.RxnGraph:
        """Method to reduce species with bad geometries from graph"""
        comp_out = model_lookup_by_table_name(self.energy_output_tablename)
        relevance_model = model_lookup_by_table_name(self.energy_relevance_tablename)
        relevant_ids = db_session.query(relevance_model.id)
        sids = db_session.query(comp_out.id).filter(
                        ~(comp_out.good_geometry & comp_out.successful) & comp_out.id.in_(relevant_ids)).all()
        # removing IDs from graph
        for sid in sids:
            sid = sid[0]
            if rxn_graph.has_specie_id(sid):
                specie = rxn_graph.get_specie_from_id(sid)
                rxn_graph = rxn_graph.remove_specie(specie)
        return rxn_graph

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="{}_rxn_graph_path".format(self.target_graph)).one()[0]
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        print("BEFORE REDUCTION")
        print("n species =", len(rxn_graph.species))
        print("n reactions =", len(rxn_graph.reactions))
        # removing species with bad geometry or failed computation
        rxn_graph = self.reduce_bad_geometries(db_session, rxn_graph)
        # updating energy values for species
        rxn_graph = self.update_specie_energies(db_session, rxn_graph)
        # applying reducer on graph
        rxn_graph = self.reducer.apply(rxn_graph)
        rxn_graph.save(rxn_graph_path)
        print("AFTER REDUCTION")
        print("n species =", len(rxn_graph.species))
        print("n reactions =", len(rxn_graph.reactions))
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
        self.update_species_comp = ReadSpeciesFromUnchargedGraph()
        super().__init__()

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="uncharged_rxn_graph_path").one()[0]
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        # setting up & enumerating
        stop_cond = tn.iterate.stop_conditions.MaxIterNumber(1)
        iterator = tn.iterate.Iterator(rxn_graph)
        rxn_graph = iterator.enumerate_reactions(self.conversion_filters, 
                                                    self.ac_filters, 
                                                    stop_cond, 
                                                    verbose=1)
        # updating SQL
        entries = self.update_species_comp.execute(db_session, rxn_graph)
        # saving graph to disk
        iter_count = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="macro_iteration").one()[0]
        res_dir = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="results_dir").one()[0]
        local_file = os.path.join(res_dir, iter_count, "crud_uncharged.rxn")
        rxn_graph.save(local_file)
        shutil.copyfile(local_file, rxn_graph_path)
        return entries


class RedoxReactionEnumeration (Computation):

    """Method to enumerate possible redox reactions in network"""

    name = "redox_reaction_enumeration"
    tablename = None
    
    def __init__(self, 
                    max_reduction,
                    max_oxidation,
                    charge_filters) -> None:
        self.max_reduction = max_reduction
        self.max_oxidation = max_oxidation
        self.charge_filters = charge_filters
        self.update_species_comp = ReadSpeciesFromChargedGraph()
        super().__init__()

    def execute(self, db_session) -> List[SqlBase]:
        # reading uncharged reaction graph path
        rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="uncharged_rxn_graph_path").one()[0]
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        # setting up & enumerating
        charge_iterator = tn.iterate.ChargeIterator(rxn_graph, type(rxn_graph))
        charged_graph = charge_iterator.enumerate_charges(self.max_reduction, 
                                                            self.max_oxidation, 
                                                            self.charge_filters)
        # updating SQL
        entries = self.update_species_comp.execute(db_session, charged_graph)
        # saving graph to disk
        iter_count = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="macro_iteration").one()[0]
        res_dir = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="results_dir").one()[0]
        local_file = os.path.join(res_dir, iter_count, "crud_charged.rxn")
        charged_graph.save(local_file)
        # saving charged graph in separate file
        rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="charged_rxn_graph_path").one()[0]
        shutil.copyfile(local_file, rxn_graph_path)
        return entries


class UnchargeGraph (Computation):

    """Method to uncharge a charged reaction graph"""

    name = "uncharging graph"
    tablename = None

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        uncharged_rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="uncharged_rxn_graph_path").one()[0]
        uncharged_rxn_graph = tn.core.RxnGraph.from_file(uncharged_rxn_graph_path)
        charged_rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="charged_rxn_graph_path").one()[0]
        charged_rxn_graph = tn.core.RxnGraph.from_file(charged_rxn_graph_path)
        # "uncharging" charged graph
        rids = set([uncharged_rxn_graph.make_unique_id(r) for r in charged_rxn_graph.reactions])
        uncharged_rxn_graph = uncharged_rxn_graph.copy(keep_ids=rids)  
        # saving new reaction graph
        uncharged_rxn_graph.save(uncharged_rxn_graph_path)
        return []

class FindMvc (Computation):

    """Computation to find MVC species in graph"""

    name = "find_mvc"
    tablename = "mvc_species"
    __results_columns__ = {
        "iteration": Column(Integer),
    }

    def __init__(self, n_trails: int=10, max_samples: int=300, metric: str="degree"):
        self.n_trails = n_trails
        self.max_samples = max_samples
        self.metric = metric
        self.greedy_finder = tn.analyze.algorithms.vertex_cover.GreedyMvcFinder(metric)
        self.stochastic_finder = tn.analyze.algorithms.vertex_cover.StochasticMvcFinder(metric, max_samples, max_samples)
        super().__init__()

    def execute(self, db_session) -> List[SqlBase]:
        # reading graph
        rxn_graph_path = db_session.query(model_lookup_by_table_name("config").value).filter_by(name="charged_rxn_graph_path").one()[0]
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        # getting all "covered" species - species with known product energies
        # covered_species = [s[0] for s in db_session.query(model_lookup_by_table_name("energy_outputs").id).all()]
        covered_species = [rxn_graph.make_unique_id(s) for s in rxn_graph.species if s.properties["visited"]]
        # running MVC finder for n trails
        mvc = []
        min_mvc_l = len(rxn_graph.species)
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
        iter_count = \
        db_session.query(model_lookup_by_table_name("config").value).filter_by(name="macro_iteration").one()[0]
        entries = []
        for specie in mvc:
            sid = rxn_graph.make_unique_id(specie)
            if not db_session.query(exists().where(self.sql_model.id == sid)).scalar():
                print("adding {} to mvc".format(sid))
                entries.append(self.sql_model(id=sid, iteration=iter_count))
            else:
                print("{} exists in MVC table, we don't add twice !".format(sid))
                
        return entries
