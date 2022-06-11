# utils file for running reaction enumeration jobs on a cluster
# upgraded version of the enumerate reactions script
import shutil
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.sql import exists
from typing import List
import dask as da
import os
import openbabel as ob
from copy import copy
from typing import Optional
from config import torinax
from torinax.pipelines.computations import Computation, SqlBase, DaskComputation, SlurmComputation
from torinax.utils.openbabel import ob_read_file_to_molecule, molecule_to_obmol
import torinanet as tn

class ReadSpeciesFromUnchargedGraph (Computation):

    """Computation to create main specie table in Database and read it from uncharged ReactionGraph object"""

    __name__ = "uncharged_species"
    __results_columns__ = {
        "smiles": Column(String(100)),
        "ac_matrix_str": Column(String(500)),
    }

    def execute(self, db_session, rxn_graph=None) -> List[SqlBase]:
        if not rxn_graph:
            # if not provided rxn_graph, tries to read one from dist
            # basing on file path from config table in the SQL database
            rxn_graph_path = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="uncharged_rxn_graph_path")
            rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        entries = []
        for specie in self.rxn_graph.species:
            sid = self.rxn_graph.make_unique_id(specie)
            if not db_session.query(exists().where(self.sql_model.id == sid)):
                entry = self.sql_model(
                                        id=sid,
                                        ac_matrix_str=specie.ac_matrix._to_str(),
                                        smiles=specie.ac_matrix.to_specie().identifier,
                                        )
                entries.append(entry)
        return entries

class ReadSpeciesFromChargedGraph (Computation):

    """Computation to create main specie table in Database and read it from uncharged ReactionGraph object"""

    __name__ = "charged_species"
    __results_columns__ = {
        "uncharged_sid": Column(String(100), ForeignKey("uncharged_species")),
        "charge": Column(Integer),
    }

    def execute(self, db_session, rxn_graph=None) -> List[SqlBase]:
        if not rxn_graph:
            # if not provided rxn_graph, tries to read one from dist
            # basing on file path from config table in the SQL database
            rxn_graph_path = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="charged_rxn_graph_path")
            rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        entries = []
        for specie in self.rxn_graph.species:
            sid = self.rxn_graph.make_unique_id(specie)
            if not db_session.query(exists().where(self.sql_model.id == sid)):
                entry = self.sql_model(
                                        id=sid,
                                        uncharged_sid=specie._get_id_str(),
                                        charge=specie.charge,
                                        )
                entries.append(entry)
        return entries

class OpenbabelFfError (Exception):
    pass

class OpenbabelBuildError (Exception):
    pass

class BuildMolecules (DaskComputation):

    __name__ = "specie_xyz"
    __results_columns__ = {
        "xyz_path": Column(String(100)),
        "successful": Column(Boolean)
    }

    def __init__(self, dask_client, connected_molecules):
        self.connected_molecules = connected_molecules
        super().__init__(dask_client)

    @da.delayed
    @staticmethod
    def _run(sql_model: SqlBase, ac_string: str, xyz_path: str, sid: str, connected_molecules) -> SqlBase:
        ac_mat = tn.core.BinaryAcMatrix()
        ac_mat._from_str(ac_string)
        try:
            molecule = ac_mat.build_geometry(connected_molecules)
            molecule.save_to_file(xyz_path)
            return sql_model(id=sid,
                            xyz_path=xyz_path,
                            successful=True)
        except OpenbabelBuildError or OpenbabelFfError:
            return sql_model(id=sid,
                            successful=False)

    def make_futures(self, db_session):
        specie_table = db_session.metadata.tables[ReadSpeciesFromUnchargedGraph.__name__]
        # getting all sids for building
        # these sids are for species who were not built yet (id doesnt exist in specie_xyz table)
        sids = db_session.query(specie_table.id).except_(db_session.query(self.sql_model.id)).all()
        parent_res_dir = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="results_dir")
        iter_no = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="macro_iteration")
        xyz_dir = os.path.join(parent_res_dir, iter_no, "xyz")
        if not os.path.isdir(xyz_dir):
            os.mkdir(xyz_dir)
        # making list of futures for calculation
        futures = []
        for sid in sids:
            ac_str = db_session.query(specie_table).filter_by(id=sid).first().ac_matrix_str
            future = self._run(
                self.sql_model, # sql model
                ac_str,
                os.path.join(self.xyz_dir, str(sid)),
                sid,
                self.connected_molecules
            )
            futures.append(future)
        return futures


class ExternalCalculation (SlurmComputation):
    
    __results_columns__ = {
        "input_path": Column(String(100)),
        "output_path": Column(String(100)),
    }
    
    def __init__(self, slurm_client, program, input_type, comp_kwdict, output_extension, name: str="energy_outputs"):
        self.__name__ = name
        self.program = program
        self.input_type = input_type
        self.comp_kwdict = comp_kwdict
        self.output_ext = output_extension
        super().__init__(slurm_client)

    def single_calc(self, xyz_path, comp_dir, sid, charge):
        """Method to make an sql entry and command line string for single SLURM run"""
        molecule = ob_read_file_to_molecule(xyz_path)
        in_file_path = os.path.join(self.comp_dir, "inputs", sid + "." + self.input_type.extension)
        entry = self.sql_model(id=sid,
                                input_path=in_file_path,
                                output_path=os.path.join(
                                        comp_dir,
                                        sid + "_out", sid + "." + self.output_extension)
                                        )
        # calculating number of electrons
        n_elec = 0
        for atom in molecule.atoms:
            n_elec += ob.OBElementTable().GetAtomicNum(atom.symbol)
        infile = self.input_type(in_file_path)
        kwds = copy(self.comp_kwdict)
        kwds["charge"] = charge
        kwds["mult"] = (n_elec + kwds["charge"]) % 2 + 1
        infile.write_file(molecule, kwds)
        return entry, self.program.run_command(in_file_path)

    def make_cmd_list(self, db_session):
        # setting up appropriate directory for computation results
        iter_count = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="macro_iteration")
        res_dir = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="results_dir")
        comp_dir = os.path.join(res_dir, iter_count, self.__name__)
        if not os.path.isdir(comp_dir):
            os.mkdir(comp_dir)
        input_dir = os.path.join(comp_dir, "inputs")
        if not os.path.isdir(input_dir):
            os.mkdir(input_dir)
        # now continuing to main computation
        specie_table = db_session.metadata.tables[BuildMolecules.__name__]
        charge_table = db_session.metadata.tables[ReadSpeciesFromChargedGraph.__name__]
        sids = db_session.query(charge_table.id).except_(db_session.query(self.sql_model.id)).all()
        entries = []
        cmds = []
        for sid in sids:
            uncharged_sid = db_session.query(charge_table).filter_by(id=sid).first().uncharged_sid
            xyz = db_session.query(specie_table).filter_by(id=uncharged_sid).first().xyz_path
            charge = db_session.query(charge_table).filter_by(id=sid).first().charge
            entry, cmd_str = self.single_calc(xyz, sid, charge)
            entries.append(entry)
            cmds.append(cmd_str)
        return entries, cmds

class EstimateCharges (Computation):

    """Method to estimate charges of species in RxnGraph"""

    pass

class ReadCompOutput (Computation):

    """Method to parse data generated by external programs"""
    
    __name__ = "energy_outputs"
    __results_columns__ = {
        "energy": Column(String(100)),
        "good_geometry": Column(Boolean),
        "successful": Column(Boolean)
    }

    def __init__(self, dask_client, output_type, comp_output_table_name: str="comp_outputs"):
        self.output_type = output_type
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


    def _run(self, sid: str, output_path: str, xyz_path: str) -> SqlBase:
        # reading molecule from xyz file
        original_mol = ob_read_file_to_molecule(xyz_path)
        # reading output
        output = self.output_type(output_path)
        out_mol = output.read_specie()
        out_d = output.read_scalar_data()
        # returning SQL entry with results
        return self.sql_model(
            id=sid,
            energy=out_d["final_energy"],
            successful=out_d["finished_normally"],
            good_geometry=self.compare_species(original_mol, out_mol)
        )


    def make_futures(self, db_session):
        comp_table = db_session.metadata.tables[self.comp_output_table_name]
        build_table = db_session.metadata.tables[BuildMolecules.__name__]
        # getting all sids for building
        # these sids are for species who were not built yet (id doesnt exist in specie_xyz table)
        sids = db_session.query(comp_table.id).except_(db_session.query(self.sql_model.id)).all()
        # making list of futures for calculation
        futures = []
        for sid in sids:
            xyz_path = db_session.query(build_table.xyz_path).filter_by(id=sid).first()
            output_path = db_session.query(comp_table.output_path).filter_by(id=sid).first()
            future = self.dask_client.submit(self._run,
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

    __name__ = None
    
    def __init__(self, target_graph: str, sid_query_func: callable, local_file_name: Optional[str]=None) -> None:
        if not target_graph.lower() in ["charged", "uncharged"]:
            raise ValueError("invalid target graph '{}'. allowed values 'charged' and 'uncharged'".format(target_graph))
        self.target_graph = target_graph.lower()
        self.sid_query_func = sid_query_func
        self.local_file_name = local_file_name
        super().__init__()

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        rxn_graph_path = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="{}_rxn_graph_path".format(self.target_graph))
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        # finding specie IDs to remove
        sids = self.sid_query_func(db_session)
        # removing IDs from graph
        for sid in sids:
            if rxn_graph.has_specie_id(sid):
                specie = rxn_graph.get_specie_from_id(sid)
                rxn_graph = rxn_graph.remove_specie(specie)
        # saving reduced graph
        rxn_graph.save(rxn_graph_path)
        # if a local copy is desired, making one
        if self.local_file_name:
            iter_count = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="macro_iteration")
            res_dir = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="results_dir")
            local_file = os.path.join(res_dir, iter_count, self.local_file_name)
            shutil.copyfile(local_file, rxn_graph_path)
        return []

class ReduceGraphByEnergyReducer (Computation):

    """Method to reduce a reaction graph by an energy reducer (in torinanet.analyze.network_reduction)."""
    
    __name__ = None
    
    def __init__(self, reducer, local_file_name: Optional[str]=None) -> None:
        self.reducer = reducer
        self.local_file_name = local_file_name
        super().__init__()

    @staticmethod
    def update_specie_energies(db_session, rxn_graph) -> tn.core.RxnGraph:
        """Method to update the species energies in the reaction graph from the computation"""
        comp_out = db_session.metadata.tables[ReadCompOutput.__name__]
        sids_energies = db_session.query(comp_out.id, comp_out.energy).filter(
                        comp_out.good_geometry & comp_out.successful).all()
        for sid, energy in sids_energies:
            if rxn_graph.has_specie_id(sid):
                specie = rxn_graph.get_specie_from_id(sid)
                specie.properties["energy"] = energy
        return rxn_graph

    @staticmethod
    def reduce_bad_geometries(db_session, rxn_graph) -> tn.core.RxnGraph:
        """Method to reduce species with bad geometries from graph"""
        comp_out = db_session.metadata.tables[ReadCompOutput.__name__]
        sids = db_session.query(comp_out.id).filter(
                        ~(comp_out.good_geometry & comp_out.successful)).all()
        # removing IDs from graph
        for sid in sids:
            if rxn_graph.has_specie_id(sid):
                specie = rxn_graph.get_specie_from_id(sid)
                rxn_graph = rxn_graph.remove_specie(specie)
        return rxn_graph

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        rxn_graph_path = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="{}_rxn_graph_path".format(self.target_graph))
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        # removing species with bad geometry or failed computation
        rxn_graph = self.reduce_bad_geometries(db_session, rxn_graph)
        # updating energy values for species
        rxn_graph = self.update_specie_energies(db_session, rxn_graph)
        # applying reducer on graph
        rxn_graph = self.reducer.apply(rxn_graph)
        rxn_graph.save(rxn_graph_path)
        if self.local_file_name:
            iter_count = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="macro_iteration")
            res_dir = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="results_dir")
            local_file = os.path.join(res_dir, iter_count, self.local_file_name)
            shutil.copyfile(local_file, rxn_graph_path)



class ElementaryReactionEnumeration (Computation):

    """Method to enumerate elementary reactions"""

    __name__ = None
    
    def __init__(self, 
                    conversion_filters: List[tn.iterate.conversion_matrix_filters.ConvFilter],
                    ac_filters: List[tn.iterate.ac_matrix_filters.AcMatrixFilter]) -> None:
        self.conversion_filters = conversion_filters
        self.ac_filters = ac_filters
        self.update_species_comp = ReadSpeciesFromUnchargedGraph()
        super().__init__()

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        rxn_graph_path = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="uncharged_rxn_graph_path")
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
        iter_count = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="macro_iteration")
        res_dir = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="results_dir")
        local_file = os.path.join(res_dir, iter_count, "crud_uncharged.rxn")
        rxn_graph.save(local_file)
        shutil.copyfile(local_file, rxn_graph_path)
        return entries


class RedoxReactionEnumeration (Computation):

    """Method to enumerate possible redox reactions in network"""

    __name__ = None
    
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
        # reading reaction graph path
        rxn_graph_path = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="charged_rxn_graph_path")
        rxn_graph = tn.core.RxnGraph.from_file(rxn_graph_path)
        # setting up & enumerating
        charge_iterator = tn.iterate.ChargeIterator(rxn_graph, type(rxn_graph))
        charged_graph = charge_iterator.enumerate_charges(self.max_reduction, 
                                                            self.max_oxidation, 
                                                            self.charge_filters)
        # updating SQL
        entries = self.update_species_comp.execute(db_session, charged_graph)
        # saving graph to disk
        iter_count = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="macro_iteration")
        res_dir = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="results_dir")
        local_file = os.path.join(res_dir, iter_count, "crud_charged.rxn")
        charged_graph.save(local_file)
        shutil.copyfile(local_file, rxn_graph_path)
        return entries


class UnchargeGraph (Computation):

    """Method to uncharge a charged reaction graph"""

    __name__ == None

    def execute(self, db_session) -> List[SqlBase]:
        # reading reaction graph path
        uncharged_rxn_graph_path = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="uncharged_rxn_graph_path")
        uncharged_rxn_graph = tn.core.RxnGraph.from_file(uncharged_rxn_graph_path)
        charged_rxn_graph_path = db_session.query(db_session.metadata.tables["config"].value).filter_by(name="charged_rxn_graph_path")
        charged_rxn_graph = tn.core.RxnGraph.from_file(charged_rxn_graph_path)
        # "uncharging" charged graph
        rids = set([uncharged_rxn_graph.make_unique_id(r) for r in charged_rxn_graph])
        uncharged_rxn_graph = uncharged_rxn_graph.copy(keep_ids=rids)  
        # saving new reaction graph
        uncharged_rxn_graph.save(uncharged_rxn_graph_path)
        return []

