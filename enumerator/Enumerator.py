from optparse import Option
from sqlalchemy import Column, String
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import importlib
import os
from typing import List, Optional
from dask.distributed import Client
from config import torinanet as tn
from config import torinax as tx
from tx.pipelines.computations import SqlBase, run_computations
from tx.clients import SlurmClient
from . import computations as comps


class ConfigValue (SqlBase):

    __tablename__ = "config"

    name = Column(String(100), primary_key=True)
    value = Column(String(100))


class Enumerator:

    """Class to handle enumeration of elementary reactions"""

    def __init__(self, rxn_graph, pipeline, n_iter: int, results_dir: str):
        self.rxn_graph = rxn_graph
        self.pipeline = pipeline
        self.n_iter = n_iter
        self.results_dir = results_dir
        db_path = os.path.join(results_dir, "main.db")
        engine = create_engine("sqlite3///{}".format(db_path))
        SqlBase.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()
        self.load_settings()

    def load_settings(self):
        """Method to load settings of enumerator to the SQL database"""
        # properly define results dir
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        self.session.add(ConfigValue(name="results_dir", value=self.results_dir))
        # properly saves reaction graph
        rxn_graph_path = os.path.join(self.results_dir, "uncharged.rxn")
        if self.rxn_graph.use_charge:
            self.rxn_graph = self.rxn_graph.uncharge()
        self.rxn_graph.save(rxn_graph_path)
        self.session.add(ConfigValue(name="uncharged_rxn_graph_path", value=rxn_graph_path))
        self.session.add(ConfigValue(name="charged_rxn_graph_path", value=os.path.join(self.results_dir, "charged.rxn")))
        self.session.add(ConfigValue(name="macro_iteration", value="0"))
        self.session.commit()

    def enumerate(self):
        for counter in range(self.n_iter):
            print("STARTING MACRO-ITERATION", counter + 1)
            # updating macro-iteration value
            self.session.add(ConfigValue(name="macro_iteration", value=str(counter + 1)))
            # creating results dir for macro-iteration
            res_dir = os.path.join(self.results_dir, str(counter + 1))
            if not os.path.isdir(res_dir):
                os.mkdir(res_dir)
            run_computations(self.pipeline, db_session=self.session)
        print("==== DONE ENUMERATING ====")

class SimpleEnumerator (Enumerator):

    def __init__(self, rxn_graph: tn.core.RxnGraph, 
                        n_iter: int, 
                        results_dir: str,
                        dask_client: Client,
                        slurm_client: SlurmClient,
                        max_changing_bonds: int=2,
                        ac_filters: Optional[List[tn.iterate.ac_matrix_filters.AcMatrixFilter]]=None,
                        max_reduction: int=0,
                        max_oxidation: int=0,
                        max_abs_charge: int=1,
                        slurm_config_path: Optional[str]=None,
                        program=None,
                        connected_molecules=None,
                        comp_kwdict: Optional[dict]=None,
                        input_type: Optional[tx.io.FileParser]=None,
                        output_type: Optional[tx.io.FileParser]=None,
                        reaction_energy_th: float=40,
                        use_shortest_path: bool=True,
                        sp_energy_th: float=60
                        ):
        # making conversion filters
        conversion_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(max_changing_bonds),
                                tn.iterate.conversion_matrix_filters.OnlySingleBonds()] 
        # making default ac_matrix filters
        if not ac_filters:
            ac_filters = [
                tn.iterate.ac_matrix_filters.MaxBondsPerAtom(),
                tn.iterate.ac_matrix_filters.MaxAtomsOfElement({4: 4, 6: 4})
            ]
        # making charge filters
        charge_filters = [tn.iterate.charge_filters.MaxAbsCharge(max_abs_charge)]
        # parsing external computation kwargs - defaults to ORCA basic energy computation
        if not program:
            if not slurm_config_path:
                slurm_config_path = os.path.join(os.path.dirname(os.path.dirname(tx.__file__)), "scripts", "slurm", "slurm_config.py")
            spec = importlib.util.spec_from_file_location("slurm_config", slurm_config_path)
            slurm_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(slurm_config)
            program = slurm_config.job_dict["ORCA"]["program"]
        if not input_type:
            input_type = tx.io.OrcaIn
        if not output_type:
            output_type = tx.io.OrcaOut
        if not comp_kwdict:
            comp_kwdict = {"input_text": "! OPT"}
        # making build computation & build filter
        build_comp = comps.BuildMolecules(dask_client, connected_molecules)
        build_filter = lambda db_session: db_session.query(build_comp.sql_model.id).filter_by(successful=False)
        # making final computation pipeline
        pipeline = [
            # enumerate elementary reactions & update new specie data in DB
            comps.ElementaryReactionEnumeration(conversion_filters, ac_filters),
            # estimate initial geometries for species
            build_comp,
            # filter molecules without build
            comps.ReduceGraphByCriterion("uncharged", build_filter),
            # enumerate redox reactions & update charge information on species in DB
            comps.RedoxReactionEnumeration(max_reduction, max_oxidation, charge_filters),
            # calculate energies
            comps.ExternalCalculation("energy_calc", slurm_client, program, input_type, comp_kwdict, output_type.extension),
            # read computation results
            comps.ReadCompOutput(dask_client, output_type),
            # energy based reduction
            comps.ReduceGraphByEnergyReducer(
                tn.analyze.network_reduction.SimpleEnergyReduction(reaction_energy_th, use_shortest_path, sp_energy_th), 
                "reduced_graph.rxn"),
            # uncharging charged graph
            comps.UnchargeGraph()
        ]
        super().__init__(rxn_graph, pipeline, n_iter, results_dir)
