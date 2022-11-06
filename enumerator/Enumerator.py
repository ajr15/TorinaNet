from sqlalchemy import Column, String
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import importlib
import os
from typing import List, Optional
from dask.distributed import Client
import torinanet as tn
import torinax as tx
from torinax.pipelines.computations import SqlBase, run_computations, model_lookup_by_table_name
from torinax.clients import SlurmClient
from . import computations as comps


class ConfigValue (SqlBase):

    __tablename__ = "config"

    name = Column(String(100), primary_key=True)
    value = Column(String(100))


class Enumerator:

    """Class to handle enumeration of elementary reactions"""

    def __init__(self, rxn_graph, pipeline, n_iter: int, results_dir: str, reflect: bool=True):
        self.rxn_graph = rxn_graph
        self.pipeline = pipeline
        self.n_iter = n_iter
        self.results_dir = os.path.abspath(results_dir)
        db_path = os.path.abspath(os.path.join(results_dir, "main.db"))
        if os.path.isfile(db_path) and not reflect:
            os.remove(db_path)
            engine = create_engine("sqlite:///{}".format(db_path))
            SqlBase.metadata.create_all(engine)
        else:
            engine = create_engine("sqlite:///{}".format(db_path))
            SqlBase.metadata.create_all(engine)
        
        self.session = sessionmaker(bind=engine)()
        self.load_settings(overwrite=True)

    def _load_setting(self, name: str, value: Optional[str]=None, overwrite: bool=False):
        """Method to safely load a setting from and to main.db file"""
        query = self.session.query(ConfigValue).filter_by(name=name)
        # if setting is not set, put the value
        if query.count() == 0:
            self.session.add(ConfigValue(name=name, value=value))
        elif overwrite:
            query.update({"value": value})
        # else, don't do anything and use the existing setting

    def load_settings(self, overwrite: bool=False):
        """Method to load settings of enumerator to the SQL database"""
        # properly define results dir
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        # properly saves reaction graph
        if self.rxn_graph.use_charge:
            # saving charged graph
            rxn_graph_path = os.path.join(self.results_dir, "charged.rxn")
            self.rxn_graph.save(rxn_graph_path)
            # uncharging and saving uncharged graph
            uncharged_rxn_graph = self.rxn_graph.__class__(use_charge=False)
            for r in self.rxn_graph.reactions:
                uncharged_rxn_graph.add_reaction(r)
            uncharged_rxn_graph.set_source_species(self.rxn_graph.source_species)                
            rxn_graph_path = os.path.join(self.results_dir, "uncharged.rxn")
            uncharged_rxn_graph.save(rxn_graph_path)
        else:
            rxn_graph_path = os.path.join(self.results_dir, "uncharged.rxn")
            self.rxn_graph.save(rxn_graph_path)
        # loading config settings
        self._load_setting("results_dir", self.results_dir, overwrite=True)
        self._load_setting("uncharged_rxn_graph_path", rxn_graph_path, overwrite)
        self._load_setting("charged_rxn_graph_path", os.path.join(self.results_dir, "charged.rxn"), overwrite)
        self._load_setting("macro_iteration", "0", overwrite)
        # resetting dynamic tables
        for comp in self.pipeline:
            # delete the "relevance" recods for external computations - these are set on each run independently
            if isinstance(comp, comps.ExternalCalculation):
                self.session.query(comp.relevance_model).delete()
                # adding reactant relevance entries (always reactants are relevant)
                entries = []
                for s in self.rxn_graph.source_species:
                    entries.append(comp.relevance_model(id=s._get_charged_id_str(), iteration=0))
                self.session.add_all(entries)
        self.session.commit()

    def load_uncharged_graph(self) -> tn.core.RxnGraph:
        """Method to load the most recent uncharged reaction graph from database"""
        rxn_graph_path = self.session.query(ConfigValue.value).filter_by(name="uncharged_rxn_graph_path").one()[0]
        return tn.core.RxnGraph.from_file(rxn_graph_path)

    def load_charged_graph(self) -> tn.core.RxnGraph:
        """Method to load the most recent charged reaction graph from database"""
        rxn_graph_path = self.session.query(ConfigValue.value).filter_by(name="charged_rxn_graph_path").one()[0]
        return tn.core.RxnGraph.from_file(rxn_graph_path)

    def pre_enumerate(self):
        """Method to run before enumeration process"""
        pass

    def enumerate(self):
        self.pre_enumerate()
        macro_iteration = int(self.session.query(ConfigValue.value).filter_by(name="macro_iteration").one()[0])
        for counter in range(macro_iteration, self.n_iter):
            print("STARTING MACRO-ITERATION", counter + 1)
            # updating macro-iteration value
            self.session.query(ConfigValue).filter_by(name="macro_iteration").update({"value": str(counter + 1)})
            self.session.commit()
            # creating results dir for macro-iteration
            res_dir = os.path.join(self.results_dir, str(counter + 1))
            if not os.path.isdir(res_dir):
                os.mkdir(res_dir)
            run_computations(self.pipeline, db_session=self.session)
        print("==== DONE ENUMERATING ====")

class SimpleEnumerator (Enumerator):

    """Basic enumerator with energy reduction scheme (classic and MVC based) and surrogate kinetic reduction scheme"""

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
                    reaction_energy_th: float=0.064, # Ha = 40 kcal/mol
                    min_electron_energy: Optional[float]=None, 
                    use_shortest_path: bool=True,
                    sp_energy_th: float=0.096, # Ha = 60 kcal/mol
                    use_mvc: bool=True,
                    max_mvc_samples: int=300,
                    n_mvc_trails: int=5,
                    mvc_metric: str="degree",
                    use_kinetics: bool=True,
                    reaction_rate_th: float=2.11e-17, # rate corresponding to "effective barrier" of 40 kcal/mol in 298 K
                    rate_constant_property: str="k", 
                    simulation_time: float=100, 
                    timestep: float=0.01, 
                    reactant_concs: Optional[str]=None, 
                    temperature: float=298,
                    kinetic_solver_kwargs: dict={},
                    reflect: bool=True):
        # parsing input kwargs
        self.parse_inputs(max_changing_bonds,
                        ac_filters,
                        max_abs_charge,
                        slurm_config_path,
                        program,
                        comp_kwdict,
                        input_type,
                        output_type)
        self.max_abs_charge = max_abs_charge
        self.slurm_client = slurm_client
        self.dask_client = dask_client
        self.connected_molecules = connected_molecules
        # making build computation & build filter
        build_comp = comps.BuildMolecules(dask_client, connected_molecules)
        build_filter = lambda db_session: db_session.query(build_comp.sql_model.id).filter_by(successful=False)
        # making final computation pipeline
        pipeline = [
            # enumerate elementary reactions & update new specie data in DB
            comps.ElementaryReactionEnumeration(self.conversion_filters, self.ac_filters),
            # estimate initial geometries for species
            build_comp,
            # filter molecules without build
            comps.ReduceGraphByCriterion("uncharged", build_filter),
            # enumerate redox reactions & update charge information on species in DB
            comps.RedoxReactionEnumeration(max_reduction, max_oxidation, self.charge_filters)]
        if use_mvc:
            # finding minimal vertex cover
            pipeline += [comps.FindMvc(n_trails=n_mvc_trails, max_samples=max_mvc_samples, metric=mvc_metric),
                            # calculate energies for MVC species
                            comps.ExternalCalculation(slurm_client, self.program, self.input_type,
                                                    self.comp_kwdict, self.output_type.extension,
                                                    specie_tablename="mvc_species",
                                                    name="energy_comp"),
                            # read computation results for MVC species
                            comps.ReadCompOutput(dask_client, self.output_type, comp_output_table_name="energy_comp")]
        if use_mvc or min_electron_energy:
            # energy based reduction with MVC species data
            pipeline += [comps.ReduceGraphByEnergyReducer(
                            tn.analyze.network_reduction.EnergyReduction.MinEnergyReduction(reaction_energy_th,
                                                                                            min_electron_energy,
                                                                                            use_shortest_path,
                                                                                            sp_energy_th),
                            "charged",
                            "mvc_energy_reduced_graph.rxn",
                            energy_comp_table_name="energy_comp")]
            if use_kinetics:
                pipeline += [comps.ReduceGraphByEnergyReducer(
                                tn.analyze.network_reduction.KineticReduction.SimpleKineticsReduction(reaction_rate_th, 
                                                                                                    rate_constant_property, 
                                                                                                    simulation_time, 
                                                                                                    timestep, 
                                                                                                    reactant_concs, 
                                                                                                    temperature,
                                                                                                    energy_conversion_factor=4.359744e-18, 
                                                                                                    specie_energy_property_name="energy", 
                                                                                                    estimate_max_constants=True,
                                                                                                    **kinetic_solver_kwargs),
                                "charged",
                                "mvc_kinetic_reduced_graph.rxn",
                                energy_comp_tablename="energy_comp")]
        # now, calculating energies for every specie in graph
        pipeline += [comps.ExternalCalculation(slurm_client, self.program, self.input_type, self.comp_kwdict, self.output_type.extension, name="energy_comp"),
            # read computation results
            comps.ReadCompOutput(dask_client, self.output_type, comp_output_table_name="energy_comp"),
            # energy based reduction for all species
            comps.ReduceGraphByEnergyReducer(
                tn.analyze.network_reduction.EnergyReduction.SimpleEnergyReduction(reaction_energy_th,
                                                                                use_shortest_path,
                                                                                sp_energy_th),
                "charged",
                "energy_reduced_graph.rxn",
                energy_comp_table_name="energy_comp")]
        if use_kinetics:
            pipeline += [comps.ReduceGraphByEnergyReducer(
                            tn.analyze.network_reduction.KineticReduction.SimpleKineticsReduction(reaction_rate_th, 
                                                                                                rate_constant_property, 
                                                                                                simulation_time, 
                                                                                                timestep, 
                                                                                                reactant_concs, 
                                                                                                temperature,
                                                                                                energy_conversion_factor=4.359744e-18, 
                                                                                                specie_energy_property_name="energy", 
                                                                                                estimate_max_constants=True,
                                                                                                comp_output_table_name="energy_comp"
                                                                                                **kinetic_solver_kwargs),
                            "charged",
                            "kinetic_reduced_graph.rxn")]
            # uncharging charged graph
        pipeline += [comps.UnchargeGraph()]
        super().__init__(rxn_graph, pipeline, n_iter, results_dir, reflect)

    def pre_enumerate(self):
        """Pre-enumeration calculation for calculating source specie's energies and basic sizes for
        atomization energy estimates"""
        res_dir = os.path.join(self.results_dir, str(0))
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)
        # add reactant molecules & atoms to specie's table
        specie_model = model_lookup_by_table_name("uncharged_species")
        charge_model = model_lookup_by_table_name("charged_species")
        # making all species irrelevant for comuputation
        self.session.query(charge_model).update({"relevant": False})
        entries = []
        for s in self.rxn_graph.source_species:
            # adding reactants
            sid = self.rxn_graph.make_unique_id(s)
            if self.session.query(specie_model).filter_by(id=sid).count() == 0:
              s_entry = specie_model(id=sid, ac_matrix_str=s.ac_matrix._to_str(), smiles=s.ac_matrix.to_specie().identifier)
              entries.append(s_entry)
            if self.session.query(charge_model).filter_by(id=s._get_charged_id_str()).count() == 0:
              c_entry = charge_model(id=s._get_charged_id_str(), charge=s.charge, uncharged_sid=sid)
              entries.append(c_entry)
            else:
              # if specie entry exists, make it relevant
              self.session.query(charge_model).filter_by(id=s._get_charged_id_str()).update({"relevant": True})
        self.session.add_all(entries)
        self.session.commit() 
        # building molecules, calculating energies, reading results
        pipeline = [
            # estimate initial geometries for species
            comps.BuildMolecules(self.dask_client,
                                 self.connected_molecules),
            # calculate energies for MVC species
            comps.ExternalCalculation(self.slurm_client,
                                      self.program,
                                      self.input_type,
                                      self.comp_kwdict,
                                      self.output_type.extension,
                                      name="energy_comp"),
            # read computation results for MVC species
            comps.ReadCompOutput(self.dask_client, self.output_type, comp_output_table_name="energy_comp"),
        ]
        run_computations(pipeline, db_session=self.session)



    def parse_inputs(self,
                        max_changing_bonds: int=2,
                        ac_filters: Optional[List[tn.iterate.ac_matrix_filters.AcMatrixFilter]]=None,
                        max_abs_charge: int=1,
                        slurm_config_path: Optional[str]=None,
                        program=None,
                        comp_kwdict: Optional[dict]=None,
                        input_type: Optional[tx.io.FileParser]=None,
                        output_type: Optional[tx.io.FileParser]=None
                ):
        # making conversion filters
        self.conversion_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(max_changing_bonds),
                                tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
        # making default ac_matrix filters
        if not ac_filters:
            self.ac_filters = [
                tn.iterate.ac_matrix_filters.MaxBondsPerAtom(),
                tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4})
            ]
        else:
            self.ac_filters = ac_filters
        # making charge filters
        self.charge_filters = [tn.iterate.charge_filters.MaxAbsCharge(max_abs_charge)]
        # parsing external computation kwargs - defaults to ORCA basic energy computation
        if not program:
            if not slurm_config_path:
                slurm_config_path = os.path.join(os.path.dirname(os.path.dirname(tx.__file__)), "scripts", "slurm", "slurm_config.py")
            spec = importlib.util.spec_from_file_location("slurm_config", slurm_config_path)
            slurm_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(slurm_config)
            self.program = slurm_config.job_dict["orca"]["program"]
        else:
            self.program = program
        if not input_type:
            self.input_type = tx.io.OrcaIn
        else:
            self.input_type = input_type
        if not output_type:
            self.output_type = tx.io.OrcaOut
        else:
            self.output_type = output_type
        if not comp_kwdict:
            self.comp_kwdict = {"input_text": "! OPT"}
        else:
            self.comp_kwdict = comp_kwdict
