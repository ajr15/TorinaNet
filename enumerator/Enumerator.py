from sqlalchemy import Column, String
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import importlib
import os
from typing import List, Optional
import torinanet as tn
import torinax as tx
from torinax.pipelines.computations import SqlBase, run_computations, model_lookup_by_table_name
from torinax.clients import SlurmClient
from torinax.utils import atomic_numer_to_symbol
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
        # properly save reaction graph
        rxn_graph_path = os.path.join(self.results_dir, "rxn_graph.rxn")
        self.rxn_graph.save(rxn_graph_path)
        # loading config settings
        self._load_setting("results_dir", self.results_dir, overwrite=True)
        self._load_setting("rxn_graph_path", rxn_graph_path, overwrite)
        self._load_setting("macro_iteration", "0", overwrite)
        # resetting log tables - these are re-made every calculation
        log_table = model_lookup_by_table_name("log")
        self.session.query(log_table).delete()
        # adding reactant relevance entries (always reactants are relevant)
        # entries = []
        # for s in self.rxn_graph.source_species:
        #     print(s.identifier, 0, "reactant")
        #     entries.append(log_table(id=s.identifier, iteration=0, source="reactant"))
        # self.session.add_all(entries)
        # self.session.commit()

    def load_rxn_graph(self) -> tn.core.RxnGraph:
        """Method to load the most recent uncharged reaction graph from database"""
        rxn_graph_path = self.session.query(ConfigValue.value).filter_by(name="rxn_graph_path").one()[0]
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
                    slurm_client: SlurmClient,
                    max_breaking_bonds: int=2,
                    max_forming_bonds: int=2,
                    ac_filters: Optional[List[tn.iterate.ac_matrix_filters.AcMatrixFilter]]=None,
                    slurm_config_path: Optional[str]=None,
                    program=None,
                    comp_kwdict: Optional[dict]=None,
                    input_type: Optional[tx.io.FileParser]=None,
                    output_type: Optional[tx.io.FileParser]=None,
                    reaction_energy_th: float=0.064, # Ha = 40 kcal/mol
                    use_shortest_path: bool=True,
                    sp_energy_th: float=0.096, # Ha = 60 kcal/mol
                    molrank_specie_th: float=0.01,
                    molrank_reactions_th: float=1e-3,
                    molrank_temperature: float=298, # Ha = 40 kcal/mol
                    molrank_energy_scaling_factor: float=30,
                    molrank_energy_conversion_factor: float=4.359744e-18, # Ha to J (must convert to J)
                    molrank_reducer_after: int=0, # iteration after which to apply the leaf reducer
                    min_atomic_energy: Optional[float]=None, 
                    use_mvc: bool=False,
                    max_mvc_samples: int=300,
                    n_mvc_trails: int=5,
                    mvc_metric: str="degree",
                    reflect: bool=True):
        # parsing input kwargs
        self.parse_inputs(max_forming_bonds,
                        max_breaking_bonds,
                        ac_filters,
                        slurm_config_path,
                        program,
                        comp_kwdict,
                        input_type,
                        output_type)
        self.slurm_client = slurm_client
        # making final computation pipeline
        pipeline = [
            # enumerate elementary reactions & update new specie data in DB
            comps.ElementaryReactionEnumeration(self.conversion_filters, self.ac_filters),
            # estimate initial geometries for species
            comps.BuildMolecules(),
            # filter molecules without build
            comps.ReduceGraphByCriterion(self.filter_bad_build_species)]
        if use_mvc:
            # finding minimal vertex cover
            pipeline += [comps.FindMvc(n_trails=n_mvc_trails, max_samples=max_mvc_samples, metric=mvc_metric),
                            # calculate energies for MVC species
                            comps.ExternalCalculation(slurm_client, self.program, self.input_type,
                                                    self.comp_kwdict, self.output_type.extension,
                                                    comp_source="mvc"),
                            # read computation results for MVC species
                            comps.ReadCompOutput(self.output_type),
                            # filter species with bad geometry
                            comps.ReduceGraphByCriterion(self.filter_bad_geometry_species)
                            ]
        if use_mvc or min_atomic_energy:
            # energy based reduction with MVC species data
            pipeline += [comps.ReduceGraphByEnergyReducer(
                            tn.analyze.network_reduction.EnergyReduction.AtomicEnergyReducer(reaction_energy_th,
                                                                                            min_atomic_energy,
                                                                                            use_shortest_path,
                                                                                            sp_energy_th),
                            "mvc_energy_reduced_graph.rxn",
                            use_atomization_energies=True)]
        # now, calculating energies for every specie in graph
        pipeline += [comps.ExternalCalculation(slurm_client, self.program, self.input_type, self.comp_kwdict, self.output_type.extension),
            # read computation results
            comps.ReadCompOutput(self.output_type),
            # filter out species with bad geometries
            comps.ReduceGraphByCriterion(self.filter_bad_geometry_species),
            # energy based reduction for all species
            comps.ReduceGraphByEnergyReducer(
                tn.analyze.network_reduction.EnergyReduction.SimpleEnergyReduction(reaction_energy_th,
                                                                                use_shortest_path,
                                                                                sp_energy_th),
                "energy_reduced_graph.rxn"),
            comps.ReduceGraphByEnergyReducer(
                tn.analyze.network_reduction.KineticReduction.MolRankReduction(rank_th=molrank_specie_th, 
                                                                                target="species",
                                                                                rate_constant_property="k", 
                                                                                estimate_max_constants=True,
                                                                                temperature=molrank_temperature,
                                                                                activation_energy_scaling_factor=molrank_energy_scaling_factor,
                                                                                energy_conversion_factor=molrank_energy_conversion_factor),
                "molrank_species_reduced_graph.rxn",
                apply_after_iter=molrank_reducer_after),
            comps.ReduceGraphByEnergyReducer(
                tn.analyze.network_reduction.KineticReduction.MolRankReduction(rank_th=molrank_reactions_th, 
                                                                                target="reactions",
                                                                                rate_constant_property="k", 
                                                                                estimate_max_constants=True,
                                                                                temperature=molrank_temperature,
                                                                                activation_energy_scaling_factor=molrank_energy_scaling_factor,
                                                                                energy_conversion_factor=molrank_energy_conversion_factor),
                "molrank_reactions_reduced_graph.rxn",
                apply_after_iter=molrank_reducer_after)]
        super().__init__(rxn_graph, pipeline, n_iter, results_dir, reflect)

    @staticmethod
    def filter_bad_geometry_species(db_session):
        # we run here with raw SQL as using the models is somewhat unstable
        q = """SELECT smiles FROM species WHERE NOT (good_geometry == 1 AND successful == 1) AND smiles IN (SELECT id FROM log)"""
        return [s[0] for s in db_session.execute(q)]

    @staticmethod
    def filter_bad_build_species(db_session):
        # we run here with raw SQL as using the models is somewhat unstable
        q = """SELECT smiles FROM species WHERE xyz ISNULL"""
        return [s[0] for s in db_session.execute(q)]

    def pre_enumerate(self):
        """Pre-enumeration calculation for calculating source specie's energies and basic sizes for
        atomization energy estimates"""
        res_dir = os.path.join(self.results_dir, str(0))
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)
        # making preparations for MVC-based computations, they require reactant specie calculation prior to the run
        # add reactant molecules & atoms to specie's table
        specie_table = model_lookup_by_table_name("species")
        # making all species irrelevant for comuputation
        self.session.query(specie_table).update({"relevant": False})
        entries = []
        atoms = [] # keeping tracks of atoms in the system, to submit computations for them as well
        for s in self.rxn_graph.source_species:
            # adding reactants 
            if self.session.query(specie_table).filter_by(smiles=s.identifier).count() == 0:
                s_entry = specie_table(hash_key=comps.ReadSpeciesFromGraph.specie_hash_func(s), smiles=s.identifier, charge=0, relevant=True)
                entries.append(s_entry)
            else:
                # if reactant is in table specie, make it relevant for computation
                self.session.query(specie_table).filter_by(smiles=s.identifier).update({"relevant": True})
            # going over atomic species & adding atom specie computation for atomic energy comp of MVC
            for atom in s.ac_matrix.get_atoms():
                sp = tn.core.BinaryAcMatrix.from_specie(tn.core.Specie("[{}]".format(atomic_numer_to_symbol(atom)))).to_specie()
                if not atom in atoms:
                    atoms.append(atom)
                    if self.session.query(specie_table).filter_by(smiles=sp.identifier).count() == 0:
                        s_entry = specie_table(hash_key=comps.ReadSpeciesFromGraph.specie_hash_func(sp), smiles=sp.identifier, charge=0, relevant=True)
                        entries.append(s_entry)
                    else:
                        # if reactant is in table specie, make it relevant for computation
                        self.session.query(specie_table).filter_by(smiles=sp.identifier).update({"relevant": True})
        self.session.add_all(entries)
        self.session.commit() 
        # building molecules, calculating energies, reading results
        pipeline = [
            # estimate initial geometries for species
            comps.BuildMolecules(),
            # calculate energies for MVC species
            comps.ExternalCalculation(self.slurm_client,
                                      self.program,
                                      self.input_type,
                                      self.comp_kwdict,
                                      self.output_type.extension,
                                      comp_source="reactant"),
            # read computation results for MVC species
            comps.ReadCompOutput(self.output_type),
        ]
        run_computations(pipeline, db_session=self.session)

    def parse_inputs(self,
                        max_forming_bonds: int=2,
                        max_breaking_bonds: int=2,
                        ac_filters: Optional[List[tn.iterate.ac_matrix_filters.AcMatrixFilter]]=None,
                        slurm_config_path: Optional[str]=None,
                        program=None,
                        comp_kwdict: Optional[dict]=None,
                        input_type: Optional[tx.io.FileParser]=None,
                        output_type: Optional[tx.io.FileParser]=None
                ):
        # making conversion filters
        self.conversion_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(max_breaking_bonds + max_forming_bonds),
                                    tn.iterate.conversion_matrix_filters.MaxFormingAndBreakingBonds(max_forming_bonds, max_breaking_bonds),
                                    tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
        # making default ac_matrix filters
        if not ac_filters:
            self.ac_filters = [
                tn.iterate.ac_matrix_filters.MaxBondsPerAtom(),
                tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4}),
                tn.iterate.ac_matrix_filters.MaxComponents(2)
            ]
        else:
            self.ac_filters = ac_filters
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
