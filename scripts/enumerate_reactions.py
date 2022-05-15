"""Main script for enumerating elementary reactions"""
import sys; sys.path += ["/home/shaharpit/Personal/TorinaX", "/home/shaharpit/Personal/TorinaNet"]
import argparse
import os
import json
import pandas as pd
import dask as da
from dask_jobqueue import SLURMCluster
import importlib
from dask.distributed import Client
import dask.dataframe as dd
import openbabel as ob
import collections.abc
import torinanet as tn
import torinax as tx
from torinax.utils.openbabel import obmol_to_molecule, molecule_to_obmol


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def parse_input_dictionary(d: dict):
    """Method to parse the input dictionary into dictionary with python objects"""
    # parse elemantary reaction enumeration input variables
    ac_filters = []
    for ac_filter_d in d["elementary_reactions_enumeration"]["ac_matrix_filters"]:
        ac_filter = getattr(tn.iterate.ac_matrix_filters, ac_filter_d["filter_name"])
        if ac_filter_d["filter_name"] == "MaxAtomsOfElement":
            if "max_atoms_of_element_dict" in ac_filter_d["filter_kwargs"]:
                ac_filter_d["filter_kwargs"]["max_atoms_of_element_dict"] = {int(k): v for k, v in ac_filter_d["filter_kwargs"]["max_atoms_of_element_dict"].items()}
        ac_filter = ac_filter(**ac_filter_d["filter_kwargs"])
        ac_filters.append(ac_filter)
    d["elementary_reactions_enumeration"]["ac_matrix_filters"] = ac_filters
    # parse system block
    # parse connected molecules
    connected_mols = {}
    for k, mol_d in d["system"]["connected_molecules"].items():
        mol = obmol_to_molecule(mol_d["xyz"])
        connected_mols[k] = {"mol": mol, 
                            "binding_atom": mol_d["binding_atom"],
                            "charge": mol_d["charge"]}
    d["system"]["connected_molecules"] = connected_mols
    # parse reactants
    reactants = []
    for smiles in d["system"]["reactants"]:
        reactants.append(tn.core.Specie(smiles))
    d["system"]["reactants"] = reactants
    # parse energy calculation block
    # find program
    config_file_path = d["system"]["slurm_client_config"]["client_config_file"]
    if config_file_path == "default":
        config_file_path = os.path.join(os.path.dirname(os.path.dirname(tx.__file__)), "scripts", "slurm", "slurm_config.py")
    spec = importlib.util.spec_from_file_location("slurm_config", config_file_path)
    slurm_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slurm_config)
    d["energy_calculation"]["computation"]["program"] = slurm_config.job_dict[d["energy_calculation"]["computation"]["program"]]
    # find input type
    type_str = d["energy_calculation"]["computation"]["input_type"]
    pkg = importlib.import_module("torinax.io.{}".format(type_str, type_str))
    d["energy_calculation"]["computation"]["input_type"] = getattr(pkg, type_str)
    # find output type
    type_str = d["energy_calculation"]["computation"]["output_type"]
    pkg = importlib.import_module("torinax.io.{}".format(type_str, type_str))
    d["energy_calculation"]["computation"]["output_type"] = getattr(pkg, type_str)
    return d

def estimate_charge(ac_string: str, connected_molecules: dict) -> int:
    """Method to estimate the total charge of a molecule given ac matrix string.
    RETURNS:
        (int) total charge of a molecule, or None"""
    # read molecule
    ac = tn.core.BinaryAcMatrix.from_str(ac_string)
    # check charge of connected molecules
    added_charge = 0
    for i in range(len(ac)):
        if ac.get_atom(i) <= 0:
            added_charge += connected_molecules[ac.get_atom(i)]["charge"]
    # reading to obmol
    obmol = ac.to_obmol()
    # find charges
    charge_model = ob.OBChargeModel.FindType("mmff94")
    if charge_model.ComputeCharges(obmol):
        charges = charge_model.GetFormalCharges()
        return int(sum(charges)) + added_charge


def estimate_charges(dask_client, rxn_graph, specie_df, connected_molecules_charges):
    # estimate charges of molecules
    _estimate_charge = da.delayed(pure=True)(estimate_charge)
    for sid in specie_df.index:
        if specie_df.loc[sid, "charge"].isnull():
            specie_df.loc[sid, "charge"] = _estimate_charge(specie_df.loc[sid, "ac_matrix_str"], connected_molecules_charges)
    # running in parallel on dask cluster & updating specie df
    specie_df = dask_client.compute(specie_df)
    # updating charges for reaction graph
    for sid in specie_df.index:
        specie = rxn_graph.get_specie_from_id(sid)
        specie.charge = specie_df.loc[sid, "charge"]

def charge_reduce_graph(rxn_graph):
    """Reduce reactions that violate charge conservation law, used only when estimating charges (no charge enumeration)"""
    res = rxn_graph.copy()
    for rxn in rxn_graph.reactions:
        r_charge = sum([s.charge for s in rxn.reactants])
        p_charge = sum([s.charge for s in rxn.products])
        if r_charge != p_charge:
            res = res.remove_reaction(rxn)
    return res

class OpenbabelBuildError (Exception):
    pass

class OpenbabelFfError (Exception):
    pass


def build_molecule(xyz_path: str, ac_string: str, sid: str, specie_df, connected_molecules):
    ac_mat = tn.core.BinaryAcMatrix.from_str(ac_string)
    try:
        molecule = ac_mat.build_geometry(connected_molecules)
        molecule.save_to_file(xyz_path)
        specie_df.loc[sid, "built_successfully"] = True
        specie_df.loc[sid, "xyz_path"] = xyz_path
    except OpenbabelBuildError or OpenbabelFfError:
        specie_df.loc[sid, "built_successfully"] = False


def build_molecules(dask_client, xyz_directory: str, specie_df: dd.DataFrame, connected_molecules):
    # find molecules to build
    build_rows = specie_df[specie_df["built_successfully"].isnull()]
    # submit to client build jobs
    ajr = []
    for row in build_rows.iterrows():
        xyz_path = os.path.join(xyz_directory, row.index + ".xyz")
        ajr.append(dask_client.submit(build_molecule, 
                                        xyz_path, 
                                        row["ac_matrix_str"], 
                                        row.index, 
                                        specie_df,
                                        connected_molecules,
                                        pure=False))
    dask_client.compute(ajr, scheduler="processes")


def calculate_row_cmd(specie_df, row, program, input_type, input_file_dir, comp_kwdict, output_extension) -> str:
    """Method to make the shell commands and files required for energy calculation of a specie. returns the command string"""
    molecule = obmol_to_molecule(row["xyz_path"])
    in_file_path = os.path.join(input_file_dir, row["sid"] + "." + input_type.extension)
    specie_df.loc[row["sid"], "input_file"] = in_file_path
    specie_df.loc[row["sid"], "output_file"] = os.path.join(os.path.dirname(input_file_dir), 
                                                            row["sid"] + "_out", row["sid"] + "." + output_extension)
    infile = input_type(in_file_path)
    infile.write_file(molecule, comp_kwdict)
    return program.run_command(in_file_path)
    

def calculate_energies(slurm_client, specie_df, input_type, input_file_dir, program, comp_kwdict):
    # find molecules to caculate energies for
    energy_rows = specie_df[specie_df["built_successfully"] & specie_df["total_energy"].isnull() & specie_df["good_geometry"].isnull()]
    # build input files
    cmds = []
    for row in energy_rows.to_dict(orient="records"):
        cmds.append(calculate_row_cmd(row, program, input_type, input_file_dir, comp_kwdict))
    # submit to client
    slurm_client.submit(cmds)
    # wating for task completion
    slurm_client.wait()


def read_energy_outputs(specie_df, output_type):
    energy_rows = specie_df[specie_df["built_successfully"] & specie_df["total_energy"].isnull() & specie_df["good_geometry"].isnull()]
    for sid, outfile in zip(energy_rows.index, energy_rows["output_file"]):
        f = output_type(outfile)
        outdict = f.read_scalar_data()
        if not outdict["finished_normally"]:
            specie_df.loc[sid, "good_geometry"] = False
        else:
            # checking if optimized to a correct structure
            outspecie = f.read_specie()
            inspecie = obmol_to_molecule(specie_df.loc[sid, "xyz_path"])
            if not compare_species(inspecie, outspecie):
                specie_df.loc[sid, "good_geometry"] = False
            else:
                specie_df.loc[sid, "good_geometry"] = True
                specie_df.loc[sid, "total_energy"] = outdict["total_energy"]


def compare_species(specie1, specie2) -> bool:
    # reading to openbabel & guessing bonds
    ob1 = molecule_to_obmol(specie1)
    ob1.PerceiveBondOrders()
    ob2 = molecule_to_obmol(specie2)
    ob2.PerceiveBondOrders()
    # making SMILES
    conv = ob.OBConversion()
    conv.SetOutFormat("smi")
    smiles1 = conv.WriteString(ob1)
    smiles2 = conv.WriteString(ob2)
    # if smiles are equal -> structures are equal
    return smiles1 == smiles2


def enumerate_elementary_reactions(rxn_graph, ac_filters, max_changing_bonds):
    print(ac_filters[0])
    conversion_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(max_changing_bonds),
                            tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
    stop_cond = tn.iterate.stop_conditions.MaxIterNumber(1)
    iterator = tn.iterate.Iterator(rxn_graph)
    rxn_graph = iterator.enumerate_reactions(conversion_filters, ac_filters, stop_cond, verbose=1)
    return rxn_graph

def uncharge_graph(charged_rxn_graph, uncharged_rxn_graph):
    """Method to 'uncharge' a charged reaction graph - removing charged species/reactions and reducing graph size"""
    rids = set([r._get_id_str() for r in charged_rxn_graph])
    return uncharged_rxn_graph.copy(keep_ids=rids)

def enumerate_redox_reactions(rxn_graph, max_reduction, max_oxidation, max_abs_charge):
    charge_iterator = tn.Iterate.ChargeIterator(rxn_graph, type(rxn_graph))
    return charge_iterator.enumerate_charges(max_reduction, 
                                            max_oxidation, 
                                            [tn.iterate.charge_filters.MaxAbsCharge(max_abs_charge)])
    
def network_energy_reduction(rxn_graph, specie_df, energy_th):
    # removing "bad species" from graph
    bad_species_ids = specie_df[(~specie_df["good_geometry"])].index
    for sid in bad_species_ids:
        if rxn_graph.has_specie_id(sid):
            s = rxn_graph.get_specie_from_id(sid)
            rxn_graph = rxn_graph.remove_specie(s)
    # loading new energies to reaction graph
    for sid in specie_df.index:
        energy = specie_df.loc[sid, "total_energy"]
        if rxn_graph.has_specie_id(sid):
            specie = rxn_graph.get_specie_from_id(sid)
            if not "energy" in specie.properties:
                specie.properties["energy"] = energy
    # energy reduction
    reduced_graph = reaction_energy_reduction(rxn_graph, energy_th)
    return reduced_graph

def reaction_energy_reduction(rxn_graph, energy_th):
    """Method to reduce a reaction graph based on reaction energy differences"""
    res = rxn_graph.copy()
    for rxn in rxn_graph.reactions:
        r_energy = calc_reaction_energy(rxn)
        if r_energy > energy_th:
            res = res.remove_reaction(rxn)
    return res

def calc_reaction_energy(reaction):
    """Calculate energy difference between species and reactants in a reaction"""
    reactants_e = sum([s.properties["energy"] for s in reaction.reactants])
    products_e = sum([s.properties["energy"] for s in reaction.products])
    return products_e - reactants_e


def update_df_from_rxn_graph(specie_df: dd.DataFrame, rxn_graph: tn.core.RxnGraph):
    existing_species = da.persist(set(specie_df.index))
    new_species = dd.from_pandas(pd.DataFrame(), None)
    for specie in rxn_graph.species:
        sid = rxn_graph.make_unique_id(specie)
        if not sid in existing_species:
            new_species = new_species.append({"sid": sid, "ac_matrix_str": specie.ac_matrix.to_str()}, 
                                            ignore_index=True)
    new_species = new_species.set_index("sid")
    return specie_df.append(new_species)


def main():
    # parser to read input json file
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=str, help="input json file for computation")
    input_json = parser.parse_args().input_json
    # some init code
    counter = 1
    species_df = dd.from_pandas(pd.DataFrame(columns=[
        "smiles",
        "ac_matrix_str",
        "charge",
        "built_successfully",
        "good_geometry",
        "total_energy",
        "xyz_path",
        "input_path",
        "output_path"
    ]), npartitions=1)
    # reading input
    with open(input_json, "r") as f:
        input_d = json.load(f)
    # reading default values
    with open(os.path.join(os.path.dirname(__file__), "defaults.json"), "r") as f:
        defaults_dict = json.load(f)
    # updating defaults values with user input
    input_d = recursive_dict_update(defaults_dict, input_d)
    input_d = parse_input_dictionary(input_d)
    parent_results_dir = input_d["system"]["results_dir"]
    # setting up dask client
    dask_cluster = SLURMCluster(**input_d["system"]["dask_client_config"])
    dask_client = Client(dask_cluster)
    dask_cluster.adapt(minimum_jobs=0, maximum_jobs=10)
    # setting up slurm client
    slurm_d = input_d["system"]["slurm_client_config"]
    slurm_client = tx.clients.SlurmClient(slurm_d["cpus_per_task"], slurm_d["memory_per_task"], slurm_d["job_name"])
    # set up reaction graph
    rxn_graph = tn.core.RxnGraph()
    source_species = []
    rxn_graph.set_source_species(input_d["system"]["reactants"], force=True)
    # start main loop
    while True:
        # setting parent step dir
        res_dir = os.path.join(parent_results_dir, str(counter))
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)
        xyz_dir = os.path.join(res_dir, "xyz")
        if not os.path.isdir(xyz_dir):
            os.mkdir(xyz_dir)
        comp_dirs = os.path.join(res_dir, "comp")
        if not os.path.isdir(comp_dirs):
            os.mkdir(comp_dirs)
        inputs_dir = os.path.join(comp_dirs, "inputs")
        if not os.path.isdir(inputs_dir):
            os.mkdir(inputs_dir)
        specie_df_csv_path = os.path.join(parent_results_dir, "specie_df.csv")
        # enumerating elementary reactions
        rxn_graph = enumerate_elementary_reactions(rxn_graph, input_d["elementary_reactions_enumeration"]
                                                                        ["ac_matrix_filters"],
                                                   input_d["elementary_reactions_enumeration"]
                                                    ["conversion_matrix_filters"]["max_changing_bonds"])
        rxn_graph.save(os.path.join(parent_results_dir, "uncharged_rxn_graph.rxn"))
        # update dataframe with new species
        specie_df = update_df_from_rxn_graph(specie_df, rxn_graph)
        # write dataframe to disk
        specie_df.to_csv(specie_df_csv_path)
        # if redox reactions are required, enumerate redox
        if input_d["system"]["enumerate_redox_reactions"]:    
            charged_rxn_graph = enumerate_redox_reactions(rxn_graph, **input_d["redox_reactions_enumeration"])
        else:
            # in case no redox reaction enumeration is made, charges are estimated emperically
            # estimating charges
            estimate_charges(dask_client, rxn_graph, specie_df, input_d["system"]["connected_molecules"])
            # reducing reactions that violate charge conservation law
            charged_rxn_graph = charge_reduce_graph(rxn_graph)
        # saving charged graph
        charged_rxn_graph.save(os.path.join(parent_results_dir, "charged_rxn_graph.rxn"))
        # update dask dataframe with new species & charges
        specie_df = update_df_from_rxn_graph(specie_df, charged_rxn_graph)
        # write dataframe to disk
        specie_df.to_csv(specie_df_csv_path)
        # build new molecules
        build_molecules(xyz_dir, species_df, charged_rxn_graph, input_d["system"]["connected_molecules"])
        # write dataframe to disk
        specie_df.to_csv(specie_df_csv_path)
        # submitting calculation
        comp_d = input_d["energy_calculation"]["computation"]
        calculate_energies(slurm_client, specie_df, comp_d["input_type"], inputs_dir, comp_d["program"], comp_d["comp_kwargs"])
        # write dataframe to disk
        specie_df.to_csv(specie_df_csv_path)
        # energy reduction of graph
        charged_rxn_graph = network_energy_reduction(charged_rxn_graph, specie_df)
        # save graph
        charged_rxn_graph.save(os.path.join(parent_results_dir, "charged_rxn_graph.rxn"))
        # fallback to uncharged reaction graph
        if input_d["system"]["enumerate_redox_reactions"]:
            rxn_graph = uncharge_graph(charged_rxn_graph, rxn_graph)
            # save new grap
            rxn_graph.save(os.path.join(parent_results_dir, "uncharged_rxn_graph.rxn"))

def test():
    pars

if __name__ == "__main__":
    main()