"""Module to contain different methods for generation of atom connectivity matrices"""

def full_bond_ac_matrix(specie):
    rdmol = specie.parse_identifier()
    ac = np.zeros(rdmol.GetNumAtoms(), rdmol.GetNumAtoms())
    # writing off diagonal values - bond orders between atoms
    for bond in rdmol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        t = bond.GetTypeAsDouble()
        ac[i, j] = t
        ac[j, i] = t
    # writing diagonal values - atom numbers
    for i in range(rdmol.GetNumAtoms()):
        ac[i, i] = rdmol.GetAtomWithIdx(i).GetAtomicNum()
    return ac

def binary_ac_matrix(specie):
    rdmol = specie.parse_identifier()
    ac = np.zeros(rdmol.GetNumAtoms(), rdmol.GetNumAtoms())
    # writing off diagonal values - bond orders between atoms
    for bond in rdmol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ac[i, j] = 1
        ac[j, i] = 1
    # writing diagonal values - atom numbers
    for i in range(rdmol.GetNumAtoms()):
        ac[i, i] = rdmol.GetAtomWithIdx(i).GetAtomicNum()
    return ac
