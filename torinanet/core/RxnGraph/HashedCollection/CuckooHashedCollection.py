from typing import Callable, Any, Iterator, Tuple
from copy import copy
from itertools import combinations, chain
from rdkit.DataStructs.cDataStructs import BitVectToText
from rdkit.Chem import rdFingerprintGenerator
from enum import Enum
import hashlib
from .HashedCollection import HashedCollection
from ...Reaction import Reaction


class CuckooHashedCollection (HashedCollection):

    """General hash collection using the cuckoo hash algorithm
    Parameters
    ----------
    - hash_func_generator (Callable[[], Callable]): a function that generates a generator of hash function pairs (take object and return integer key)
    - max_trails (int): max amont of kickouts used in the hash procedure
    - max_rehashes (int): max amont of rehashes used in the hash procedure"""

    def __init__(self, hash_func_generator: Iterator[Tuple[Callable[[Any], int]]], max_trails: int=100, max_rehashes: int=5):
        self.max_rehashes = max_rehashes
        self.max_trails = max_trails
        # building a picklable hash func generator - a list with max rehashes number of functions
        self.hash_func_generator = []
        for _ in range(max_rehashes + 2):
            self.hash_func_generator.append(next(hash_func_generator))
        self._current_func_idx = 0
        self._hash_func1, self._hash_func2 = self.hash_func_generator[self._current_func_idx]
        self._table1 = {}
        self._table2 = {}
        self._rehash_counter = 0

    def add(self, obj):
        # add file to collection only if it not there
        key_table = self._find_key_and_table(obj)
        if key_table is None:
            for _ in range(self.max_trails):
                x = self._add(obj)
                # break only if there is no new object to assign
                if x is None:
                    return
            # if the loop fails to reasign the objects, a rehash is made
            self.rehash()
            self.add(obj)

    def delete(self, obj):
        key_table = self._find_key_and_table(obj)
        if key_table is None:
            raise KeyError(obj)
        else:
            table, key = key_table
            del table[key]

    def get(self, obj):
        key_table = self._find_key_and_table(obj)
        if key_table is None:
            raise KeyError(obj)
        else:
            table, key = key_table
            return table[key]

    def get_key(self, obj):
        key_table = self._find_key_and_table(obj)
        if key_table is None:
            raise KeyError(obj)
        else:
            table, key = key_table
            return key

    def has(self, obj) -> bool:
        key_table = self._find_key_and_table(obj)
        return key_table is not None

    def objects(self) -> Iterator[Any]:
        return chain(self._table1.values(), self._table2.values())

    def keys(self) -> Iterator[int]:
        return chain(self._table1.keys(), self._table2.keys())

    def to_dict(self) -> dict:
        return {
            "max_rehashes": self.max_rehashes,
            "max_trails": self.max_trails,
            "hash_func_generator": self.hash_func_generator
        }

    def rehash(self):
        """Method to rehash the collection with new hash functions. triggered by the add method"""
        # do not rehash over max amount of times
        if self._rehash_counter <= self.max_rehashes:
            self._rehash_counter += 1
            # now, rehashing - resetting hash functions
            self._current_func_idx += 1
            self._hash_func1, self._hash_func2 = self.hash_func_generator[self._current_func_idx]
            # getting all objects
            objects = list(self.objects())
            # deleting old hashing
            self._table1 = {}
            self._table2 = {}
            # rehashing everything
            for obj in objects:
                self.add(obj)
        else:
            raise RuntimeError("The collection was rehashed the maximal amount of times")

    def _add(self, obj: Any):
        h1 = self._hash_func1(obj)
        h2 = self._hash_func2(obj)
        # if there is space in the hashes, normally insert it to the free place
        if h1 not in self._table1:
            self._table1[h1] = obj
            return None
        elif h2 not in self._table2:
            self._table2[h2] = obj
            return None
        # if both hashes are occupied, apply the cuckoo method - kick out and re-assign
        else:
            # copies current occupying object into a variable
            occupying = self._table1[h1]
            if hasattr(occupying, "copy"):
                occupying = occupying.copy()
            else:
                occupying = copy(occupying)
            # reassign hash
            self._table1[h1] = obj
            # return kicked out object for reassignment
            return occupying

    def _find_key_and_table(self, obj):
        """Private method to return the dict and hash-key of an object. if object is not found, returns None"""
        # if hash key is in table 1 and the stored value is same, return the stored value
        h1 = self._hash_func1(obj)
        if h1 in self._table1 and self._table1[h1] == obj:
            return self._table1, h1
        # else check second table
        h2 = self._hash_func2(obj)
        if h2 in self._table2 and self._table2[h2] == obj:
            return self._table2, h2
        # else returns None
        return None

    def __len__(self):
        return len(self._table1) + len(self._table2)


class FuncWrapper:

    def __init__(self, f):
        self.f = f
    
    def __call__(self, *args, **kwds):
        return self.f(*args, **kwds)

class FingerprintGenerators (Enum):
    
    RDKIT = FuncWrapper(rdFingerprintGenerator.GetRDKitFPGenerator)
    MORGAN = FuncWrapper(rdFingerprintGenerator.GetMorganGenerator)
    TOPOLOGICAL = FuncWrapper(rdFingerprintGenerator.GetTopologicalTorsionGenerator)
    ATOMPAIRS = FuncWrapper(rdFingerprintGenerator.GetAtomPairGenerator)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)

def generator_to_hash(hash_generator, fp_size: int, n_bits_per_feature: int):
    if hash_generator == FingerprintGenerators.RDKIT:
        gen = hash_generator(fpSize=fp_size, numBitsPerFeature=n_bits_per_feature)
    else:
        gen = hash_generator(fpSize=fp_size)
    return lambda x: int(BitVectToText(gen.GetFingerprint(x.to_rdmol())), 2)

def specie_hash_function_generator(fp_size: int, n_bits_per_feature: int):
    # first make normal combinations of all pairs of generators
    combos = combinations(FingerprintGenerators, 2)
    for combo in combos:
        gen1 = generator_to_hash(combo[0], fp_size, n_bits_per_feature)
        gen2 = generator_to_hash(combo[1], fp_size, n_bits_per_feature)
        yield gen1, gen2
    # later make only pairs with RDKIT fingerprint with increasing bits per feature
    while True:
        n_bits_per_feature += 1
        for gen in [FingerprintGenerators.MORGAN, FingerprintGenerators.TOPOLOGICAL, FingerprintGenerators.ATOMPAIRS]:
            gen1 = generator_to_hash(gen, fp_size, n_bits_per_feature)
            gen2 = generator_to_hash(FingerprintGenerators.RDKIT, fp_size, n_bits_per_feature)
            yield gen1, gen2

class CuckooSpecieCollection (CuckooHashedCollection):

    """Hashed collection of species using the Cuckoo hashing method"""

    def __init__(self, fp_size: int=128, n_bits_per_feature: int=2, max_trails: int = 100, max_rehashes: int = 5):
        self.fp_size = fp_size
        self.n_bits_per_feature = n_bits_per_feature
        self.max_trails = max_trails
        self.max_rehashes = max_rehashes
        hash_func_generator = specie_hash_function_generator(fp_size, n_bits_per_feature)
        super().__init__(hash_func_generator, max_trails, max_rehashes)
    
    def to_dict(self) -> dict:
        return {
            "fp_size": self.fp_size,
            "n_bits_per_feature": self.n_bits_per_feature,
            "max_trails": self.max_trails,
            "max_rehashes": self.max_rehashes
        }

def hash_rxn(string_hash_func, rxn: Reaction, specie_collection: HashedCollection):
    s = "{}={}".format(
        ".".join([specie_collection.get_key(s) for s in rxn.reactants]),
        ".".join([specie_collection.get_key(s) for s in rxn.products])
    )
    return string_hash_func(s)

def reaction_hash_function_generator(specie_collection):
    hashes = list(hashlib.algorithms_available)
    for combo in combinations(hashes, 2):
        hash1 = lambda s: int(hashlib.new(combo[0]).update(s).hexdigest(), 16)
        hash2 = lambda s: int(hashlib.new(combo[1]).update(s).hexdigest(), 16)
        gen1 = lambda rxn: hash_rxn(hash1, rxn, specie_collection)
        gen2 = lambda rxn: hash_rxn(hash2, rxn, specie_collection)
        yield gen1, gen2


class CuckooReactionCollection (CuckooHashedCollection):

    """Hashed collection of reactions using the Cuckoo hashing method. Based on specie hashing of another specie collection (must have static specie hash keys)."""

    def __init__(self, specie_collection: HashedCollection, max_trails: int = 100, max_rehashes: int = 5):
        self.specie_collection = specie_collection
        super().__init__(reaction_hash_function_generator(specie_collection), max_trails, max_rehashes)

    def has(self, reaction: Reaction) -> bool:
        # setting has method to return False if not all species in reactions are in specie collection
        if not all([self.specie_collection.has(s) for s in chain(reaction.reactants, reaction.products)]):
            return False
        # if all species exist, return normal result
        return super().has(reaction)

    def to_dict(self) -> dict:
        raise NotImplementedError("Currently this collection does not support to_dict method")


def hash_rxn_indep(specie_hash_func, rxn: Reaction):
    s = "{}={}".format(
        ".".join(sorted([str(specie_hash_func(s)) for s in rxn.reactants])),
        ".".join(sorted([str(specie_hash_func(s)) for s in rxn.products]))
    ).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest(), 16)

def reaction_hash_function_generator_indep(fp_size: int, n_bits_per_feature: int):
    hashes = specie_hash_function_generator(fp_size, n_bits_per_feature)
    for combo in hashes:
        gen1 = lambda rxn: hash_rxn_indep(combo[0], rxn)
        gen2 = lambda rxn: hash_rxn_indep(combo[1], rxn)
        yield gen1, gen2
        

class IndepCuckooReactionCollection (CuckooHashedCollection):

    """Hashed collection of reactions using the Cuckoo hashing method. Using fingerprints as specie hash directly."""

    def __init__(self, fp_size: int=128, n_bits_per_feature: int=2, max_trails: int = 100, max_rehashes: int = 5):
        self.fp_size = fp_size
        self.n_bits_per_feature = n_bits_per_feature
        self.max_trails = max_trails
        self.max_rehashes = max_rehashes
        super().__init__(reaction_hash_function_generator_indep(fp_size, n_bits_per_feature), max_trails, max_rehashes)

    def to_dict(self) -> dict:
        return {
            "fp_size": self.fp_size,
            "n_bits_per_feature": self.n_bits_per_feature,
            "max_trails": self.max_trails,
            "max_rehashes": self.max_rehashes
        }

