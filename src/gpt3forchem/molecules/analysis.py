import re
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from collections import defaultdict
import numpy as np

from bidict import bidict

_BINS = {
    0: (-4.989399999999997, 0.098),
    1: (0.098, 1.7063),
    2: (1.7063, 2.8406000000000025),
    3: (2.8406000000000025, 4.168000000000004),
    4: (4.168000000000004, 14.856799999999968),
}

_ELEMENTS = ["C", "H", "O", "N", "P", "S"]
atomic_symbols = bidict(
    {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Tb": 65,
        "Dy": 66,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W": 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Pb": 82,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
        "Fr": 87,
        "Ra": 88,
        "Ac": 89,
        "Th": 90,
        "Pa": 91,
        "U": 92,
        "Np": 93,
        "Pu": 94,
        "Am": 95,
        "Cm": 96,
        "Bk": 97,
        "Cf": 98,
        "Es": 99,
        "Fm": 100,
        "Md": 101,
        "No": 102,
        "Lr": 103,
        "Rf": 104,
        "Db": 105,
        "Sg": 106,
        "Bh": 107,
        "Hs": 108,
        "Mt": 109,
        "Ds": 110,
        "Rg": 111,
        "Uub": 112,
        "Uut": 113,
        "Uuq": 114,
        "Uup": 115,
        "Uuh": 116,
        "Uuo": 118,
    }
)


def composition(molecule):
    """Get the composition of an RDKit molecule:
    Atomic counts, including hydrogen atoms, and any charge.
    For example, fluoride ion (chemical formula F-, SMILES string [F-])
    returns {9: 1, 0: -1}.

    :param molecule: The molecule to analyze
    :type some_input: An RDKit molecule
    :rtype: A dictionary.
    """
    # Check that there is a valid molecule
    if molecule:

        # Add hydrogen atoms--RDKit excludes them by default
        molecule_with_Hs = Chem.AddHs(molecule)
        comp = defaultdict(lambda: 0)

        # Get atom counts
        for atom in molecule_with_Hs.GetAtoms():
            atom_symb = atomic_symbols.inverse[atom.GetAtomicNum()]
            comp[atom_symb] += 1

        return dict(comp)


def extract_mol_text_from_completion(completion):
    return completion.split("@")[0].strip()


def get_log_p_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MolLogP(mol)
    except Exception:
        return None


def get_num_element(string, element):
    num = re.findall(f"([\d+]) {element}", string)
    try:
        num = int(num[0])
    except Exception:
        num = 0
    return num


def get_composition_from_string(string, composition):
    comp = {}
    for element in composition:
        num = get_num_element(string, element)
        comp[element] = num
    return comp


def get_distance(prediction, bin, bins=_BINS):
    in_bin = (prediction >= bins[bin][0]) & (prediction < bins[bin][1])
    if in_bin:
        loss = 0
    else:
        # compute the minimum distance to bin
        left_edge_distance = abs(prediction - bins[bin][0])
        right_edge_distance = abs(prediction - bins[bin][1])
        loss = min(left_edge_distance, right_edge_distance)
    return loss


def get_composition_mistmatch(smiles, prompt_composition):
    found_composition = composition(Chem.MolFromSmiles(smiles))
    return composition_mismatch(prompt_composition, found_composition)


def composition_mismatch(composition: dict, found: dict):
    try:
        distances = []

        # We also might have the case the there are keys that the input did not contain
        all_keys = set(composition.keys()) & set(found.keys())

        expected_len = []
        found_len = []

        for key in all_keys:
            try:
                expected = composition[key]
            except KeyError:
                expected = 0
            expected_len.append(expected)
            try:
                f = found[key]
            except KeyError:
                f = 0
            found_len.append(f)

            distances.append(np.abs(expected - f))

        expected_len = sum(expected_len)
        found_len = sum(found_len)
        return {
            "distances": distances,
            "min": np.min(distances),
            "max": np.max(distances),
            "mean": np.mean(distances),
            "expected_len": expected_len,
            "found_len": found_len,
        }
    except Exception as e:
        return {
            "distances": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "expected_len": np.nan,
            "found_len": np.nan,
        }


def get_log_p_from_string(string):
    logP = re.findall(r"logP of (\d+)", string)
    return int(logP[0])


def analyze_completion(prompt, completion):
    completion = completion["choices"][0]["text"]
    smiles = extract_mol_text_from_completion(completion)
    requested_composition = get_composition_from_string(prompt, _ELEMENTS)
    requested_logp = get_log_p_from_string(prompt)

    predicted_log_p = get_log_p_from_smiles(smiles)
    distance = get_distance(predicted_log_p, requested_logp)
    composition_mismatch = get_composition_mistmatch(smiles, requested_composition)

    res = {
        "prompt": prompt,
        "completion": completion,
        "smiles": smiles,
        "predicted_log_p": predicted_log_p,
        "requested_log_p": requested_logp,
        "requested_composition": requested_composition,
        "distance": distance,
    }

    res.update(composition_mismatch)
    return res
