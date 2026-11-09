import numpy as np
from scipy.spatial.distance import cdist
from . import chemistry as chem

ELEMENTS = ("C", "N", "O", "S")

CHANNELS = ["heavy_min"] + [
    f"{a}__{b}" for a in ("backbone", "sidechain") for b in ("backbone", "sidechain")
]

CHANNELS += [f"element_{a}__{b}" for a in ELEMENTS for b in ELEMENTS]
CHANNELS += [
    "polar_NO__polar_NO",
    "donor__acceptor",
    "acceptor__donor",
    "donor_acceptor_either",
    "nonpolar_C__nonpolar_C",
    "positive__negative",
    "negative__positive",
    "opposite_charge_either",
    "same_charge",
    "aromatic__aromatic",
    "sulfur__any",
    "any__sulfur",
]

GROUPS = dict(
    heavy=[0],
    backbone_sidechain=list(range(1, 5)),
    element_pairs=list(range(5, 21)),
    chemical_roles=list(range(21, 33)),
)


def pack(residues):
    coords, rid, names, comps, elements = [], [], [], [], []
    for i, residue in enumerate(residues):
        for name, (element, xyz, _) in residue["atoms"].items():
            if element in ("H", "D"):
                continue
            coords.append(xyz)
            rid.append(i)
            names.append(name)
            comps.append(residue["comp"])
            elements.append(element)
    names, comps, elements = map(np.asarray, (names, comps, elements))
    masks = {"any": np.ones(len(rid), bool)}
    masks["backbone"] = np.isin(names, list(chem.BACKBONE))
    masks["sidechain"] = ~masks["backbone"]
    for element in ELEMENTS:
        masks[element] = elements == element
    masks["polar"] = np.isin(elements, ["N", "O"])
    masks["donor"] = np.array(
        [(n == "N" and c != "PRO") or n in chem.SIDE_DONOR.get(c, ()) for n, c in zip(names, comps)]
    )
    masks["acceptor"] = np.array(
        [n in ("O", "OXT") or n in chem.SIDE_ACCEPTOR.get(c, ()) for n, c in zip(names, comps)]
    )
    masks["nonpolar"] = np.array([n in chem.NONPOLAR_CARBON[c] for n, c in zip(names, comps)])
    charge = np.array(
        [
            chem.CHARGED_ATOMS.get(c, (0, ()))[0]
            if n in chem.CHARGED_ATOMS.get(c, (0, ()))[1]
            else 0
            for n, c in zip(names, comps)
        ]
    )
    masks["positive"], masks["negative"] = charge > 0, charge < 0
    masks["aromatic"] = np.array(
        [any(n in ring for ring in chem.RINGS.get(c, ())) for n, c in zip(names, comps)]
    )
    return np.asarray(coords, np.float64), np.asarray(rid, np.int32), masks


def atom_selections(left, right):
    pairs = [("any", "any")]
    pairs += [(a, b) for a in ("backbone", "sidechain") for b in ("backbone", "sidechain")]
    pairs += [(a, b) for a in ELEMENTS for b in ELEMENTS]
    pairs += [
        ("polar", "polar"),
        ("donor", "acceptor"),
        ("acceptor", "donor"),
        None,
        ("nonpolar", "nonpolar"),
        ("positive", "negative"),
        ("negative", "positive"),
        None,
        None,
        ("aromatic", "aromatic"),
        ("S", "any"),
        ("any", "S"),
    ]
    selections = []
    for pair in pairs:
        if pair is None:
            selections.append(None)
        else:
            a, b = pair
            selections.append(left[a][:, None] & right[b][None, :])
    selections[24] = selections[22] | selections[23]
    selections[28] = selections[26] | selections[27]
    selections[29] = (left["positive"][:, None] & right["positive"][None, :]) | (
        left["negative"][:, None] & right["negative"][None, :]
    )
    assert len(selections) == len(CHANNELS) == 33
    return selections


def compute(cif, row):
    cdr, pep, nalpha, starts = chem._parse_residues(cif, row)
    xyz, rid, lm = pack(cdr)
    pxyz, prid, rm = pack(pep)
    distances = cdist(xyz, pxyz, metric="euclidean")  # float64, Angstrom
    codes = rid[:, None] * len(pep) + prid[None, :]
    output = np.full((len(cdr), len(pep), len(CHANNELS)), np.inf, np.float64)
    for k, selection in enumerate(atom_selections(lm, rm)):
        reduced = np.full(len(cdr) * len(pep), np.inf, np.float64)
        np.minimum.at(reduced, codes[selection], distances[selection])
        output[:, :, k] = reduced.reshape(len(cdr), len(pep))
    output[~np.isfinite(output)] = np.nan
    # Both complete disjoint partitions must reconstruct the all-heavy minimum.
    np.testing.assert_allclose(
        np.nanmin(output[:, :, 1:5], axis=-1), output[:, :, 0], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.nanmin(output[:, :, 5:21], axis=-1), output[:, :, 0], rtol=0, atol=0
    )
    assert np.isfinite(output[:, :, 0]).all() and np.all(output[np.isfinite(output)] >= 0)
    return output.astype(np.float32), nalpha, starts
