"""Strict AF3 mmCIF geometry extraction for existing CDR3--peptide structures.

Only stdlib and NumPy are needed. Chemistry is a fixed, label-independent
standard-amino-acid approximation. No hydrogens are inferred and no interaction
here is an experimentally established hydrogen bond, salt bridge, or pi stack.
"""

from __future__ import annotations

import math
import shlex
import numpy as np

VERSION = "geometry-1.1"
BIT_DEFINITIONS = {
    0: "any heavy atom distance <= 4.0 Angstrom",
    1: "any heavy atom distance <= 5.0 Angstrom",
    2: "any N/O--N/O distance <= 3.5 Angstrom",
    3: "any N/O--N/O distance <= 4.0 Angstrom",
    4: "compatible N/O donor--acceptor distance <= 3.5 Angstrom; no H/angle test",
    5: "nonpolar C--C distance <= 4.0 Angstrom",
    6: "nonpolar C--C distance <= 4.5 Angstrom",
    7: "nonpolar C--C distance <= 5.0 Angstrom",
    8: "opposite charged side-chain heavy atoms distance <= 4.0 Angstrom",
    9: "opposite charged side-chain heavy atoms distance <= 5.5 Angstrom",
    10: "same charged side-chain heavy atoms distance <= 4.0 Angstrom",
    11: "aromatic parallel proximity; complete ring geometry",
    12: "aromatic T-shaped proximity; complete ring geometry",
    13: "N/O--N/O <= 3.5 Angstrom involving at least one backbone atom",
    14: "N/O--N/O <= 3.5 Angstrom with both atoms in side chains",
    15: "any heavy atom distance < 2.0 Angstrom; potential clash",
}
AROMATIC_CRITERIA = {
    "centroid_distance_max_angstrom": 5.5,
    "acute_normal_angle_parallel_max_degree": 30.0,
    "acute_normal_angle_t_shape_min_degree": 60.0,
    "minimum_of_two_lateral_offsets_max_angstrom": 2.0,
    "plane": "least-squares SVD through complete ring heavy atoms",
    "multiple_rings_per_residue": "any qualifying ring pair sets one residue-pair bit",
}
CHEMISTRY_NOTES = {
    "nonpolar_carbon": "fixed atom-name table: carbon attached only to C/H; excludes CA, carbonyl C, and carbons directly attached to N/O/S",
    "polar": "N/O atoms; a broader category than donor--acceptor compatibility",
    "donor_acceptor": "fixed N/O atom typing; no hydrogens, protonation, tautomer or angular test; His ND1/NE2 are potential donor/acceptors",
    "charged": "Asp/Glu carboxylate O negative; Lys NZ and Arg NE/NH1/NH2 positive; His and termini excluded; atom-distance rather than charge-centre test",
    "backbone": "N, CA, C, O, OXT; only N/O enter polar tests",
    "plddt": "arithmetic mean of CIF B_iso_or_equiv over all atoms in each residue; CIF stores two decimals",
    "boundary": "all distances <= except potential clash < 2; calculations float64",
}

AA3_TO_1 = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
    )
)
SIDECHAIN = {
    "ALA": "CB",
    "ARG": "CB CG CD NE CZ NH1 NH2",
    "ASN": "CB CG OD1 ND2",
    "ASP": "CB CG OD1 OD2",
    "CYS": "CB SG",
    "GLN": "CB CG CD OE1 NE2",
    "GLU": "CB CG CD OE1 OE2",
    "GLY": "",
    "HIS": "CB CG ND1 CD2 CE1 NE2",
    "ILE": "CB CG1 CG2 CD1",
    "LEU": "CB CG CD1 CD2",
    "LYS": "CB CG CD CE NZ",
    "MET": "CB CG SD CE",
    "PHE": "CB CG CD1 CD2 CE1 CE2 CZ",
    "PRO": "CB CG CD",
    "SER": "CB OG",
    "THR": "CB OG1 CG2",
    "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH",
    "VAL": "CB CG1 CG2",
}
EXPECTED_HEAVY = {res: set(("N CA C O " + atoms).split()) for res, atoms in SIDECHAIN.items()}
NONPOLAR_CARBON = {
    res: frozenset(names.split())
    for res, names in {
        "ALA": "CB",
        "ARG": "CB CG",
        "ASN": "CB",
        "ASP": "CB",
        "CYS": "",
        "GLN": "CB CG",
        "GLU": "CB CG",
        "GLY": "",
        "HIS": "CB",
        "ILE": "CB CG1 CG2 CD1",
        "LEU": "CB CG CD1 CD2",
        "LYS": "CB CG CD",
        "MET": "CB",
        "PHE": "CB CG CD1 CD2 CE1 CE2 CZ",
        "PRO": "CB CG",
        "SER": "",
        "THR": "CG2",
        "TRP": "CB CG CD2 CE3 CZ2 CZ3 CH2",
        "TYR": "CB CG CD1 CD2 CE1 CE2",
        "VAL": "CB CG1 CG2",
    }.items()
}
SIDE_DONOR = {
    res: frozenset(names.split())
    for res, names in {
        "ARG": "NE NH1 NH2",
        "ASN": "ND2",
        "GLN": "NE2",
        "HIS": "ND1 NE2",
        "LYS": "NZ",
        "SER": "OG",
        "THR": "OG1",
        "TRP": "NE1",
        "TYR": "OH",
    }.items()
}
SIDE_ACCEPTOR = {
    res: frozenset(names.split())
    for res, names in {
        "ASN": "OD1",
        "ASP": "OD1 OD2",
        "GLN": "OE1",
        "GLU": "OE1 OE2",
        "HIS": "ND1 NE2",
        "SER": "OG",
        "THR": "OG1",
        "TYR": "OH",
    }.items()
}
CHARGED_ATOMS = {
    "ASP": (-1, frozenset(["OD1", "OD2"])),
    "GLU": (-1, frozenset(["OE1", "OE2"])),
    "LYS": (1, frozenset(["NZ"])),
    "ARG": (1, frozenset(["NE", "NH1", "NH2"])),
}
RINGS = {
    res: tuple(tuple(r.split()) for r in rings)
    for res, rings in {
        "PHE": ["CG CD1 CE1 CZ CE2 CD2"],
        "TYR": ["CG CD1 CE1 CZ CE2 CD2"],
        "HIS": ["CG ND1 CE1 NE2 CD2"],
        "TRP": ["CG CD1 NE1 CE2 CD2", "CD2 CE2 CZ2 CH2 CZ3 CE3"],
    }.items()
}
BACKBONE = frozenset(["N", "CA", "C", "O", "OXT"])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _atom_rows(cif_bytes):
    """Read one AF3 atom-site loop; reject unsupported wrapping, never guess."""
    lines = (
        cif_bytes.decode("utf-8").splitlines()
        if isinstance(cif_bytes, bytes)
        else cif_bytes.splitlines()
    )
    start = next((i for i, line in enumerate(lines) if line.startswith("_atom_site.")), None)
    _require(start is not None, "Missing atom_site loop")
    columns = []
    i = start
    while i < len(lines) and lines[i].startswith("_atom_site."):
        columns.append(lines[i].strip().removeprefix("_atom_site."))
        i += 1
    _require(len(set(columns)) == len(columns), "Duplicate atom_site columns")
    required = [
        "group_PDB",
        "type_symbol",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_seq_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "pdbx_PDB_model_num",
    ]
    _require(set(required) <= set(columns), "Required atom_site columns missing")
    indices = [columns.index(k) for k in required]
    rows = []
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == "#" or line == "loop_" or line.startswith("_") or line.startswith("data_"):
            break
        # Standard AF3 rows need only str.split. Quoted CIF values use shlex.
        parts = shlex.split(line, posix=True) if ('"' in line or "'" in line) else line.split()
        _require(len(parts) == len(columns), "Unsupported wrapped or malformed atom_site row")
        _require(parts[indices[0]] in ("ATOM", "HETATM"), "Unexpected atom_site record")
        if parts[indices[5]] in ("C", "D", "E"):
            rows.append(tuple(parts[j] for j in indices))
    _require(bool(rows), "No target-chain atoms found")
    return rows


def _parse_residues(cif_bytes, source_row):
    expected = {
        "C": source_row["peptide"],
        "D": source_row["tcra_trimmed"],
        "E": source_row["tcrb_trimmed"],
    }
    _require(all(isinstance(s, str) and s for s in expected.values()), "Empty sequence metadata")
    residues = {c: {} for c in expected}
    for _, element, name, alt, comp, chain, seqid, x, y, z, occ, b, model in _atom_rows(cif_bytes):
        _require(alt in (".", "?"), "Alternate locations unsupported")
        _require(model == "1", "Multiple/noncanonical model number unsupported")
        _require(comp in AA3_TO_1, "Nonstandard residue in target chain")
        num = int(seqid)
        _require(1 <= num <= len(expected[chain]), "Residue number outside metadata sequence")
        _require(
            AA3_TO_1[comp] == expected[chain][num - 1], "Residue identity does not match metadata"
        )
        _require(float(occ) == 1.0, "Nonunit atom occupancy")
        pos = (float(x), float(y), float(z))
        plddt = float(b)
        _require(all(math.isfinite(v) for v in pos) and math.isfinite(plddt), "Nonfinite atom data")
        _require(0 <= plddt <= 100, "pLDDT outside [0,100]")
        _require(element in ("C", "N", "O", "S", "H", "D"), "Unsupported element")
        if element not in ("H", "D"):
            _require(name in EXPECTED_HEAVY[comp] or name == "OXT", "Unexpected heavy atom name")
            _require(element == name[0], "Atom name and element disagree")
            _require(name != "OXT" or num == len(expected[chain]), "Nonterminal OXT atom")
        residue = residues[chain].setdefault(num, {"comp": comp, "atoms": {}})
        _require(residue["comp"] == comp, "Mixed residue identities")
        _require(name not in residue["atoms"], "Duplicate atom name / alternate conformation")
        residue["atoms"][name] = (element, pos, plddt)
    for chain, seq in expected.items():
        _require(
            set(residues[chain]) == set(range(1, len(seq) + 1)), "Missing target-chain residues"
        )
        for residue in residues[chain].values():
            present = {n for n, (e, _, _) in residue["atoms"].items() if e not in ("H", "D")}
            _require(EXPECTED_HEAVY[residue["comp"]] <= present, "Missing standard heavy atoms")
    selected = []
    starts = []
    for chain, prefix in [("D", "tcra"), ("E", "tcrb")]:
        cdr = source_row[prefix + "_cdr3"]
        seq = expected[chain]
        _require(isinstance(cdr, str) and cdr and seq.count(cdr) == 1, "CDR3 must map uniquely")
        first = seq.index(cdr) + 1
        selected.append([residues[chain][n] for n in range(first, first + len(cdr))])
        starts.append(first)
    return (
        selected[0] + selected[1],
        [residues["C"][n] for n in range(1, len(expected["C"]) + 1)],
        len(selected[0]),
        starts,
    )


def _pack(residues):
    coords = []
    rid = []
    polar = []
    hydrophobic = []
    donor = []
    acceptor = []
    charge = []
    backbone = []
    sums = []
    counts = []
    rings = []
    for i, residue in enumerate(residues):
        comp, atoms = residue["comp"], residue["atoms"]
        # Integer hundredths retain the exact saved CIF B values and allow auditing.
        sums.append(sum(round(a[2] * 100) for a in atoms.values()))
        counts.append(len(atoms))
        sign, charge_names = CHARGED_ATOMS.get(comp, (0, ()))
        for name, (element, xyz, _) in atoms.items():
            if element in ("H", "D"):
                continue
            coords.append(xyz)
            rid.append(i)
            polar.append(element in ("N", "O"))
            hydrophobic.append(name in NONPOLAR_CARBON[comp])
            donor.append((name == "N" and comp != "PRO") or name in SIDE_DONOR.get(comp, ()))
            acceptor.append(name in ("O", "OXT") or name in SIDE_ACCEPTOR.get(comp, ()))
            charge.append(sign if name in charge_names else 0)
            backbone.append(name in BACKBONE)
        for names in RINGS.get(comp, ()):
            xyz = np.asarray([atoms[name][1] for name in names], dtype=np.float64)
            centroid = xyz.mean(axis=0)
            _, singular, vt = np.linalg.svd(xyz - centroid, full_matrices=False)
            _require(singular[1] > 0.1, "Degenerate aromatic ring plane")
            normal = vt[-1]
            _require(
                np.isfinite(normal).all() and abs(np.linalg.norm(normal) - 1) < 1e-10,
                "Invalid ring normal",
            )
            rings.append((i, centroid, normal))
    return {
        "coords": np.asarray(coords, np.float64),
        "rid": np.asarray(rid, np.int32),
        "polar": np.asarray(polar, bool),
        "hydro": np.asarray(hydrophobic, bool),
        "donor": np.asarray(donor, bool),
        "acceptor": np.asarray(acceptor, bool),
        "charge": np.asarray(charge, np.int8),
        "backbone": np.asarray(backbone, bool),
        "plddt_sum": np.asarray(sums, np.int64),
        "plddt_count": np.asarray(counts, np.int32),
        "rings": rings,
    }


def _aromatic_masks(left, right, mask):
    parallel = tshape = 0
    for i, c1, n1 in left:
        for j, c2, n2 in right:
            delta = c2 - c1
            dist2 = float(delta @ delta)
            if dist2 > 5.5**2:
                continue
            cosine = min(1.0, max(0.0, abs(float(n1 @ n2))))
            angle = math.degrees(math.acos(cosine))
            offset1 = max(0.0, dist2 - float(delta @ n1) ** 2)
            offset2 = max(0.0, dist2 - float(delta @ n2) ** 2)
            if min(offset1, offset2) > 2.0**2:
                continue
            if angle <= 30.0:
                mask[i, j] |= np.uint16(1 << 11)
                parallel += 1
            if angle >= 60.0:
                mask[i, j] |= np.uint16(1 << 12)
                tshape += 1
    return parallel, tshape


def extract_geometry(cif_bytes, source_row):
    """Return residue-pair masks/counts, pLDDT and mapping audit.

    Rows are CDR3alpha followed by CDR3beta in sequence order; columns are the
    peptide in sequence order. n_alpha splits the CDR3 rows. counts order is
    heavy<=4, N/O polar<=3.5, nonpolar-carbon<=4.5 atom-pair counts. All positive
    mask bits count each residue pair once regardless of atom/ring multiplicity.
    """
    cdr, peptide, n_alpha, starts = _parse_residues(cif_bytes, source_row)
    left, right = _pack(cdr), _pack(peptide)
    nleft, nright = len(cdr), len(peptide)
    delta = left["coords"][:, None, :] - right["coords"][None, :, :]
    d2 = np.einsum("ijk,ijk->ij", delta, delta)
    paircode = left["rid"][:, None] * nright + right["rid"][None, :]
    mask = np.zeros((nleft, nright), dtype=np.uint16)

    def count(condition):
        values = np.bincount(paircode[condition], minlength=nleft * nright)
        _require(values.max(initial=0) <= np.iinfo(np.uint16).max, "Atom-pair count overflow")
        return values.reshape(nleft, nright).astype(np.uint16)

    def add(bit, condition):
        values = count(condition)
        mask[values > 0] |= np.uint16(1 << bit)
        return values

    counts = np.empty((3, nleft, nright), dtype=np.uint16)
    counts[0] = add(0, d2 <= 4.0**2)
    add(1, d2 <= 5.0**2)
    polar = left["polar"][:, None] & right["polar"][None, :]
    counts[1] = add(2, polar & (d2 <= 3.5**2))
    add(3, polar & (d2 <= 4.0**2))
    compatible = (left["donor"][:, None] & right["acceptor"][None, :]) | (
        left["acceptor"][:, None] & right["donor"][None, :]
    )
    add(4, compatible & (d2 <= 3.5**2))
    hydro = left["hydro"][:, None] & right["hydro"][None, :]
    add(5, hydro & (d2 <= 4.0**2))
    counts[2] = add(6, hydro & (d2 <= 4.5**2))
    add(7, hydro & (d2 <= 5.0**2))
    charge_product = left["charge"][:, None] * right["charge"][None, :]
    add(8, (charge_product == -1) & (d2 <= 4.0**2))
    add(9, (charge_product == -1) & (d2 <= 5.5**2))
    add(10, (charge_product == 1) & (d2 <= 4.0**2))
    backbone = left["backbone"][:, None] | right["backbone"][None, :]
    add(13, polar & backbone & (d2 <= 3.5**2))
    add(14, polar & ~backbone & (d2 <= 3.5**2))
    add(15, d2 < 2.0**2)
    aromatic_counts = _aromatic_masks(left["rings"], right["rings"], mask)
    result = {
        "mask": mask,
        "counts": counts,
        "n_alpha": n_alpha,
        "plddt_cdr3": left["plddt_sum"] / left["plddt_count"] / 100.0,
        "plddt_peptide": right["plddt_sum"] / right["plddt_count"] / 100.0,
        "plddt_cdr3_sum": left["plddt_sum"],
        "plddt_cdr3_count": left["plddt_count"],
        "plddt_peptide_sum": right["plddt_sum"],
        "plddt_peptide_count": right["plddt_count"],
        "audit": {
            "version": VERSION,
            "n_cdr3": nleft,
            "n_peptide": nright,
            "n_alpha": n_alpha,
            "cdr3_starts_1based": starts,
            "n_cdr3_heavy_atoms": len(left["coords"]),
            "n_peptide_heavy_atoms": len(right["coords"]),
            "n_cdr3_rings": len(left["rings"]),
            "n_peptide_rings": len(right["rings"]),
            "parallel_ring_pair_hits": aromatic_counts[0],
            "t_shape_ring_pair_hits": aromatic_counts[1],
            "all_target_residue_identities_and_heavy_atoms_verified": True,
            "altlocs_duplicates_missing_atoms_nonstandard_residues_rejected": True,
        },
    }
    return result
