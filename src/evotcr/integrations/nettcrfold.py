"""Pinned upstream checkout and lossless conversion from saved prediction archives."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np

UPSTREAM = "https://github.com/mnielLab/NetTCRfold-1.0"
REVISION = "ae17ea27a8345c48046531463a49e90272bc33da"


def read_archive(root: Path, ref: dict) -> bytes:
    from evotcr.artifacts import safe_path

    if "path" in ref:
        blob = safe_path(root, ref["path"]).read_bytes()
    else:
        with tarfile.open(safe_path(root, ref["shard"]), "r:") as shard:
            member = shard.getmember(ref["member"])
            if not member.isfile() or member.size > 256 << 20:
                raise ValueError("Invalid structure shard member")
            blob = shard.extractfile(member).read()
    if hashlib.sha256(blob).hexdigest() != ref["sha256"]:
        raise ValueError("Structure archive checksum mismatch")
    return blob


def install(destination: Path) -> None:
    if not destination.exists():
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM, str(destination)], check=True)
    actual_url = subprocess.check_output(
        ["git", "-C", str(destination), "remote", "get-url", "origin"], text=True
    ).strip()
    if actual_url.removesuffix(".git") != UPSTREAM:
        raise ValueError("Unexpected NetTCRFold source repository")
    subprocess.run(["git", "-C", str(destination), "fetch", "origin", REVISION], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", REVISION], check=True)


def extract(task: dict, root: Path) -> dict:
    """Read three verified original prediction.tar.zst files without extracting paths.

    task has record.chains, record.metadata.cdr3_chain_ranges_zero_based_halfopen,
    and refs=[{path,sha256,seed}]. Chain A=MHC,C=peptide,D=alpha,E=beta.
    """
    import zstandard
    from .distances import compute
    from .confidence import confidence_scores

    refs = sorted(task["refs"], key=lambda r: r["seed"])
    if len(refs) != 3 or len({r["seed"] for r in refs}) != 3:
        raise ValueError("Exactly three distinct structure seeds are required")
    ch = task["record"]["chains"]
    rg = task["record"]["metadata"]["cdr3_chain_ranges_zero_based_halfopen"]
    row = dict(
        peptide=ch["C"],
        hla_trimmed=ch["A"],
        tcra_trimmed=ch["D"],
        tcrb_trimmed=ch["E"],
        tcra_cdr3=ch["D"][slice(*rg["D"])],
        tcrb_cdr3=ch["E"][slice(*rg["E"])],
    )
    qa, qb, values = [], [], []
    wanted = {"model.cif", "raw_confidences.npz", "summary_confidences.json"}
    for ref in refs:
        blob = read_archive(root, ref)
        members = {}
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(blob)) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for member in tar:
                    if member.name in wanted:
                        if member.name in members or not member.isfile() or member.size > 256 << 20:
                            raise ValueError("Invalid prediction archive member")
                        members[member.name] = tar.extractfile(member).read()
        required = wanted - {"raw_confidences.npz"} if "confidence_values" in ref else wanted
        if not required.issubset(members):
            raise ValueError("Prediction archive is incomplete")
        if (
            "model_cif_sha256" in ref
            and hashlib.sha256(members["model.cif"]).hexdigest() != ref["model_cif_sha256"]
        ):
            raise ValueError("Original CIF checksum mismatch")
        dist, na, _ = compute(members["model.cif"], row)
        if dist.shape[1] != 9 or max(na, len(dist) - na) > 20:
            raise ValueError("Published descriptor schema supports 9-mer peptides, CDR3 <=20")
        q = np.where(
            np.isfinite(dist), np.minimum(np.rint(np.nan_to_num(dist) * 8), 254), 255
        ).astype(np.uint8)
        a = np.full((20, 9, 33), 255, np.uint8)
        b = a.copy()
        a[:na] = q[:na]
        b[: len(dist) - na] = q[na:]
        qa.append(a)
        qb.append(b)
        if "raw_confidences.npz" not in members:
            confidence = np.asarray(ref["confidence_values"], dtype=np.float64)
            if confidence.shape != (7,) or not np.isfinite(confidence).all():
                raise ValueError("Compact structures require the seven unchanged confidence inputs")
            values.append(confidence)
            continue
        raw = np.load(io.BytesIO(members["raw_confidences.npz"]), allow_pickle=False)
        tc = raw["token_chain_ids"].astype(str)
        P = np.round(raw["contact_probs"].astype(float), 2)
        E = np.round(raw["pae"].astype(float), 1)
        alpha = np.flatnonzero(tc == "D")
        beta = np.flatnonzero(tc == "E")
        pep = np.flatnonzero(tc == "C")
        ca = alpha[slice(*rg["D"])]
        cb = beta[slice(*rg["E"])]
        tcr = np.r_[alpha, beta]
        cdr = np.r_[ca, cb]
        summary = json.loads(members["summary_confidences.json"])
        cf = {k: raw[k] for k in ["token_chain_ids", "token_res_ids", "atom_chain_ids"]}
        cf.update(pae=E, contact_probs=P, atom_plddts=np.round(raw["atom_plddts"].astype(float), 2))
        conf, _ = confidence_scores(cf, summary, members["model.cif"].decode(), row)
        values.append(
            [
                P[np.ix_(beta, pep)].max(),
                -(E[np.ix_(cb, pep)].mean() + E[np.ix_(pep, cb)].mean()) / 2,
                conf["plddt_peptide_mean"],
                -min(E[np.ix_(tcr, pep)].min(), E[np.ix_(pep, tcr)].min()),
                P[np.ix_(tcr, pep)].max(),
                -min(E[np.ix_(cdr, pep)].min(), E[np.ix_(pep, cdr)].min()),
                summary["iptm"],
            ]
        )
    return dict(
        qa=np.stack(qa),
        qb=np.stack(qb),
        values=np.array(values),
        length=np.array([len(row["tcra_cdr3"]), len(row["tcrb_cdr3"])]),
        cdr3a=row["tcra_cdr3"],
        cdr3b=row["tcrb_cdr3"],
        peptide=row["peptide"],
    )


def reduce_confidence(scalar7: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    if scalar7.ndim != 3 or scalar7.shape[1:] != (3, 7) or len(scalar7) % 10:
        raise ValueError("Expected complete groups: [10N,3,7]")
    base = scalar7.max(1)
    base[:, 2] = scalar7[:, 0, 2]
    base[:, 6] = scalar7[:, 0, 6]
    base[:, 3:7] = (
        rankdata(base.reshape(-1, 10, 7)[:, :, 3:7], axis=1, method="average") / 10
    ).reshape(-1, 4)
    return base
