"""Historical equations with direct model input; unavailable CDR1/2 regions omitted."""

import numpy as np
import gemmi

NAMES = {
    "mhc": "MHC",
    "peptide": "肽",
    "alpha": "TCRα",
    "beta": "TCRβ",
    "tcr": "TCRα/β",
    "cdr3alpha": "CDR3α",
    "cdr3beta": "CDR3β",
    "cdr3": "CDR3α/β",
    "cdralpha": "CDR1/2/3α",
    "cdrbeta": "CDR1/2/3β",
    "cdr": "全部CDR",
    "complex": "整个复合物",
}

CHAIN = {"mhc": "A", "peptide": "C", "alpha": "D", "beta": "E"}

INTERFACES = [
    ("tcr", "peptide"),
    ("alpha", "peptide"),
    ("beta", "peptide"),
    ("cdr3", "peptide"),
    ("cdr3alpha", "peptide"),
    ("cdr3beta", "peptide"),
    ("cdralpha", "peptide"),
    ("cdrbeta", "peptide"),
    ("tcr", "mhc"),
    ("tcr", "pmhc"),
    ("peptide", "mhc"),
    ("alpha", "beta"),
]

NAMES["pmhc"] = "pMHC"


def catalogue():
    specs = []

    def add(key, label, family, sign, definition, native=False):
        specs.append(
            dict(
                metric=key,
                label=label,
                family=family,
                sign=sign,
                direction="越高越好" if sign == 1 else "越低越好",
                definition=definition,
                native=native,
            )
        )

    for key, label, sign in [
        ("iptm", "整体 ipTM", 1),
        ("ptm", "整体 pTM", 1),
        ("ranking_score", "AF3 ranking score", 1),
        ("fraction_disordered", "无序区域比例（诊断项）", 1),
        ("has_clash", "严重碰撞标志（诊断项）", -1),
    ]:
        add(key, label, "整体原生置信度", sign, "manifest.best_by_af3_ranking." + key, True)
    add(
        "confidence_only_rank",
        "0.8×ipTM + 0.2×pTM",
        "整体派生置信度",
        1,
        "使用保存的ipTM/pTM，去除ranking score中的无序与碰撞项；无拟合权重",
    )
    for c in CHAIN:
        for kind in ["iptm", "ptm"]:
            add(
                "chain_" + kind + "_" + c,
                NAMES[c] + " 链级 " + kind,
                "链级原生置信度",
                1,
                "chain_" + kind + "对应链的原生值",
                True,
            )
    chains = list(CHAIN)
    for i, l in enumerate(chains):
        for r in chains[i + 1 :]:
            pair = l + "_" + r
            title = NAMES[l] + "–" + NAMES[r]
            add(
                "pair_iptm_" + pair,
                title + " 链对 ipTM",
                "链对原生置信度",
                1,
                "chain_pair_iptm非对角元素；验证两方向一致",
                True,
            )
            add(
                "pair_pae_min_bidir_" + pair,
                title + " 原生最小PAE双向平均",
                "链对原生PAE",
                -1,
                "两方向chain_pair_pae_min的算术平均",
            )
        for r in chains:
            add(
                "pair_pae_min_" + l + "_" + r,
                NAMES[l] + "→" + NAMES[r] + " 原生最小PAE",
                "链对原生PAE",
                -1,
                "chain_pair_pae_min；箭头表示矩阵行链→列链",
                True,
            )
    for stat in ["mean", "min", "max"]:
        add(
            "tcr_peptide_pair_iptm_" + stat,
            "TCRα/β–肽 链对ipTM " + stat,
            "链对派生置信度",
            1,
            "α–肽与β–肽链对ipTM的" + stat + "；两链等权",
        )
    add(
        "pae_complex_mean",
        "整个复合物 PAE均值",
        "PAE区域统计",
        -1,
        "完整PAE矩阵均值，含链内与对角线",
    )
    for l, r in INTERFACES:
        pair = l + "_" + r
        title = NAMES[l] + "–" + NAMES[r]
        definitions = {
            "mean": ("双向平均PAE", -1, "S=(PAE[L,R]+PAE[R,L].T)/2；mean(S)，所有残基对等权"),
            "forward": (
                "行→列平均PAE",
                -1,
                "mean(PAE[L,R])；行是" + NAMES[l] + "，列是" + NAMES[r],
            ),
            "reverse": (
                "列→行平均PAE",
                -1,
                "mean(PAE[R,L])；行是" + NAMES[r] + "，列是" + NAMES[l],
            ),
            "median": ("对称PAE中位数", -1, "median(S)；S为逐残基对双向平均PAE"),
            "q10": ("对称PAE第10百分位", -1, "quantile(S,0.1)，NumPy线性插值；分位数预先固定"),
            "min": ("对称PAE最小值", -1, "min(S)，先逐残基对平均方向，再取最小值"),
            "frac_lt10": ("PAE<10Å残基对比例", 1, "mean(S<10)，阈值10Å预先固定"),
            "contact_weighted": (
                "接触概率加权PAE",
                -1,
                "sum(S*C)/sum(C)，C为AF3接触概率；零权重时回退为mean(S)",
            ),
        }
        for stat, (label, sign, definition) in definitions.items():
            add("pae_" + pair + "_" + stat, title + " " + label, "PAE区域统计", sign, definition)
        for stat, label, definition in [
            ("mean", "平均接触概率", "mean(C)"),
            ("sum", "接触概率总和", "sum(C)，相当于期望接触残基对数；受区域长度影响"),
            ("max", "最大接触概率", "max(C)"),
            (
                "right_best_mean",
                "右侧残基最佳接触概率均值",
                "mean(max(C,axis=0))，每个右侧残基取与左侧的最大接触概率",
            ),
        ]:
            add("contact_" + pair + "_" + stat, title + " " + label, "接触概率", 1, definition)
        add(
            "plddt_contact_" + pair,
            title + " 接触加权pLDDT",
            "pLDDT区域统计",
            1,
            "两侧各以最大跨界面接触概率为残基权重；将两侧加权均值等权平均，零权重回退该侧残基均值",
        )
    for region in [k for k in NAMES if k != "pmhc"]:
        for stat in ["mean", "q10"]:
            add(
                "plddt_" + region + "_" + stat,
                NAMES[region] + " pLDDT " + ("均值" if stat == "mean" else "第10百分位"),
                "pLDDT区域统计",
                1,
                "先对每个残基的所有原子pLDDT平均，再对区域残基"
                + ("等权平均" if stat == "mean" else "取第10百分位"),
            )
    for region in ["complex", *CHAIN]:
        add(
            "atom_plddt_" + region + "_mean",
            NAMES[region] + " 原子加权pLDDT均值",
            "pLDDT原子统计",
            1,
            "直接对该区域全部原子的atom_plddts平均",
        )
    assert len(set(s["metric"] for s in specs)) == len(specs)
    return specs


SPECS = catalogue()


def confidence_scores(conf, summary, cif_text, row):
    tokens = np.asarray(conf["token_chain_ids"])
    resids = np.asarray(conf["token_res_ids"], int)
    pae = np.asarray(conf["pae"], float)
    contacts = np.asarray(conf["contact_probs"], float)
    assert pae.shape == contacts.shape == (len(tokens), len(tokens))
    assert np.isfinite(pae).all() and (pae >= 0).all()
    assert np.isfinite(contacts).all() and ((contacts >= 0) & (contacts <= 1)).all()
    assert np.max(np.abs(contacts - contacts.T)) < 1e-8
    ordered = list(dict.fromkeys(tokens.tolist()))
    assert ordered == ["A", "C", "D", "E"] or ordered == ["A", "B", "C", "D", "E"]
    if "chain_ids" in summary:
        assert summary["chain_ids"] == ordered
    regions = {}
    chaincols = {
        "mhc": ("mhc_chain1_aa", "hla_trimmed"),
        "peptide": ("peptide_aa", "peptide"),
        "alpha": ("tcr_alpha_variable_aa", "tcra_trimmed"),
        "beta": ("tcr_beta_variable_aa", "tcrb_trimmed"),
    }
    for key, (mf, sf) in chaincols.items():
        ids = np.flatnonzero(tokens == CHAIN[key])
        regions[key] = ids
        assert len(ids) == len(row[sf])
        np.testing.assert_array_equal(resids[ids], np.arange(1, len(ids) + 1))
    for ab, source in [("alpha", "tcra"), ("beta", "tcrb")]:
        allcdr = []
        for n in [1, 2, 3]:
            seq = row[source + "_trimmed"]
            cdr = row.get(source + "_cdr" + str(n), "")
            if not cdr:
                continue
            assert cdr and seq.count(cdr) == 1
            start = seq.index(cdr)
            ids = regions[ab][start : start + len(cdr)]
            allcdr.extend(ids)
            if n == 3:
                regions["cdr3" + ab] = ids
        assert len(set(allcdr)) == len(allcdr)
        if all(row.get(source + "_cdr" + str(n)) for n in [1, 2, 3]):
            regions["cdr" + ab] = np.asarray(allcdr, int)
    for key, l, r in [
        ("tcr", "alpha", "beta"),
        ("cdr3", "cdr3alpha", "cdr3beta"),
        ("cdr", "cdralpha", "cdrbeta"),
        ("pmhc", "mhc", "peptide"),
    ]:
        if l in regions and r in regions:
            regions[key] = np.concatenate([regions[l], regions[r]])
    regions["complex"] = np.arange(len(tokens))
    block = gemmi.cif.read_string(cif_text).sole_block()
    atom_table = block.find("_atom_site.", ["label_asym_id", "label_seq_id", "B_iso_or_equiv"])
    atomchains = np.asarray([str(a[0]) for a in atom_table])
    atomres = np.asarray([int(a[1]) for a in atom_table])
    atomplddt = np.asarray(conf["atom_plddts"], float)
    np.testing.assert_array_equal(atomchains, np.asarray(conf["atom_chain_ids"]))
    assert len(atomchains) == len(atomplddt)
    cifplddt = np.asarray([float(a[2]) for a in atom_table])
    assert np.max(np.abs(cifplddt - atomplddt)) < 0.011
    assert np.isfinite(atomplddt).all() and ((atomplddt >= 0) & (atomplddt <= 100)).all()
    token_map = {(str(c), int(n)): i for i, (c, n) in enumerate(zip(tokens, resids))}
    atomtoken = np.asarray([token_map[(c, int(n))] for c, n in zip(atomchains, atomres)])
    counts = np.bincount(atomtoken, minlength=len(tokens))
    assert (counts > 0).all()
    residue_plddt = np.bincount(atomtoken, weights=atomplddt, minlength=len(tokens)) / counts
    best = summary
    v = {}
    for key in ["iptm", "ptm", "ranking_score", "fraction_disordered", "has_clash"]:
        v[key] = float(best[key])
        assert abs(v[key] - float(summary[key])) < 0.011
    v["confidence_only_rank"] = 0.8 * v["iptm"] + 0.2 * v["ptm"]
    chain_positions = [ordered.index(ch) for ch in CHAIN.values()]
    pmin = np.asarray(best["chain_pair_pae_min"], float)[np.ix_(chain_positions, chain_positions)]
    piptm = np.asarray(best["chain_pair_iptm"], float)[np.ix_(chain_positions, chain_positions)]
    assert pmin.shape == piptm.shape == (4, 4) and np.max(abs(piptm - piptm.T)) < 1e-8
    chains = list(CHAIN)
    for i, l in enumerate(chains):
        for kind in ["iptm", "ptm"]:
            v["chain_" + kind + "_" + l] = float(best["chain_" + kind][chain_positions[i]])
        assert abs(piptm[i, i] - v["chain_ptm_" + l]) < 1e-12
        for j, r in enumerate(chains):
            # This archive stores full PAE to 0.1 A, while the native minima use 0.01 A.
            assert abs(pmin[i, j] - pae[np.ix_(regions[l], regions[r])].min()) < 0.061
            v["pair_pae_min_" + l + "_" + r] = float(pmin[i, j])
            if j > i:
                v["pair_iptm_" + l + "_" + r] = float(piptm[i, j])
                v["pair_pae_min_bidir_" + l + "_" + r] = float((pmin[i, j] + pmin[j, i]) / 2)
    vals = [v["pair_iptm_peptide_alpha"], v["pair_iptm_peptide_beta"]]
    for stat in ["mean", "min", "max"]:
        v["tcr_peptide_pair_iptm_" + stat] = float(getattr(np, stat)(vals))
    v["pae_complex_mean"] = float(pae.mean())
    fallbacks = 0
    for l, r in INTERFACES:
        if l not in regions or r not in regions:
            continue
        li, ri = regions[l], regions[r]
        pair = l + "_" + r
        fwd = pae[np.ix_(li, ri)]
        rev = pae[np.ix_(ri, li)].T
        sym = (fwd + rev) / 2
        c = contacts[np.ix_(li, ri)]
        stats = {
            "mean": float((fwd.mean() + rev.mean()) / 2),
            "forward": float(fwd.mean()),
            "reverse": float(rev.mean()),
            "median": float(np.median(sym)),
            "q10": float(np.quantile(sym, 0.1)),
            "min": float(sym.min()),
            "frac_lt10": float((sym < 10).mean()),
            "contact_weighted": float((sym * c).sum() / c.sum())
            if c.sum() > 0
            else float(sym.mean()),
        }
        fallbacks += int(c.sum() == 0)
        for stat, value in stats.items():
            v["pae_" + pair + "_" + stat] = value
        for stat, value in [
            ("mean", c.mean()),
            ("sum", c.sum()),
            ("max", c.max()),
            ("right_best_mean", c.max(axis=0).mean()),
        ]:
            v["contact_" + pair + "_" + stat] = float(value)
        wl, wr = c.max(axis=1), c.max(axis=0)
        lp = (
            float(np.average(residue_plddt[li], weights=wl))
            if wl.sum() > 0
            else float(residue_plddt[li].mean())
        )
        rp = (
            float(np.average(residue_plddt[ri], weights=wr))
            if wr.sum() > 0
            else float(residue_plddt[ri].mean())
        )
        v["plddt_contact_" + pair] = (lp + rp) / 2
    for region in [k for k in NAMES if k != "pmhc"]:
        if region not in regions:
            continue
        arr = residue_plddt[regions[region]]
        v["plddt_" + region + "_mean"] = float(arr.mean())
        v["plddt_" + region + "_q10"] = float(np.quantile(arr, 0.1))
    for region in ["complex", *CHAIN]:
        arr = atomplddt if region == "complex" else atomplddt[atomchains == CHAIN[region]]
        v["atom_plddt_" + region + "_mean"] = float(arr.mean())
    assert all(np.isfinite(x) for x in v.values())
    return v, sorted(set(s["metric"] for s in SPECS) - set(v))
