"""The 17 published factors. Recipe order is part of the model contract."""

from .registry import Factor, Registry

NAMES = [
    "contact_beta_max",
    "neg_pae_cdr3beta_bidir",
    "plddt_peptide",
    "R10_neg_pae_tcr_min",
    "R10_contact_tcr_max",
    "R10_neg_pae_cdr3_min",
    "R10_iptm",
    "seq_cognate",
    "seq_HLA",
    "seq_masked",
    "nearest_positive",
    "nearest_negative",
    "structure_HLA",
    "structure_cognate",
    "structure_hierarchy",
    "union_HLA",
    "LJ_HLA",
]
DEFINITIONS = [
    "TCRβ–肽最大接触概率；三种结构取最大值",
    "CDR3β–肽双向平均PAE取负；三种结构取最大值",
    "首个固定seed的肽残基平均pLDDT",
    "TCR–肽最小PAE取负，三seed最大值后同TCR十肽排名",
    "TCR–肽最大接触概率，三seed最大值后同TCR十肽排名",
    "CDR3–肽最小PAE取负，三seed最大值后同TCR十肽排名",
    "首个固定seed的ipTM，同TCR十肽排名",
    "配对CDR3固定距离→训练中位数校准RBF→cognate岭回归",
    "同HLA序列核，训练子集中心化标签→岭回归",
    "观察掩码中心化的序列核岭回归",
    "距训练子集同靶点设计阳性的最小距离取负",
    "距训练子集同HLA其他靶点假定阴性的最小距离取负",
    "6250维结构描述子→训练PCA→同HLA核岭回归",
    "6250维结构描述子→训练PCA→cognate核岭回归",
    "5320维组内中心化→训练PCA→全局与靶点残差分层回归",
    "F1+F3共5320维→训练PCA→同HLA核岭回归",
    "F1+F3的lj/dmin列→训练PCA→同HLA核岭回归",
]


def registry() -> Registry:
    r = Registry()
    for i, name in enumerate(NAMES):

        def compute(context, index=i):
            # A shared frozen transform is evaluated once by the feature service.
            if "runtime" in context:
                return context["runtime"].evaluate(index)
            return context["raw17"][:, index]

        r.register(
            Factor(
                name,
                "1.0.0",
                "confidence" if i < 7 else "sequence" if i < 12 else "structure",
                DEFINITIONS[i],
                (),
                i >= 7,
                compute,
            )
        )
    return r


def diagnostics(raw):
    import numpy as np

    sd = np.std(raw, axis=0)
    z = (raw - np.mean(raw, axis=0)) / np.maximum(sd, 1e-12)
    corr = z.T @ z / len(z)
    np.fill_diagonal(corr, 0)
    return {
        "max_absolute_offdiagonal_correlation": float(np.max(np.abs(corr))),
        "constant_columns": np.flatnonzero(sd == 0).tolist(),
        "orthogonality_proven": False,
        "correlation_offdiagonal": corr.tolist(),
    }
