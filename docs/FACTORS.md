# 因子库

以下 17 个是发布模型实际使用的预测因子。F1/F3/F5 的 6250 维是描述子，不能把维数当成独立因子或独立实验样本。

| 槽位 | 名称 | 定义 | 需训练拟合 |
|---:|---|---|---|
| 0 | `contact_beta_max` | TCRβ–肽最大接触概率；三种结构取最大值 | 否 |
| 1 | `neg_pae_cdr3beta_bidir` | CDR3β–肽双向平均PAE取负；三种结构取最大值 | 否 |
| 2 | `plddt_peptide` | 首个固定seed的肽残基平均pLDDT | 否 |
| 3 | `R10_neg_pae_tcr_min` | TCR–肽最小PAE取负，三seed最大值后同TCR十肽排名 | 否 |
| 4 | `R10_contact_tcr_max` | TCR–肽最大接触概率，三seed最大值后同TCR十肽排名 | 否 |
| 5 | `R10_neg_pae_cdr3_min` | CDR3–肽最小PAE取负，三seed最大值后同TCR十肽排名 | 否 |
| 6 | `R10_iptm` | 首个固定seed的ipTM，同TCR十肽排名 | 否 |
| 7 | `seq_cognate` | 配对CDR3固定距离→训练中位数校准RBF→cognate岭回归 | 是 |
| 8 | `seq_HLA` | 同HLA序列核，训练子集中心化标签→岭回归 | 是 |
| 9 | `seq_masked` | 观察掩码中心化的序列核岭回归 | 是 |
| 10 | `nearest_positive` | 距训练子集同靶点设计阳性的最小距离取负 | 是 |
| 11 | `nearest_negative` | 距训练子集同HLA其他靶点假定阴性的最小距离取负 | 是 |
| 12 | `structure_HLA` | 6250维结构描述子→训练PCA→同HLA核岭回归 | 是 |
| 13 | `structure_cognate` | 6250维结构描述子→训练PCA→cognate核岭回归 | 是 |
| 14 | `structure_hierarchy` | 5320维组内中心化→训练PCA→全局与靶点残差分层回归 | 是 |
| 15 | `union_HLA` | F1+F3共5320维→训练PCA→同HLA核岭回归 | 是 |
| 16 | `LJ_HLA` | F1+F3的lj/dmin列→训练PCA→同HLA核岭回归 | 是 |

## 注册与共享计算

`Factor(name, version, family, definition, dependencies, fitted, compute)` 是注册单元。名称唯一，依赖由 DAG 解析；重复名称、循环依赖、非有限输出都拒绝。插件可通过 Python entry point `evotcr.factors` 加载，默认不自动执行外部插件。

`FrozenFactors` 将 PCA 投影和核矩阵作为共享中间产物缓存；选择单个因子只计算其依赖，不要求先计算全部 17 项。置信度与最近参考因子也可独立执行。训练接口是 `TrainableFactor.fit(FitView, config)` / `transform(features, state)`，FitView 只包含指定训练行。当前发布只加载训练时保存的状态，不提供新的拟合或超参搜索入口；插件合同保留以支持独立算子开发。注册不会自动更换冻结模型使用的因子。

## 固定描述子

F1 为 2464 维，F3 为 2856 维，F5 为 930 维，总计 6250。通道和精确列序在 `resources/raw_factor_catalog.json`。几何阈值、原子类型、AA 性质表是版本化定义，PCA、标准化和监督回归是训练状态。

LJ/dmin 通道表示固定距离变换，不是独立分子动力学自由能计算；化学近似与 pLDDT/PAE 都存在建模误差。因子名称提供可追溯含义，不证明因果解释或统计正交。

## 固定组合与正交诊断

17 个因子全部启用，组合和内部参数固定在 `configs/frozen_factors.json`。改变任何一项都不再是本次发布版本。应在训练分割上运行 `evotcr factors diagnose factors.npz diagnostics.json` 查看相关性、常量列和最大非对角相关；诊断结果不会自动宣称因子正交。

## 插件安装示例

```toml
[project.entry-points."evotcr.factors"]
my_factors = "my_package:register"
```

参见 `examples/custom_factor.py`。插件算子必须遵循行/组作用域，并为任何可拟合状态保存训练指纹。
