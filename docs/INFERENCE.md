# 冻结推理合同

## 从结构端到端构建

```bash
python scripts/download_release.py --model
# tasks.jsonl 必须按 TCR 分组，每组包含完整 10 个同 HLA 肽。
evotcr build-query tasks.jsonl data/refill212 weights/refill212 --run-root runs/query
# 上一步打印 query.npz 的完整路径。
evotcr predict-all weights/refill212 QUERY_NPZ predictions-five.npz
```

`tasks.jsonl` 使用公开数据格式，也可使用独立 NetTCRFold 任务。分片结构和独立归档两种引用均支持。紧凑发布包读取完整 CIF 与精确保存的七项逐 seed 置信度，原始 NetTCRFold 归档则直接从矩阵提取这些标量。构建器在每组内按 target ID 排序，并在输出目录的 `row_order.json` 保存查询行到原任务行的零基索引映射；评价标签须用该映射对齐。公开训练任务原本按肽序列排序，训练描述子继续保留原始顺序。每个结构核验 SHA256；流水线先提取三结构的几何与置信度，再构建 F1/F3/F5 和序列距离。已完成阶段按内容指纹复用，损坏缓存拒绝使用。

## 已有描述子的接口

`query.npz` 必须仅包含以下五个字段，不得夹带标签：

| 键 | 形状 | 含义 |
|---|---|---|
| F | `(10*N,6250)` | 固定 F1/F3/F5 列序，未用查询批次拟合任何统计 |
| base | `(10*N,7)` | 七项置信度；按固定规则合并三 seed；后四项为组内十肽排名 |
| D | `(N,1000)` | 配对 CDR3 对模型保存的 1000 个训练参考的**未缩放距离** |
| g | `(10*N,)` 整数 | `panel.json` 中 target ID，同组严格递增 |
| h | `(N,)` 整数 | 训练定义的 HLA 索引 0 或 1 |

D 使用固定 BLOSUM62 派生替换代价、中心补空至 20 位、空位代价 4、两链求和。参考顺序必须和模型 `train.npz` 一致。`evotcr.sequence.paired_distances` 实现这一原始定义；核带宽由保存的训练距离中位数及选中因子参数决定。

输出包含 `score`、`raw17` 和 `g`。score 是组内相对 softmax，不能解释为真实结合概率。改变十肽面板本身会改变组内排名/softmax，这是模型语义；加入、移除或重排其他完整 TCR 组不应改变已有组的分数（允许浮点计算容差）。

发布模型的主种子固定 1109。五 seed 是预定重复拟合，不是按外部测试挑选的模型集合。推理前默认校验所有权重文件；不存在查询时再估计 PCA、标准差或参考距离带宽的步骤。


## 五种子报告

`evotcr predict-all` 输出 `scores[5,10N]`、`seeds[5]`、`raw17` 和 `g`。顺序固定为 1109–1113。`evotcr evaluate` 分别评价五个模型，再报告指标的均值和样本标准差（ddof=1）。它不会平均五个模型的预测来产生另一个未经验证的集成成绩。

## 结构重建校验

下载完整数据后，可运行 `python scripts/check_structure_rebuild.py --group 0` 重建一个完整组并与原始描述子及五模型输出比较；报告写入 `runs/structure-check/report.json`。`--group` 仅选择要核验的数据组，不改变任何模型参数。
