# 数据注册与完整性

公开数据集：[Tony1233/EvoTCR-1000-Refill212](https://huggingface.co/datasets/Tony1233/EvoTCR-1000-Refill212)。使用 `registries/release.json` 的固定 commit，避免主分支更新导致结果改变。

1000 组、10000 配对、30000 条三结构引用；去重得到 29940 个原始归档。重复引用属于不同配对记录，不能据此少算结构覆盖。外层 60 个 `compact-*.tar` 分片包含完整原始 CIF 与原始 summary；两者字节不变。内层归档重新压缩，加入来源说明；七项逐 seed 置信度既保存在数值数组，也保存在结构引用的 `confidence_values`。完整 N×N PAE/contact 矩阵未包含在主发布包中。

| 文件 | 内容 |
|---|---|
| `metadata/pairs.parquet` | 每行配对的序列、HLA、肽、设计亲本、假定标签、折与三个结构引用 |
| `metadata/tasks.jsonl` | 结构任务、CDR3 链范围，与训练描述子同顺序 |
| `metadata/panel.json` | 20 个 target ID 的映射 |
| `training/F.npy`、`names.npy` | 10000×6250 固定描述子和精确列顺序 |
| `training/scalar7_seeds.npy` | 10000×3×7，尚未进行三结构合并的置信度 |
| `training/train.npz` | 未缩放序列距离 D、base、y/g/cg/h/HL、当前 fold/cluster 和 CDR3 |
| `training/fold_assignment.npz` | 当前五折与设计亲本来源 |
| `training/selected1000.parquet` | 原始选择与 212 替换来源；内部服务器路径被移除 |
| `DATASET.json` | 数据注册规范；明确 `role=train` 和标签语义 |
| `MANIFEST.json` | 每份发布文件的字节数和 SHA256 |

`selected1000` 的原始 CSV 指纹为 `f032bbf7c6da7a86a82a4c60dcfa269d7853bacb729bb4f29d969d6fe81d3615`；公开 Parquet 因格式转换与路径移除具有新的指纹，见 MANIFEST。1000 个匹配是设计阳性，9000 个错配是假定负例，不能重命名为“实验确认”。`experimental_binding_label` 若为空，即缺少实验标签。

注册版本不可原地覆盖。资产路径不能逃出数据根目录；校验长度与 SHA256 后再使用。数据角色可用于训练隔离，但不是对恶意脚本的访问控制。亲本与完全相同的 CDR3 对跨折会使验证偏高，因此 `cluster` 必须整体落在同一折。

提取单个内部归档时，以 `pairs.structures` 的 `shard` 定位外层 tar、`member` 定位内容，校验紧凑归档 `sha256` 和原始 CIF 的 `model_cif_sha256`；`original_archive_sha256` 对应原大矩阵归档。请勿把不可信 tar 的任意路径解压到工作目录。`build-query` 内置按成员读取并验证。

## 许可和来源

结构来自 NetTCRFold/AlphaFold 3 计算。随附 `ALPHAFOLD3_OUTPUT_TERMS.md` 与 `Legally Binding Terms of Use.txt`。发布的数据是选择、去重索引、压缩分片和固定描述子的派生产物；该处理不赋予商业再许可权。训练用 IMMREP 关联靶点来自已有实验设计上下文，独立测试标签不混入本数据集。

紧凑包是当前固定模型的完整推理输入集，不是所有原始 AF3 内部张量的镜像。需要新增基于完整 PAE/contact 矩阵的因子时，须另外获取原矩阵或重新预测；本发布不进行此类扩展或搜索。
