# NetTCRFold 配套与权重

## 版本与边界

上游：https://github.com/mnielLab/NetTCRfold-1.0

本仓库锁定源码 commit `ae17ea27a8345c48046531463a49e90272bc33da`，通过 `third_party/NetTCRfold-1.0` 子模块提供。

```bash
git submodule update --init --recursive
# 也可显式获取锁定版本：
evotcr nettcrfold install third_party/NetTCRfold-1.0
```

上游 pipeline 使用 SLURM、Conda、Apptainer 和 GPU；其资源下载量远大于 EvoTCR 小 MLP。

## 三类“权重”分别下载

| 类型 | 获取方式 | 随本仓库分发 |
|---|---|---|
| EvoTCR 因子 PCA/核/参考参数 | Hugging Face `Tony1233/EvoTCR-Refill212-MLP`，固定 revision + SHA256 | 通过下载器 |
| EvoTCR 五个 MLP | 同一模型仓库，`models/seed1109` 至 `seed1113` | 通过下载器 |
| AlphaFold 3 模型参数 | 按 [Google 官方说明](https://github.com/google-deepmind/alphafold3#obtaining-model-parameters) 申请并直接接收 | **不分发** |

上游 `install.sh` 明确要求手动申请 AF3 参数；不应以不明来源镜像或转存取代官方申请。NetTCRFold 的 `alphafold3_tcrpmhc` 目录保留上游 CC BY-NC-SA 4.0 源码许可；代码、参数和输出分别有各自条款。

## 上游运行步骤

1. 按上游安装所需 Conda、Apptainer、数据库、化学组分与容器；可阅读并执行其 `install.sh`。该脚本会下载大文件，且部分步骤使用 SSH Git 地址。
2. 将官方收到的 AF3 参数按官方说明准备在 `${AF3_RESOURCES_DIR}/weights/`。锁定上游的 `run_alphafold3_tcrpmhc.sh` 将该目录挂载为 `/mnt/models`，并传入 `--model_dir=/mnt/models`。不把参数提交到 Git。
3. 从 `pipeline/configs/envExample.cfg` 复制成 `env.cfg`，填写 `PROJECT_ROOT`、`CONDA_SH`、`AF3_RESOURCES_DIR`、`AF3_IMAGE` 等本机路径。
4. 复制 `configMinimal.cfg`，设置 `DATA_DIR` 和 `INPUT_FILE`。输入 CSV 至少有 `TRA_aa`、`TRB_aa`、`peptide`/`Epitope_aa`，以及 MHC 序列 `MHCA_aa` 或 `allele`；以锁定上游 README 为准。
5. 在集群环境执行：

```bash
cd third_party/NetTCRfold-1.0/pipeline
sbatch workflow.sh configs/configMinimal.cfg
```

## 接入 EvoTCR

`evotcr nettcrfold extract task.json ARCHIVE_ROOT compact.npz` 从三份经 SHA256 校验的结构归档读取 CIF 和 summary。原始 NetTCRFold 归档直接提取完整矩阵中的七项置信度；公开紧凑归档使用引用中随附的同一组 `confidence_values`。它不把 tar 成员路径直接解压到文件系统。

task 格式与公开数据的 `metadata/tasks.jsonl` 单行一致。原始归档引用可为 `{path,sha256,seed}`；公开分片引用为 `{shard,member,sha256,seed,confidence_values,model_cif_sha256,original_archive_sha256}`。链约定：A=MHC，C=肽，D=TCRα，E=TCRβ；CDR3 范围是零起点、左闭右开。

距离计算使用重原子坐标与固定化学原子类型，量化到 0.125 Å（255 表示缺失），构成 33 个距离通道。它们包括氢键供受体兼容性的几何近似，但没有显式氢、质子化与完整角度判据，不能把每个计数当作实际氢键。

七项置信度的取整、seed 合并、组内排名与原始训练保持一致。完整管线由 `build-query` 串联结构提取、6250 描述子、七项置信度、未缩放 CDR3 距离和冻结模型输入；详见 [INFERENCE.md](INFERENCE.md)。
