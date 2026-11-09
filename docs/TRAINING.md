# 冻结模型使用

模型配置位于 `configs/frozen_mlp.json` 和 `configs/frozen_factors.json`，权重和因子状态从 Hugging Face 下载。

```bash
python scripts/download_release.py --model
evotcr predict-all weights/refill212 query.npz predictions-five.npz
evotcr evaluate predictions-five.npz labels.npz evaluation.json
```

输入格式见 [INFERENCE.md](INFERENCE.md)，数据文件见 [DATASET.md](DATASET.md)。
