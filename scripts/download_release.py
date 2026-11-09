"""Fetch the repository's pinned public release and verify its complete manifest."""

import argparse
import json
from pathlib import Path
from evotcr.hub import download

p = argparse.ArgumentParser()
p.add_argument("--model", action="store_true")
p.add_argument("--dataset", action="store_true")
a = p.parse_args()
if not (a.model or a.dataset):
    p.error("Choose --model and/or --dataset")
root = Path(__file__).resolve().parents[1]
release = json.loads((root / "registries/release.json").read_text())
for name, enabled, kind, folder in [
    ("model", a.model, "model", "weights/refill212"),
    ("dataset", a.dataset, "dataset", "data/refill212"),
]:
    if enabled:
        item = release[name]
        if not item.get("revision"):
            raise SystemExit(
                "完整数据仍在上传。请在发布完成后更新仓库，再下载数据；冻结模型已可下载。"
            )
        result = download(item["repo_id"], item["revision"], root / folder, kind)
        print(f"{name}: verified {result}")
