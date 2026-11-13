# Cataract-101 helpers

Prepare the CataNet-style frame pool (PNG + CSV), then train:

```bash
python -u scripts/cataract101/prepare_catanet_frames.py \
  --dataset_root "$RSDBENCH_DATA_ROOT/cataract-101"

python -u run.py --model CataNet \
  --protocol configs/protocol/cataract101_main.yaml \
  --model_cfg configs/experiment/cataract101_CataNet.yaml \
  --exp_name cataract101_CataNet
```

See the repository README for protocols, metrics, and how to add other datasets.
