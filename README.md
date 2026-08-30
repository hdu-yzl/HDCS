# HDCS

Official PyTorch implementation of **Hierarchical Differentiated Cross-scenario Sharing (HDCS)** for data-sparse multi-scenario recommendation.

HDCS models cross-scenario transfer at two complementary levels:

1. A scenario-level hard selector learns **where to transfer** each source instance.
2. A shared-specific soft allocator learns **how to transfer** by balancing shared and target-specific representations.

The implementation supports DNN, DeepFM, and DCN backbones on the three-scenario AliCCP and four-scenario AliMama settings.

## Repository structure

```text
.
|-- modules/
|   |-- layers.py          # Shared embedding, MLP, FM, and cross layers
|   |-- SM3_model.py       # General HDCS implementation and three-scenario models
|   `-- SM4_model.py       # Four-scenario model wrappers
|-- dataloader/
|   `-- Dtfloader.py       # Streaming TFRecord-to-PyTorch readers
|-- datatransform/
|   |-- Alimama_preprocess.py  # Merge and chronologically split AliMama tables
|   `-- Alimama2tf.py          # Encode AliMama CSV files as TFRecords
|-- SM3_final_trainer.py   # AliCCP training and evaluation
|-- SM4_final_trainer.py   # AliMama training and evaluation
`-- requirements.txt
```

## Environment

The code was tested on Windows with Python 3.12. Install the dependencies with the project interpreter:

```powershell
& 'D:\anaconda\envs\py312\python.exe' -m pip install -r requirements.txt
```

CUDA is optional. Pass `--cuda -1` to run on CPU.

## Datasets

The datasets are not redistributed in this repository.

- [AliCCP](https://tianchi.aliyun.com/dataset/408)
- [AliMama](https://tianchi.aliyun.com/dataset/56)

### Expected data layout

```text
data/
|-- ali-ccp/
|   |-- stats/defaults.pkl
|   `-- tfrecord/
|       |-- train_*.tfrecord
|       |-- val_*.tfrecord
|       `-- test_*.tfrecord
`-- ali-mama/
    |-- train_data.csv
    |-- test_data.csv
    |-- stats_2/defaults.pkl
    `-- threshold_2/
        |-- train_*.tfrecord
        |-- val_*.tfrecord
        `-- test_*.tfrecord
```

Each TFRecord example must contain:

- `feature`: fixed-length integer feature IDs (18 fields for AliCCP, 16 for AliMama);
- `label`: one binary click label;
- `domain`: one zero-based scenario ID.

AliCCP preprocessing follows the public dataset release and is not included because distributions of the downloaded package differ. Prepare the three splits using the schema above.

### AliMama preprocessing

Place `raw_sample.csv`, `ad_feature.csv`, and `user_profile.csv` under `data/`, then run:

```powershell
& 'D:\anaconda\envs\py312\python.exe' datatransform\Alimama_preprocess.py
& 'D:\anaconda\envs\py312\python.exe' datatransform\Alimama2tf.py
```

The first command uses interactions before `2017-05-13` for training and the final day for testing. The second command samples 10% of the training portion for validation, applies a frequency threshold of 2, and writes the four-scenario TFRecords expected by `SM4_final_trainer.py`.

## Training

### AliCCP (three scenarios)

```powershell
& 'D:\anaconda\envs\py312\python.exe' SM3_final_trainer.py --model dnn --dataset ali-ccp
```

### AliMama (four scenarios)

```powershell
& 'D:\anaconda\envs\py312\python.exe' SM4_final_trainer.py --model dnn --dataset ali-mama
```

Set `--model` to `dnn`, `deepfm`, or `dcn`. By default, each command runs five seeds (`2022`-`2026`) and reports per-scenario AUC and Logloss as mean +/- standard deviation. For a quick single-run check:

```powershell
& 'D:\anaconda\envs\py312\python.exe' SM4_final_trainer.py --model deepfm --dataset ali-mama --seeds 2022
```

Useful options:

| Option | Default | Description |
| --- | ---: | --- |
| `--bsize` | `4096` | Training batch size |
| `--dim` | `16` | Feature embedding dimension |
| `--mlp_dims` | `1024 512 256` | Backbone MLP widths |
| `--hdcs_dims` | `512 256 128` | Selector/allocator FNN widths |
| `--threshold` | `0.5` | Hard transfer selection threshold |
| `--lambda_tr` | `0.3` | Cross-scenario transfer-loss weight |
| `--lambda1` | `0.2` | KL regularization weight |
| `--lambda2` | `1.1` | Routing sparsity weight |
| `--cuda` | `0` | CUDA device; use `-1` for CPU |

Checkpoints are written under `checkpt/` by default, with one file per seed. Dataset files, checkpoints, logs, and Python caches are excluded from version control.

## Implementation notes

- Native prediction is independent of the selector and allocator.
- Hard routing applies only to off-diagonal source-target pairs and uses a straight-through estimator.
- The training objective is `L_nat + lambda_tr L_tr + lambda_1 L_KL + lambda_2 L_spa`.
- KL regularization is computed from the soft transfer probabilities before discretization.
- Evaluation uses the native-only path, so the routing modules are not executed at inference time.

## Citation

Citation information will be added after publication. If you use this repository before then, please cite the repository URL.

## License

This project is released under the [MIT License](LICENSE).
