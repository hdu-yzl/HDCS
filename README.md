# HDCS

PyTorch implementation of **Hierarchical Differentiated Cross-scenario Sharing (HDCS)** for multi-scenario recommendation, with training entry points for AliCCP (three scenarios) and AliMama (four scenarios).

HDCS combines shared and scenario-specific representations, an instance-target hard selector with straight-through gradients, and a soft allocator for cross-scenario transfer. Inference uses the native scenario prediction path.

## Code structure

```text
SM3_final_trainer.py          # AliCCP training entry point
SM4_final_trainer.py          # AliMama training entry point
modules/SM3_model.py          # Core HDCS and DNN/DeepFM/DCN backbones
modules/SM4_model.py          # Four-scenario model wrappers
modules/HDCS_model.py         # Configurable representations, routing and loss
modules/layers.py            # Embedding, MLP, FM and cross layers
dataloader/Dtfloader.py       # Batched TFRecord readers
configs/                     # Dataset DNN runtime configurations
tests/                       # Synthetic model and training behavior checks
datatransform/               # AliMama data preparation
```

The configurable model includes residual branches, scenario input adapters, optional additive feature logits, and prediction-head sharing. The DNN trainer supports separate optimizer parameter groups, moving-average weights, mixed precision, optional CUDA Graph execution, validation-based checkpoint saving, and per-scenario evaluation.

## Environment

Use Linux with Python 3.10 or later:

```bash
python3 -m pip install -r requirements.txt
```

CUDA is optional; pass `--cuda -1` for CPU execution. CUDA Graph execution requires CUDA and a model without BatchNorm or dropout.

## Datasets

Dataset files are not redistributed. Prepare [AliCCP](https://tianchi.aliyun.com/dataset/408) and [AliMama](https://tianchi.aliyun.com/dataset/56) in this layout:

```text
data/
|-- ali-ccp/
|   |-- stats/defaults.pkl
|   `-- tfrecord/
|       |-- train_*.tfrecord
|       |-- val_*.tfrecord
|       `-- test_*.tfrecord
`-- ali-mama/
    |-- stats_2/defaults.pkl
    `-- threshold_2/
        |-- train_*.tfrecord
        |-- validation_*.tfrecord
        `-- test_*.tfrecord
```

Each example contains `feature` (18 integer feature IDs for AliCCP or 16 for AliMama), `label` (one binary float), and `domain` (one zero-based scenario ID stored as a float). `defaults.pkl` stores the field vocabulary metadata used by the model. Both `val_` and `validation_` prefixes are accepted for validation records.

For AliMama, place `raw_sample.csv`, `ad_feature.csv`, and `user_profile.csv` under `data/`, then run:

```bash
python3 datatransform/Alimama_preprocess.py
python3 datatransform/Alimama2tf.py
```

AliCCP preprocessing depends on the public release layout; prepare its splits using the schema above.

## Training

The existing entry points use the dataset JSON configuration for DNN training and create missing caches automatically:

```bash
python3 SM3_final_trainer.py --model dnn --dataset ali-ccp --cuda 0
python3 SM4_final_trainer.py --model dnn --dataset ali-mama --cuda 0
```

DNN configurations are `configs/hdcs_dnn_aliccp.json` and `configs/hdcs_dnn_alimama.json`. Model and training settings come from these files. Use `--config` to supply another JSON file; `--seeds`, `--cuda`, `--cache`, `--output-dir`, and `--validation-only` control execution. For a single seed:

```bash
python3 SM4_final_trainer.py --model dnn --seeds 2022 --output-dir output/alimama_run
```

Each seed writes checkpoints and evaluation artifacts under its own output directory. Use a fresh output directory for each invocation. Cache files preserve feature IDs, row order, labels and scenario IDs, and record source-file metadata for consistency checks.

DeepFM and DCN use the streaming trainer and its command-line model/training options:

```bash
python3 SM3_final_trainer.py --model deepfm --seeds 2022
python3 SM4_final_trainer.py --model dcn --seeds 2022
```

Use `--help` on either entry point for the available arguments.

## Verification

```bash
python3 -m unittest discover -s tests -v
```

The checks use synthetic inputs to exercise model paths, gradients, optimizer behavior and CUDA Graph execution when CUDA is available.

## License

Released under the [MIT License](LICENSE).
