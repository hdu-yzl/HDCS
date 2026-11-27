# HDCS

PyTorch implementation of **Hierarchical Differentiated Cross-scenario Sharing** for multi-scenario recommendation on AliCCP (3 scenarios) and AliMama (4 scenarios). Supports DNN, DeepFM, and DCN backbones.

## Project structure

```text
SM3_final_trainer.py       # AliCCP training entry point
SM4_final_trainer.py       # AliMama training entry point
modules/                  # HDCS models and backbone layers
dataloader/              # TFRecord readers
datatransform/            # AliMama preprocessing
tests/                   # Model and training checks
```

## Environment

Install `requirements.txt` in your Python environment. The PowerShell examples below use this project's existing Windows environment:

```powershell
& 'd:\anaconda\envs\py312\python.exe' -m pip install -r requirements.txt
```

Use `--cuda 0` for GPU training or `--cuda -1` for CPU.

## Data

Download [AliCCP](https://tianchi.aliyun.com/dataset/408) and [AliMama](https://tianchi.aliyun.com/dataset/56). Dataset files are not included.

```text
data/
|-- ali-ccp/
|   |-- stats/defaults.pkl
|   `-- tfrecord/{train,val,test}_*.tfrecord
`-- ali-mama/
    |-- stats_2/defaults.pkl
    `-- threshold_2/{train,val,test}_*.tfrecord
```

Each record contains `feature` (18 integer IDs for AliCCP; 16 for AliMama), `label` (one binary float), and `domain` (one zero-based scenario ID stored as a float). `defaults.pkl` maps each field to its vocabulary size, excluding its reserved OOV ID.

For AliMama, put `raw_sample.csv`, `ad_feature.csv`, and `user_profile.csv` in `data/`, then run:

```powershell
& 'd:\anaconda\envs\py312\python.exe'datatransform/Alimama_preprocess.py
& 'd:\anaconda\envs\py312\python.exe'datatransform/Alimama2tf.py
```

AliCCP preprocessing is not included; prepare its splits in the format above.

## Training

Run commands from the repository root:

```powershell
& 'd:\anaconda\envs\py312\python.exe' SM3_final_trainer.py --model dnn --cuda 0
& 'd:\anaconda\envs\py312\python.exe' SM4_final_trainer.py --model dnn --cuda 0
```

For all backbones, edit parameter defaults in `build_parser()` of the corresponding trainer, or override them on the command line with flags such as `--lr`, `--bsize`, `--train_epoch`, `--threshold`, and `--lambda_tr`. Defaults are starting points for validation tuning.

All backbones use the same TFRecord training flow within each dataset trainer. Checkpoints are saved under `checkpt/` with a suffix for each seed; use `--save_dir` to choose a different path for each experiment. Pass `--seeds 2022` for one run. The default seeds are 2022?2026.

To train DeepFM or DCN:

```powershell
& 'd:\anaconda\envs\py312\python.exe' SM3_final_trainer.py --model deepfm --seeds 2022 --cuda 0
& 'd:\anaconda\envs\py312\python.exe' SM4_final_trainer.py --model dcn --seeds 2022 --cuda 0
```

All backbones select checkpoints and stop early using validation **pooled AUC** across all scenarios. Test evaluation reports per-scenario AUC and Logloss. Use `--help` to list available options.

## Tests

```powershell
& 'd:\anaconda\envs\py312\python.exe' -m unittest discover -s tests -v
```

## License

[MIT](LICENSE)
