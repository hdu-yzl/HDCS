## HDCS

Experiment code for the paper:

Chaohua Yang, Kailiang Hao, Dugang Liu, Xing Tang, Xiuqiang He, Zhong Ming. **Hierarchical Differentiated Cross-scenario Sharing for Data-sparse Multi-scenario Recommendation.**

**Please cite our paper if you use this code. Thanks!**

## Requirement

See [requirements.txt](requirements.txt).

## Data Preprocessing

Please download the original data ([AliMama](https://tianchi.aliyun.com/dataset/56) and [AliCCP](https://tianchi.aliyun.com/dataset/408)) and place them under `data/`.

For AliMama, place `raw_sample.csv`, `ad_feature.csv`, and `user_profile.csv` in `data/`, then run the following commands from the repository root:

# Process the original data
python datatransform/Alimama_preprocess.py

# Convert to TFRecord
python datatransform/Alimama2tf.py


## Usage

Examples of running HDCS from the repository root:


python trainer.py --dataset ali-mama --model dnn --cuda 0

python trainer.py --dataset ali-ccp --model dnn --cuda 0


## Variants

HDCS supports **DNN**, **DeepFM**, and **DCN** backbones. Set `--model` to `dnn`, `deepfm`, or `dcn` in `trainer.py`. Select the dataset with `--dataset ali-ccp` or `--dataset ali-mama`.

## Contact

If you have any issues or ideas, please open an [issue].
