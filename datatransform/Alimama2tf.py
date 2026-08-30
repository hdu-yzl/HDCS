"""Convert preprocessed AliMama CSV files to HDCS TFRecords."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


LABEL_COLUMN = "clk"
DOMAIN_COLUMN = "domain_id"


def build_parser():
    parser = argparse.ArgumentParser(description="Convert AliMama CSV files to TFRecord")
    parser.add_argument("--train_csv", type=Path, default=Path("data/ali-mama/train_data.csv"))
    parser.add_argument("--test_csv", type=Path, default=Path("data/ali-mama/test_data.csv"))
    parser.add_argument("--stats_dir", type=Path, default=Path("data/ali-mama/stats_2"))
    parser.add_argument("--record_dir", type=Path, default=Path("data/ali-mama/threshold_2"))
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--records_per_file", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=2021)
    return parser


def _bucket_price(value):
    if pd.isna(value):
        return 0
    value = float(value)
    return int(np.floor(np.log(value) ** 2)) if value > 2 else int(value)


def _feature_key(field, value):
    """Build stable keys despite pandas converting integer rows to floats."""
    if pd.isna(value):
        value = "<NA>"
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    return f"{field}_{value}"


def _serialize(label, features, domain):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "feature": tf.train.Feature(int64_list=tf.train.Int64List(value=features)),
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
                "domain": tf.train.Feature(float_list=tf.train.FloatList(value=[domain])),
            }
        )
    )
    return example.SerializeToString()


def _fit_feature_map(train_data, feature_columns, threshold):
    feature_map = {}
    defaults = {}
    field_offset = {}
    offset = 0

    for field in feature_columns:
        counts = train_data[field].value_counts(dropna=False)
        values = list(counts.index) if field == "price" else list(counts[counts >= threshold].index)
        for index, value in enumerate(values):
            feature_map[_feature_key(field, value)] = offset + index
        # Reserve the final ID in every field for out-of-vocabulary values.
        offset += len(values) + 1
        defaults[field] = len(values)
        field_offset[field] = offset
    return feature_map, defaults, field_offset


def _write_split(data, split, output_dir, feature_columns, feature_map, field_offset, records_per_file):
    output_dir.mkdir(parents=True, exist_ok=True)
    for part, start in enumerate(range(0, len(data), records_per_file)):
        chunk = data.iloc[start : start + records_per_file]
        output_path = output_dir / f"{split}_{part:04d}.tfrecord"
        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=str(output_path.name)):
                features = [
                    feature_map.get(_feature_key(field, row[field]), field_offset[field] - 1)
                    for field in feature_columns
                ]
                writer.write(_serialize(float(row[LABEL_COLUMN]), features, float(row[DOMAIN_COLUMN])))


def convert(args):
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    train_data = pd.read_csv(args.train_csv)
    test_data = pd.read_csv(args.test_csv)
    if "price" in train_data:
        train_data["price"] = train_data["price"].map(_bucket_price)
        test_data["price"] = test_data["price"].map(_bucket_price)

    feature_columns = [
        column for column in train_data.columns if column not in {LABEL_COLUMN, DOMAIN_COLUMN}
    ]
    if len(feature_columns) != 16:
        raise ValueError(f"AliMama requires 16 feature fields, found {len(feature_columns)}")
    invalid_domains = sorted(
        (set(train_data[DOMAIN_COLUMN]) | set(test_data[DOMAIN_COLUMN])) - {0, 1, 2, 3}
    )
    if invalid_domains:
        raise ValueError(f"Expected AliMama domain IDs 0-3, found {invalid_domains}")

    validation = train_data.sample(frac=args.validation_ratio, random_state=args.seed)
    training = train_data.drop(validation.index)
    feature_map, defaults, field_offset = _fit_feature_map(training, feature_columns, args.threshold)

    args.stats_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in (
        ("feat_map.pkl", feature_map),
        ("defaults.pkl", defaults),
        ("offset.pkl", field_offset),
    ):
        with (args.stats_dir / filename).open("wb") as stream:
            pickle.dump(value, stream)

    _write_split(training, "train", args.record_dir, feature_columns, feature_map, field_offset, args.records_per_file)
    _write_split(validation, "val", args.record_dir, feature_columns, feature_map, field_offset, args.records_per_file)
    _write_split(test_data, "test", args.record_dir, feature_columns, feature_map, field_offset, args.records_per_file)


def main():
    convert(build_parser().parse_args())


if __name__ == "__main__":
    main()
