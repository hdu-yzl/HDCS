"""Merge the public AliMama tables and create chronological train/test CSVs."""

import argparse
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "user_id",
    "cms_segid",
    "cms_group_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "occupation",
    "new_user_class_level",
    "adgroup_id",
    "cate_id",
    "campaign_id",
    "customer",
    "brand",
    "price",
    "pid",
    "shopping_level",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Preprocess the AliMama dataset")
    parser.add_argument("--raw_sample", type=Path, default=Path("data/raw_sample.csv"))
    parser.add_argument("--ad_feature", type=Path, default=Path("data/ad_feature.csv"))
    parser.add_argument("--user_profile", type=Path, default=Path("data/user_profile.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/ali-mama"))
    parser.add_argument("--test_date", default="2017-05-13")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    return parser


def preprocess(args):
    raw_samples = pd.read_csv(args.raw_sample)
    ad_features = pd.read_csv(args.ad_feature)
    user_profiles = pd.read_csv(args.user_profile)

    raw_samples.columns = ["user_id", "time_stamp", "adgroup_id", "pid", "nonclk", "clk"]
    user_profiles.columns = [
        "user_id",
        "cms_segid",
        "cms_group_id",
        "final_gender_code",
        "age_level",
        "pvalue_level",
        "shopping_level",
        "occupation",
        "new_user_class_level",
    ]
    raw_samples["time_stamp"] = (
        pd.to_datetime(raw_samples["time_stamp"], unit="s", utc=True)
        .dt.tz_convert(args.timezone)
        .dt.tz_localize(None)
    )

    merged = raw_samples.merge(user_profiles, on="user_id", how="left")
    merged = merged.merge(ad_features, on="adgroup_id", how="left").fillna(0)
    merged["domain_id"] = merged["shopping_level"].astype("int64")

    invalid_domains = sorted(set(merged["domain_id"]) - {0, 1, 2, 3})
    if invalid_domains:
        raise ValueError(f"Expected AliMama domain IDs 0-3, found {invalid_domains}")

    columns = FEATURE_COLUMNS + ["clk", "domain_id"]
    test_start = pd.Timestamp(args.test_date)
    train_data = merged.loc[merged["time_stamp"] < test_start, columns]
    test_data = merged.loc[merged["time_stamp"] >= test_start, columns]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_data.csv"
    test_path = args.output_dir / "test_data.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    print(f"Wrote {len(train_data):,} training rows to {train_path}")
    print(f"Wrote {len(test_data):,} test rows to {test_path}")


def main():
    preprocess(build_parser().parse_args())


if __name__ == "__main__":
    main()
