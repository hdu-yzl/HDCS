"""TensorFlow TFRecord readers that yield PyTorch tensors.

The on-disk format is shared by both datasets: each example contains a sparse
feature vector, a binary label, and a numeric scenario/domain identifier.
"""

from pathlib import Path

import tensorflow as tf
import torch


class _TFRecordLoader:
    def __init__(self, tfrecord_path, field_count):
        self.tfrecord_path = Path(tfrecord_path)
        self.field_count = field_count
        self.description = {
            "feature": tf.io.FixedLenFeature([field_count], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.float32),
            "domain": tf.io.FixedLenFeature([1], tf.float32),
        }

    def _parse(self, serialized_example):
        example = tf.io.parse_single_example(serialized_example, self.description)
        return example["feature"], example["label"], example["domain"]

    def _files(self, split):
        files = sorted(self.tfrecord_path.glob(f"{split}*.tfrecord"))
        if not files:
            raise FileNotFoundError(
                f"No TFRecords matching '{split}*.tfrecord' in {self.tfrecord_path}"
            )
        return [str(path) for path in files]

    def get_data(self, split, batch_size=1, shuffle=False, seed=2022):
        """Yield one split without materializing the full dataset in memory."""
        dataset = tf.data.TFRecordDataset(self._files(split))
        dataset = dataset.map(self._parse, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=100_000,
                seed=seed,
                reshuffle_each_iteration=True,
            )
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        for features, labels, domains in dataset:
            yield (
                torch.from_numpy(features.numpy()),
                torch.from_numpy(labels.numpy()),
                torch.from_numpy(domains.numpy()),
            )

class Ali_CCPLoader(_TFRecordLoader):
    """AliCCP reader for the 18-field representation used by this project."""

    def __init__(self, tfrecord_path):
        super().__init__(tfrecord_path, field_count=18)


class Ali_MamaLoader(_TFRecordLoader):
    """AliMama reader for the 16-field representation used by this project."""

    def __init__(self, tfrecord_path):
        super().__init__(tfrecord_path, field_count=16)
