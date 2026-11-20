"""TensorFlow TFRecord readers that yield PyTorch tensors.

The on-disk format is shared by both datasets: each example contains a sparse
feature vector, a binary label, and a numeric scenario/domain identifier.
"""

from pathlib import Path
import os

import tensorflow as tf
import torch

tf.config.experimental.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("TF_NUM_INTRAOP_THREADS", "4")))
tf.config.threading.set_inter_op_parallelism_threads(int(os.environ.get("TF_NUM_INTEROP_THREADS", "2")))


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
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=100_000,
                seed=seed,
                reshuffle_each_iteration=True,
            )
        # Parsing whole batches avoids one Python/TF scheduling operation per row.
        # Shuffle still operates on individual records before batching.
        dataset = dataset.batch(batch_size)
        def parse_batch(serialized):
            example = tf.io.parse_example(serialized, self.description)
            return example["feature"], example["label"], example["domain"]
        dataset = dataset.map(parse_batch, num_parallel_calls=2)
        options = tf.data.Options()
        options.experimental_threading.private_threadpool_size = 4
        dataset = dataset.with_options(options).prefetch(2)

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
