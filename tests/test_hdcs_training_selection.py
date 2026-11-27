"""Checkpoint selection when pooled and macro AUC disagree."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from modules.hdcs_training import run


class PooledAUCSelection(unittest.TestCase):
    def test_pooled_auc_controls_weights_checkpoint_early_stop_and_summary(self):
        options = dict(latent_dim=4, feat_num=32, field_num=3, domain_num=3,
                       model='dnn', mlp_dropout=0., use_bn=False,
                       mlp_dims=[8], hdcs_dims=[8], threshold=.5)

        class SyntheticData:
            dataset = 'ali-ccp'
            rows = 6

            def batches(self, size, shuffle_seed=None):
                yield (torch.arange(18).reshape(6, 3),
                       torch.tensor([0., 1.] * 3).reshape(-1, 1),
                       torch.tensor([0., 0., 1., 1., 2., 2.]).reshape(-1, 1))

        # Epoch 0: pooled prefers live, macro prefers EMA.
        # Epoch 1: EMA beats live, but neither beats epoch 0's pooled AUC.
        # Macro would instead choose epoch 1 as the best checkpoint.
        evaluations = [dict(auc=.80, macro_auc=.60), dict(auc=.70, macro_auc=.85),
                       dict(auc=.65, macro_auc=.70), dict(auc=.75, macro_auc=.90)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(dataset='ali-ccp', epochs=3, patience=1, batch_size=6,
                          lr=.001, lambda_tr=.3, lambda1=.2, lambda2=1.1,
                          ema_decay=.9, ema_interval=1)
            config_path = root / 'input.json'
            config_path.write_text(json.dumps(config))
            output = root / 'output'
            args = argparse.Namespace(config=config_path, output=output, cache=root,
                                      evaluate_checkpoint=None)
            with patch('modules.hdcs_training.model_options', return_value=options), \
                 patch('modules.hdcs_training.CachedData', return_value=SyntheticData()), \
                 patch('modules.hdcs_training.torch.cuda.is_available', return_value=False), \
                 patch('modules.hdcs_training.evaluate', side_effect=evaluations) as evaluate, \
                 contextlib.redirect_stdout(io.StringIO()):
                run(args)

            history = json.loads((output / 'history.json').read_text())
            result = json.loads((output / 'result.json').read_text())
            checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
            metadata = json.loads((output / 'config.json').read_text())
            self.assertEqual(evaluate.call_count, 4)
            self.assertEqual(result['trained_epochs'], 2)
            self.assertEqual(result['best_epoch'], 0)
            self.assertEqual(result['validation']['auc'], .80)
            self.assertEqual(checkpoint['epoch'], 0)
            self.assertEqual(checkpoint['selected_weights'], 'live')
            self.assertEqual(history[0]['selected_weights'], 'live')
            self.assertEqual(history[1]['selected_weights'], 'ema')
            self.assertEqual(metadata['checkpoint_metric'], 'validation pooled auc')


if __name__ == '__main__':
    unittest.main()
