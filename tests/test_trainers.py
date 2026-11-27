"""Shared training behavior across both datasets and all three backbones."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import SM3_final_trainer as sm3
import SM4_final_trainer as sm4


class TrainerChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    @staticmethod
    def loader(domains):
        class SyntheticLoader:
            def get_data(self, split, **kwargs):
                yield (torch.arange(domains * 6).reshape(domains * 2, 3) % 32,
                       torch.tensor([0., 1.] * domains).reshape(-1, 1),
                       torch.arange(domains).repeat_interleave(2).float().reshape(-1, 1))
        return SyntheticLoader()

    def test_all_entry_points_train_save_reload_and_evaluate(self):
        for module, domains in [(sm3, 3), (sm4, 4)]:
            for backbone in ('dnn', 'deepfm', 'dcn'):
                with self.subTest(domains=domains, backbone=backbone), tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / 'model.pt'
                    args = module.build_parser().parse_args([
                        '--model', backbone, '--cuda', '-1', '--seeds', '7',
                        '--train_epoch', '1', '--bsize', '8', '--dim', '4',
                        '--mlp_dims', '16', '8', '--hdcs_dims', '8',
                        '--threshold', '0', '--save_dir', str(checkpoint),
                    ])
                    with patch.object(module, 'get_feature_dimensions', return_value=[32, 32, 32]), \
                         patch.object(module, 'get_data_loader', return_value=self.loader(domains)):
                        module.run(args)
                    saved = torch.load(module.checkpoint_for_seed(checkpoint, 7), weights_only=True)
                    self.assertIn('embedding.embedding', saved)
                    self.assertTrue(all(torch.isfinite(value).all() for value in saved.values()))

    def test_validation_uses_pooled_auc_not_macro_auc(self):
        # Perfect ranking inside each domain, imperfect ranking across domains.
        for module, domains in [(sm3, 3), (sm4, 4)]:
            with self.subTest(domains=domains):
                trainer = module.Trainer.__new__(module.Trainer)
                trainer.batch_size = 8
                trainer.dataloader = self.loader(domains)
                predictions = np.arange(1, domains * 2 + 1) / (domains * 2 + 1)
                trainer.eval_on_batch = lambda *args: predictions.reshape(-1, 1)
                auc, _ = trainer.evaluate_validation('val')
                self.assertAlmostEqual(auc, roc_auc_score([0, 1] * domains, predictions))
                self.assertLess(auc, 1.)

    def test_early_stop_restores_best_checkpoint_for_every_backbone(self):
        for module, domains in [(sm3, 3), (sm4, 4)]:
            for backbone in ('dnn', 'deepfm', 'dcn'):
                with self.subTest(domains=domains, backbone=backbone), tempfile.TemporaryDirectory() as directory:
                    trainer = module.Trainer.__new__(module.Trainer)
                    trainer.opt = {'data_dir': directory}
                    trainer.seed, trainer.batch_size = 7, 8
                    trainer.epochs, trainer.patience = 5, 1
                    trainer.domain_list = list(range(domains))
                    trainer.device = torch.device('cpu')
                    trainer.model_path = Path(directory) / 'best.pt'
                    trainer.dataloader = self.loader(domains)
                    trainer.logger = sm3.get_logger('test-early-stop')
                    models = module.SM3_model if module is sm3 else module.SM4_model
                    options = dict(latent_dim=4, feat_num=32, field_num=3, domain_num=domains,
                                   mlp_dropout=0., use_bn=False, mlp_dims=[8], hdcs_dims=[8], cross=2)
                    trainer.network = models.getModel(backbone, options)
                    parameter = next(trainer.network.parameters())

                    def train_batch(*args):
                        with torch.no_grad():
                            parameter.add_(1.)
                        return dict.fromkeys(('total', 'native', 'transfer', 'kl', 'sparsity'), 0.)

                    initial = parameter.detach().clone()
                    with patch.object(trainer, 'train_on_batch', side_effect=train_batch) as train, \
                         patch.object(trainer, 'evaluate_validation', side_effect=[(.8, .4), (.7, .3)]), \
                         patch.object(trainer, 'evaluate_test', return_value=(dict.fromkeys(range(domains), .6),
                                                                            dict.fromkeys(range(domains), .4))):
                        trainer.train()
                    self.assertEqual(train.call_count, 2)
                    torch.testing.assert_close(parameter, initial + 1.)


if __name__ == '__main__':
    unittest.main()
