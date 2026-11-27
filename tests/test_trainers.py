"""Shared training behavior across both datasets and all three backbones."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import trainer as module


class TrainerChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_dataset_defaults_and_explicit_overrides(self):
        paths = set()
        for dataset, lr, l2 in [('ali-ccp', 1e-3, 1e-7), ('ali-mama', 3e-4, 3e-6)]:
            for backbone in ('dnn', 'deepfm', 'dcn'):
                args = module.resolve_args(module.build_parser().parse_args([
                    '--dataset', dataset, '--model', backbone,
                ]))
                self.assertEqual((args.lr, args.l2), (lr, l2))
                paths.add(args.save_dir)
                overridden = module.resolve_args(module.build_parser().parse_args([
                    '--dataset', dataset, '--model', backbone,
                    '--lr', '0.02', '--l2', '0', '--save_dir', 'custom.pt',
                ]))
                self.assertEqual((overridden.lr, overridden.l2), (.02, 0.))
                self.assertEqual(overridden.save_dir, Path('custom.pt'))
        self.assertEqual(len(paths), 6)

    @staticmethod
    def loader(domains):
        class SyntheticLoader:
            def get_data(self, split, **kwargs):
                yield (torch.arange(domains * 6).reshape(domains * 2, 3) % 32,
                       torch.tensor([0., 1.] * domains).reshape(-1, 1),
                       torch.arange(domains).repeat_interleave(2).float().reshape(-1, 1))
        return SyntheticLoader()

    def test_all_entry_points_train_save_reload_and_evaluate(self):
        for dataset, domains in [('ali-ccp', 3), ('ali-mama', 4)]:
            for backbone in ('dnn', 'deepfm', 'dcn'):
                with self.subTest(domains=domains, backbone=backbone), tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / 'model.pt'
                    args = module.build_parser().parse_args([
                        '--dataset', dataset, '--model', backbone, '--cuda', '-1', '--seeds', '7',
                        '--train_epoch', '1', '--bsize', '8', '--dim', '4',
                        '--mlp_dims', '16', '8', '--hdcs_dims', '8',
                        '--threshold', '0', '--save_dir', str(checkpoint),
                    ])
                    with patch.object(module, 'get_feature_dimensions', return_value=[32, 32, 32]) as dimensions, \
                         patch.object(module, 'get_data_loader', return_value=self.loader(domains)) as loader:
                        module.run(args)
                    dimensions.assert_called_once_with(module.DATASETS[dataset]['stats'])
                    loader.assert_called_once_with(dataset, module.DATASETS[dataset]['records'])
                    saved = torch.load(module.checkpoint_for_seed(checkpoint, 7), weights_only=True)
                    self.assertIn('embedding.embedding', saved)
                    self.assertIn(f'prediction_heads.{domains - 1}.weight', saved)
                    self.assertNotIn(f'prediction_heads.{domains}.weight', saved)
                    self.assertTrue(all(torch.isfinite(value).all() for value in saved.values()))

    def test_validation_uses_pooled_auc_not_macro_auc(self):
        # Perfect ranking inside each domain, imperfect ranking across domains.
        for dataset, domains in [('ali-ccp', 3), ('ali-mama', 4)]:
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
        for dataset, domains in [('ali-ccp', 3), ('ali-mama', 4)]:
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
                    trainer.logger = module.get_logger('test-early-stop')
                    models = module.hdcs
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
