"""Behavior checks for opt-in four-scene adaptation and selective supervision."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from modules.SM4_model import getModel, ConfiguredMaskDNN
from modules.HDCS_model import loss, GraphStep
from train_script.hdcs_train import augment_features, learning_rate_factor, average_parameters


class FourSceneAdaptation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2022)
        torch.set_num_threads(2)
        self.options = dict(latent_dim=4, feat_num=64, field_num=3, domain_num=3,
            mlp_dropout=0., use_bn=False, mlp_dims=[16, 8], hdcs_dims=[8],
            threshold=.5, initial_probability=.8, residual=True,
            input_adapter_dims=[12, 8])
        self.config = dict(lambda_tr=.5, lambda1=.15, lambda2=.1,
                           lambda_native_transfer=.2, native_domain_weights=[2., 2., 1., .5])
        self.x = torch.randint(0, 64, (16, 3))
        self.domain = (torch.arange(16) % 4).float().reshape(-1, 1)
        self.label = ((torch.arange(16) // 4) % 2).float().reshape(-1, 1)

    def test_four_scene_native_inference_and_all_core_gradients(self):
        model = getModel('dnn', dict(self.options, configured=True))
        self.assertIsInstance(model, ConfiguredMaskDNN)
        self.assertEqual(model.domain_num, 4)
        output = model(self.x, self.domain)
        self.assertEqual(tuple(output['native_logits'].shape), (16, 4))
        torch.testing.assert_close(model.predict_native(self.x, self.domain),
            output['native_logits'].gather(1, self.domain.long()))
        source = torch.nn.functional.one_hot(self.domain.long().reshape(-1), 4).bool()
        self.assertTrue(bool((output['hard_decision'][source] == 0).all()))
        loss(output, self.label, self.domain, self.config)['total'].backward()
        for name, parameter in model.named_parameters():
            if (name.endswith('weight') and ('prediction_heads' in name or
                    'input_adapters' in name or 'allocator.output' in name or 'selector_heads' in name)):
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(parameter.grad.norm().item(), 0., name)

    def test_native_balance_changes_only_own_scene_supervision(self):
        logits = torch.zeros(4, 4, requires_grad=True)
        domains = torch.arange(4).float().reshape(-1, 1)
        cross = 1 - torch.eye(4)
        output = dict(native_logits=logits, transfer_logits=torch.zeros(4, 4),
            cross_scenario_mask=cross, hard_decision=torch.zeros(4, 4),
            transfer_probability=torch.full((4, 4), .8), allocation_weight=cross * .5)
        config = dict(lambda_tr=0., lambda1=0., lambda2=0., native_domain_weights=[2., 2., 1., .5])
        loss(output, torch.ones(4, 1), domains, config)['total'].backward()
        self.assertTrue(bool((logits.grad[cross.bool()] == 0).all()))
        self.assertAlmostEqual(float(logits.grad[0, 0] / logits.grad[3, 3]), 4.)

    def test_native_transfer_supervises_only_selected_cross_scene_pairs(self):
        model = ConfiguredMaskDNN(self.options)
        output = model(self.x, self.domain)
        hard = torch.zeros_like(output['hard_decision'])
        hard[0, 2] = 1.
        output['hard_decision'] = hard
        config = dict(lambda_tr=0., lambda1=0., lambda2=0.)
        base = loss(output, self.label, self.domain, config)['total']
        auxiliary = loss(output, self.label, self.domain, dict(config, lambda_native_transfer=.2))['total'] - base
        gradient = torch.autograd.grad(auxiliary, output['native_logits'])[0]
        self.assertGreater(abs(gradient[0, 2].item()), 0.)
        gradient[0, 2] = 0.
        self.assertTrue(bool((gradient == 0).all()))

    def test_linear_residual_is_additive_without_changing_selection_or_allocation(self):
        model = ConfiguredMaskDNN(dict(self.options, linear_residual='shared'))
        before = model(self.x, self.domain)
        with torch.no_grad():
            model.wide_weight.fill_(.02)
        after = model(self.x, self.domain)
        # Three fields, each adding .02, to every native and transferred logit.
        for key in ['native_logits', 'transfer_logits']:
            torch.testing.assert_close(after[key] - before[key], torch.full_like(after[key], .06))
        for key in ['hard_decision', 'transfer_probability', 'allocation_weight']:
            torch.testing.assert_close(before[key], after[key])
        torch.testing.assert_close(model.predict_native(self.x, self.domain),
            after['native_logits'].gather(1, self.domain.long()))
        loss(after, self.label, self.domain, self.config)['total'].backward()
        self.assertGreater(model.wide_weight.grad.norm().item(), 0.)
        self.assertGreater(model.allocator.output.weight.grad.norm().item(), 0.)

    def test_shared_readout_keeps_all_four_target_branches_trainable(self):
        model = ConfiguredMaskDNN(dict(self.options, prediction_sharing='last_pair'))
        self.assertIs(model.prediction_heads[2], model.prediction_heads[3])
        self.assertIsNot(model.prediction_heads[0], model.prediction_heads[1])
        output = model(self.x, self.domain)
        loss(output, self.label, self.domain, self.config)['total'].backward()
        for branch in model.scenario_specific_branches:
            self.assertGreater(branch.mlp[0].weight.grad.norm().item(), 0.)
        self.assertGreater(model.allocator.output.weight.grad.norm().item(), 0.)
        torch.testing.assert_close(model.predict_native(self.x, self.domain),
            output['native_logits'].gather(1, self.domain.long()))

    def test_shared_prediction_shortcut_has_zero_initial_effect_and_receives_gradient(self):
        model = ConfiguredMaskDNN(dict(self.options, shared_readout_scale=.3))
        before = model(self.x, self.domain)
        with torch.no_grad():
            model.shared_readout.bias.fill_(.2)
        after = model(self.x, self.domain)
        for key in ['native_logits', 'transfer_logits']:
            torch.testing.assert_close(after[key]-before[key], torch.full_like(after[key], .06))
        loss(after, self.label, self.domain, self.config)['total'].backward()
        self.assertGreater(model.shared_readout.weight.grad.norm().item(), 0.)
        self.assertGreater(model.allocator.output.weight.grad.norm().item(), 0.)

    def test_feature_masking_is_reproducible_and_preserves_original_rows(self):
        original = self.x.clone()
        first = augment_features(self.x, .25, 99, np.random.RandomState(2022))
        second = augment_features(self.x, .25, 99, np.random.RandomState(2022))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(self.x, original))
        selected = first == 99
        self.assertTrue(bool(selected.any()))
        self.assertTrue(bool((~selected).any()))
        self.assertTrue(torch.equal(first[~selected], original[~selected]))

    def test_adapter_scale_preserves_native_inference_and_all_core_gradients(self):
        model = ConfiguredMaskDNN(dict(self.options, input_adapter_scale=.3))
        output = model(self.x, self.domain)
        torch.testing.assert_close(model.predict_native(self.x, self.domain),
                                   output['native_logits'].gather(1, self.domain.long()))
        loss(output, self.label, self.domain, self.config)['total'].backward()
        for branch in model.input_adapters:
            self.assertGreater(branch[0].weight.grad.norm().item(), 0.)
        self.assertGreater(model.allocator.output.weight.grad.norm().item(), 0.)

    def test_smooth_activation_keeps_native_inference_and_core_gradients(self):
        for activation in ['silu', 'gelu', 'leaky_relu']:
            model = ConfiguredMaskDNN(dict(self.options, activation=activation))
            output = model(self.x, self.domain)
            torch.testing.assert_close(model.predict_native(self.x, self.domain),
                                       output['native_logits'].gather(1, self.domain.long()))
            loss(output, self.label, self.domain, self.config)['total'].backward()
            for name, parameter in model.named_parameters():
                if name.endswith('weight') and ('allocator.output' in name or 'selector_heads' in name):
                    self.assertTrue(torch.isfinite(parameter.grad).all(), (activation, name))
                    self.assertGreater(parameter.grad.norm().item(), 0., (activation, name))

    def test_consistency_teaches_only_selected_native_pairs(self):
        native = torch.zeros(4, 4, requires_grad=True)
        transfer = torch.full((4, 4), 2., requires_grad=True)
        hard = torch.zeros(4, 4, requires_grad=True)
        with torch.no_grad():
            hard[1, 2] = 1.
        cross = 1. - torch.eye(4)
        output = dict(native_logits=native, transfer_logits=transfer, hard_decision=hard,
            cross_scenario_mask=cross, transfer_probability=cross*.8+.1,
            allocation_weight=cross*.5)
        config = dict(lambda_tr=0., lambda1=0., lambda2=0.)
        domains = torch.arange(4).float().reshape(-1, 1)
        labels = torch.zeros(4, 1)
        base = loss(output, labels, domains, config)['total']
        auxiliary = loss(output, labels, domains, dict(config, consistency_weight=.3))['total'] - base
        gradients = torch.autograd.grad(auxiliary, [native, transfer, hard], allow_unused=True)
        self.assertLess(gradients[0][1, 2].item(), 0.)
        gradients[0][1, 2] = 0.
        self.assertEqual(gradients[0].abs().sum().item(), 0.)
        for gradient in gradients[1:]:
            self.assertTrue(gradient is None or gradient.abs().sum().item() == 0.)

    def test_cosine_warmup_and_final_learning_rates(self):
        rates = [learning_rate_factor(i, 100, .1, .2) for i in range(100)]
        self.assertAlmostEqual(rates[0], .1)
        self.assertAlmostEqual(rates[9], 1.)
        self.assertAlmostEqual(rates[10], 1.)
        self.assertAlmostEqual(rates[-1], .2)
        self.assertTrue(all(a >= b for a, b in zip(rates[10:], rates[11:])))

    def test_dense_average_preserves_latest_sparse_tables(self):
        dense = torch.tensor([2., 4.], requires_grad=True)
        sparse = torch.tensor([1., 3.], requires_grad=True)
        average_parameters([dense], [torch.tensor([4., 8.])], .75,
                           [(sparse, torch.tensor([9., 10.]))])
        torch.testing.assert_close(dense, torch.tensor([2.5, 5.]))
        torch.testing.assert_close(sparse, torch.tensor([9., 10.]))
        self.assertIsNone(dense.grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_embedding_epsilon_reaches_real_captured_training(self):
        import argparse
        import contextlib
        import io
        import json
        import tempfile
        from unittest.mock import patch
        from train_script.hdcs_train import run
        options = dict(self.options, domain_num=4)
        torch.manual_seed(2022)
        initial = ConfiguredMaskDNN(options).embedding.embedding.detach().clone()
        distances = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root/'cache'
            cache.mkdir()
            for split in ['train', 'val']:
                (cache/(split+'.json')).write_text(json.dumps(dict(rows=16, fields=3,
                    domains=4, dataset='ali-mama', source_files=[])))
                self.x.numpy().astype('int32').tofile(str(cache/(split+'_feature.bin')))
                self.label.numpy().astype('uint8').tofile(str(cache/(split+'_label.bin')))
                self.domain.numpy().astype('uint8').tofile(str(cache/(split+'_domain.bin')))
            for index, epsilon in enumerate([1e-8, 1.]):
                config = dict(seed=2022, dataset='ali-mama', epochs=1, batch_size=16,
                    lr=.001, embedding_lr=.001, embedding_eps=epsilon, l2=0.,
                    lambda_tr=.5, lambda1=.15, lambda2=.1, cuda_graph=True)
                path = root/('config'+str(index)+'.json')
                path.write_text(json.dumps(config))
                output = root/('trial'+str(index))
                args = argparse.Namespace(config=path, output=output, cache=cache,
                                          evaluate_checkpoint=None, eval_split='val')
                with patch('train_script.hdcs_train.model_options', return_value=options), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                saved = torch.load(str(output/'best.pt'), map_location='cpu')
                distances.append((saved['state_dict']['embedding.embedding']-initial).norm().item())
        self.assertGreater(distances[0], 0.)
        self.assertLess(distances[1], distances[0]*.5)

    def test_ranking_loss_pairs_only_within_scenario_and_handles_absent_classes(self):
        # Scene 0 has both classes; scene 1 has positives only, scene 2 has
        # negatives only and scene 3 is absent. Only scene 0 receives ranking.
        domains = torch.tensor([0., 0., 1., 1., 2., 2.]).reshape(-1, 1)
        labels = torch.tensor([1., 0., 1., 1., 0., 0.]).reshape(-1, 1)
        native = torch.zeros(6, 4, requires_grad=True)
        cross = 1 - torch.nn.functional.one_hot(domains.long().reshape(-1), 4).float()
        output = dict(native_logits=native, transfer_logits=native*0., hard_decision=cross,
            cross_scenario_mask=cross, transfer_probability=cross*.8+.1, allocation_weight=cross*.5)
        config = dict(lambda_tr=0., lambda1=0., lambda2=0.)
        base = loss(output, labels, domains, config)['total']
        auxiliary = loss(output, labels, domains, dict(config, ranking_weight=.1))['total'] - base
        gradient = torch.autograd.grad(auxiliary, native)[0]
        self.assertAlmostEqual(auxiliary.item(), .1 * np.log(2.) / 4, places=6)
        self.assertLess(gradient[0, 0].item(), 0.)
        self.assertGreater(gradient[1, 0].item(), 0.)
        gradient[:2, 0] = 0.
        self.assertEqual(gradient.abs().sum().item(), 0.)

    def test_transfer_prior_correction_restores_preselection_class_mixture(self):
        # Six negatives and two positives per target; half the negatives are
        # selected, while both positives are selected. Corrected BCE restores
        # the original 3:1 gradient mixture without creating unselected data.
        label = torch.tensor([0.]*6+[1.]*2).reshape(-1, 1)
        hard = torch.tensor([1., 1., 1., 0., 0., 0., 1., 1.]).reshape(-1, 1).expand(-1, 4)
        logits = torch.zeros(8, 4, requires_grad=True)
        cross = torch.ones_like(logits)
        output = dict(native_logits=torch.zeros_like(logits), transfer_logits=logits,
            cross_scenario_mask=cross, hard_decision=hard,
            transfer_probability=cross*.8, allocation_weight=cross*.5)
        config = dict(lambda_tr=1., lambda1=0., lambda2=0., transfer_prior_correction=True)
        value = loss(output, label, torch.zeros(8, 1), config)['transfer']
        value.backward()
        self.assertAlmostEqual(float(value), float(torch.log(torch.tensor(2.))), places=6)
        self.assertAlmostEqual(float(logits.grad[:6].sum() / -logits.grad[6:].sum()), 3.)
        self.assertTrue(bool((logits.grad[3:6] == 0).all()))

    def test_selector_advantage_changes_only_gate_gradient(self):
        logits = torch.full((2, 4), -1., requires_grad=True)
        hard = torch.full((2, 4), 1., requires_grad=True)
        source = torch.zeros_like(logits, requires_grad=True)
        output = dict(native_logits=source, transfer_logits=logits,
            cross_scenario_mask=torch.ones_like(logits), hard_decision=hard,
            transfer_probability=torch.full_like(logits, .8), allocation_weight=logits*.0+.5)
        config = dict(lambda_tr=1., lambda1=0., lambda2=0.)
        base = loss(output, torch.zeros(2, 1), torch.zeros(2, 1), config)['transfer']
        adjusted = loss(output, torch.zeros(2, 1), torch.zeros(2, 1), dict(config, selector_advantage=True))['transfer']
        torch.testing.assert_close(base, adjusted)
        first = torch.autograd.grad(base, [logits, hard], retain_graph=True)
        second = torch.autograd.grad(adjusted, [logits, hard, source], allow_unused=True)
        torch.testing.assert_close(first[0], second[0])
        self.assertTrue(bool((first[1] > 0).all()))
        self.assertTrue(bool((second[1] < 0).all()))
        self.assertIsNone(second[2])

    def test_class_correction_does_not_amplify_selector_gradient_when_requested(self):
        label = torch.tensor([0.]*6+[1.]*2).reshape(-1, 1)
        hard = torch.tensor([1., 1., 1., 0., 0., 0., 1., 1.]).reshape(-1, 1).expand(-1, 4).clone().requires_grad_()
        logits = torch.zeros(8, 4, requires_grad=True)
        output = dict(native_logits=torch.zeros_like(logits), transfer_logits=logits,
            cross_scenario_mask=torch.ones_like(logits), hard_decision=hard,
            transfer_probability=torch.full_like(logits, .8), allocation_weight=torch.full_like(logits, .5))
        config = dict(lambda_tr=1., lambda1=0., lambda2=0.)
        regular = loss(output, label, torch.zeros(8, 1), config)['transfer']
        corrected = loss(output, label, torch.zeros(8, 1), dict(config,
            transfer_prior_correction=True, prior_gate_unweighted=True))['transfer']
        first = torch.autograd.grad(regular, [logits, hard], retain_graph=True)
        second = torch.autograd.grad(corrected, [logits, hard])
        torch.testing.assert_close(first[1], second[1])
        torch.testing.assert_close(first[0][:3]*2., second[0][:3])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_balanced_four_scene_cuda_graph_matches_eager_adam(self):
        self.check_graph('adam', False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_linear_residual_cuda_graph_matches_eager_adamw(self):
        self.check_graph('adamw', True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_prior_corrected_advantage_cuda_graph_matches_eager_adam(self):
        self.config.update(transfer_prior_correction=True, selector_advantage=True,
                           selection_balance_weight=.3, prior_gate_unweighted=True)
        self.check_graph('adam', True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_consistency_and_adapter_scaling_cuda_graph_matches_eager(self):
        self.config['consistency_weight'] = .3
        self.options['input_adapter_scale'] = .3
        self.check_graph('adam', True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_ranking_cuda_graph_matches_eager(self):
        self.config['ranking_weight'] = .03
        self.check_graph('adam', True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_silu_cuda_graph_matches_eager(self):
        self.options['activation'] = 'silu'
        self.check_graph('adam', True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_changing_learning_rate_cuda_graph_matches_eager(self):
        self.check_graph('adam', True, schedule=True)

    def check_graph(self, optimizer_name, wide, schedule=False):
        if wide:
            self.options.update(linear_residual='shared', shared_readout_scale=.3)
        eager = ConfiguredMaskDNN(self.options).cuda()
        captured = ConfiguredMaskDNN(self.options).cuda()
        captured.load_state_dict(eager.state_dict())
        optimizer = torch.optim.AdamW if optimizer_name == 'adamw' else torch.optim.Adam
        first = optimizer(eager.parameters(), lr=.001, weight_decay=.01)
        second = optimizer(captured.parameters(), lr=.001, weight_decay=.01)
        batch = (self.x.cuda(), self.label.cuda(), self.domain.cuda())
        step = GraphStep(captured, second, self.config, batch)
        for index in range(3):
            if schedule:
                rate = .001 * learning_rate_factor(index, 3, 0., .2)
                first.param_groups[0]['lr'] = rate
                second.param_groups[0]['lr'].fill_(rate)
            first.zero_grad(set_to_none=True)
            loss(eager(batch[0], batch[2]), batch[1], batch[2], self.config)['total'].backward()
            first.step()
            step.step(batch)
            torch.cuda.synchronize()
            for name, value in eager.state_dict().items():
                torch.testing.assert_close(value, captured.state_dict()[name], rtol=3e-4, atol=3e-6, msg=name)


if __name__ == '__main__':
    unittest.main()
