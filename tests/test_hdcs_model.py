"""Behavior checks for preserving HDCS and training useful selective transfer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from modules.SM3_model import HDCSModel
from modules.HDCS_model import ConfigurableHDCS, loss


class HDCSChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2022)
        torch.set_num_threads(2)
        self.opt=dict(latent_dim=4,feat_num=64,field_num=3,domain_num=3,
                      mlp_dropout=0.,use_bn=False,mlp_dims=[16,8],hdcs_dims=[8],threshold=.5)
        self.x=torch.randint(0,64,(12,3))
        self.domain=torch.tensor([0,1,2]*4).reshape(-1,1).float()
        self.y=torch.tensor([0,0,1,0,1,0]*2).reshape(-1,1).float()

    def test_default_forward_and_native_inference_preserve_upstream(self):
        old=HDCSModel(self.opt,'dnn').eval()
        new=ConfigurableHDCS(self.opt).eval()
        new.load_state_dict(old.state_dict())
        for name,value in old(self.x,self.domain).items():
            torch.testing.assert_close(value,new(self.x,self.domain)[name])
        torch.testing.assert_close(old.predict_native(self.x,self.domain),new.predict_native(self.x,self.domain))

    def test_original_source_and_allocation_paths_are_preserved(self):
        model=ConfigurableHDCS(dict(self.opt,initial_probability=.9,residual=True))
        output=model(self.x,self.domain)
        source=torch.nn.functional.one_hot(self.domain.long().reshape(-1),3).bool()
        self.assertTrue(torch.equal(output['hard_decision'][source],torch.zeros(12)))
        self.assertTrue(torch.equal(output['hard_decision'][~source],torch.ones(24)))
        total=loss(output,self.y,self.domain,dict(lambda_tr=.3,lambda1=.1,lambda2=.01))['total']
        total.backward()
        self.assertGreater(float(model.allocator.output.weight.grad.norm()),0.)
        self.assertGreater(float(model.embedding.embedding.grad.norm()),0.)

    def test_detached_selector_regularization_does_not_change_backbone(self):
        model=ConfigurableHDCS(dict(self.opt,detach_selector=True))
        output=model(self.x,self.domain)
        regularizer=loss(output,self.y,self.domain,dict(lambda_tr=.3,lambda1=.1,lambda2=.01))['kl']
        gradients=torch.autograd.grad(regularizer, [model.backbone_dnn.mlp[0].weight, model.selector_heads[0].weight],allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertGreater(float(gradients[1].norm()),0.)

    def test_bernoulli_regularizer_pushes_closed_gates_toward_retention_target(self):
        p=torch.full((12,3),.15,requires_grad=True)
        cross=1-torch.nn.functional.one_hot(self.domain.long().reshape(-1),3).float()
        output=dict(native_logits=torch.zeros(12,3),transfer_logits=torch.zeros(12,3),
                    transfer_probability=p,hard_decision=torch.zeros(12,3),
                    cross_scenario_mask=cross,allocation_weight=torch.zeros(12,3))
        reg=loss(output,self.y,self.domain,dict(lambda_tr=.3,lambda1=.1,lambda2=.01,
                 regularizer='bernoulli',target_rate=.7))['kl']
        reg.backward()
        self.assertTrue(bool((p.grad[cross.bool()]<0).all()))
        self.assertTrue(bool((p.grad[~cross.bool()]==0).all()))

    def test_small_input_adapter_preserves_native_path_and_receives_training_gradients(self):
        model=ConfigurableHDCS(dict(self.opt,input_adapter_dim=8,residual=True,initial_probability=.8))
        output=model(self.x,self.domain)
        torch.testing.assert_close(model.predict_native(self.x,self.domain),
                                   output['native_logits'].gather(1,self.domain.long()))
        loss(output,self.y,self.domain,dict(lambda_tr=.3,lambda1=.1,lambda2=.01))['total'].backward()
        self.assertGreater(float(model.input_adapters[0][0].weight.grad.norm()),0.)


if __name__=='__main__':
    unittest.main()
