"""Hierarchical differentiated cross-scenario sharing models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import CrossNetwork, FactorizationMachine, FeatureEmbedding, MultiLayerPerceptron


class StraightThroughGate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, value):
        return (value > 0).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SoftAllocator(nn.Module):

    def __init__(self, input_dim, hidden_dims, dropout, use_bn):
        super().__init__()
        self.network = MultiLayerPerceptron(
            input_dim,
            hidden_dims,
            dropout,
            output_layer=False,
            use_bn=use_bn,
        )
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, transfer_representation):
        batch_size, domain_num, input_dim = transfer_representation.shape
        flattened = transfer_representation.reshape(batch_size * domain_num, input_dim)
        alpha = torch.sigmoid(self.output(self.network(flattened)))
        return alpha.reshape(batch_size, domain_num)


class HDCSModel(nn.Module):

    def __init__(self, opt, backbone):
        super().__init__()
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.field_num = opt["field_num"]
        self.domain_num = opt["domain_num"]
        self.backbone = backbone
        self.threshold = opt.get("threshold", 0.5)
        self.dropout = opt["mlp_dropout"]
        self.use_bn = opt["use_bn"]
        self.backbone_dims = list(opt.get("mlp_dims", [1024, 512, 256]))
        self.hdcs_dims = list(opt.get("hdcs_dims", [512, 256, 128]))
        self.embedding_dim = self.field_num * self.latent_dim

        self.embedding = FeatureEmbedding(self.feature_num, self.latent_dim)
        self.backbone_dnn = MultiLayerPerceptron(
            self.embedding_dim,
            self.backbone_dims,
            self.dropout,
            output_layer=False,
            use_bn=self.use_bn,
        )

        if backbone == "dnn":
            backbone_output_dim = self.backbone_dims[-1]
        elif backbone == "deepfm":
            self.fm = FactorizationMachine(reduce_sum=False)
            backbone_output_dim = self.backbone_dims[-1] + self.latent_dim
        elif backbone == "dcn":
            self.cross = CrossNetwork(self.embedding_dim, opt["cross"])
            backbone_output_dim = self.backbone_dims[-1] + self.embedding_dim
        else:
            raise ValueError(f"Invalid backbone type: {backbone}")

        self.representation_dim = self.hdcs_dims[-1]


        self.shared_projector = self._make_hdcs_fnn(backbone_output_dim)
        self.scenario_specific_branches = nn.ModuleList(
            self._make_hdcs_fnn(self.representation_dim) for _ in range(self.domain_num)
        )
        self.prediction_heads = nn.ModuleList(
            nn.Linear(self.representation_dim, 1) for _ in range(self.domain_num)
        )


        self.selector_shared = self._make_hdcs_fnn(self.representation_dim)
        self.selector_target_branches = nn.ModuleList(
            self._make_hdcs_fnn(self.representation_dim) for _ in range(self.domain_num)
        )
        self.selector_heads = nn.ModuleList(
            nn.Linear(self.representation_dim, 1) for _ in range(self.domain_num)
        )


        self.allocator = SoftAllocator(
            self.representation_dim,
            self.hdcs_dims,
            self.dropout,
            self.use_bn,
        )

        self.apply(self._initialize_module)

    def _make_hdcs_fnn(self, input_dim):
        return MultiLayerPerceptron(
            input_dim,
            self.hdcs_dims,
            self.dropout,
            output_layer=False,
            use_bn=self.use_bn,
        )

    @staticmethod
    def _initialize_module(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _domain_indices(domain):
        return domain.reshape(-1).round().long()

    def _encode_backbone(self, features):
        embedded = self.embedding(features)
        flattened = embedded.reshape(embedded.shape[0], self.embedding_dim)
        deep_representation = self.backbone_dnn(flattened)

        if self.backbone == "dnn":
            return deep_representation
        if self.backbone == "deepfm":
            fm_representation = self.fm(embedded)
            return torch.cat((deep_representation, fm_representation), dim=1)

        cross_representation = self.cross(flattened)
        return torch.cat((deep_representation, cross_representation), dim=1)

    def _native_representations(self, features):
        backbone_representation = self._encode_backbone(features)
        shared_representation = self.shared_projector(backbone_representation)
        specific_representations = torch.stack(
            [branch(shared_representation) for branch in self.scenario_specific_branches],
            dim=1,
        )
        native_logits = torch.cat(
            [
                head(specific_representations[:, index, :])
                for index, head in enumerate(self.prediction_heads)
            ],
            dim=1,
        )
        return shared_representation, specific_representations, native_logits

    def predict_native(self, features, domain):
        _, _, native_logits = self._native_representations(features)
        domain_indices = self._domain_indices(domain)
        return native_logits.gather(1, domain_indices.unsqueeze(1))

    def forward(self, features, domain):
        domain_indices = self._domain_indices(domain)
        shared_representation, specific_representations, native_logits = self._native_representations(features)


        fused_representation = specific_representations.mean(dim=1)
        selector_common = self.selector_shared(fused_representation)


        selector_representations = torch.stack(
            [branch(selector_common) for branch in self.selector_target_branches],
            dim=1,
        )
        selector_logits = torch.cat(
            [
                head(selector_representations[:, index, :])
                for index, head in enumerate(self.selector_heads)
            ],
            dim=1,
        )
        transfer_probability = torch.sigmoid(selector_logits)

        source_one_hot = F.one_hot(domain_indices, num_classes=self.domain_num).to(transfer_probability.dtype)
        cross_scenario_mask = 1.0 - source_one_hot
        hard_decision = StraightThroughGate.apply(transfer_probability - self.threshold)
        hard_decision = hard_decision * cross_scenario_mask

        allocation_weight = self.allocator(selector_representations) * cross_scenario_mask
        mixed_representation = (
            allocation_weight.unsqueeze(-1) * shared_representation.unsqueeze(1)
            + (1.0 - allocation_weight.unsqueeze(-1)) * specific_representations
        )
        transfer_logits = torch.cat(
            [
                head(mixed_representation[:, index, :])
                for index, head in enumerate(self.prediction_heads)
            ],
            dim=1,
        )

        return {
            "native_logits": native_logits,
            "transfer_logits": transfer_logits,
            "transfer_probability": transfer_probability,
            "hard_decision": hard_decision,
            "allocation_weight": allocation_weight,
            "cross_scenario_mask": cross_scenario_mask,
        }


class MaskDNN(HDCSModel):
    def __init__(self, opt):
        super().__init__(opt, backbone="dnn")


class MaskDeepFM(HDCSModel):
    def __init__(self, opt):
        super().__init__(opt, backbone="deepfm")


class MaskDeepCross(HDCSModel):
    def __init__(self, opt):
        super().__init__(opt, backbone="dcn")


def getOptim(network, optim, lr, l2):
    optimizer_name = optim.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(network.parameters(), lr=lr, weight_decay=l2)
    if optimizer_name == "adam":
        return torch.optim.Adam(network.parameters(), lr=lr, weight_decay=l2)
    raise ValueError(f"Invalid optimizer type: {optim}")


def getModel(model: str, opt):
    model_name = model.lower()
    if model_name == "deepfm":
        return MaskDeepFM(opt)
    if model_name == "dcn":
        return MaskDeepCross(opt)
    if model_name == "dnn":
        return MaskDNN(opt)
    raise ValueError(f"Invalid model type: {model}")
