"""Reusable neural-network layers for the HDCS backbones."""

import torch
import torch.nn.functional as F


class FeatureEmbedding(torch.nn.Module):
    """Shared sparse feature embedding table (paper Eq. 7)."""

    def __init__(self, feature_num, latent_dim):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.empty(feature_num, latent_dim))
        torch.nn.init.xavier_uniform_(self.embedding)

    def forward(self, features):
        return F.embedding(features, self.embedding)


class FactorizationMachine(torch.nn.Module):
    """Second-order factorization-machine interaction layer."""

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, embeddings):
        square_of_sum = torch.sum(embeddings, dim=1) ** 2
        sum_of_square = torch.sum(embeddings**2, dim=1)
        interactions = 0.5 * (square_of_sum - sum_of_square)
        if self.reduce_sum:
            interactions = torch.sum(interactions, dim=1, keepdim=True)
        return interactions


class MultiLayerPerceptron(torch.nn.Module):
    """Configurable MLP used by backbones, selectors, and allocators."""

    def __init__(
        self,
        input_dim,
        mlp_dims,
        dropout,
        output_layer=True,
        use_bn=False,
        use_ln=False,
    ):
        super().__init__()
        layers = []
        for output_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, output_dim))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(output_dim))
            if use_ln:
                layers.append(torch.nn.LayerNorm(output_dim))
            layers.extend((torch.nn.ReLU(), torch.nn.Dropout(p=dropout)))
            input_dim = output_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class CrossNetwork(torch.nn.Module):
    """Explicit feature-crossing network used by the DCN backbone."""

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.weights = torch.nn.ModuleList(
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        )
        self.biases = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        )

    def forward(self, inputs):
        crossed = inputs
        for weight, bias in zip(self.weights, self.biases):
            crossed = inputs * weight(crossed) + bias + crossed
        return crossed
