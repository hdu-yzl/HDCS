"""Configurable HDCS representations, routing, allocation and training loss."""
import math
import torch
import torch.nn.functional as F
from modules.SM3_model import HDCSModel, StraightThroughGate


class ConfigurableHDCS(HDCSModel):
    def __init__(self, opt):
        super().__init__(opt, backbone=opt.get('model', 'dnn'))
        self.detach_selector = opt.get('detach_selector', False)
        self.residual = opt.get('residual', False)
        self.residual_scale = opt.get('residual_scale', 1.)
        self.temperature = opt.get('temperature', 1.0)
        prediction_sharing = opt.get('prediction_sharing', 'none')
        if prediction_sharing == 'last_pair':
            self.prediction_heads[-1] = self.prediction_heads[-2]
        elif prediction_sharing == 'all':
            for index in range(1, self.domain_num):
                self.prediction_heads[index] = self.prediction_heads[0]
        elif prediction_sharing != 'none':
            raise ValueError('prediction_sharing must be none, last_pair or all')
        self.linear_residual = opt.get('linear_residual', 'none')
        self.linear_residual_scale = opt.get('linear_residual_scale', 1.)
        if self.linear_residual not in ('none', 'shared', 'scenario'):
            raise ValueError('linear_residual must be none, shared or scenario')
        self.wide_weight = None
        if self.linear_residual != 'none':
            width = 1 if self.linear_residual == 'shared' else self.domain_num
            self.wide_weight = torch.nn.Parameter(torch.zeros(self.feature_num, width))
        adapter_dim = opt.get('input_adapter_dim', 0)
        self.input_adapter_scale = float(opt.get('input_adapter_scale', 1.))
        self.input_adapters = None
        adapter_dims = opt.get('input_adapter_dims')
        if adapter_dims:
            def make_adapter():
                layers = []
                dimensions = [self.embedding_dim] + list(adapter_dims) + [self.representation_dim]
                for source, target in zip(dimensions, dimensions[1:]):
                    layers.extend([torch.nn.Linear(source, target), torch.nn.ReLU()])
                return torch.nn.Sequential(*layers)
            self.input_adapters = torch.nn.ModuleList([make_adapter() for _ in range(self.domain_num)])
            self.input_adapters.apply(self._initialize_module)
        elif adapter_dim:
            self.input_adapters = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(self.embedding_dim,adapter_dim),torch.nn.ReLU(),
                                    torch.nn.Linear(adapter_dim,self.representation_dim),torch.nn.ReLU())
                for _ in range(self.domain_num)])
            self.input_adapters.apply(self._initialize_module)
        if opt.get('linear_init') == 'pytorch':
            for module in self.modules():
                if isinstance(module,torch.nn.Linear):
                    module.reset_parameters()
        initial_probability = opt.get('initial_probability')
        if initial_probability is not None:
            bias = torch.logit(torch.tensor(float(initial_probability))).item()
            for head in self.selector_heads:
                torch.nn.init.constant_(head.bias, bias)
        self.shared_readout_scale = opt.get('shared_readout_scale', 0.)
        self.shared_readout = None
        if self.shared_readout_scale:
            self.shared_readout = torch.nn.Linear(self.representation_dim, 1)
            torch.nn.init.zeros_(self.shared_readout.weight)
            torch.nn.init.zeros_(self.shared_readout.bias)
        activation = opt.get('activation', 'relu')
        if activation != 'relu':
            factories = {'silu': torch.nn.SiLU, 'gelu': torch.nn.GELU,
                         'leaky_relu': lambda: torch.nn.LeakyReLU(.1)}
            if activation not in factories:
                raise ValueError('Unsupported activation: ' + activation)
            def replace_relu(parent):
                for name, child in parent.named_children():
                    if isinstance(child, torch.nn.ReLU):
                        setattr(parent, name, factories[activation]())
                    else:
                        replace_relu(child)
            replace_relu(self)

    def _native_representations(self, features):
        backbone = self._encode_backbone(features)
        shared = self.shared_projector(backbone)
        if self.residual and shared.shape[-1] == backbone.shape[-1]:
            shared = self.residual_scale * shared + backbone
        specifics = torch.stack([branch(shared) for branch in self.scenario_specific_branches], dim=1)
        if self.residual:
            specifics = self.residual_scale * specifics + shared.unsqueeze(1)
        if self.input_adapters is not None:
            embedded = self.embedding(features).reshape(len(features),self.embedding_dim)
            specifics = specifics + self.input_adapter_scale * torch.stack(
                [adapter(embedded) for adapter in self.input_adapters],dim=1)
        native = torch.cat([head(specifics[:, k]) for k, head in enumerate(self.prediction_heads)], dim=1)
        if self.wide_weight is not None:
            native = native + self._wide_logits(features)
        if self.shared_readout is not None:
            native = native + self.shared_readout_scale * self.shared_readout(shared)
        return shared, specifics, native

    def _wide_logits(self, features):
        # A small additive feature effect, shared by native/transfer predictions.
        # The HDCS representations, selector and allocation remain unchanged.
        value = _EmbeddingGradient.apply(features, self.wide_weight).sum(dim=1)
        return self.linear_residual_scale * value

    def forward(self, features, domain):
        shared, specifics, native = self._native_representations(features)
        fused = specifics.mean(dim=1)
        if self.detach_selector:
            fused = fused.detach()
        common = self.selector_shared(fused)
        transfer_representation = torch.stack([branch(common) for branch in self.selector_target_branches], dim=1)
        logits = torch.cat([head(transfer_representation[:, k]) for k, head in enumerate(self.selector_heads)], dim=1)
        probability = torch.sigmoid(logits.float() * self.temperature)
        cross = 1.0 - F.one_hot(self._domain_indices(domain), self.domain_num).to(probability.dtype)
        decision = StraightThroughGate.apply(probability - self.threshold) * cross
        allocation = self.allocator(transfer_representation) * cross
        mixed = allocation.unsqueeze(-1) * shared.unsqueeze(1) + (1.0 - allocation.unsqueeze(-1)) * specifics
        transfer = torch.cat([head(mixed[:, k]) for k, head in enumerate(self.prediction_heads)], dim=1)
        if self.wide_weight is not None:
            transfer = transfer + self._wide_logits(features)
        if self.shared_readout is not None:
            transfer = transfer + self.shared_readout_scale * self.shared_readout(shared)
        return dict(native_logits=native, transfer_logits=transfer,
                    transfer_probability=probability, hard_decision=decision,
                    allocation_weight=allocation, cross_scenario_mask=cross)


def loss(outputs, label, domain, config):
    label = label.reshape(-1, 1).float()
    native_logits = outputs['native_logits'].float()
    source = domain.long().reshape(-1, 1)
    if config.get('native_domain_weights'):
        # Fixed weights are calculated from training counts only. Keeping the
        # constants as scalar operations also supports CUDA graph replay.
        weights = config['native_domain_weights']
        assert len(weights) == native_logits.shape[1] and all(w > 0 for w in weights)
        row_weight = torch.zeros_like(label)
        for k, weight in enumerate(weights):
            row_weight = row_weight + (source == k).to(label.dtype) * weight
        native = (F.binary_cross_entropy_with_logits(native_logits.gather(1, source), label,
                                                     reduction='none') * row_weight).mean()
    else:
        native = F.binary_cross_entropy_with_logits(native_logits.gather(1, source), label)
    cross = outputs['cross_scenario_mask']
    count = cross.sum().clamp_min(1)
    transfer_bce = F.binary_cross_entropy_with_logits(outputs['transfer_logits'].float(), label.expand_as(cross), reduction='none')
    hard = outputs['hard_decision']
    transfer_weights = torch.ones_like(cross)
    if config.get('transfer_prior_correction', False):
        # Restore each target's pre-selection class mixture. The detached
        # normalization changes data supervision without rewarding gate closure.
        transfer_weights = torch.zeros_like(cross)
        for class_label in (0., 1.):
            group = (label == class_label).to(cross.dtype) * cross
            available = group.sum(dim=0, keepdim=True)
            selected = (group * hard.detach()).sum(dim=0, keepdim=True)
            denominator = torch.maximum(selected, .2 * available).clamp_min(1.)
            transfer_weights = transfer_weights + group * (available / denominator)
    transfer_terms = transfer_bce * hard
    if config.get('selector_advantage', False):
        # Compare a target's transfer loss with the source prediction's loss.
        # Forward value and predictor gradients stay unchanged; only the gate
        # receives a relative-loss signal instead of preferring easy negatives.
        baseline = F.binary_cross_entropy_with_logits(native_logits.gather(1, source),
                                                       label, reduction='none').detach()
        transfer_terms = transfer_terms + baseline * (hard.detach() - hard)
    if config.get('prior_gate_unweighted', False):
        # Class correction weights the transferred predictor examples. Keep its
        # batch-dependent factor out of the gate objective, where amplification
        # would otherwise drive all transfer probabilities below threshold.
        gate_cost = transfer_bce.detach()
        if config.get('selector_advantage', False):
            gate_cost = gate_cost - baseline
        transfer = (transfer_bce * hard.detach() * transfer_weights
                    + gate_cost * (hard-hard.detach())).sum() / count
    else:
        transfer = (transfer_terms * transfer_weights).sum() / count
    p = outputs['transfer_probability'].clamp(1e-6, 1 - 1e-6)
    if config.get('regularizer', 'original') == 'bernoulli':
        # Finite target retention gives both lower and upper pressure on p.
        target = float(config.get('target_rate', .7))
        kl = ((target * (math.log(target) - p.log())
               + (1-target) * (math.log(1-target) - (1-p).log())) * cross).sum() / count
    else:
        kl = (-p.log() * cross).sum() / count
    sparsity = (hard * cross).sum() / count
    total = native + config['lambda_tr'] * transfer + config['lambda1'] * kl + config['lambda2'] * sparsity
    if config.get('consistency_weight', 0.):
        # Selected mixed predictions teach the inference path. Detaching both
        # the teacher and selection leaves the original allocator/gate loss
        # intact, rather than rewarding gate closure or changing inference.
        teacher = outputs['transfer_logits'].float().detach().sigmoid()
        consistency = F.binary_cross_entropy_with_logits(native_logits, teacher, reduction='none')
        total = total + config['consistency_weight'] * (consistency * hard.detach()).sum() / count
    if config.get('ranking_weight', 0.):
        # Each positive is paired with a shuffled-batch negative in its own
        # scenario. Equal scenario averaging matches macro AUC. All indexing
        # has static shapes, including scenes with no positives/negatives.
        own = native_logits.gather(1, source).reshape(-1)
        flat_label, flat_source = label.reshape(-1), source.reshape(-1)
        order = torch.argsort(flat_source * 2 + flat_label.long())
        row = torch.arange(len(label), device=own.device)
        ranking = own.sum() * 0.
        for k in range(native_logits.shape[1]):
            start = (flat_source < k).long().sum()
            negative_count = ((flat_source == k) & (flat_label == 0)).long().sum()
            indices = start + (row * 131 + 17).remainder(negative_count.clamp_min(1))
            negative = own[order[indices.clamp_max(len(label)-1)]]
            positive_mask = ((flat_source == k) & (flat_label == 1)).to(own.dtype)
            positive_mask = positive_mask * (negative_count > 0).to(own.dtype)
            ranking = ranking + (F.softplus(negative-own) * positive_mask).sum() / positive_mask.sum().clamp_min(1.)
        total = total + config['ranking_weight'] * ranking / native_logits.shape[1]
    if config.get('selection_balance_weight', 0.):
        rates = []
        for class_label in (0., 1.):
            group = (label == class_label).to(cross.dtype) * cross
            rates.append((p * group).sum(dim=0) / group.sum(dim=0).clamp_min(1.))
        total = total + config['selection_balance_weight'] * (rates[0]-rates[1]).square().mean()
    if config.get('lambda_native_transfer', 0.):
        # Supervise the inference branch on selected transferred examples too.
        # The original mixed-representation loss still trains the allocator.
        native_cross_bce = F.binary_cross_entropy_with_logits(native_logits,
                            label.expand_as(cross), reduction='none')
        native_transfer = (native_cross_bce * hard).sum() / count
        total = total + config['lambda_native_transfer'] * native_transfer
    if config.get('allocation_prior_weight', 0.):
        alpha = outputs['allocation_weight'].clamp(1e-6, 1-1e-6)
        prior = ((-.5*alpha.log()-.5*(1-alpha).log() - 0.6931471805599453) * cross).sum()/count
        total = total + config['allocation_prior_weight'] * prior
    return dict(total=total, native=native, transfer=transfer, kl=kl, sparsity=sparsity,
                probability=(p * cross).sum()/count, allocation=(outputs['allocation_weight'] * cross).sum()/count)


class _EmbeddingGradient(torch.autograd.Function):
    """Identical dense embedding derivative without host-sized unique buffers."""
    @staticmethod
    def forward(ctx, indices, weight):
        ctx.save_for_backward(indices)
        ctx.shape = weight.shape
        return F.embedding(indices,weight)

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        result=gradient.new_zeros(ctx.shape)
        result.index_add_(0,indices.reshape(-1),gradient.reshape(-1,ctx.shape[-1]))
        return None,result


class _GraphEmbedding(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.embedding=original.embedding

    def forward(self,indices):
        return _EmbeddingGradient.apply(indices,self.embedding)


class GraphStep:
    def __init__(self, model, optimizer, config, example):
        assert not config.get('amp', False)
        assert not model.use_bn, 'CUDA graph parity is validated for non-BatchNorm trials only'
        assert config.get('dropout', 0.) == 0.
        assert config.get('final_temperature', 1.) == 1.
        assert not config.get('clip', 0.)
        if not isinstance(model.embedding,_GraphEmbedding):
            model.embedding=_GraphEmbedding(model.embedding)
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.inputs = tuple(t.clone() for t in example)
        self.keys = ['total','native','transfer','kl','sparsity','probability','allocation']
        self.sums = torch.zeros(len(self.keys), device=example[0].device)
        self.size = len(example[0])
        initial = {name: value.detach().clone() for name,value in model.state_dict().items()}
        # Allocate Adam states before capture, keeping all scalars on-device.
        for group in optimizer.param_groups:
            group['capturable'] = True
            group['lr'] = torch.tensor(float(group['lr']), device=example[0].device)
        optimizer.defaults['capturable'] = True
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                values = loss(model(self.inputs[0],self.inputs[2]),self.inputs[1],self.inputs[2],config)
                values['total'].backward()
                optimizer.step()
        torch.cuda.current_stream().wait_stream(stream)
        # Warmup must not count as training: restore weights, BN buffers and Adam.
        model.load_state_dict(initial)
        for state in optimizer.state.values():
            for name,value in state.items():
                if torch.is_tensor(value):
                    value.zero_()
        optimizer.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.values = loss(model(self.inputs[0],self.inputs[2]),self.inputs[1],self.inputs[2],config)
            self.values['total'].backward()
            optimizer.step()
            self.sums.add_(torch.stack([self.values[k].detach() for k in self.keys]) * self.size)
        # CUDA capture does not execute its kernels. Start metrics from zero.
        self.sums.zero_()

    def reset(self):
        self.sums.zero_()

    def step(self, example):
        for target,value in zip(self.inputs,example):
            target.copy_(value)
        self.graph.replay()

    def metrics(self, rows):
        return dict(zip(self.keys,(self.sums/rows).cpu().tolist()))
