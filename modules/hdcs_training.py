"""Cached HDCS training and checkpoint evaluation for AliCCP and AliMama."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import copy
import json
import hashlib
import math
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temp.replace(path)


def dataset_spec(name):
    if name == 'ali-ccp':
        return dict(dataset=name, fields=18, domains=3, stats='stats', records='tfrecord')
    if name == 'ali-mama':
        return dict(dataset=name, fields=16, domains=4, stats='stats_2', records='threshold_2')
    raise ValueError('Unknown dataset: '+name)


def prepare_cache(directory, splits, dataset='ali-ccp'):
    # Materialization preserves exact feature IDs, labels, domains and file order.
    from dataloader.Dtfloader import Ali_CCPLoader, Ali_MamaLoader
    spec = dataset_spec(dataset)
    loader_type = Ali_CCPLoader if dataset == 'ali-ccp' else Ali_MamaLoader
    loader = loader_type(str(ROOT / 'data' / dataset / spec['records']))
    directory.mkdir(parents=True, exist_ok=True)
    for split in splits:
        meta = directory / (split + '.json')
        if meta.exists():
            assert json.loads(meta.read_text()).get('dataset','ali-ccp') == dataset, 'Cache dataset mismatch'
            print('Cache exists:', split, flush=True)
            continue
        paths = [directory / (split + '_' + name + '.bin') for name in ('feature', 'label', 'domain')]
        streams = [p.open('wb') for p in paths]
        count = 0
        try:
            source_split = split
            for features, labels, domains in loader.get_data(source_split, batch_size=65536):
                arrays = [features.numpy().astype('int32'), labels.numpy().astype('uint8'), domains.numpy().astype('uint8')]
                assert np.isin(labels.numpy(), [0, 1]).all()
                assert np.isin(domains.numpy(), list(range(spec['domains']))).all()
                for array, stream in zip(arrays, streams):
                    array.tofile(stream)
                count += len(features)
                if count % (65536 * 100) == 0:
                    print('Cached', split, count, flush=True)
        finally:
            for stream in streams:
                stream.close()
        if count == 0:
            raise RuntimeError('No rows found for split: ' + split)
        originals = [{ 'path': str(Path(p).resolve()), 'bytes': Path(p).stat().st_size,
                       'mtime_ns': Path(p).stat().st_mtime_ns} for p in loader._files(source_split)]
        write_json(meta, dict(rows=count, fields=spec['fields'], domains=spec['domains'], dataset=dataset, source_files=originals,
                             format='feature int32; label/domain uint8; original row order'))
        print('Cached', split, count, 'COMPLETE', flush=True)


class CachedData:
    def __init__(self, directory, split):
        self.split = split
        meta = json.loads((directory / (split + '.json')).read_text())
        for source in meta['source_files']:
            stat = Path(source['path']).stat()
            assert stat.st_size == source['bytes'] and stat.st_mtime_ns == source['mtime_ns'], 'Source changed'
        self.rows = meta['rows']
        self.dataset = meta.get('dataset','ali-ccp')
        self.domains = meta.get('domains',3)
        self.feature = np.memmap(directory/(split+'_feature.bin'), dtype='int32', mode='r', shape=(self.rows, meta['fields']))
        self.label = np.memmap(directory/(split+'_label.bin'), dtype='uint8', mode='r', shape=(self.rows, 1))
        self.domain = np.memmap(directory/(split+'_domain.bin'), dtype='uint8', mode='r', shape=(self.rows, 1))

    def batches(self, size, shuffle_seed=None):
        indices = None if shuffle_seed is None else np.random.RandomState(shuffle_seed).permutation(self.rows)
        for start in range(0, self.rows, size):
            index = slice(start, start+size) if indices is None else indices[start:start+size]
            yield (torch.from_numpy(self.feature[index].astype('int64')),
                   torch.from_numpy(self.label[index].astype('float32')),
                   torch.from_numpy(self.domain[index].astype('float32')))


def augment_features(features, probability, mask_id, rng):
    """Mask input values on a copy; labels, domains and the cache stay intact."""
    if not probability:
        return features
    masked = features.clone()
    masked.numpy()[rng.random_sample(tuple(features.shape)) < probability] = mask_id
    return masked


def learning_rate_factor(step, steps, warmup_fraction=0., minimum=0.):
    """Linear warmup followed by cosine decay, with exact end points."""
    assert steps > 0 and 0. <= warmup_fraction < 1. and 0. <= minimum <= 1.
    warmup_steps = int(steps * warmup_fraction)
    if step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, steps - warmup_steps - 1)
    return minimum + (1. - minimum) * .5 * (1. + math.cos(math.pi * min(1., progress)))


@torch.no_grad()
def average_parameters(target, source, beta, copy_only=()):
    """Average dense weights while optionally retaining final sparse tables."""
    torch._foreach_mul_(target, beta)
    torch._foreach_add_(target, source, alpha=1-beta)
    for destination, current in copy_only:
        destination.copy_(current)


def evaluate(model, data, device, batch_size=32768):
    model.eval()
    predictions = np.empty(data.rows, dtype='float32')
    cursor = 0
    with torch.no_grad():
        for feature, label, domain in data.batches(batch_size):
            feature, domain = feature.to(device), domain.to(device)
            native = model.predict_native(feature, domain)
            prediction = native.float().sigmoid().cpu().numpy().reshape(-1)
            predictions[cursor:cursor+len(prediction)] = prediction
            cursor += len(prediction)
    labels = np.asarray(data.label).reshape(-1)
    domains = np.asarray(data.domain).reshape(-1)
    result = {'auc': float(roc_auc_score(labels, predictions)),
              'logloss': float(log_loss(labels, predictions.astype('float64'), labels=[0, 1]))}
    for k in range(data.domains):
        index = domains == k
        result['auc_s'+str(k+1)] = float(roc_auc_score(labels[index], predictions[index]))
        result['logloss_s'+str(k+1)] = float(log_loss(labels[index], predictions[index].astype('float64'), labels=[0, 1]))
    result['macro_auc'] = float(np.mean([result['auc_s'+str(k)] for k in range(1,data.domains+1)]))
    result['macro_logloss'] = float(np.mean([result['logloss_s'+str(k)] for k in range(1,data.domains+1)]))
    return result


def model_options(config):
    import pickle
    spec = dataset_spec(config.get('dataset','ali-ccp'))
    with (ROOT/'data'/spec['dataset']/spec['stats']/'defaults.pkl').open('rb') as f:
        defaults = pickle.load(f)
    return dict(latent_dim=config.get('dim', 16), feat_num=sum(x+1 for x in defaults.values()),
                field_num=len(defaults), domain_num=spec['domains'], model='dnn', cross=3, fusion='mean',
                mlp_dims=config.get('mlp_dims', [1024,512,256]), hdcs_dims=config.get('hdcs_dims', [512,256,128]),
                mlp_dropout=config.get('dropout', 0.), use_bn=config.get('bn', False),
                threshold=config.get('threshold', .5), detach_selector=config.get('detach_selector', False),
                residual=config.get('residual', False), initial_probability=config.get('initial_probability'),
                input_adapter_dim=config.get('input_adapter_dim',0),linear_init=config.get('linear_init','xavier'),
                input_adapter_scale=config.get('input_adapter_scale',1.),
                input_adapter_dims=config.get('input_adapter_dims'),
                linear_residual=config.get('linear_residual','none'),
                linear_residual_scale=config.get('linear_residual_scale',1.),
                prediction_sharing=config.get('prediction_sharing','none'),
                shared_readout_scale=config.get('shared_readout_scale',0.),
                activation=config.get('activation','relu'),
                residual_scale=config.get('residual_scale',1.))


def run(args):
    from modules.HDCS_model import ConfigurableHDCS, loss
    config = json.loads(args.config.read_text())
    seed = int(config.get('seed', 2022))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        config.update(cuda_graph=False, amp=False)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output/'result.json').exists():
        raise FileExistsError('Result already exists: '+str(output))
    if config.get('kind', 'hdcs') != 'hdcs':
        raise ValueError('This trainer supports HDCS only')
    options = model_options(config)
    if options['domain_num'] == 4:
        from modules.SM4_model import ConfiguredMaskDNN
        model = ConfiguredMaskDNN(options)
    else:
        model = ConfigurableHDCS(options)
    model = model.to(device)
    ema_model = None
    if config.get('ema_decay',0.) and not args.evaluate_checkpoint:
        assert not options['use_bn'], 'EMA requires a model without BatchNorm'
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        ema_parameters = list(ema_model.parameters())
        live_parameters = list(model.parameters())
        ema_copy_only = []
        if config.get('ema_dense_only', False):
            pairs = [(name, target, source) for (name, target), source in
                     zip(ema_model.named_parameters(), model.parameters())]
            sparse_names = {'embedding.embedding', 'wide_weight'}
            ema_copy_only = [(target, source) for name, target, source in pairs if name in sparse_names]
            ema_parameters = [target for name, target, source in pairs if name not in sparse_names]
            live_parameters = [source for name, target, source in pairs if name not in sparse_names]

    ema_last_update = 0
    def update_ema(global_step):
        nonlocal ema_last_update
        if ema_model is None:
            return
        interval=config.get('ema_interval',25)
        if global_step%interval and global_step%batch_count:
            return
        beta=0. if global_step<=config.get('ema_warmup',1000) else config['ema_decay']**(global_step-ema_last_update)
        average_parameters(ema_parameters, live_parameters, beta, ema_copy_only)
        ema_last_update=global_step
    checkpoint = output/'best.pt'
    if args.evaluate_checkpoint:
        state = torch.load(str(args.evaluate_checkpoint), map_location=device)
        if 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)
        data = CachedData(args.cache, args.eval_split)
        assert data.dataset == config.get('dataset','ali-ccp'), 'Cache dataset mismatch'
        evaluation = evaluate(model, data, device)
        write_json(output/'result.json', dict(config=config, split=args.eval_split, metrics=evaluation))
        print('EVALUATED', json.dumps(evaluation), flush=True)
        return
    train = CachedData(args.cache, 'train')
    validation = CachedData(args.cache, 'val')
    assert train.dataset == validation.dataset == config.get('dataset','ali-ccp'), 'Cache dataset mismatch'
    feature_dropout = config.get('feature_dropout', 0.)
    assert 0. <= feature_dropout < 1.
    if feature_dropout:
        mask_id = int(config['mask_feature_id'])
        # Use an allocated, otherwise unused ID, never a real category. No test
        # rows or labels are inspected to configure this training augmentation.
        assert max(int(train.feature.max()), int(validation.feature.max())) < mask_id < options['feat_num']
        augmentation_rng = np.random.RandomState(seed+17000)
    auxiliary = [(n,p) for n,p in model.named_parameters() if n.startswith('selector_')]
    main = [p for n,p in model.named_parameters() if not n.startswith('selector_') and n != 'wide_weight'
            and not ('embedding_lr' in config and n == 'embedding.embedding')]
    groups = [
        dict(params=main, lr=config['lr'], weight_decay=config.get('l2', 1e-7)),
        dict(params=[p for n,p in auxiliary], lr=config.get('selector_lr', config['lr']), weight_decay=0.)]
    if 'embedding_lr' in config:
        groups.append(dict(params=[model.embedding.embedding], lr=config['embedding_lr'],
                           weight_decay=config.get('embedding_l2',config.get('l2',1e-7)),
                           eps=config.get('embedding_eps',1e-8),
                           betas=tuple(config.get('embedding_betas', [.9,.999]))))
    if model.wide_weight is not None:
        groups.append(dict(params=[model.wide_weight], lr=config.get('linear_lr',config['lr']),
                           weight_decay=config.get('linear_l2',config.get('l2',1e-7))))
    optimizer_type = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[config.get('optimizer','adam')]
    optimizers = [optimizer_type(groups)]
    amp = bool(config.get('amp', False)) and device.type=='cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    graph_step = None
    history, best, stale = [], -math.inf, 0
    epochs = int(config.get('epochs', 3))
    batch_size = int(config.get('batch_size', 4096))
    batch_count = math.ceil(train.rows/batch_size)
    schedule = config.get('lr_schedule', 'constant')
    assert schedule in ('constant', 'cosine')
    base_learning_rates = [[float(g['lr']) for g in optimizer.param_groups] for optimizer in optimizers]
    write_json(output/'config.json', dict(config=config, model_options=options, cache=str(args.cache),
               torch=torch.__version__, python=sys.version, device=str(device), pid=os.getpid(),
               checkpoint_metric='validation pooled auc'))
    started = time.time()
    for epoch in range(epochs):
        model.train()
        sums, count = {}, 0
        graph_rows = 0
        if graph_step is not None:
            graph_step.reset()
        seed_for_order = seed+epoch if config.get('shuffle', False) else None
        if config.get('decay', 1.) != 1. and epoch:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group['lr'] *= config['decay']
        for step, (feature, label, domain) in enumerate(train.batches(batch_size, seed_for_order)):
            if schedule == 'cosine':
                factor = learning_rate_factor(epoch * batch_count + step, epochs * batch_count,
                    config.get('warmup_fraction', 0.), config.get('minimum_lr_fraction', .1))
                for optimizer, rates in zip(optimizers, base_learning_rates):
                    for group, rate in zip(optimizer.param_groups, rates):
                        if torch.is_tensor(group['lr']):
                            group['lr'].fill_(rate * factor)
                        else:
                            group['lr'] = rate * factor
            if feature_dropout:
                feature = augment_features(feature, feature_dropout, mask_id, augmentation_rng)
            feature, label, domain = feature.to(device), label.to(device), domain.to(device)
            if config.get('cuda_graph', False) and len(feature) == batch_size:
                if graph_step is None:
                    from modules.HDCS_model import GraphStep
                    graph_step = GraphStep(model, optimizers[0], config, (feature,label,domain))
                graph_step.step((feature,label,domain))
                update_ema(epoch*batch_count+step+1)
                graph_rows += len(feature)
                count += len(feature)
                if (step+1)%1000 == 0:
                    print('epoch',epoch,'step',step+1,'/',batch_count,'graph',True,
                          'elapsed_s',round(time.time()-started),flush=True)
                continue
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            progress = (epoch+step/batch_count)/max(1,epochs)
            model.temperature = config.get('final_temperature',1.) ** progress
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(feature, domain)
                losses = loss(outputs, label, domain, config)
            scaler.scale(losses['total']).backward()
            if config.get('clip', 0):
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()
            update_ema(epoch*batch_count+step+1)
            for key,value in losses.items():
                sums[key] = sums.get(key, 0.) + float(value.detach())*len(feature)
            count += len(feature)
            if (step+1)%1000 == 0:
                print('epoch',epoch,'step',step+1,'/',batch_count,'loss',float(losses['total'].detach()),
                      'elapsed_s',round(time.time()-started),flush=True)
        validation_metrics = evaluate(model, validation, device)
        selected_model = model
        selected_weights = 'live'
        live_validation = validation_metrics
        ema_validation = None
        if ema_model is not None:
            ema_validation = evaluate(ema_model, validation, device)
            if config.get('ema_only',False) or ema_validation['auc']>validation_metrics['auc']:
                validation_metrics=ema_validation
                selected_model=ema_model
                selected_weights='ema'
        if graph_step is not None and graph_rows:
            for key,value in graph_step.metrics(graph_rows).items():
                sums[key] = sums.get(key, 0.) + value*graph_rows
        item = dict(epoch=epoch, training={k:v/count for k,v in sums.items()},
                    validation=validation_metrics, elapsed_s=time.time()-started)
        if ema_validation is not None:
            item.update(live_validation=live_validation,ema_validation=ema_validation,selected_weights=selected_weights)
        history.append(item)
        score = validation_metrics['auc']
        if score > best:
            best, stale = score, 0
            saved = dict(state_dict=selected_model.state_dict(), config=config, model_options=options,
                         epoch=epoch, validation=validation_metrics,selected_weights=selected_weights)
            torch.save(saved, str(checkpoint))
        else:
            stale += 1
        write_json(output/'history.json', history)
        print('VALIDATION', json.dumps(item),flush=True)
        if stale >= config.get('patience', 2):
            break
    best_item = max(history, key=lambda x:x['validation']['auc'])
    write_json(output/'result.json', dict(config=config, best_epoch=best_item['epoch'],
               validation=best_item['validation'], checkpoint=str(checkpoint),
               elapsed_s=time.time()-started, trained_epochs=len(history), test_evaluated=False))
    if config.get('evaluate_test',False):
        # The checkpoint policy and configuration are fixed before test access.
        saved = torch.load(str(checkpoint),map_location=device)
        model.load_state_dict(saved['state_dict'])
        test = CachedData(args.cache,'test')
        assert test.dataset == config.get('dataset','ali-ccp'), 'Cache dataset mismatch'
        metrics = evaluate(model,test,device)
        write_json(output/'test_result.json',dict(config=config,split='test',metrics=metrics,
                   checkpoint=str(checkpoint),selected_weights=saved.get('selected_weights','live')))
        print('TEST',json.dumps(metrics),flush=True)



def run_configured(args):
    """Launch the dataset DNN configuration once for each requested seed."""
    import subprocess
    if args.model != 'dnn':
        raise ValueError('--config is supported by the DNN model only')
    dataset = args.dataset.lower()
    filename = 'hdcs_dnn_aliccp.json' if dataset == 'ali-ccp' else 'hdcs_dnn_alimama.json'
    config_path = args.config or ROOT / 'configs' / filename
    config = json.loads(config_path.resolve().read_text())
    if config.get('dataset', dataset) != dataset:
        raise ValueError('Configuration dataset does not match --dataset')
    config['dataset'] = dataset
    config['evaluate_test'] = not args.validation_only
    cache = (args.cache or ROOT / 'cache' / dataset).resolve()
    output = (args.output_dir or ROOT / 'output' / dataset).resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if args.cuda == -1 else str(args.cuda),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-u', str(ROOT / 'modules' / 'hdcs_training.py')]
    splits = ['train', 'val'] + (['test'] if config['evaluate_test'] else [])
    missing = [split for split in splits if not (cache / (split + '.json')).exists()]
    if missing:
        subprocess.run(command + ['--cache', str(cache), '--dataset', dataset,
                       '--prepare'] + missing, cwd=str(ROOT), env=env, check=True)
    for seed in args.seeds:
        directory = output / ('seed' + str(seed))
        directory.mkdir(parents=True, exist_ok=False)
        selected = dict(config, seed=seed)
        if args.cuda == -1:
            selected.update(cuda_graph=False, amp=False)
        requested = directory / 'requested_config.json'
        write_json(requested, selected)
        subprocess.run(command + ['--cache', str(cache), '--config', str(requested),
                       '--output', str(directory)], cwd=str(ROOT), env=env, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--prepare', nargs='+', choices=['train','val','test'])
    p.add_argument('--dataset', choices=['ali-ccp','ali-mama'], default='ali-ccp', help='Dataset for cache preparation; training uses the config dataset')
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--evaluate-checkpoint', type=Path)
    p.add_argument('--eval-split', choices=['val','test'], default='val')
    args=p.parse_args()
    if args.prepare:
        prepare_cache(args.cache, args.prepare, args.dataset)
    else:
        if args.config is None or args.output is None:
            p.error('--config and --output required for training')
        run(args)


if __name__=='__main__':
    main()
