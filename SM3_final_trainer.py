"""Training entry point for the three-scenario AliCCP dataset."""

import argparse
import logging
import os
import pickle
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from modules import SM3_model


def build_parser():
    parser = argparse.ArgumentParser(description="Train three-scenario HDCS")
    parser.add_argument("--dataset", type=str, default="ali-ccp")
    parser.add_argument("--model", type=str, choices=("dnn", "deepfm", "dcn"), default="dnn")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-7)
    parser.add_argument("--bsize", type=int, default=4096)
    parser.add_argument("--optim", type=str, choices=("Adam", "SGD", "adam", "sgd"), default="Adam")
    parser.add_argument("--train_epoch", type=int, default=20)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpt/hdcs_sm3.pt"))

    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--mlp_dims", type=int, nargs="+", default=[1024, 512, 256])
    parser.add_argument("--hdcs_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--mlp_bn", dest="mlp_bn", action="store_true", default=True)
    parser.add_argument("--no-mlp_bn", dest="mlp_bn", action="store_false")
    parser.add_argument("--cross", type=int, default=3)

    parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=0)
    parser.add_argument("--threshold", "--thre", dest="threshold", type=float, default=0.5)
    parser.add_argument("--lambda_tr", type=float, default=0.3)
    parser.add_argument("--lambda1", type=float, default=0.2)
    parser.add_argument("--lambda2", type=float, default=1.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--config", type=Path,
                        help="DNN JSON configuration; defaults to the dataset configuration")
    parser.add_argument("--cache", type=Path,
                        help="TFRecord cache directory; defaults to cache/<dataset>")
    parser.add_argument("--output-dir", type=Path,
                        help="Per-seed DNN outputs; use a fresh directory")
    parser.add_argument("--validation-only", action="store_true",
                        help="Skip test evaluation for the configured DNN path")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_id):
    if device_id == -1:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --cuda -1 for CPU")
    torch.cuda.set_device(device_id)
    return torch.device(f"cuda:{device_id}")


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt="[{asctime}]:{message}", style="{"))
        logger.addHandler(handler)
    return logger


def get_feature_dimensions(stats_path):
    defaults_path = Path(stats_path) / "defaults.pkl"
    with defaults_path.open("rb") as stream:
        defaults = pickle.load(stream)
    return [value + 1 for value in defaults.values()]


def get_data_loader(dataset, data_dir):
    # Import lazily so model/loss checks do not require TensorFlow.
    from dataloader import Dtfloader

    if dataset.lower() == "ali-ccp":
        return Dtfloader.Ali_CCPLoader(data_dir)
    raise ValueError(f"Unsupported three-scenario dataset: {dataset}")


def compute_hdcs_loss(outputs, label, domain, lambda_tr, lambda1, lambda2):
    """Paper Eqs. (20)-(24), averaged over samples/valid transfer pairs."""
    label = label.reshape(-1, 1).float()
    domain_indices = domain.reshape(-1).round().long()

    native_logits = outputs["native_logits"]
    native_prediction = native_logits.gather(1, domain_indices.unsqueeze(1))
    native_loss = F.binary_cross_entropy_with_logits(native_prediction, label)

    transfer_label = label.expand_as(outputs["transfer_logits"])
    transfer_loss_per_pair = F.binary_cross_entropy_with_logits(
        outputs["transfer_logits"],
        transfer_label,
        reduction="none",
    )
    cross_scenario_mask = outputs["cross_scenario_mask"]
    valid_pair_count = cross_scenario_mask.sum().clamp_min(1.0)
    hard_decision = outputs["hard_decision"]

    transfer_loss = (transfer_loss_per_pair * hard_decision).sum() / valid_pair_count

    transfer_probability = outputs["transfer_probability"].clamp(min=1e-8, max=1.0)
    kl_loss = (-torch.log(transfer_probability) * cross_scenario_mask).sum() / valid_pair_count
    sparsity_loss = (hard_decision * cross_scenario_mask).sum() / valid_pair_count

    total_loss = (
        native_loss
        + lambda_tr * transfer_loss
        + lambda1 * kl_loss
        + lambda2 * sparsity_loss
    )
    return {
        "total": total_loss,
        "native": native_loss,
        "transfer": transfer_loss,
        "kl": kl_loss,
        "sparsity": sparsity_loss,
    }


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.seed = opt["seed"]
        set_seed(self.seed)

        self.device = get_device(opt["cuda"])
        self.batch_size = opt["bsize"]
        self.epochs = opt["train_epoch"]
        self.patience = opt["patience"]
        self.lambda_tr = opt["lambda_tr"]
        self.lambda1 = opt["lambda1"]
        self.lambda2 = opt["lambda2"]
        self.domain_list = list(range(opt["model_opt"]["domain_num"]))
        self.model_path = opt["save_dir"]
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.dataloader = get_data_loader(opt["dataset"], opt["data_dir"])
        self.network = SM3_model.getModel(opt["model"], opt["model_opt"]).to(self.device)
        self.optimizer = SM3_model.getOptim(
            self.network,
            opt["optimizer"],
            opt["lr"],
            opt["l2"],
        )
        self.logger = get_logger(f"hdcs-{opt['model']}-seed-{self.seed}")

    def train_on_batch(self, label, data, domain):
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        data = data.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        outputs = self.network(data, domain)
        losses = compute_hdcs_loss(
            outputs,
            label,
            domain,
            self.lambda_tr,
            self.lambda1,
            self.lambda2,
        )
        losses["total"].backward()
        self.optimizer.step()
        return {name: value.detach().item() for name, value in losses.items()}

    def eval_on_batch(self, data, domain):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            domain = domain.to(self.device)
            native_logit = self.network.predict_native(data, domain)
            return torch.sigmoid(native_logit).cpu().numpy()

    def evaluate_validation(self, split):
        predictions, labels = [], []
        for feature, label, domain in self.dataloader.get_data(split, batch_size=self.batch_size * 10):
            predictions.append(self.eval_on_batch(feature, domain))
            labels.append(label.numpy())
        y_pred = np.concatenate(predictions).reshape(-1).astype("float64")
        y_true = np.concatenate(labels).reshape(-1).astype("float64")
        return metrics.roc_auc_score(y_true, y_pred), metrics.log_loss(y_true, y_pred)

    def evaluate_test(self, split):
        predictions = {domain: [] for domain in self.domain_list}
        labels = {domain: [] for domain in self.domain_list}

        for feature, label, domain in self.dataloader.get_data(split, batch_size=self.batch_size * 10):
            prediction = self.eval_on_batch(feature, domain).reshape(-1)
            label_array = label.numpy().reshape(-1)
            domain_array = domain.numpy().reshape(-1).round().astype("int64")
            for target_domain in self.domain_list:
                indices = np.nonzero(domain_array == target_domain)[0]
                if indices.size:
                    predictions[target_domain].append(prediction[indices])
                    labels[target_domain].append(label_array[indices])

        auc, logloss = {}, {}
        for target_domain in self.domain_list:
            y_pred = np.concatenate(predictions[target_domain]).astype("float64")
            y_true = np.concatenate(labels[target_domain]).astype("float64")
            auc[target_domain] = metrics.roc_auc_score(y_true, y_pred)
            logloss[target_domain] = metrics.log_loss(y_true, y_pred)
        return auc, logloss

    def train(self):
        self.logger.info("-----------------Begin Train-----------------")
        best_auc = float("-inf")
        stale_epochs = 0

        for epoch_index in range(self.epochs):
            epoch_losses = {name: 0.0 for name in ("total", "native", "transfer", "kl", "sparsity")}
            batch_count = 0

            train_batches = self.dataloader.get_data(
                "train",
                batch_size=self.batch_size,
                shuffle=True,
                seed=self.seed + epoch_index,
            )
            for step, (feature, label, domain) in enumerate(train_batches, start=1):
                batch_count = step
                batch_losses = self.train_on_batch(label, feature, domain)
                for name, value in batch_losses.items():
                    epoch_losses[name] += value
                if step % 1000 == 0:
                    self.logger.info(
                        "Epoch %d | Step %d | total=%.6f native=%.6f transfer=%.6f kl=%.6f spa=%.6f",
                        epoch_index,
                        step,
                        batch_losses["total"],
                        batch_losses["native"],
                        batch_losses["transfer"],
                        batch_losses["kl"],
                        batch_losses["sparsity"],
                    )

            if batch_count == 0:
                raise RuntimeError(f"No training TFRecords found in {self.opt['data_dir']}")
            epoch_losses = {name: value / batch_count for name, value in epoch_losses.items()}
            validation_auc, validation_loss = self.evaluate_validation("val")
            self.logger.info(
                "Epoch %d | total=%.6f native=%.6f transfer=%.6f kl=%.6f spa=%.6f | Val AUC=%.6f Logloss=%.6f",
                epoch_index,
                epoch_losses["total"],
                epoch_losses["native"],
                epoch_losses["transfer"],
                epoch_losses["kl"],
                epoch_losses["sparsity"],
                validation_auc,
                validation_loss,
            )

            if validation_auc > best_auc:
                best_auc = validation_auc
                stale_epochs = 0
                torch.save(self.network.state_dict(), self.model_path)
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    self.logger.info("Early stop at epoch %d", epoch_index)
                    break

        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(state_dict)
        test_auc, test_logloss = self.evaluate_test("test")
        for target_domain in self.domain_list:
            self.logger.info(
                "Test S%d | AUC=%.6f Logloss=%.6f",
                target_domain + 1,
                test_auc[target_domain],
                test_logloss[target_domain],
            )
        return test_auc, test_logloss


def checkpoint_for_seed(base_path, seed):
    base_path = Path(base_path)
    if base_path.suffix:
        return base_path.with_name(f"{base_path.stem}_seed{seed}{base_path.suffix}")
    return base_path / f"hdcs_sm3_seed{seed}.pt"


def run(args):
    if args.dataset.lower() != "ali-ccp":
        raise ValueError("SM3_final_trainer.py supports the three-scenario AliCCP dataset only")

    if args.model == "dnn" or args.config is not None:
        from modules.hdcs_training import run_configured
        return run_configured(args)

    field_dimensions = get_feature_dimensions("./data/ali-ccp/stats")
    data_dir = "./data/ali-ccp/tfrecord"
    model_opt = {
        "latent_dim": args.dim,
        "feat_num": sum(field_dimensions),
        "field_num": len(field_dimensions),
        "domain_num": 3,
        "mlp_dropout": args.mlp_dropout,
        "use_bn": args.mlp_bn,
        "mlp_dims": args.mlp_dims,
        "hdcs_dims": args.hdcs_dims,
        "cross": args.cross,
        "threshold": args.threshold,
    }

    all_auc = {domain: [] for domain in range(3)}
    all_logloss = {domain: [] for domain in range(3)}
    for seed in args.seeds:
        opt = {
            "model_opt": model_opt,
            "dataset": args.dataset,
            "model": args.model,
            "lr": args.lr,
            "l2": args.l2,
            "bsize": args.bsize,
            "optimizer": args.optim,
            "data_dir": data_dir,
            "save_dir": checkpoint_for_seed(args.save_dir, seed),
            "cuda": args.cuda,
            "train_epoch": args.train_epoch,
            "patience": args.patience,
            "lambda_tr": args.lambda_tr,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "seed": seed,
        }
        test_auc, test_logloss = Trainer(opt).train()
        for domain in range(3):
            all_auc[domain].append(test_auc[domain])
            all_logloss[domain].append(test_logloss[domain])

    logger = get_logger("hdcs-summary")
    for domain in range(3):
        auc_mean = statistics.mean(all_auc[domain])
        auc_std = statistics.stdev(all_auc[domain]) if len(all_auc[domain]) > 1 else 0.0
        loss_mean = statistics.mean(all_logloss[domain])
        loss_std = statistics.stdev(all_logloss[domain]) if len(all_logloss[domain]) > 1 else 0.0
        logger.info(
            "S%d over %d seeds | AUC=%.6f +/- %.6f | Logloss=%.6f +/- %.6f",
            domain + 1,
            len(args.seeds),
            auc_mean,
            auc_std,
            loss_mean,
            loss_std,
        )


def main():
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
