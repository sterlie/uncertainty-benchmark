import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from swa_gaussian.swag.posteriors import SWAG
import os

from src.methods import register_method
from src.methods.method import Method
from src.methods.utils import multi_label_uncertainty, multi_class_uncertainty
from src.metrics import F1Score, Accuracy, Precision, Recall


@register_method('swag')
class Swag(Method):
    def __init__(self, config):
        super(Swag, self).__init__(config)
        self.swag_model = SWAG(
            self.model.__class__,
            no_cov_mat=False,
            max_num_models=20,
            config=config,
        )
        self.swag_model.to(self.device)
        self.sub_population = config.method.get('sub_population', 1.0)
        self.scale = config.method.get('scale', 0.1)
        self.swag_batch_size = config.method.get('swag_batch_size', 64)
        self.uncertainty_per_class = config.method.get('uncertainty_per_class', False)
        self.train_loader = None
        self.model_dir = "swag"
        self.eps = 1e-12

    def init_optimizer(self):
        arguments = OmegaConf.to_container(self.config.optimizer)
        arguments.pop("name", None)
        arguments.pop("epochs", None)
        scheduler_arguments = arguments.pop("scheduler", None)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=arguments["lr"] * 100,
            weight_decay=arguments.get("weight_decay", 0.0),
            momentum=arguments.get("momentum", 0.9),
        )
        self.scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                scheduler_arguments["milestones"],
                scheduler_arguments["gamma"],
            )
            if scheduler_arguments is not None
            else None
        )

    def _make_metrics(self):
        return [
            F1Score(self.config, "classification", num_classes=self.num_classes, seeking=False),
            Accuracy(self.config, "classification", num_classes=self.num_classes),
            Precision(self.config, "classification", num_classes=self.num_classes),
            Recall(self.config, "classification", num_classes=self.num_classes),
        ]

    def train_epoch(self, loader, criterion):
        self.model.train()
        metrics = self._make_metrics()
        total_loss = 0.0
        total_metrics = [0.0] * len(metrics)
        total_samples = 0

        for inputs, targets, *_ in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.is_multilabel:
                targets = targets.float()
            else:
                targets = targets.squeeze(1).long() if targets.ndim == 2 else targets.long()

            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predictions = torch.sigmoid(outputs) if self.is_multilabel else torch.softmax(outputs, dim=-1)
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric.evaluate(predictions, targets).item() * inputs.size(0)
            total_samples += inputs.size(0)

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / len(loader)
        metrics_str = ", ".join(f"{m.name}: {total_metrics[i] / total_samples:.4f}" for i, m in enumerate(metrics))
        print(f'Training...   Loss: {avg_loss:.4f}, {metrics_str}')
        return {"loss": avg_loss, "accuracy": total_metrics[1] / total_samples}

    def eval(self, loader, model, criterion):
        model.eval()
        metrics = self._make_metrics()
        total_loss = 0.0
        total_metrics = [0.0] * len(metrics)
        total_samples = 0

        with torch.no_grad():
            for inputs, targets, *_ in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.is_multilabel:
                    targets = targets.float()
                else:
                    targets = targets.squeeze(1).long() if targets.ndim == 2 else targets.long()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                predictions = torch.sigmoid(outputs) if self.is_multilabel else torch.softmax(outputs, dim=-1)
                for i, metric in enumerate(metrics):
                    total_metrics[i] += metric.evaluate(predictions, targets).item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = total_loss / len(loader)
        metrics_str = ", ".join(f"{m.name}: {total_metrics[i] / total_samples:.4f}" for i, m in enumerate(metrics))
        print(f'Evaluation... Loss: {avg_loss:.4f}, {metrics_str}')
        return {"loss": avg_loss, "accuracy": total_metrics[1] / total_samples}

    def run_model(self, inputs, use_swag=False, **kwargs):
        if use_swag:
            return self.swag_model(inputs)
        return self.model(inputs)

    def run_predictions(self, test_loader):
        all_predictions = []
        all_targets = []
        self.swag_model.eval()
        with torch.no_grad():
            for inputs, targets, *_ in test_loader:
                preds = self.predict(inputs)
                all_predictions.append(preds.cpu())
                all_targets.append(targets)
        return {
            "predictions": torch.cat(all_predictions, dim=0),
            "targets": torch.cat(all_targets, dim=0),
        }

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        self.swag_model.eval()
        self.swag_model.to(self.device)
        with torch.no_grad():
            return self.run_model(inputs, use_swag=True)

    def _check_bn(self, module, flag):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            flag[0] = True

    def check_bn(self, model):
        flag = [False]
        model.apply(lambda module: self._check_bn(module, flag))
        return flag[0]
    #
    # def reset_bn(self, module):
    #     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    #         module.running_mean = torch.zeros_like(module.running_mean)
    #         module.running_var = torch.ones_like(module.running_var)
    def reset_bn(self, module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean.zero_()
            module.running_var.fill_(1.0)

    def _set_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]

    def _get_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum

    def bn_update(self, loader, model, **kwargs):
        """
            BatchNorm buffers update (if any).
            Performs 1 epochs to estimate buffers average using train dataset.

            :param loader: train dataset loader for buffers average estimation.
            :param model: model being update
            :return: None
        """
        if not self.check_bn(model):
            return

        model.train()
        momenta = {}
        model.apply(self.reset_bn)
        model.apply(lambda module: self._get_momenta(module, momenta))
        model.to(self.device)
        if hasattr(model, "base"):
            model.base.to(self.device)
        n = 0
        #num_batches = len(loader)

        with torch.no_grad():
            for input, _ in loader:
                input = input.to(self.device)
                input_var = torch.autograd.Variable(input)
                b = input_var.data.size(0)

                momentum = b / (n + b)
                for module in momenta.keys():
                    module.momentum = momentum

                model(input_var, **kwargs)
                n += b

        model.apply(lambda module: self._set_momenta(module, momenta))

    #def build_method(self, rebuild=False, **kwargs):
    #    self.train_loader = kwargs['train_loader']
    #    if not rebuild:
    #        if 'pretrained' in kwargs:
    #            pretrained_model = torch.load(kwargs['pretrained'])
    #        else:
    #            pretrained_model = torch.load(os.path.join(self.config.output.path, self.config.method.name, 'model', 'swag_model.pt'))
    #        self.swag_model.load_state_dict(pretrained_model)
    #        self.swag_model.to(self.device)
    #        return
    #    self.train_uncertainty_method(kwargs['train_loader'], kwargs['valid_loader'])
    #    output_save_dir = os.path.join(self.config.output.path, self.config.method.name, 'model')
    #    os.makedirs(output_save_dir, exist_ok=True)
    #    if 'model_name' in kwargs:
    #        filename = os.path.join(output_save_dir,kwargs['model_name'])
    #    else:
    #        filename = os.path.join(output_save_dir, 'swag_model.pt')
    #    torch.save(self.swag_model.state_dict(), filename)


    def train_uncertainty_method(self, train_loader, val_loader):
        self.train_loader = train_loader
        model_name = self.config.model.name
        dataset_name = self.config.dataset.name
        path = (
            Path(os.getcwd())
            / "models"
            / dataset_name
            / self.model_dir
            / "checkpoints"
            / f"swag_model_{model_name}.pt"
        )
        if path.exists():
            self.swag_model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            print(f"Loaded pretrained swag model from {path}")
            return

        if self.is_multilabel:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        sgd_ens_preds = None
        n_ensembled = 0.0
        swa_start = self.config.method.swa_start

        for epoch in range(self.config.method.epochs):
            train_res = self.train_epoch(train_loader, criterion)
            test_res = self.eval(val_loader, self.model, criterion)

            if (epoch + 1) > swa_start:
                self.swag_model.collect_model(self.model)
                self.swag_model.sample(scale=0.0, cov=True)
                self.bn_update(train_loader, self.swag_model)

                sgd_res = self.run_predictions(val_loader)
                sgd_preds = sgd_res["predictions"]
                if sgd_ens_preds is None:
                    sgd_ens_preds = sgd_preds.clone()
                else:
                    sgd_ens_preds = (
                        sgd_ens_preds * n_ensembled / (n_ensembled + 1)
                        + sgd_preds / (n_ensembled + 1)
                    )
                n_ensembled += 1

                swag_res = self.eval(val_loader, self.swag_model, criterion)
                print(
                    f"epoch: {epoch+1}/{self.config.method.epochs}, "
                    f"swag_loss: {swag_res['loss']:.4f}, swag_acc: {swag_res['accuracy']:.4f}"
                )

            print(
                f"epoch: {epoch+1}/{self.config.method.epochs}, "
                f"train_loss: {train_res['loss']:.4f}, train_acc: {train_res['accuracy']:.4f}, "
                f"test_loss: {test_res['loss']:.4f}, test_acc: {test_res['accuracy']:.4f}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.swag_model.state_dict(), path)
        print(f"Saved swag model to {path}")



    def inference(self, loader):
        n_samples = self.config.method.sample_size
        n_data = len(loader.dataset)

        predictions = torch.zeros((n_samples, n_data, self.num_classes))
        if self.is_multilabel:
            labels = torch.zeros(n_data, self.num_classes)
        else:
            labels = torch.zeros(n_data)

        sub_size = int(len(self.train_loader.dataset) * self.sub_population)
        indices = torch.randperm(len(self.train_loader.dataset))[:sub_size]
        sub_loader = DataLoader(
            Subset(self.train_loader.dataset, indices),
            batch_size=self.swag_batch_size, shuffle=True,
        )

        for i in range(n_samples):
            self.swag_model.sample(scale=self.scale, cov=True)
            self.bn_update(sub_loader, self.swag_model)
            self.swag_model.eval()
            torch.manual_seed(i)
            k = 0
            with torch.no_grad():
                for batch in tqdm(loader):
                    inputs = batch[0].to(self.device)
                    targets = batch[1]
                    output = self.swag_model(inputs)
                    probs = torch.sigmoid(output) if self.is_multilabel else F.softmax(output, dim=1)
                    predictions[i, k:k + inputs.size(0)] = probs.cpu()
                    if i == 0:
                        labels[k:k + inputs.size(0)] = targets
                    k += inputs.size(0)

        return predictions, labels

    # ------------------------------------------------------------------ #
    # New lifecycle hooks called by the experiment runner                  #
    # ------------------------------------------------------------------ #

    def train_model(self, train_loader, val_loader, **kwargs):
        """SWAG trains the base model internally — no separate pre-train step."""
        self.train_loader = train_loader
        self.train_uncertainty_method(train_loader, val_loader)

    def train_base_model(self, train_loader, val_loader, loss_weight=None):
        # SWAG trains the base model inside train_uncertainty_method; skip the base class loop.
        return

    def save_model(self, path: str) -> None:
        """Save base model checkpoint and SWAG posterior state alongside it."""
        self.save_pretrained_model(path)
        swag_path = path.replace('.pt', '_swag.pt')
        torch.save(self.swag_model.state_dict(), swag_path)

    def load_model(self, path: str, train_loader=None, val_loader=None) -> None:
        """Load base model and restore SWAG posterior.

        If the SWAG state file is missing and a train_loader is provided,
        the posterior is refit automatically (same behaviour as the old
        ``build_method(rebuild=True)`` path).
        """
        self.load_pretrained_model(path)
        swag_path = path.replace('.pt', '_swag.pt')
        if os.path.exists(swag_path):
            self.swag_model.load_state_dict(
                torch.load(swag_path, weights_only=True, map_location=self.device)
            )
            self.swag_model.to(self.device)
        elif train_loader is not None:
            print(f"SWAG state not found at {swag_path}. Refitting from training data...")
            self.train_uncertainty_method(train_loader, val_loader)
        else:
            raise FileNotFoundError(
                f"SWAG state not found at {swag_path} and no train_loader provided to refit."
            )
        if train_loader is not None:
            self.train_loader = train_loader