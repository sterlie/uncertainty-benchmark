import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from swa_gaussian.swag.posteriors import SWAG
import os

from src.methods import register_method
from src.methods.method import Method


@register_method('swag')
class Swag(Method):
    def __init__(self, config):
        super(Swag, self).__init__(config)
        self.swag_model = SWAG(
            self.model.__class__,
            no_cov_mat=False,
            max_num_models=20,
            config=config
        )
        self.swag_model.to(self.device)

        self.eps = 1e-12

    def train_epoch(self, loader, criterion):
        loss_sum = 0.0
        correct = 0.0
        verb_stage = 0

        num_objects_current = 0
        num_batches = len(loader)

        self.model.train()

        for i, (input, target) in enumerate(loader):
            input, target = input.to(self.device), target.to(self.device)

            output = self.model(input)
            loss = criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.data.item() * input.size(0)

            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

            num_objects_current += input.size(0)

        return {
            "loss": loss_sum / num_objects_current,
            "accuracy": correct / num_objects_current * 100.0,
        }

    def eval(self, loader, model, criterion):
        loss_sum = 0.0
        correct = 0.0
        num_objects_total = len(loader.dataset)

        model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(loader):
                input, target = input.to(self.device), target.to(self.device)

                output = model(input)
                loss = criterion(output, target)

                loss_sum += loss.item() * input.size(0)

                pred = output.data.argmax(1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()

        return {
            "loss": loss_sum / num_objects_total,
            "accuracy": correct / num_objects_total * 100.0,
        }

    def predict(self, loader):
        predictions = list()
        targets = list()

        self.model.eval()

        offset = 0
        with torch.no_grad():
            for input, target in loader:
                # input = input.cuda(non_blocking=True)
                input =input.to(self.device)
                output = self.model(input)

                batch_size = input.size(0)
                predictions.append(F.softmax(output, dim=1).cpu().numpy())
                targets.append(target.numpy())
                offset += batch_size

        return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}

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
        num_batches = len(loader)

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

    def build_method(self, rebuild=False, **kwargs):
        self.train_loader = kwargs['train_loader']
        if not rebuild:
            if 'pretrained' in kwargs:
                pretrained_model = torch.load(kwargs['pretrained'])
            else:
                pretrained_model = torch.load(os.path.join(self.config.output.path, self.config.method.name, 'model', 'swag_model.pt'))
            self.swag_model.load_state_dict(pretrained_model)
            self.swag_model.to(self.device)
            return
        self.train_uncertainty_method(kwargs['train_loader'], kwargs['valid_loader'])
        output_save_dir = os.path.join(self.config.output.path, self.config.method.name, 'model')
        os.makedirs(output_save_dir, exist_ok=True)
        if 'model_name' in kwargs:
            filename = os.path.join(output_save_dir,kwargs['model_name'])
        else:
            filename = os.path.join(output_save_dir, 'swag_model.pt')
        torch.save(self.swag_model.state_dict(), filename)


    def train_uncertainty_method(self, train_loader, valid_loader):
        if self.config.weighted:
            if 'weights' in self.config.dataset:
                weights = torch.tensor(self.config.dataset.weights).float().to(self.device)
            else:
                labels = torch.tensor(train_loader.dataset.labels)
                class_counts = torch.bincount(labels, minlength=self.num_classes)
                class_counts[class_counts == 0] = 1

                N = labels.size(0)
                weights = N / (self.num_classes * class_counts.float())
                weights = weights.to(self.device)
            print("weights", weights)
        else:
            weights = None
        criterion = nn.CrossEntropyLoss(weights)
        sgd_ens_preds = None
        sgd_targets = None
        n_ensembled = 0.0
        for epoch in range(self.config.method.epochs):

            if (epoch + 1) > self.config.method.swa_start:
                for group in self.optimizer.param_groups:
                    group['lr'] = self.config.optimizer.lr * self.config.method.lr_increase_factor

            train_res = self.train_epoch(train_loader, criterion)

            test_res = self.eval(valid_loader, self.model, criterion)
            if (epoch + 1) > self.config.method.swa_start:
                # sgd_preds, sgd_targets = utils.predictions(loaders["test"], model)
                sgd_res = self.predict(valid_loader)
                sgd_preds = sgd_res["predictions"]
                sgd_targets = sgd_res["targets"]
                print("updating sgd_ens")
                if sgd_ens_preds is None:
                    sgd_ens_preds = sgd_preds.copy()
                else:
                    # TODO: rewrite in a numerically stable way
                    sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                            n_ensembled + 1
                    ) + sgd_preds / (n_ensembled + 1)
                n_ensembled += 1
                self.swag_model.collect_model(self.model)

                self.swag_model.sample(0.5, True)
                self.bn_update(train_loader, self.swag_model)
                swag_res = self.eval(valid_loader, self.swag_model, criterion)
                print(f"epoch: {epoch+1}/{self.config.method.epochs}, swag_loss: {swag_res['loss']}, swag_acc: {swag_res['accuracy']}")

            print(f"epoch: {epoch+1}/{self.config.method.epochs}, train_loss: {train_res['loss']}, train_acc: {train_res['accuracy']}, test_loss: {test_res['loss']}, test_acc: {test_res['accuracy']}")

    def inference(self, loader):

        eps = 1e-12
        n_samples = self.config.method.sample_size
        n_data = len(loader.dataset)

        # store per-sample predictive probabilities
        predictions = torch.zeros((n_samples, n_data, self.num_classes))
        labels = torch.zeros(n_data)

        for i in range(n_samples):
            self.swag_model.train()
            self.swag_model.sample(scale=0.5, cov=True)
            self.bn_update(self.train_loader, self.swag_model)
            self.swag_model.eval()

            k = 0
            for input, target in tqdm(loader):
                input = input.to(self.device)
                self.swag_model.sample(scale=0.5, cov=True)
                torch.manual_seed(i)
                with torch.no_grad():
                    output = self.swag_model(input)
                    probs = F.softmax(output, dim=1)
                    predictions[i, k:k + input.size(0), :] = probs
                    labels[k:k + input.size(0)] = target.to(self.device)
                k += input.size(0)

        #
        return predictions, labels
