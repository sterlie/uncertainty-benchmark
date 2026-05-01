from typing import Optional

import torch
from omegaconf import OmegaConf
from torchmetrics.functional import f1_score, precision, recall, auroc, average_precision, accuracy
from sklearn.metrics import precision_recall_curve, f1_score as sklearn_f1_score


class Metric:
    def __init__(self, config, task, **kwargs):
        self.config = config
        self.task = task
        self.threshold_name = f"{task}_threshold"
        self.multi_label = (config.dataset.name in ["nih", "chexpert", "vin_chest"]) and (config.dataset.chosen_disease is None)
        self.reduction = not config.dataset.get('uncertainty_per_class', False)

        assert task in ["classification", "ood", "misclassify"]
        if task == "classification":
            self.threshold = 0.5
        elif task == "ood":
            self.threshold = config.method.get("ood_threshold", 0.0)
        elif task == "misclassify":
            self.threshold = config.method.get("misclassify_threshold", config.method.get("ood_threshold", 0.0))
        return

    def check_outputs(self, outputs: torch.Tensor, threshold: Optional[float] = None):
        if outputs.dim() > 1 and outputs.shape[-1] > 1 and (not self.multi_label):
            outputs = outputs.argmax(dim=1)
        elif outputs.dtype.is_floating_point:
            if threshold is not None:
                outputs = (outputs > threshold).long()
        else:
            outputs = outputs.long()

        return outputs

    def evaluate(self, outputs: torch.Tensor, targets: torch.Tensor):
        self.threshold = self.config.method.get(self.threshold_name, self.threshold)
        return None


class Accuracy(Metric):
    def __init__(self, config, task, num_classes: int = 2):
        super(Accuracy, self).__init__(config, task)
        self.name = "Accuracy"
        self.num_classes = num_classes

    def evaluate(self, outputs: torch.Tensor, targets: torch.Tensor):
        super(Accuracy, self).evaluate(outputs, targets)
        if targets.shape[0] == 0:
            return torch.tensor(0.0)
        predictions = self.check_outputs(outputs, threshold=self.threshold)
        if self.num_classes <= 2:
            accuracy_result = accuracy(predictions, targets, task="binary", num_classes=1, average="macro")
        elif self.multi_label:
            accuracy_result = accuracy(predictions, targets, task="multilabel", num_labels=self.num_classes, average="macro")
        else:
            accuracy_result = accuracy(predictions, targets, task="multiclass", num_classes=self.num_classes, average="macro")
        return accuracy_result


class F1Score(Metric):
    def __init__(self, config, task, num_classes: int = 2, seeking=True):
        super(F1Score, self).__init__(config, task)
        self.name = "F1Score"
        self.num_classes = num_classes
        self.seeking = seeking
        self.eps = 1e-8

    def evaluate(self, outputs: torch.Tensor, targets: torch.Tensor):
        super(F1Score, self).evaluate(outputs, targets)
        if targets.shape[0] == 0:
            return torch.tensor(0.0)
        predictions = self.check_outputs(outputs, threshold=self.threshold)

        if self.num_classes <= 2:
            if not self.seeking:
                f1_score_result = f1_score(predictions, targets.long(), task="binary", num_classes=self.num_classes, average="macro")
            else:
                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
                outputs = outputs[..., -1] if outputs.ndim >= 2 and outputs.shape[-1] == 2 else outputs.squeeze()
                # print(predictions.shape, targets.shape, torch.sum(targets), (predictions == targets).sum())
                precisions, recalls, thresholds = precision_recall_curve(targets, outputs)
                f1_scores = 2 * precisions * recalls / (precisions + recalls + self.eps)
                best_index = f1_scores.argmax()
                best_threshold = thresholds[best_index].item()
                f1_score_result = f1_scores[best_index]
                if self.threshold_name in self.config.method:
                    self.config.method = OmegaConf.merge(self.config.method, {self.threshold_name: best_threshold})
                else:
                    OmegaConf.update(self.config, f"method.{self.threshold_name}", best_threshold, force_add=True)
                print(f"Setting {self.threshold_name} to {best_threshold}")
        else:
            if self.multi_label:
                f1_score_result = f1_score(predictions, targets.long(), task="multilabel", num_labels=self.num_classes, average="macro")
            else:
                f1_score_result = f1_score(predictions, targets.long(), task="multiclass", num_classes=self.num_classes, average="macro")
        return f1_score_result


class Precision(Metric):
    def __init__(self, config, task, num_classes: int = 2):
        super(Precision, self).__init__(config, task)
        self.name = "Precision"
        self.num_classes = num_classes

    def evaluate(self, outputs: torch.Tensor, targets: torch.Tensor):
        super(Precision, self).evaluate(outputs, targets)
        if targets.shape[0] == 0:
            return torch.tensor(0.0)
        predictions = self.check_outputs(outputs, threshold=self.threshold)
        if self.num_classes <= 2:
            precision_result = precision(predictions, targets.long(), task="binary", num_classes=1, average="macro")
        else:
            if self.multi_label:
                precision_result = precision(predictions, targets.long(), task="multilabel", num_labels=self.num_classes, average="macro")
            else:
                precision_result = precision(predictions, targets.long(), task="multiclass", num_classes=self.num_classes, average="macro")
        return precision_result


class Recall(Metric):
    def __init__(self, config, task, num_classes: int = 2):
        super(Recall, self).__init__(config, task)
        self.name = "Recall"
        self.num_classes = num_classes

    def evaluate(self, outputs: torch.Tensor, targets: torch.Tensor):
        super(Recall, self).evaluate(outputs, targets)
        if targets.shape[0] == 0:
            return torch.tensor(0.0)
        predictions = self.check_outputs(outputs, threshold=self.threshold)
        if self.num_classes <= 2:
            recall_result = recall(predictions, targets.long(), task="binary", num_classes=1, average="macro")
        else:
            if self.multi_label:
                recall_result = recall(predictions, targets.long(), task="multilabel", num_labels=self.num_classes, average="macro")
            else:
                recall_result = recall(predictions, targets.long(), task="multiclass", num_classes=self.num_classes, average="macro")
        return recall_result


class AUROC(Metric):
    def __init__(self, config, task, num_classes: int = 2):
        super(AUROC, self).__init__(config, task)
        self.name = "AUROC"
        self.num_classes = num_classes

    def evaluate(self, outputs: torch.Tensor, targets: torch.Tensor):
        super(AUROC, self).evaluate(outputs, targets)
        if targets.shape[0] == 0:
            return torch.tensor(0.0)
        predictions = self.check_outputs(outputs) if self.task != "classification" else outputs
        if self.num_classes <= 2:
            if (not self.reduction) and (targets.ndim > 1):
                auroc_result = auroc(predictions, targets.long(), task="binary", num_classes=1, average="macro")
            else:
                auroc_result = auroc(predictions[:, 1] if predictions.ndim == 2 else predictions, targets.long(), task="binary", num_classes=1, average="macro")
        else:
            # targets = torch.nn.functional.one_hot(targets.long(), num_classes=self.num_classes) if targets.ndim == 1 and predictions.ndim > 1 else targets.long()
            if self.multi_label:
                auroc_result = auroc(predictions, targets.long(), task="multilabel", num_labels=self.num_classes, average="macro")
            else:
                auroc_result = auroc(predictions, targets.long(), task="multiclass", num_classes=self.num_classes, average="macro")
        return auroc_result


class AveragePrecision(Metric):
    def __init__(self, config, task, num_classes: int = 2):
        super(AveragePrecision, self).__init__(config, task)
        self.name = "AveragePrecision"
        self.num_classes = num_classes

    def evaluate(self, outputs: torch.Tensor, targets: torch.Tensor):
        super(AveragePrecision, self).evaluate(outputs, targets)
        if targets.shape[0] == 0:
            return torch.tensor(0.0)
        predictions = self.check_outputs(outputs) if self.task != "classification" else outputs

        if self.num_classes <= 2:
            if (not self.reduction) and (targets.ndim > 1):
                average_precision_result = average_precision(predictions, targets.long(), task="binary", num_classes=1, average="macro")
                # pd = torch.zeros([predictions.shape[0], predictions.shape[-1], predictions.shape[-1]]).to(predictions.device)
                # tg = torch.zeros([predictions.shape[0], predictions.shape[-1], predictions.shape[-1]]).to(predictions.device)
                # for i in range(predictions.shape[-1]):
                #     pd[:, i, i] = predictions[:, i]
                #     tg[:, i, i] = targets[:, i]
                # print(average_precision(pd, tg.long(), task="multilabel", num_labels=predictions.shape[-1], average="none"))
                # print(pd.min(), pd.max())
                # pd = (((pd - pd.min()) / (pd.max() - pd.min())) > 0.5).float()
                # tg = (tg > 0.5).long()
                # print(precision(pd, tg, task="multilabel", num_labels=predictions.shape[-1], average="none"))
                # print(recall(pd, tg, task="multilabel", num_labels=predictions.shape[-1], average="none"))
                # print(f1_score(pd, tg, task="multilabel", num_labels=predictions.shape[-1], average="none"))
            else:
                average_precision_result = average_precision(predictions[:, 1] if predictions.ndim == 2 else predictions, targets.long(), task="binary", num_classes=1, average="macro")
        else:
            if self.multi_label:
                average_precision_result = average_precision(predictions, targets.long(), task="multilabel", num_labels=self.num_classes, average="macro")
            else:
                average_precision_result = average_precision(predictions, targets.long(), task="multiclass", num_classes=self.num_classes, average="macro")
        return average_precision_result
