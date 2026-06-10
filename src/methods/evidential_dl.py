import torch
from src.methods import Method, register_method
import torch.nn.functional as F

def kl_divergence(alpha, num_classes):
    ones = torch.ones([1, num_classes], dtype=torch.float32)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return A + kl_div

def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step
):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step
        )
    )
    return loss

def relu_evidence(y):
    return F.relu(y)

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

#@register_method("evidential_dl")
class EvidentialDeepLearning(Method):
    def __init__(self, config):
        super(EvidentialDeepLearning, self).__init__(config)
        self.num_classes = config.dataset.get('num_classes', 10)

    def build_base_model(self, retrain=False, **kwargs):
        pass

    def train_uncertainty_method(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
        print(self.model)
        optimizer = self.optimizer
        criterion = edl_digamma_loss

        # Training
        epochs = self.config.optimizer.get('epochs', 10)
        print("Any trainable params:",
              any(p.requires_grad for p in self.model.parameters()))
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):

                labels = one_hot_embedding(targets, self.num_classes)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(
                    outputs, labels.float(), epoch, self.num_classes, 10
                )
                match = torch.reshape(torch.eq(preds, targets).float(), (-1, 1))
                acc = torch.mean(match)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1
                u = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)

                total_evidence = torch.sum(evidence, 1, keepdim=True)
                mean_evidence = torch.mean(total_evidence)
                mean_evidence_succ = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * match
                ) / torch.sum(match + 1e-20)
                mean_evidence_fail = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * (1 - match)
                ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += torch.sum(preds == targets.data)

            avg_loss = total_loss / len(train_loader)

            # -------- VALIDATION --------
            self.model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    labels = one_hot_embedding(targets, self.num_classes)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(
                        outputs, labels.float(), epoch, self.num_classes, 10
                    )
                    match = torch.reshape(torch.eq(preds, targets).float(), (-1, 1))
                    acc = torch.mean(match)
                    evidence = relu_evidence(outputs)
                    alpha = evidence + 1
                    u = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)

                    total_evidence = torch.sum(evidence, 1, keepdim=True)
                    mean_evidence = torch.mean(total_evidence)
                    mean_evidence_succ = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * match
                    ) / torch.sum(match + 1e-20)
                    mean_evidence_fail = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * (1 - match)
                    ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == targets).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / len(val_loader.dataset)

            print(
                f'Epoch {epoch + 1}/{epochs} - Train: Loss: {avg_loss:.4f}, Accuracy: {total_correct / len(train_loader.dataset):.4f}\nValidation: Loss {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}')

    def inference(self, loader: torch.utils.data.DataLoader):
        """Make predictions in a conventional manner.
        Args:
            inputs: Input tensor

        Returns:
            The predictions of the model
        """
        predictions = []
        for inputs, _ in loader:
            output = self.model(inputs)
            evidence = relu_evidence(output)
            alpha = evidence + 1
            uncertainty = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)
            _, preds = torch.max(output, 1)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()

            predictions.append(preds)

        predictions = torch.cat(predictions)
        return predictions

    def measure_uncertainty(self, loader: torch.utils.data.DataLoader):
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_uncertainty = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                output = self.model(inputs)
                evidence = relu_evidence(output)
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                prob = alpha / S
                # vacuity uncertainty: K / S, one scalar per sample
                uncertainty = (self.num_classes / S).squeeze(1)  # [B]
                all_predictions.append(prob.cpu())
                all_targets.append(targets.cpu() if isinstance(targets, torch.Tensor) else torch.tensor(targets))
                all_uncertainty.append(uncertainty.cpu())

        predictions = torch.cat(all_predictions, dim=0)   # [N, C]
        ground_truth = torch.cat(all_targets, dim=0)      # [N] or [N, C]
        total_uncertainty = torch.cat(all_uncertainty, dim=0)  # [N]

        return {
            "predictions": predictions,
            "predicted_labels": predictions.argmax(dim=-1),
            "ground_truth": ground_truth,
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": total_uncertainty,
            "epistemic_uncertainty": torch.zeros_like(total_uncertainty),
            "out_of_distribution": total_uncertainty,
        }

