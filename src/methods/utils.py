import torch
import torch.nn.functional as F


def bernoulli_entropy(p, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return entropy


def bernoulli_entropy_from_logits(logits, eps=1e-6):
    # softplus(x) = log(1 + exp(x))
    entropy = (
        F.softplus(logits) +
        F.softplus(-logits) -
        logits * torch.sigmoid(logits)
    )
    return entropy


def multi_label_uncertainty(predictions, mean_prediction, reduction=True, sigmoid=True, eps=1e-6):
    # Predictive entropy
    p_t = torch.sigmoid(predictions) if sigmoid else predictions
    mean_p = p_t.mean(dim=0)
    total_uncertainty = bernoulli_entropy(mean_p, eps)

    # Aleatoric entropy
    aleatoric_uncertainty = bernoulli_entropy(p_t, eps).mean(dim=0)

    # Epistemic uncertainty (mutual information)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    if reduction:
        total_uncertainty = total_uncertainty.mean(dim=1)
        aleatoric_uncertainty = aleatoric_uncertainty.mean(dim=1)
        epistemic_uncertainty = epistemic_uncertainty.mean(dim=1)
    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty


def multi_class_uncertainty(predictions, mean_prediction, eps=1e-8):
    # Total uncertainty (entropy of mean prediction)
    total_uncertainty = -torch.sum(mean_prediction * torch.log(mean_prediction + eps), dim=1)

    # Aleatoric uncertainty (expected entropy)
    entropies = -torch.sum(predictions * torch.log(predictions + eps), dim=2)
    aleatoric_uncertainty = entropies.mean(dim=0)

    # Epistemic uncertainty (mutual information)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
    mean_pred_batch = mean_prediction.unsqueeze(0).repeat(predictions.shape[0], 1, 1)
    epistemic_uncertainty_kl = 0.5 * (
        F.kl_div(mean_pred_batch.log(), predictions, reduction="none").sum(dim=2).mean(dim=0) +
        F.kl_div(predictions.log(), mean_pred_batch, reduction="none").sum(dim=2).mean(dim=0))
    # print(epistemic_uncertainty.mean(), epistemic_uncertainty_kl.mean(), epistemic_uncertainty.mean() / epistemic_uncertainty_kl.mean())
    epistemic_uncertainty = epistemic_uncertainty_kl

    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty


def multi_label_uncertainty_evidential_dl(alpha, reduction=True, use_plus_one=False, eps=1e-8):
    # alpha is (N, C, 2)
    sum_alpha = torch.sum(alpha, dim=2)
    # probabilities is (N, C)
    probabilities = alpha[:, :, 1] / (sum_alpha + eps)
    if use_plus_one:
        total_uncertainty = (
            torch.digamma(sum_alpha + eps) -
            (alpha[:, :, 0] / (sum_alpha + eps)) * torch.digamma(alpha[:, :, 0] + eps) -
            (alpha[:, :, 1] / (sum_alpha + eps)) * torch.digamma(alpha[:, :, 1] + eps)
        )
        aleatoric_uncertainty = (
            torch.digamma(sum_alpha + 1) -
            (alpha[:, :, 0] / (sum_alpha + eps)) * torch.digamma(alpha[:, :, 0] + 1) -
            (alpha[:, :, 1] / (sum_alpha + eps)) * torch.digamma(alpha[:, :, 1] + 1)
        )
    else:
        total_uncertainty = (
            torch.digamma(sum_alpha + eps) -
            (alpha[:, :, 0] / (sum_alpha + eps)) * torch.digamma(alpha[:, :, 0] + eps) -
            (alpha[:, :, 1] / (sum_alpha + eps)) * torch.digamma(alpha[:, :, 1] + eps)
        )
        aleatoric_uncertainty = bernoulli_entropy(probabilities, eps)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    if reduction:
        total_uncertainty = total_uncertainty.mean(dim=1)
        aleatoric_uncertainty = aleatoric_uncertainty.mean(dim=1)
        epistemic_uncertainty = epistemic_uncertainty.mean(dim=1)
    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty


def multi_class_uncertainty_evidential_dl(alpha, use_plus_one=False, eps=1e-8):
    # alpha is (N, C)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    probabilities = alpha / (sum_alpha + eps)

    if use_plus_one:
        total_uncertainty = -torch.sum(
            probabilities * (torch.digamma(sum_alpha + eps) - torch.digamma(alpha + eps)),
            dim=1,
        )
        aleatoric_uncertainty = -torch.sum(
            probabilities * (torch.digamma(sum_alpha + 1) - torch.digamma(alpha + 1)),
            dim=1,
        )
    else:
        total_uncertainty = torch.sum(
            probabilities * (torch.digamma(sum_alpha + eps) - torch.digamma(alpha + eps)),
            dim=1,
        )
        aleatoric_uncertainty = -torch.sum(probabilities * torch.log(probabilities + eps), dim=1)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty
