import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def get_NLL_score(probs, targets):
  # loss_fn = nn.NLLLoss()
  loss = np.log(np.array(probs)[np.arange(probs.shape[0]), targets])
  # loss = loss_fn(probs, targets)
  return -loss.mean(), loss.var(), loss, np.array(probs)[np.arange(probs.shape[0]), targets]


def get_acc_score(probs, targets):
  acc = (probs == targets).float().mean()
  return acc

def get_f1_score(probs, targets):
    f1 = f1_score(targets, probs)
    return f1

def get_auc_score(probs, targets):
    auroc = roc_auc_score(targets, probs)
    return auroc