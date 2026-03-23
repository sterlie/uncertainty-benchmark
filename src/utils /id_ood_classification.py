from sklearn.metrics import confusion_matrix
from utils.metrics import *

def evaluate(true_labels, predicted_labels):
    ## confusion matrix
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    tn = conf_mat[0, 0]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]
    tp = conf_mat[1, 1]
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)



