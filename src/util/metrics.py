import torch
import numpy as np
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score


def acc_s(y_true, y_pred):
    correct = np.sum(np.equal(y_true, y_pred))
    total = len(y_true)
    return correct / total


def f1_s(y_true, y_pred, average="micro"):
    # print("Precision: ", precision_score(y_true, y_pred, average=average))
    # print("Recall: ", recall_score(y_true, y_pred, average=average))
    return f1_score(y_true, y_pred, average=average)


def eer_s(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1.0)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return eer, eer_threshold


def calculate_metrics(y_true, y_pred):
    acc = acc_s(y_true, y_pred)
    f1 = f1_s(y_true, y_pred)
    eer, _ = eer_s(y_true, y_pred)

    return acc, f1, eer
