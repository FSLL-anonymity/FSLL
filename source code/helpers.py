# -*- coding:utf-8 -*-
import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def split_train(data, train_num):
    np.random.seed(95)
    shuffled_indices = np.random.permutation(len(data))
    train_indices = shuffled_indices[:train_num]
    test_indices = shuffled_indices[train_num:]
    return data[train_indices], data[test_indices]


def split_label(data, past_label_num):   # full-label Y data, past_label_num
    np.random.seed(95)
    shuffled_indices = np.random.permutation(data.shape[1])
    past_label_indices = shuffled_indices[:past_label_num]
    new_label_indices = shuffled_indices[past_label_num:]
    return data[:, past_label_indices], data[:, new_label_indices]


def get_torch_acc(prediction, ground_truth):
    rounded = 4
    prediction = torch.sigmoid(prediction)
    prediction = prediction.to('cpu')
    ground_truth = ground_truth.to('cpu')
    threshold = 0.5
    F1_score = round(f1_score(ground_truth, prediction > threshold, average='micro', zero_division=0), rounded)
    AUC = round(roc_auc_score(ground_truth, prediction, average='micro'), rounded)

    task_acc = [F1_score, AUC]

    return task_acc


def precision_at_ks(true_Y, pred_Y):
    ks = [1, 3, 5]
    result = {}
    true_labels = [set(true_Y[i, :].nonzero()[0]) for i in range(true_Y.shape[0])]
    label_ranks = np.fliplr(np.argsort(pred_Y, axis=1))
    for k in ks:
        pred_labels = label_ranks[:, :k]
        precs = [len(t.intersection(set(p))) / k
                 for t, p in zip(true_labels, pred_labels)]
        result[k] = round(np.mean(precs), 4)
    return result


def one_error(ground_truth, prediction):
    true_labels = [set(ground_truth[i, :].nonzero()[0]) for i in range(ground_truth.shape[0])]
    label_ranks = np.fliplr(np.argsort(prediction, axis=1))
    pred_labels = label_ranks[:, :1]
    precs = [(1 - len(t.intersection(set(p))))
             for t, p in zip(true_labels, pred_labels)]
    result = np.mean(precs)
    return result
