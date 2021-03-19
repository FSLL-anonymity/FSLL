# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmeta.utils.gradient_based import gradient_update_parameters
from helpers import get_torch_acc

metrics = ['F1 score', 'AUC']


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(phase, query_x_i, base_label_ID, novel_label_ID, y):
        # Sets model to TRAIN mode
        model.train()
        optimizer.zero_grad()
        # Makes predictions
        yhat = model(phase, query_x_i, base_label_ID, novel_label_ID)
        # Computes loss
        loss = loss_fn(yhat, y)
        # Computes gradients
        loss.backward(retain_graph=True)
        # Updates parameters and zeroes gradients
        optimizer.step()
        # Returns the loss
        return yhat

    # Returns the function that will be called inside the train loop
    return train_step


def make_train_step_thre(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(query_x_i, y, y_pre):
        # Sets model to TRAIN mode
        model.train()
        optimizer.zero_grad()
        # Makes predictions
        y_thre = model(query_x_i)
        # Computes loss
        loss = loss_fn(torch.sigmoid(y_pre - y_thre), y.float())
        # Computes gradients
        loss.backward(retain_graph=True)
        # Updates parameters and zeroes gradients
        optimizer.step()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t_predict, y_t):
        ey_t = y_t_predict - y_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def batch_train(params, ep, loader, criterion, optimizer, optimizer_threshold, classifier_model, threshold_model, fe_model=False, grad=True):
    epoch_losses = torch.tensor(0., device=params.device)
    task_accuracy = []
    threshold_f = LogCoshLoss()
    train_step = make_train_step(classifier_model, criterion, optimizer)
    train_step_thre = make_train_step_thre(threshold_model, threshold_f, optimizer_threshold)

    for batch in (loader(ep)):
        support_x = batch[0].to(params.device)
        support_y = batch[1].to(params.device)
        query_x = batch[2].to(params.device)
        query_y = batch[3].to(params.device)
        base_label_IDlist = batch[4].to(params.device)
        novel_label_IDlist = batch[5].to(params.device)

        batch_size, K, x_dim = support_x.size()
        K_q = query_x.size(1)

        if fe_model:
            support_x = support_x.view(batch_size * K, x_dim)
            query_x = query_x.view(batch_size * K_q, x_dim)
            support_x = fe_model(support_x).view(batch_size, K, -1)
            query_x = fe_model(query_x).view(batch_size, K_q, -1)

        outer_loss = torch.tensor(0., device=params.device)
        outer_thre_loss = torch.tensor(0., device=params.device)

        for task_id, (support_x_i, support_y_i, query_x_i, query_y_i, base_label_ID, novel_label_ID) \
                in enumerate(zip(support_x, support_y, query_x, query_y, base_label_IDlist, novel_label_IDlist)):
            # task_id = batch_size_id
            if fe_model:
                if params.weightCombine:
                    z_logit, i_logit, a_logit = classifier_model(loader.phase, support_x_i, base_label_ID, novel_label_ID)
                    support_logit = params.weight_z * z_logit + params.weight_i * i_logit + params.weight_a * a_logit
                else:
                    support_logit = classifier_model(loader.phase, support_x_i, base_label_ID, novel_label_ID)
            else:
                support_logit = classifier_model(support_x_i)

            inner_loss = criterion(support_logit, support_y_i)

            classifier_model.zero_grad()
            theta = gradient_update_parameters(classifier_model, inner_loss,
                                               step_size=params.step_size,
                                               first_order=params.first_order)

            support_logit = classifier_model(loader.phase, support_x_i, base_label_ID, novel_label_ID, theta=theta)
            threshold = threshold_model(support_x_i)
            thresholds = threshold + params.threshold
            threshold_loss = threshold_f(torch.sigmoid(support_logit-thresholds), support_y_i.float())
            threshold_model.zero_grad()
            threshold_theta = gradient_update_parameters(threshold_model, threshold_loss, step_size=params.step_size,
                                                         first_order=params.first_order)

            optimizer.zero_grad()
            optimizer_threshold.zero_grad()
            query_logit = classifier_model(loader.phase, query_x_i, base_label_ID, novel_label_ID, theta=theta)
            query_threshold = threshold_model(query_x_i, theta=threshold_theta)
            query_thresholds = query_threshold + params.threshold

            outer_loss += criterion(query_logit, query_y_i)
            outer_thre_loss += threshold_f(torch.sigmoid(query_logit-query_thresholds), query_y_i.float())

            if not grad:
                torch.save(classifier_model.state_dict(), '../model/SIN_model')
                for i in range(3):
                    train_step(loader.phase, support_x_i, base_label_ID, novel_label_ID, support_y_i)
                classifier_model.eval()
                query_logit = classifier_model(loader.phase, query_x_i, base_label_ID, novel_label_ID)
                classifier_model.load_state_dict(torch.load('../model/SIN_model'), strict=True)
                with torch.no_grad():
                    task_accuracy.append(get_torch_acc((query_logit - query_threshold), query_y_i))

        outer_loss.div_(batch_size)
        outer_thre_loss.div_(batch_size)
        epoch_losses += outer_loss

        if grad:
            outer_loss.backward(retain_graph=True)
            optimizer.step()
            outer_thre_loss.backward()
            optimizer_threshold.step()

    epoch_losses = epoch_losses.div_(loader.__len__()).item()
    epoch_accuracy = np.mean(task_accuracy, 0)

    if not grad:
        print('train loss={0:.4f}'.format(epoch_losses))
        for i in range(len(epoch_accuracy)):
            print(metrics[i] + ":", epoch_accuracy[i])

    return epoch_losses, epoch_accuracy


def meta_train(params, fe_model, classifier_model, threshold_model, loader_train, loader_test):

    if torch.cuda.is_available():
        classifier_model = classifier_model.cuda()
        if fe_model:
            fe_model = fe_model.cuda()
            threshold_model = threshold_model.cuda()

    criterion = nn.MultiLabelSoftMarginLoss()
    # optimizer
    optimizer_threshold = optim.Adam(threshold_model.parameters(), weight_decay=params.classifier_L2)
    optimizer = optim.Adam(classifier_model.parameters(), weight_decay=params.classifier_L2) 

    for ep in range(1, params.classifier_epoch+1):
        params.currentEpoch = ep
        print("----epoch: %2d---- " % int(ep))
        if fe_model:
            fe_model.train()
            threshold_model.train()
        classifier_model.train()
        batch_train(params, ep, loader_train, criterion, optimizer, optimizer_threshold, classifier_model,
                                                   threshold_model, fe_model=fe_model, grad=True)


    print("--------begin validation--------")
    fe_model.eval()
    threshold_model.eval()
    classifier_model.eval()
    batch_train(params, ep, loader_test, criterion, optimizer, optimizer_threshold, classifier_model,
                threshold_model, fe_model=fe_model, grad=False)

