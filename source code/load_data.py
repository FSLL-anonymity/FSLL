# -*- coding:utf-8 -*-
from helpers import split_train, split_label
import numpy as np

def load_dataset(dataset, phase='train'):
    if dataset == 'delicious':
        data_dir = '../datasets/delicious/pkl/'
        train_X = np.load(data_dir + 'delicious-train-features.pkl', allow_pickle=True)
        train_Y = np.load(data_dir + 'delicious-train-labels.pkl', allow_pickle=True)
        test_X = np.load(data_dir + 'delicious-test-features.pkl', allow_pickle=True)
        test_Y = np.load(data_dir + 'delicious-test-labels.pkl', allow_pickle=True)
        num_train = train_Y.shape[0]

        label_vec = np.loadtxt('../datasets/delicious/delicious_961word2vec.txt')
        del_index = np.loadtxt('../datasets/delicious/delicious_14del_word_index.txt').astype(int)
        label_text = np.loadtxt('../datasets/delicious/delicious_961label_text.txt', dtype=str)

        num_base_label = 589  # 60%
        num_val_label = 196  # 20%

        base_label_vec, novel_label_vec = split_train(label_vec, num_base_label+num_val_label)
        base_label_vec, val_label_vec = split_train(base_label_vec, num_base_label)
        base_label_text, novel_label_text = split_train(np.array(label_text), num_base_label+num_val_label)
        base_label_text, val_label_text = split_train(base_label_text, num_base_label)

        def label_train_val_test(Labels):
            Labels = Labels[:, 8:]
            Labels = np.delete(Labels, del_index, axis=1)

            meta_train_labels, meta_test_labels = split_label(Labels, num_base_label+num_val_label)
            meta_train_labels, val_labels = split_label(meta_train_labels, num_base_label)

            return meta_train_labels, val_labels, meta_test_labels

        # obtain training dataset of past labels
        meta_train_X = train_X
        meta_train_labels, _, _ = label_train_val_test(train_Y)

        # obtain new-label dataset from training dataset and testing dataset.
        # in testing, extract few new-label examples (support set) from training dataset, and evaluate the model on (support set sampled from) testing dataset
        meta_test_X = np.r_[train_X, test_X]
        val_X = np.r_[train_X, test_X]
        Labels = np.r_[train_Y, test_Y]
        _, val_labels, meta_test_labels = label_train_val_test(Labels)

        # remove non-labelled instance
        # meta_train_X = Data[meta_train_labels.sum(1).nonzero()[0]]
        # val_X = Data[val_labels.sum(1).nonzero()[0]]
        # meta_test_X = Data[meta_test_labels.sum(1).nonzero()[0]]

    if phase == 'train':
        return meta_train_X, meta_train_labels, base_label_vec, base_label_text, num_train
    if phase == 'val':
        return val_X, val_labels, val_label_vec, val_label_text, num_train
    if phase == 'test':
        return meta_test_X, meta_test_labels, novel_label_vec, novel_label_text, num_train

