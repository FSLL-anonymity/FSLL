# -*- coding:utf-8 -*-
import numpy as np
import random
import torch
import torchnet as tnt


class FewShotDataLoader:
    def __init__(self, dataset,
                 N=5,  # number of novel categories (N-way)
                 num_base_label=-1,  # number of base categories
                 K=1,  # number of training examples per novel category (K-shot)
                 K_q=13,  # number of test examples for per novel category
                 batch_size=10,  # number of training episodes per batch
                 num_workers=4,
                 epoch_size=2000):

        self.dataset = dataset
        self.num_train = self.dataset.num_train
        self.phase = self.dataset.phase
        max_possible_N = (self.dataset.num_cats_base if self.phase == 'train'
                                else self.dataset.num_cats_novel)
        assert (N >= 0 and N <= max_possible_N)
        self.N = N

        max_possible_num_base_label = self.dataset.num_cats_base
        num_base_label = num_base_label if num_base_label >= 0 else max_possible_num_base_label

        if self.phase == 'train' and num_base_label > 0:
            num_base_label -= self.N    # 'train' 时进行减N操作
            max_possible_num_base_label -= self.N

        assert (num_base_label >= 0 and num_base_label <= max_possible_num_base_label)
        self.num_base_label = num_base_label
        self.K = K
        self.K_q = K_q
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase == 'test') or (self.phase == 'val')
        self.is_test_mode = self.phase == 'test'

    def sampleInstanceIdsFrom(self, label_id):
        sample_size = self.K + self.K_q
        instance_id_list = self.dataset.instancesID2label[label_id]
        if self.phase == 'train':
            if len(instance_id_list) <= sample_size:
                if len(instance_id_list) <= self.K:
                    raise ('samples less than K, no query set')
                else:
                    S_Q = instance_id_list
            else:
                S_Q = random.sample(instance_id_list, sample_size)
            # only training dataset, without testing dataset
        else:
            instance_id_list_train = [id for id in instance_id_list if id < self.num_train]
            instance_id_list_test = [id for id in instance_id_list if id >= self.num_train]
            S = random.sample(instance_id_list_train, self.K)

            if len(instance_id_list_test) <= self.K_q:
                Q = instance_id_list_test
            else:
                Q = random.sample(instance_id_list_test, self.K_q)
            S_Q = S + Q

        return S_Q

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples 'sample_size' number of unique categories picked from the
        'cat_set' set of categories.'cat_set' can be either 'base' or 'novel'.
        """
        if cat_set == 'base':
            labelIds = self.dataset.labelIds_base
        elif cat_set == 'novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognize category set {}'.format(cat_set))

        assert (len(labelIds) >= sample_size)

        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, num_base_label, N):
        """
        Samples 'num_base_label' number of base categories and 'N'  number of novel categories.
        """
        if self.is_eval_mode:
            assert (N <= self.dataset.num_cats_novel)

            base_label_IDlist = sorted(self.sampleCategories('base', num_base_label))
            novel_label_IDlist = sorted(self.sampleCategories('novel', N))
        else:
            cats_ids = self.sampleCategories('base', num_base_label + N)
            assert (len(cats_ids) == (num_base_label + N))

            random.shuffle(cats_ids)
            novel_label_IDlist = sorted(cats_ids[:N])
            base_label_IDlist = sorted(cats_ids[N:])
        return base_label_IDlist, novel_label_IDlist

    def sample_train_and_test_examples_for_novel_categories(
            self, novel_label_IDlist, K):
        assert (len(novel_label_IDlist) > 0)

        support_instance_list = []
        query_instance_list = []

        for novel_i in range(len(novel_label_IDlist)):
            instance_list = self.sampleInstanceIdsFrom(novel_label_IDlist[novel_i])
            support_instance_list_per = instance_list[:K]
            query_instance_list_per = instance_list[K:]

            support_instance_list += support_instance_list_per
            query_instance_list += query_instance_list_per

        X_s = self.dataset.instances[support_instance_list]
        X_q = self.dataset.instances[query_instance_list]
        Y_s = torch.from_numpy(self.dataset.labels[:, novel_label_IDlist][support_instance_list]).long()
        Y_q = torch.from_numpy(self.dataset.labels[:, novel_label_IDlist][query_instance_list]).long()

        return X_s, Y_s, X_q, Y_q

    def sample_episode(self):
        """
        Sample a episode (task)
        """
        N = self.N
        num_base_label = self.num_base_label
        K_q = self.K_q
        K = self.K

        base_label_IDlist, novel_label_IDlist = self.sample_base_and_novel_categories(num_base_label, N)
        X_s, Y_s, X_q, Y_q = self.sample_train_and_test_examples_for_novel_categories(novel_label_IDlist, K)  # Q set number <= K_q
        return X_s, Y_s, X_q, Y_q, base_label_IDlist, novel_label_IDlist, num_base_label

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            X_s, Y_s, X_q, Y_q, base_label_IDlist, novel_label_IDlist, num_base_label = self.sample_episode()

            base_label_IDlist = torch.LongTensor(base_label_IDlist)
            novel_label_IDlist = torch.LongTensor(novel_label_IDlist)

            return X_s, Y_s, X_q, Y_q, base_label_IDlist, novel_label_IDlist, num_base_label

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_test_mode else self.num_workers),
            shuffle=(False if self.is_test_mode else True))

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)
