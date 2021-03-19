# -*- coding:utf-8 -*-
"""
Few-shot Streaming Label Learning
"""
import torch
from params import get_params
from dataloader import FewShotDataLoader
from model import Word_Semantic, Threshold, DNN2Word_fe
from Delicious import Delicious
from train_process import meta_train

torch.manual_seed(123)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(123)

dataset_name = "delicious"
params = get_params(dataset_name)
params.phase = 'train'
dataset_train = Delicious(dataset_name, phase=params.phase)
dataset_val = Delicious(dataset_name, phase='val')
dataset_test = Delicious(dataset_name, phase='test')

# load pre-trained past-label feature-extractor
feature_extractor_path = '../model/semantic_feature_extractor'

fe_model = DNN2Word_fe(params)
fe_model.load_state_dict(torch.load(feature_extractor_path))

print("\n---- meta-learning stage ----\n")
params.model_name = 'SIN'
params.N = 5
params.K = 1

classifier_model = Word_Semantic(params, dataset_train.word2vec_base, dataset_val.word2vec_novel,
                                             dataset_test.word2vec_novel)

loader_train = FewShotDataLoader(dataset_train, N=params.N,  # number of novel categories (N-way)
                 num_base_label=-1,  # number of base categories
                 K=params.K,  # number of training examples per novel category (K-shot)
                 K_q=params.K_q,  # number of test examples for per novel category
                 batch_size=params.batch_size,  # number of training episodes per batch
                 num_workers=4,
                 epoch_size=params.epoch_size)

loader_test = FewShotDataLoader(dataset_test, N=params.N,  # number of novel categories (N-way)
                 num_base_label=-1,  # number of base categories
                 K=params.K,  # number of training examples per novel category (K-shot)
                 K_q=params.K_q,  # number of test examples for per novel category
                 batch_size=1,
                 num_workers=4,
                 epoch_size=params.epoch_size)

threshold_model = Threshold(params)
meta_train(params, fe_model, classifier_model, threshold_model, loader_train, loader_test)
