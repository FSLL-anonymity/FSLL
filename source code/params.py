from collections import namedtuple
import time
import torch

params = namedtuple('args', [
    'word2vecDIM', 'classifier_epoch', 'dataset_name', 'device', 'batch_size', 'model_name', 'loss',
    'currentEpoch', 'batchNorm', 'algorithm', 'dw', 'time',
    'K', 'N', 'M', 'K_q', 'threshold'
    'classifier_dropout', 'classifier_L2', 'classifier_input_dim', 'classifier_output_dim', 'momentum',
    'step_size', 'first_order', 'phase', 'epoch_size', 'weightCombine', 'attentionWeight',
    'weight_z', 'weight_i', 'weight_a'
])

# default
params.time = time.strftime("%Y-%m-%d_%H:%M:%S")
params.device = 'cuda' if torch.cuda.is_available() else 'cpu'
params.currentEpoch = 0
params.batchNorm = False
params.algorithm = 'SIN'
params.classifier_dropout = 0.1
params.classifier_L2 = 1e-06
params.dw = 300
params.word2vecDIM = 300
params.threshold = 0.5
params.momentum = 0.9
params.phase = 'train'
params.classifier_epoch = 50
params.weightCombine = False
params.attentionWeight = False
params.weight_z = 1
params.weight_i = 1
params.weight_a = 1
params.K = 5
params.K_q = 13
params.step_size = 1
params.first_order = False
params.batch_size = 10
params.epoch_size = 200
params.classifier_epoch = 100
params.weightCombine = 0
params.attentionWeight = 0


def get_params(dataset_name):
    params.dataset_name = dataset_name
    if dataset_name == 'delicious':
        params.dataset_name = dataset_name
    return params
