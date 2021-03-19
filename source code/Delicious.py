import torch.utils.data as data
import torch
from load_data import load_dataset


def bulid_instancesID2label(labels):
    instancesID2label = {}
    for idx in range(labels.shape[1]):
        instancesID2label[idx] = labels[:, idx].nonzero()[0].tolist()
    return instancesID2label


class Delicious(data.Dataset):
    def __init__(self, dataset, phase='train'):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase

        if phase == 'train':
            self.instances, self.labels, self.word2vec_base, _, self.num_train = load_dataset(dataset, phase=self.phase)
            self.instancesID2label = bulid_instancesID2label(self.labels)
            self.instances = torch.from_numpy(self.instances).float()
            self.word2vec_base = torch.from_numpy(self.word2vec_base).float()
            self.num_cats_base = self.labels.shape[1]
            self.labelIds_base = sorted(self.instancesID2label.keys())
            # self.word2vec_novel = torch.tensor([]).float()

        elif phase == 'test' or 'val':
            self.instances, self.labels, self.word2vec_novel, _, self.num_train = load_dataset(dataset, phase=self.phase)
            self.instancesID2label = bulid_instancesID2label(self.labels)
            self.instances = torch.from_numpy(self.instances).float()
            self.word2vec_novel = torch.from_numpy(self.word2vec_novel).float()

            self.num_cats_novel = self.labels.shape[1]
            self.labelIds_novel = sorted(self.instancesID2label.keys())

            _, _, self.word2vec_base, _, _ = load_dataset(dataset, phase='train')
            self.num_cats_base = self.word2vec_base.shape[0]
            self.labelIds_base = [i for i in range(self.num_cats_base)]
            # self.word2vec_base = torch.from_numpy(self.word2vec_base).float()

        else:
            raise ValueError('Not valid phase {0}'.fotmat(self.phase))

    def __getitem__(self, index):
        instance, label = self.instances[index], self.labels[index]
        return instance, label

    def __len__(self):
        return len(self.instances)
