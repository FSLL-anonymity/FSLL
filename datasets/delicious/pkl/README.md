# Multilabel Classification Datasets

Please download datasets from

http://manikvarma.org/downloads/XC/XMLRepository.html

http://mulan.sourceforge.net/datasets-mlc.html

This repository contains data pickels for multilabel classification dataset for easy research. 

## File Format 
```
dataset_name
|____________ dataset_name-train-features.pkl
|____________ dataset_name-test-features.pkl
|____________ dataset_name-train-labels.pkl
|____________ dataset_name-test-labels.pkl
```

## Using the datsets 
```python
import numpy as np

def get_data(path, noise=False):
    data = np.load(path)
    if noise == True :
        data = data + np.random.normal(0, 0.001, data.shape)
    return data

x_train = get_data("./datset_name/dataset_name-train-features.pkl")
```

## Converting your own data
To convert your standard arff files to numpy pickles, which are easier to use and faster to process, use the script I have included along with the datasets.

Download the dataset and put it in this folder. Create a `count.txt` in the directory and put the dimensions in the following format :
```
features_dim
labels_dim
```
### Installation & Running
```shell
pip3 install liac-arff 
python3 to_numpy --dataset dataset_name
```
