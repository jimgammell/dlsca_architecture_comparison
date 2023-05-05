import time
import numpy as np

import datasets

def train_model(dataset_name):
    dataset_class = datasets.get_class(dataset_name)
    train_dataset = dataset_class(train=True)
    test_dataset = dataset_class(train=False)
    eg_trace, eg_label = train_dataset[0]
    print(eg_trace.shape)
    #print(eg_label.shape)
    print(len(train_dataset))
    print(len(test_dataset))
    dataset_indices = [x for x in range(len(train_dataset))]
    np.random.shuffle(dataset_indices)
    t0 = time.time()
    for idx in dataset_indices:
        _ = train_dataset[idx]
    traversal_time = time.time()-t0
    print('Time to traverse dataset: {} ({} per item)'.format(traversal_time, traversal_time/len(train_dataset)))