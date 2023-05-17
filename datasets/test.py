import time
import os
import numpy as np
from matplotlib import pyplot as plt

import config
import datasets

def test_datasets():
    traces = {}
    durations = {}
    
    for dataset_name in datasets.get_available_datasets():
        print('Testing dataset {}...'.format(dataset_name))
        dataset_class = datasets.get_class(dataset_name)
        for train in [True, False]:
            print('Testing dataset {}(train={})...'.format(dataset_name, train))
            t0 = time.time()
            dataset = dataset_class(train=train)
            print('\tConstructed dataset in {} seconds.'.format(time.time()-t0))
            print('\tDataset length: {}'.format(len(dataset)))
            print('\tDataset data shape: {}'.format(dataset.data_shape))
            tt0 = []
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            mean_trace, min_trace, max_trace = np.zeros(dataset.data_shape[1]), np.ones(dataset.data_shape[1]), -np.ones(dataset.data_shape[1])
            tt0 = []
            for idx in indices:
                t0 = time.time()
                trace, target = dataset.__getitem__(idx)
                tt0.append(time.time()-t0)
                trace = trace.numpy().squeeze()
                mean_trace += trace/len(dataset)
                min_trace = np.minimum(trace, min_trace)
                max_trace = np.maximum(trace, max_trace)
            print('\tTotal traversal time: {} seconds.'.format(sum(tt0)))
            print('\tPer-item index time: {} (min) / {}+/-{} (mean +/- stdev) / {} (max) seconds.'.format(
                min(tt0), np.mean(tt0), np.std(tt0), max(tt0)
            ))
            if not dataset_name in traces.keys():
                traces[dataset_name] = {}
            traces[dataset_name]['train' if train else 'test'] = {
                'mean_trace': mean_trace, 'min_trace': min_trace, 'max_trace': max_trace
            }
            if not dataset_name in durations.keys():
                durations[dataset_name] = {}
            durations[dataset_name]['train' if train else 'test'] = tt0
    
    print('Plotting power traces...')
    for dataset_name, trace in traces.items():
        fig, axes = plt.subplots(2, 1, figsize=(6, 10))
        for phase, ax in zip(trace.keys(), axes.flatten()):
            timesteps = np.arange(len(trace[phase]['mean_trace']))
            ax.plot(timesteps, trace[phase]['mean_trace'], color='blue', label='mean')
            ax.fill_between(timesteps, trace[phase]['min_trace'], trace[phase]['max_trace'], color='blue', alpha=0.25, label='min--max')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Power consumption')
            ax.set_title('Phase: {}'.format(phase))
            ax.legend()
        fig.suptitle('Dataset: {}'.format(dataset_name))
        fig.savefig(os.path.join(config.results_subdir('dataset_tests', dataset_name), 'power_traces.png'))
    
    print('Plotting index times...')
    for dataset_name, duration in durations.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist(duration['train'], color='blue', label='Train')
        ax.hist(duration['test'], color='red', label='Test')
        ax.set_xlabel('Duration (sec)')
        ax.set_ylabel('Count')
        ax.set_yscale('symlog', linthresh=1e0)
        ax.set_title('Time to index dataset: {}'.format(dataset_name))
        ax.legend()
        fig.savefig(os.path.join(config.results_subdir('dataset_tests', dataset_name), 'index_time.png'))