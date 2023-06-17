import time
import multiprocessing
import numpy as np
import torch
from torch import nn

import datasets
import config

class ResultsDict(dict):
    def append(self, key, value):
        if not key in self.keys():
            self[key] = np.array([])
        self[key] = np.append(self[key], [value])
    
    def update(self, new_dict):
        for key, value in new_dict.items():
            self.append(key, value)
    
    def reduce(self, reduce_fn):
        for key, value in self.items():
            if type(reduce_fn) == dict:
                self[key] = {k: f(value) for k, f in reduce_fn.items()}
            elif type(reduce_fn) == list:
                self[key] = [f(value) for f in reduce_fn]
            else:
                self[key] = reduce_fn(value)
    
    def data(self):
        return {key: value for key, value in self.items()}

def value(x):
    if not isinstance(x, torch.Tensor):
        raise NotImplementedError
    return x.detach().cpu().numpy()

def unpack_batch(batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    return x, y

def get_dataloader(dataset, batch_size=32, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=min(1, multiprocessing.cpu_count()//config.get_num_agents()), pin_memory=True
    )

def get_acc(logits, labels):
    if labels.shape == logits.shape:
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        predictions = nn.functional.softmax(logits, dim=-1)
        acc = nn.functional.cosine_similarity(predictions, labels, dim=-1).mean()
        acc = value(acc)
    else:
        if isinstance(logits, torch.Tensor):
            logits = value(logits)
        if isinstance(labels, torch.Tensor):
            labels = value(labels)
        predictions = np.argmax(logits, axis=-1)
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=-1)
        acc = np.mean(np.equal(predictions, labels))
    return acc

def get_soft_acc(logits, labels):
    predicted_dist = nn.functional.softmax(logits, dim=-1)
    if labels.ndim > 1:
        soft_acc = (labels*predicted_dist).sum(dim=-1).mean()
    else:
        soft_acc = predicted_dist[torch.arange(len(predicted_dist)), labels].mean() # get prediction[label] for each batch
    return value(soft_acc)

def get_rank(logits, labels):
    if isinstance(logits, torch.Tensor):
        logits = value(logits)
    if isinstance(labels, torch.Tensor):
        labels = value(labels)
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)
    rank = (-logits).argsort(axis=-1).argsort(axis=-1)
    correct_rank = rank[np.arange(len(rank)), labels].mean()
    return correct_rank

def get_norms(model):
    total_weight_norm, total_grad_norm = 0.0, 0.0
    for param in model.parameters():
        weight_norm = param.data.detach().norm(2).cpu().numpy()
        total_weight_norm += weight_norm**2
        if not param.requires_grad:
            continue
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().norm(2).cpu().numpy()
        total_grad_norm += grad_norm**2
    total_weight_norm, total_grad_norm = np.sqrt(total_weight_norm), np.sqrt(total_grad_norm)
    return {'weight_norm': total_weight_norm, 'grad_norm': total_grad_norm}

def run_epoch(dataloader, step_fn, *step_args, truncate_steps=None, average_batches=True, **step_kwargs):
    rv = ResultsDict()
    if truncate_steps is not None:
        assert 0 <= truncate_steps <= len(dataloader)
    for bidx, batch in enumerate(dataloader):
        step_rv = step_fn(batch, *step_args, **step_kwargs)
        rv.update(step_rv)
        if truncate_steps is not None and bidx >= truncate_steps:
            break
    if average_batches:
        rv.reduce(np.mean)
    return rv