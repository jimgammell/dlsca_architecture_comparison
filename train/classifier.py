from collections import OrderedDict
import os
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
import torchvision

import config
import datasets
import train
import models

class ClassifierTrainer:
    def __init__(
        self,
        dataset_name,
        model_name,
        model_scale={'base_channels': 64, 'stage_blocks': [2, 2, 2, 2]}, # same as ResNet-18
        optimizer_class=optim.SGD,
        optimizer_kwargs={'lr': 1e-3, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 0.0},
        l1_weight_penalty=0.0,
        gaussian_input_noise_stdev=0.0,
        loss_fn_class=nn.CrossEntropyLoss,
        loss_fn_kwargs={},
        scheduler_class=optim.lr_scheduler.OneCycleLR,
        scheduler_kwargs={'max_lr': 1e-3, 'anneal_strategy': 'cos', 'pct_start': 0.3},
        seed=None,
        device=None,
        batch_size=64,
        update_bn_steps=100,
        val_split_size=5000,
        training_set_truncate_length=None, # Number of training datapoints to include,
        data_bytes=None,
        data_repr='bytes',
        train_metrics={'acc': train.get_acc, 'soft_acc': train.get_soft_acc, 'rank': train.get_rank},
        eval_metrics={'acc': train.get_acc, 'soft_acc': train.get_soft_acc, 'rank': train.get_rank},
        average_training_batches=False
    ):
        self.seed = config.set_seed(seed)
        self.device = config.get_device(device)
        self.update_bn_steps = update_bn_steps
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.average_training_batches = average_training_batches
        self.l1_weight_penalty = l1_weight_penalty
        self.gaussian_input_noise_stdev = gaussian_input_noise_stdev
        
        dataset_class = datasets.get_class(dataset_name)
        train_dataset = dataset_class(train=True, truncate_length=training_set_truncate_length, bytes=data_bytes, data_repr=data_repr)
        test_dataset = dataset_class(train=False, bytes=data_bytes, data_repr=data_repr)
        input_shape = test_dataset.data_shape
        data_transform = nn.Upsample(size=(2**int(np.ceil(np.log2(input_shape[-1])))), mode='linear', align_corners=False)
        train_dataset.transform = test_dataset.transform = lambda x: data_transform(x.view(1, 1, -1)).view(1, -1)
        if val_split_size > 0:
            assert val_split_size < len(train_dataset)
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, (len(train_dataset)-val_split_size, val_split_size)
            )
        
        self.batch_size = batch_size
        self.get_train_dataloader = lambda: train.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size)
        self.get_val_dataloader = lambda: train.get_dataloader(val_dataset, shuffle=False, batch_size=batch_size) if val_split_size > 0 else None
        self.get_test_dataloader = lambda: train.get_dataloader(test_dataset, shuffle=False, batch_size=batch_size)
        
        model_class = models.get_class(model_name)
        head_size = {'bytes': 256, 'bits': 8, 'hw': 9}[test_dataset.data_repr]
        model_heads = OrderedDict({test_dataset.data_repr+'_'+str(byte): head_size for byte in test_dataset.bytes})
        self.input_shape = input_shape
        self.model_heads = model_heads
        self.model_scale = model_scale
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.loss_fn_kwargs = loss_fn_kwargs
        self.get_model = lambda: model_class(self.input_shape, self.model_heads, **self.model_scale).to(self.device)
        self.get_optimizer = lambda: optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
        self.get_scheduler = lambda n_epochs: scheduler_class(self.optimizer, epochs=n_epochs, steps_per_epoch=len(train_dataset)//batch_size+(1 if len(train_dataset)%batch_size!=0 else 0), **self.scheduler_kwargs) if scheduler_class is not None else None
        self.get_loss_fn = lambda: loss_fn_class(**self.loss_fn_kwargs)
        
    def reset(self, epochs_to_run):
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(epochs_to_run)
        self.loss_fn = self.get_loss_fn()
        self.train_dataloader = self.get_train_dataloader()
        self.val_dataloader = self.get_val_dataloader()
        self.test_dataloader = self.get_test_dataloader()
    
    @torch.no_grad()
    def measure_bn_stats_step(self, batch):
        traces, _ = train.unpack_batch(batch, self.device)
        self.model.train()
        _ = self.model(traces)
        return {}
    
    def train_step(self, batch):
        rv = train.ResultsDict()
        self.model.train()
        traces, labels = train.unpack_batch(batch, self.device)
        if self.gaussian_input_noise_stdev > 0:
            traces += self.gaussian_input_noise_stdev * torch.randn_like(traces)
        logits = self.model(traces)
        total_loss = 0.0
        for hidx, (head_name, head_logits) in enumerate(logits.items()):
            target = labels[head_name]
            loss = self.loss_fn(head_logits, target)
            total_loss += loss
            rv['loss__'+head_name] = train.value(loss)
            for metric_name, metric_fn in self.train_metrics.items():
                rv[metric_name+'__'+head_name] = metric_fn(head_logits, target)
        if self.l1_weight_penalty > 0:
            total_loss += self.l1_weight_penalty * sum(p.norm(p=1) for p in self.model.parameters())
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        norms = train.get_norms(self.model)
        rv['weight_norm'] = norms['weight_norm']
        rv['grad_norm'] = norms['grad_norm']
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        rv['loss'] = np.mean([v for k, v in rv.items() if 'loss__' in k])
        for metric_name in self.train_metrics.keys():
            rv[metric_name] = np.mean([v for k, v in rv.items() if metric_name+'__' in k])
        if self.scheduler is not None:
            rv['lr'] = self.scheduler.get_last_lr()
        else:
            rv['lr'] = [g['lr'] for g in self.optimizer.param_groups][0]
        return rv
    
    @torch.no_grad()
    def eval_step(self, batch):
        rv = train.ResultsDict()
        self.model.eval()
        traces, labels = train.unpack_batch(batch, self.device)
        logits = self.model(traces)
        total_loss = 0.0
        for hidx, (head_name, head_logits) in enumerate(logits.items()):
            target = labels[head_name]
            loss = self.loss_fn(head_logits, target)
            total_loss += loss
            rv['loss__'+head_name] = train.value(loss)
            for metric_name, metric_fn in self.eval_metrics.items():
                rv[metric_name+'__'+head_name] = metric_fn(head_logits, target)
        rv['loss'] = np.mean([v for k, v in rv.items() if 'loss' in k])
        for metric_name in self.eval_metrics.keys():
            rv[metric_name] = np.mean([v for k, v in rv.items() if metric_name+'__' in k])
        return rv
    
    def update_bn_stats(self, dataloader):
        if self.update_bn_steps > 0:
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.reset_running_stats()
            _ = train.run_epoch(dataloader, self.measure_bn_stats_step, truncate_steps=self.update_bn_steps)
    
    def train_epoch(self, dataloader, **kwargs):
        rv = train.run_epoch(dataloader, self.train_step, **kwargs)
        return rv
    
    def eval_epoch(self, dataloader, **kwargs):
        rv = train.run_epoch(dataloader, self.eval_step, **kwargs)
        return rv
    
    def train_model(self, epochs, metric_to_track='rank', maximize_metric=False, results_save_dir=None, model_save_dir=None, compute_max_lr=True):
        if compute_max_lr:
            lr_results_save_dir = os.path.join(results_save_dir, 'lr_sweep') if results_save_dir is not None else None
            if lr_results_save_dir is not None:
                os.makedirs(lr_results_save_dir, exist_ok=True)
            min_lr, max_lr = self.smith_lr_finder(results_save_dir=lr_results_save_dir)
            self.optimizer_kwargs['lr'] = max_lr
            self.scheduler_kwargs['max_lr'] = max_lr
            self.scheduler_kwargs['div_factor'] = max_lr/min_lr
            self.scheduler_kwargs['final_div_factor'] = max_lr/min_lr
        self.reset(epochs)
        train_rv, test_rv = train.ResultsDict(), train.ResultsDict()
        if self.val_dataloader is not None:
            val_rv = train.ResultsDict()
        best_metric, best_model = -np.inf, None
        for epoch in range(1, epochs+1):
            train_brv = self.train_epoch(self.train_dataloader, average_batches=self.average_training_batches)
            train_rv.update(train_brv)
            self.update_bn_stats(self.train_dataloader)
            if self.val_dataloader is not None:
                val_brv = self.eval_epoch(self.val_dataloader)
                val_rv.update(val_brv)
            test_brv = self.eval_epoch(self.test_dataloader)
            test_rv.update(test_brv)
            if self.val_dataloader is not None:
                metric = val_brv[metric_to_track]
            else:
                metric = test_brv[metric_to_track]
            if not maximize_metric:
                metric *= -1
            if metric > best_metric:
                print('New best model found.')
                best_metric = metric
                best_model = {k: v.cpu() for k, v in self.model.state_dict().items()}
            print('Epoch {} completed.'.format(epoch))
            for phase, rv in [('train', train_brv)]+([('val', val_brv)] if self.val_dataloader is not None else [])+[('test', test_brv)]:
                for metric_name in [k for k in rv.keys() if '__' not in k]:
                    print('{} {}: {}'.format(phase, metric_name, np.mean(rv[metric_name])), end='')
                    if any(metric_name+'__' in k for k in rv.keys()):
                        vals = [np.mean(v) for k, v in rv.items() if metric_name+'__' in k]
                        print(' ({} -- {})'.format(min(vals), max(vals)))
                    else:
                        print()
            print()
        if results_save_dir is not None:
            with open(os.path.join(results_save_dir, 'train_results.pickle'), 'wb') as F:
                pickle.dump(train_rv.data(), F)
            if self.val_dataloader is not None:
                with open(os.path.join(results_save_dir, 'val_results.pickle'), 'wb') as F:
                    pickle.dump(val_rv.data(), F)
            with open(os.path.join(results_save_dir, 'test_results.pickle'), 'wb') as F:
                pickle.dump(test_rv.data(), F)
            metric_names = [k for k in train_rv.keys() if '__' not in k]
            metric_names += [k for k in test_rv.keys() if ('__' not in k) and (k not in metric_names)]
                
            num_metrics = len(metric_names)
            (fig, axes) = plt.subplots(2, num_metrics, figsize=(num_metrics*6, 2*6))
            epochs = np.arange(len(test_rv['loss']))
            train_epochs = np.linspace(0, len(test_rv['loss'])-1, len(train_rv['loss']))
            stack = lambda rv, key: np.stack([v for k, v in rv.items() if key+'__' in k], axis=0).transpose(1, 0)
            for idx, (metric_name, ax0, ax1) in enumerate(zip(metric_names, axes[0, :], axes[1, :])):
                if metric_name in train_rv.keys():
                    ax0.plot(train_epochs, train_rv[metric_name], '.', color='blue', label='Training')
                    if any(metric_name+'__' in k for k in train_rv.keys()):
                        ax1.plot(train_epochs, stack(train_rv, metric_name), '.', color='blue')
                if metric_name in val_rv.keys():
                    ax0.plot(epochs, val_rv[metric_name] if self.val_dataloader is not None else test_rv[metric_name],
                             '-', color='red', label='Validation' if self.val_dataloader is not None else 'Testing')
                    if any(metric_name+'__' in k for k in val_rv.keys()):
                        ax1.plot(epochs, stack(val_rv if self.val_dataloader is not None else test_rv, metric_name),
                                 '-', color='red', label='Validation' if self.val_dataloader is not None else 'Testing')
                ax0.set_xlabel('Epochs')
                ax1.set_xlabel('Epochs')
                disp_metric_name = ' '.join(metric_name.split('_')).capitalize()
                ax0.set_ylabel(disp_metric_name)
                ax1.set_ylabel(disp_metric_name)
                if ('loss' in metric_name) or ('norm' in metric_name):
                    ax0.set_yscale('log')
                    ax1.set_yscale('log')
                ax0.legend()
                ax0.grid()
                ax1.grid()
            fig.suptitle('Training curves')
            fig.savefig(os.path.join(results_save_dir, 'train_curves.png'))
        if model_save_dir is not None:
            torch.save(best_model, os.path.join(model_save_dir, 'best_model.pt'))
    
    def smith_lr_finder(self, min_lr=1e-8, max_lr=1e1, lr_steps=250, total_epochs=3, avg_radius=10, results_save_dir=None):
        self.reset(1)
        self.scheduler = None
        lr_values = 10**np.linspace(np.log10(min_lr), np.log10(max_lr), lr_steps)
        rv = train.ResultsDict()
        for lr_value in lr_values:
            t0 = time.time()
            for g in self.optimizer.param_groups:
                g['lr'] = lr_value
            train_rv = self.train_epoch(self.train_dataloader, truncate_steps=total_epochs*len(self.train_dataloader)//lr_steps)
            rv.update(train_rv)
            print('learning rate: {}, training loss: {} (time taken: {} sec)'.format(lr_value, train_rv['loss'], time.time()-t0))
        if avg_radius > 0:
            smoothed_trace = []
            for i in range(avg_radius, len(rv['loss'])-avg_radius):
                smoothed_trace.append(np.mean(rv['loss'][i-avg_radius:i+avg_radius+1]))
            smoothed_trace = np.array(smoothed_trace)
        else:
            smoothed_trace = rv['loss']
        iqr = np.subtract(*np.nanpercentile(smoothed_trace, [75, 25]))
        for sidx, sample in enumerate(rv['loss']):
            if np.isfinite(sample) and sample < smoothed_trace[0]:
                pass
            else:
                maximum_lr = lr_values[sidx-avg_radius]
                break
        minimum_lr = 1e-1*lr_values[avg_radius:-avg_radius][np.nanargmin(smoothed_trace)]
        print('Learning rate range: %e -- %e'%(minimum_lr, maximum_lr))
        if results_save_dir is not None:
            with open(os.path.join(results_save_dir, 'lr_sweep_results.pickle'), 'wb') as F:
                pickle.dump(rv.data(), F)
            (fig, ax) = plt.subplots()
            ax.set_xlabel('Learning rate')
            ax.set_ylabel('Training loss')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('Learning rate sweep results')
            ax.plot(lr_values, rv['loss'], '.', color='blue', label='raw')
            ax.plot(lr_values[avg_radius:-avg_radius], smoothed_trace, color='blue', linestyle='--', label='smoothed')
            ax.set_ylim(np.nanmin(smoothed_trace)-2*iqr, smoothed_trace[0]+2*iqr)
            ax.axvspan(minimum_lr, maximum_lr, alpha=0.25, color='blue', label='Learning rate range')
            ax.axvline(lr_values[avg_radius:-avg_radius][np.nanargmin(smoothed_trace)], linestyle='--', color='black', label='Min loss')
            ax.axhline(np.nanmin(smoothed_trace), linestyle='--', color='black')
            ax.legend()
            fig.savefig(os.path.join(results_save_dir, 'lr_sweep_results.png'))
        return minimum_lr, maximum_lr
    
    def sweep_batch_size(self, min_size=1, max_size=np.inf, steps_per_test=100, results_save_dir=None):
        rv = train.ResultsDict()
        batch_size = min_size
        while batch_size < max_size:
            try:
                self.batch_size = batch_size
                self.reset(1)
                self.scheduler = None
                t0 = time.time()
                remaining_steps = steps_per_test
                while remaining_steps > 0:
                    truncate_steps = remaining_steps if remaining_steps < len(self.train_dataloader) else None
                    _ = self.train_epoch(self.train_dataloader, truncate_steps=truncate_steps)
                    remaining_steps -= len(self.train_dataloader) if truncate_steps is None else truncate_steps
                torch.cuda.synchronize()
                time_elapsed = time.time()-t0
                rv.append('batch_size', batch_size)
                rv.append('time_elapsed', time_elapsed)
                print('Time elapsed with batch size {}: {} sec'.format(batch_size, time_elapsed))
                batch_size *= 2
            except RuntimeError: # CUDA out of memory
                break
        if results_save_dir is not None:
            with open(os.path.join(results_save_dir, 'batch_size_sweep_results.pickle'), 'wb') as F:
                pickle.dump(rv.data(), F)
            (fig, ax) = plt.subplots()
            ax.set_xlabel('Batch size')
            ax.set_ylabel('Examples/sec')
            ax.set_xscale('log')
            ax.plot(rv['batch_size'], steps_per_test*rv['batch_size']/rv['time_elapsed'], marker='.', linestyle='--', color='blue', label='Throughput')
            tax = ax.twinx()
            tax.plot(rv['batch_size'], rv['time_elapsed'], marker='.', linestyle='--', color='red', label='Total time')
            tax.set_ylabel('Sec')
            ax.set_title('Throughput vs. batch size')
            ax.legend()
            tax.legend()
            fig.savefig(os.path.join(results_save_dir, 'batch_size_sweep_results.png'))