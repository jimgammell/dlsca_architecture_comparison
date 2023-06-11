from collections import OrderedDict
import os
import time
import pickle
import numpy as np
import wandb
from scipy.special import softmax
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
import torchvision

import config
import datasets
import train
import models
from external_libs.sam import SAM
from external_libs.temperature_scaling import ModelWithTemperature

class ClassifierTrainer:
    def __init__(
        self,
        dataset_name,
        model_name,
        model_kwargs={},
        optimizer_class=optim.SGD,
        optimizer_kwargs={},
        use_sam=False,
        sam_kwargs={},
        l1_weight_penalty=0.0,
        gaussian_input_noise_stdev=0.0,
        mixup_alpha=0.0,
        rescale_temperature=False,
        label_smoothing=False,
        loss_fn_class=nn.BCELoss,
        loss_fn_kwargs={},
        scheduler_class=None,
        scheduler_kwargs={},
        seed=None,
        device=None,
        batch_size=32,
        update_bn_steps=0,
        val_split_size=5000,
        n_test_traces=10000,
        final_evaluation_repetitions=1,
        selection_metric=None,
        maximize_selection_metric=True,
        training_set_truncate_length=None, # Number of training datapoints to include,
        data_bytes=None,
        data_repr='bytes',
        train_metrics={'acc': train.get_acc, 'rank': train.get_rank},
        eval_metrics={'acc': train.get_acc, 'rank': train.get_rank},
        average_training_batches=True,
        **kwargs
    ):
        if type(loss_fn_class) == str:
            loss_fn_class = getattr(nn, loss_fn_class)
        if type(scheduler_class) == str:
            scheduler_class = getattr(optim.lr_scheduler, scheduler_class)
        if type(optimizer_class) == str:
            optimizer_class = getattr(optim, optimizer_class)
        
        self.seed = config.set_seed(0) #seed)
        self.device = config.get_device(device)
        self.update_bn_steps = update_bn_steps
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.average_training_batches = average_training_batches
        self.l1_weight_penalty = l1_weight_penalty
        self.gaussian_input_noise_stdev = gaussian_input_noise_stdev
        self.batch_size = batch_size
        self.n_test_traces = n_test_traces
        self.final_evaluation_repetitions = final_evaluation_repetitions
        self.selection_metric = selection_metric
        self.maximize_selection_metric = maximize_selection_metric
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        
        dataset_class = datasets.get_class(dataset_name)
        train_dataset = dataset_class(train=True, truncate_length=training_set_truncate_length, bytes=data_bytes, data_repr=data_repr)
        test_dataset = dataset_class(train=False, bytes=data_bytes, data_repr=data_repr)
        self.test_dataset = test_dataset
        input_shape = test_dataset.data_shape
        train_dataset.transform = test_dataset.transform = None
        assert 0 < val_split_size < len(train_dataset)
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, (len(train_dataset)-val_split_size, val_split_size)
        )
        
        self.batch_size = batch_size
        self.get_train_dataloader = lambda: train.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size)
        self.get_val_dataloader = lambda: train.get_dataloader(val_dataset, shuffle=False, batch_size=batch_size)
        self.get_test_dataloader = lambda: train.get_dataloader(test_dataset, shuffle=False, batch_size=batch_size)
        #   Shuffling test dataloader so it's easier to do the final evaluation with different subsets of traces
        
        model_class = models.get_class(model_name)
        head_size = {'bytes': 256, 'bits': 8, 'hw': 9}[test_dataset.data_repr]
        model_heads = OrderedDict({test_dataset.data_repr+'_'+str(byte): head_size for byte in test_dataset.bytes})
        self.input_shape = input_shape
        self.model_heads = model_heads
        self.model_kwargs = model_kwargs
        self.model_kwargs['precise_bn'] = update_bn_steps > 0
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.loss_fn_kwargs = loss_fn_kwargs
        self.use_sam = use_sam
        self.sam_kwargs = sam_kwargs
        self.rescale_temperature = rescale_temperature
        if rescale_temperature:
            self.get_model = lambda: ModelWithTemperature(model_class(
                self.input_shape, **self.model_kwargs
            ).to(self.device), device=self.device)
        else:
            self.get_model = lambda: model_class(self.input_shape, self.model_heads, **self.model_kwargs).to(self.device)
        if use_sam:
            self.get_optimizer = lambda: SAM(self.model.parameters(), optimizer_class, **self.optimizer_kwargs, **self.sam_kwargs)
        else:
            self.get_optimizer = lambda: optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
        if (scheduler_class is None) or (scheduler_class == 'none'):
            self.get_scheduler = None
        else:
            self.get_scheduler = lambda n_epochs: scheduler_class(self.optimizer, n_epochs*len(self.train_dataloader), **self.scheduler_kwargs)
        if label_smoothing:
            self.loss_fn_kwargs['label_smoothing'] = 0.1
        self.get_loss_fn = lambda: loss_fn_class(**self.loss_fn_kwargs)
        
    def reset(self, epochs_to_run):
        self.train_dataloader = self.get_train_dataloader()
        self.val_dataloader = self.get_val_dataloader()
        self.test_dataloader = self.get_test_dataloader()
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(epochs_to_run)
        self.loss_fn = self.get_loss_fn()
    
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
        if self.mixup_alpha > 0:
            indices = torch.randperm(traces.size(0), device=self.device, dtype=torch.long)
            mu_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            traces = mu_lambda*traces + (1-mu_lambda)*traces[indices]
        if self.label_smoothing:
            labels = 0.9*labels + 0.1*torch.ones_like(labels)/(labels.size(-1)-1)
        def closure(update_rv=False):
            nonlocal rv
            logits = self.model(traces)
            total_loss = 0.0
            if self.mixup_alpha > 0:
                loss = mu_lamvda*self.loss_fn(logits, labels) + (1-mu_lambda)*self.loss_fn(logits, labels[indices])
            else:
                loss = self.loss_fn(logits, labels)
            if update_rv:
                rv['loss'] = train.value(loss)
                for metric_name, metric_fn in self.train_metrics.items():
                    rv[metric_name] = metric_fn(logits, labels)
            if self.l1_weight_penalty > 0:
                loss += self.l1_weight_penalty * sum(p.norm(p=1) for p in self.model.parameters())
            loss.backward()
            return loss
        self.optimizer.zero_grad(set_to_none=True)
        loss = closure(update_rv=True)
        norms = train.get_norms(self.model)
        rv['weight_norm'] = norms['weight_norm']
        rv['grad_norm'] = norms['grad_norm']
        if self.use_sam:
            self.optimizer.step(closure)
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
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
        loss = self.loss_fn(logits, labels)
        rv['loss'] = train.value(loss)
        for metric_name, metric_fn in self.eval_metrics.items():
            rv[metric_name] = metric_fn(logits, labels)
        return rv
    
    def update_bn_stats(self, dataloader, override_steps=-1):
        if (self.update_bn_steps > 0) or (override_steps != -1):
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.reset_running_stats()
            truncate_steps = self.update_bn_steps if override_steps == -1 else override_steps
            _ = train.run_epoch(dataloader, self.measure_bn_stats_step, truncate_steps=truncate_steps)
    
    def train_epoch(self, dataloader, **kwargs):
        rv = train.run_epoch(dataloader, self.train_step, **kwargs)
        self.model.set_temperature(self.val_dataloader)
        return rv
    
    def eval_epoch(self, dataloader, **kwargs):
        rv = train.run_epoch(dataloader, self.eval_step, **kwargs)
        return rv
    
    @torch.no_grad()
    def evaluate_model(self, truncate_at_sample=None, return_full_trace=False, target_byte=2):
        # Compute model predictions on some randomly-selected test traces
        self.model.eval()
        random_indices = np.arange(len(self.test_dataset))
        np.random.shuffle(random_indices) # We want to get random subsets of the test traces each time we run this
        test_dataset_metadata_setting = self.test_dataset.return_metadata
        self.test_dataset.return_metadata = True # We need metadata to compute the AES key corresponding to predicted intermediate variable
        logits, metadata = [], []
        for batch in self.test_dataloader:
            traces, _, metadata_ = batch
            traces = traces.to(self.device)
            logits_ = self.model(traces).cpu().numpy()
            logits.append(logits_)
            metadata.append(metadata_)
            if truncate_at_sample is not None and self.batch_size*len(logits) >= truncate_at_sample:
                break
        logits = np.concatenate([val for val in logits], axis=0)
        plaintexts = np.concatenate([val['plaintext'][:, target_byte] for val in metadata], axis=0)
        aes_keys = np.concatenate([val['key'][:, target_byte] for val in metadata], axis=0)
        if truncate_at_sample is not None:
            logits = logits[:truncate_at_sample]
            plaintexts = plaintexts[:truncate_at_sample]
            aes_keys = aes_keys[:truncate_at_sample]
        self.test_dataset.return_metadata = test_dataset_metadata_setting
        
        # Compute the rank over time, based on https://github.com/suvadeep-iitb/TransNet/blob/master/evaluation_utils_ascad.py
        def get_log_prob(predictions, plaintext):
            predictions = np.squeeze(predictions)
            n_classes = predictions.shape[0]
            keys = np.arange(n_classes, dtype=int)
            x_xor_k = np.bitwise_xor(keys, plaintext)
            z = np.take(self.test_dataset.sbox, x_xor_k)
            log_prob = np.take(predictions, z)
            return log_prob
        
        rv = {}
        predictions = softmax(logits, axis=-1)
        predictions = np.log(predictions + 1e-40)
        n_samples, n_classes = predictions.shape
        cum_log_prob = np.zeros((n_samples, n_classes))
        last_log_prob = np.zeros((1, n_classes))
        for sidx in range(n_samples):
            log_prob = get_log_prob(predictions[sidx, :], plaintexts[sidx])
            last_log_prob += log_prob
            cum_log_prob[sidx, :] = last_log_prob
            
        sorted_keys = np.argsort(-cum_log_prob, axis=1)
        key_ranks = np.zeros((n_samples,), dtype=int) - 1
        for sidx in range(n_samples):
            for cidx in range(n_classes):
                if sorted_keys[sidx, cidx] == aes_keys[sidx]:
                    key_ranks[sidx] = cidx
                    break
        if return_full_trace:
            rv.update({'full_trace': key_ranks})
        rv.update({'mean_rank': np.mean(key_ranks)})
        rv.update({'final_rank': key_ranks[-1]})
        return rv
    
    def train_model(self, epochs, results_save_dir=None, model_save_dir=None, compute_max_lr=True, generate_plots=True, report_to_wandb=False):
        if compute_max_lr:
            lr_results_save_dir = os.path.join(results_save_dir, 'lr_sweep') if results_save_dir is not None else None
            if lr_results_save_dir is not None:
                os.makedirs(lr_results_save_dir, exist_ok=True)
            min_lr, max_lr = self.smith_lr_finder(results_save_dir=lr_results_save_dir)
            self.optimizer_kwargs['lr'] = max_lr
            if self.get_scheduler is not None:
                self.scheduler_kwargs['max_lr'] = max_lr
                self.scheduler_kwargs['div_factor'] = max_lr/min_lr
                self.scheduler_kwargs['final_div_factor'] = max_lr/min_lr
        
        self.reset(epochs)
        train_rv, test_rv = train.ResultsDict(), train.ResultsDict()
        if self.val_dataloader is not None:
            val_rv = train.ResultsDict()
        
        print('Training classifier...')
        print(self.model)
        print('Number of parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))
        
        best_metric, best_model = -np.inf, None
        for epoch in range(1, epochs+1):
            t0 = time.time()
            train_brv = self.train_epoch(self.train_dataloader, average_batches=self.average_training_batches)
            train_rv.update(train_brv)
            self.update_bn_stats(self.train_dataloader)
            t0 = time.time()
            val_brv = self.eval_epoch(self.val_dataloader)
            val_rv.update(val_brv)
            t0 = time.time()
            test_brv = [self.evaluate_model(truncate_at_sample=self.n_test_traces) for _ in range(self.final_evaluation_repetitions)]
            test_brv = {key: np.mean([val[key] for val in test_brv]) for key in test_brv[0].keys()}
            test_rv.update(test_brv)
            if report_to_wandb:
                wandb.log({'epoch': epoch})
                wandb.log({'train_'+k: v for k, v in train_brv.items()})
                wandb.log({'val_'+k: v for k, v in val_brv.items()})
                wandb.log({'test_'+k: v for k, v in test_brv.items()})
            if self.selection_metric is not None:
                metric = val_brv[self.selection_metric]
                if not self.maximize_selection_metric:
                    metric *= -1
                if metric > best_metric:
                    print('New best model found.')
                    best_metric = metric
                    best_model = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    if report_to_wandb:
                        wandb.run.summary.update({'best_epoch': epoch})
                        wandb.run.summary.update({'best_train_'+k: v for k, v in train_brv.items()})
                        wandb.run.summary.update({'best_val_'+k: v for k, v in val_brv.items()})
                        wandb.run.summary.update({'best_test_'+k: v for k, v in test_brv.items()})
            print('Epoch {} completed.'.format(epoch))
            for phase, rv in [('train', train_brv), ('val', val_brv), ('test', test_brv)]:
                for metric_name in [k for k in rv.keys() if '__' not in k]:
                    print('{} {}: {}'.format(phase, metric_name, np.mean(rv[metric_name])), end='')
                    if any('__' in k and k.split('__')[0]==metric_name for k in rv.keys()):
                        vals = [np.mean(v) for k, v in rv.items() if k.split('__')[0]==metric_name in k]
                        print(' ({} -- {})'.format(min(vals), max(vals)))
                    else:
                        print()
            print()
        if self.selection_metric is None:
            best_model = {k: v.cpu() for k, v in self.model.state_dict().items()}          
            
        print('Done training. Computing performance of best model.')
        self.reset(1)
        self.model.load_state_dict({k: v.to(self.device) for k, v in best_model.items()})
        if self.update_bn_steps > 0:
            self.update_bn_stats(self.train_dataloader, override_steps=len(self.train_dataloader))
        train_final_rv = self.eval_epoch(self.train_dataloader)
        val_final_rv = self.eval_epoch(self.val_dataloader)
        test_final_rv = self.eval_epoch(self.test_dataloader)
        for phase, rv in [('train', train_final_rv), ('val', val_final_rv), ('test', test_final_rv)]:
            for metric_name in [k for k in rv.keys() if '__' not in k]:
                print('{} {}: {}'.format(phase, metric_name, np.mean(rv[metric_name])), end='')
                if any('__' in k and k.split('__')[0]==metric_name for k in rv.keys()):
                    vals = [np.mean(v) for k, v in rv.items() if k.split('__')[0]==metric_name]
                    print(' ({} -- {})'.format(min(vals), max(vals)))
                else:
                    print()
        for repetition in range(self.final_evaluation_repetitions):
            rv = self.evaluate_model(truncate_at_sample=self.n_test_traces, return_full_trace=True)
            for key, val in rv.items():
                if not key in test_final_rv.keys():
                    test_final_rv[key] = []
                test_final_rv[key].append(val)
        
        if results_save_dir is not None:
            os.makedirs(os.path.join(results_save_dir, 'intermediate_results'), exist_ok=True)
            os.makedirs(os.path.join(results_save_dir, 'final_results'), exist_ok=True)
            
            # save results
            with open(os.path.join(results_save_dir, 'intermediate_results', 'train_results.pickle'), 'wb') as F:
                pickle.dump(train_rv.data(), F)
            with open(os.path.join(results_save_dir, 'intermediate_results', 'val_results.pickle'), 'wb') as F:
                pickle.dump(val_rv.data(), F)
            with open(os.path.join(results_save_dir, 'intermediate_results', 'test_results.pickle'), 'wb') as F:
                pickle.dump(test_rv.data(), F)
            with open(os.path.join(results_save_dir, 'final_results', 'train_results.pickle'), 'wb') as F:
                pickle.dump(train_final_rv, F)
            with open(os.path.join(results_save_dir, 'final_results', 'val_results.pickle'), 'wb') as F:
                pickle.dump(val_final_rv, F)
            with open(os.path.join(results_save_dir, 'final_results', 'test_results.pickle'), 'wb') as F:
                pickle.dump(test_final_rv, F)
            metric_names = [k for k in train_rv.keys() if '__' not in k]
            metric_names += [k for k in test_rv.keys() if ('__' not in k) and (k not in metric_names)]
            
            if generate_plots:
                os.makedirs(os.path.join(results_save_dir, 'figs'), exist_ok=True)
                # plot training curves
                num_metrics = len(metric_names)
                (fig, axes) = plt.subplots(2, num_metrics, figsize=(num_metrics*6, 2*6))
                epochs = np.arange(len(val_rv['loss']))
                train_epochs = np.linspace(0, len(val_rv['loss'])-1, len(train_rv['loss']))
                stack = lambda rv, key: np.stack([v for k, v in rv.items() if key+'__' in k], axis=0).transpose(1, 0)
                for idx, (metric_name, ax0, ax1) in enumerate(zip(metric_names, axes[0, :], axes[1, :])):
                    if metric_name in train_rv.keys():
                        ax0.plot(train_epochs, train_rv[metric_name], '.', color='blue', label='Training')
                        if any('__' in k and k.split('__')[0]==metric_name for k in train_rv.keys()):
                            ax1.plot(train_epochs, stack(train_rv, metric_name), '.', color='blue')
                    if metric_name in val_rv.keys():
                        ax0.plot(epochs, val_rv[metric_name], '-', color='red', label='Validation')
                        if any('__' in k and k.split('__')[0]==metric_name for k in val_rv.keys()):
                            ax1.plot(epochs, stack(val_rv, metric_name), '-', color='red', label='Validation')
                    if metric_name in test_rv.keys():
                        ax0.plot(epochs, test_rv[metric_name], '-', color='green', label='Testing')
                        if any('__' in k and k.split('__')[0]==metric_name for k in val_rv.keys()):
                            ax1.plot(epochs, stack(test_rv, metric_name), '-', color='green', label='Testing')
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
                fig.savefig(os.path.join(results_save_dir, 'figs', 'train_curves.png'))

                # plot rank of correct key as we add traces
                (fig, axes) = plt.subplots(1, 2, figsize=(12, 6))
                samples = np.arange(len(test_final_rv['full_trace'][0]))
                mean_final_trace = np.mean(np.stack(test_final_rv['full_trace']), axis=0)
                axes[0].plot(samples, mean_final_trace, '-', color='blue')
                for val in [val for key, val in test_final_rv.items() if 'full_trace__' in key]:
                    mean_final_trace_k = np.mean(np.stack(val), axis=0)
                    axes[1].plot(samples, mean_final_trace_k, '.', color='blue')
                axes[0].set_title('Average over bytes')
                axes[1].set_title('Per-byte')
                for ax in axes.flatten():
                    ax.set_xlabel('Samples')
                    ax.set_ylabel('Rank of correct key')
                    ax.set_ylim(-1, 256)
                    ax.grid()
                fig.suptitle('Correct key rank vs. number of samples seen')
                fig.savefig(os.path.join(results_save_dir, 'figs', 'key_vs_samples.png'))
            
        # save best model
        if results_save_dir is not None:
            os.makedirs(os.path.join(results_save_dir, 'trained_models'), exist_ok=True)
            torch.save(best_model, os.path.join(results_save_dir, 'trained_models', 'best_model.pt'))
    
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