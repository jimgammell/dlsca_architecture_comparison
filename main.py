import time
import argparse
import wandb
import copy
import pickle
import os
import traceback

import datasets, datasets.test
import train, train.classifier
import models
import config

def get_datasets(args):
    if 'all' in args.dataset:
        args.dataset = datasets.get_available_datasets()
    if any(x not in datasets.get_available_datasets() for x in args.dataset):
        raise argparse.ArgumentError(
            'Error running trials. The following datasets are not recognized: \'{}\''.format(
                '\', \''.join([x for x in args.dataset if x not in datasets.get_available_datasets()])
            ))

def download_dataset(dataset_args, force=False):
    for dataset_name in dataset_args:
        t0 = time.time()
        print('Downloading dataset \'{}\'...'.format(dataset_name))
        datasets.download_dataset(dataset_name, force=force)
        print('\tDone. Time taken: {} sec.'.format(time.time()-t0))

def train_classifier(dataset_args, settings, seed=None, device=None):
    def run_(seed_, dataset_name):
        save_dir = config.results_subdir(settings['save_dir'], dataset_name, *(['seed_%d'%(seed_)] if seed_ is not None else []))
        trainer = train.classifier.ClassifierTrainer(dataset_name, seed=seed_, device=device, **settings)
        trainer.train_model(settings['total_epochs'], results_save_dir=save_dir, compute_max_lr=settings['autotune_lr'])
    for dataset_name in dataset_args:
        if len(seed) == 0:
            run_(None, dataset_name)
        else:
            for seed_ in seed:
                run_(seed_, dataset_name)

def htune_classifier(dataset_args, settings, seed=None, device=None, num_agents=1):
    assert (len(dataset_args)==1) and (len(seed)==1)
    dataset_name = dataset_args[0]
    seed = seed[0]
    wandb_config = settings['wandb_config']
    wandb_config = config.denest_dict(wandb_config)
    classifier_settings = {key: val for key, val in settings.items() if key != 'wandb_config'}
    sweep_id = wandb.sweep(
        sweep=wandb_config,
        project=settings['save_dir']
    )
    
    def run_wandb_trial_():
        wandb.init(mode='offline')
        trial_settings = copy.deepcopy(classifier_settings)
        trial_settings = config.nest_dict(trial_settings)
        save_dir = config.results_subdir(settings['save_dir'], dataset_name)
        if len(os.listdir(save_dir)) > 0:
            save_dir = os.path.join(save_dir, 'trial_%d'%(max(int(f.split('_')[-1]) for f in os.listdir(save_dir))+1))
        else:
            save_dir = os.path.join(save_dir, 'trial_0')
        try:
            trainer = train.classifier.ClassifierTrainer(dataset_name, seed=seed, device=device, **trial_settings)
        except Exception:
            traceback.print_exc()
            raise Exception('Trial code crashed.')
        trainer.train_model(settings['total_epochs'], results_save_dir=save_dir, compute_max_lr=False, generate_plots=False, report_to_wandb=True)
    
    def spawn_agent():
        wandb.agent(sweep_id, function=run_wandb_trial_)
    
    if num_agents > 1:
        procs = []
        for _ in range(num_agents):
            p = multiprocessing.Process(target=spawn_agent)
            p.start()
        for p in procs:
            p.join()
    elif num_agents == 1:
        spawn_agent()
    else:
        raise NotImplementedError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-tests', default=False, action='store_true', help='Run tests to validate dataset and model code.'
    )
    parser.add_argument(
        '--dataset', choices=datasets.get_available_datasets() + ['all'], nargs='+', default=[],
        help='Dataset to use for this trial. Valid options: \'{}\'. Pass \'all\' to download all valid options.'.format('\', \''.join(datasets.get_available_datasets()))
    )
    parser.add_argument(
        '--download', default=False, action='store_true',
        help='Download the specified datasets.'
    )
    parser.add_argument(
        '--train-classifier', choices=config.get_available_configs(), nargs='+', default=[],
        help='Classifiers to train, as defined in the respective config files.'
    )
    parser.add_argument(
        '--htune-classifier', choices=config.get_available_configs(train=False), nargs='+', default=[],
        help='Classifiers for which to tune hyperparameters, as defined in the respective config files.'
    )
    parser.add_argument(
        '--num-wandb-agents', default=1, type=int,
        help='Number of WANDB agents to run for this trial configuration.'
    )
    parser.add_argument(
        '-f', '--force', default=False, action='store_true',
        help='Force the command to take effect.'
    )
    parser.add_argument(
        '--seed', default=[], type=int, nargs='+', help='Random seed to use in this trial.'
    )
    parser.add_argument(
        '--device', default=None, help='Device to use for this trial.'
    )
    parser.add_argument(
        '--num-epochs', default=None, type=int, help='Number of epochs to train for. Overrides the value specified in the trial configuration file.'
    )
    parser.add_argument(
        '--lr', default=None, type=float, help='Override the learning rate specified in the config file with this value.'
    )
    parser.add_argument(
        '--save-dir', default=None, type=str, help='Override the results directory specified in the config file with this value.'
    )
    args = parser.parse_args()
    
    get_datasets(args)
    if args.download:
        download_dataset(args.dataset, force=args.force)
    if args.run_tests:
        datasets.test.test_datasets()
    for config_name in args.train_classifier:
        settings = config.load_config(config_name)
        if args.num_epochs is not None:
            settings['total_epochs'] = args.num_epochs
        if args.lr is not None:
            settings['optimizer_kwargs']['lr'] = args.lr
        if args.save_dir is not None:
            settings['save_dir'] = args.save_dir
        train_classifier(args.dataset, settings, seed=args.seed, device=args.device)
    for config_name in args.htune_classifier:
        settings = config.load_config(config_name, train=False)
        htune_classifier(args.dataset, settings, seed=args.seed, device=args.device, num_agents=args.num_wandb_agents)

if __name__ == '__main__':
    main()