import time
import argparse

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
    def run_(seed_):
        save_dir = config.results_subdir(settings['save_dir'], dataset_name, *(['seed_%d'%(seed_)] if seed_ is not None else []))
        trainer = train.classifier.ClassifierTrainer(dataset_name, seed=seed_, device=device, **settings)
        trainer.train_model(settings['total_epochs'], results_save_dir=save_dir, compute_max_lr=settings['autotune_lr'])
    for dataset_name in dataset_args:
        if len(seed) == 0:
            run_(None)
        else:
            for seed_ in seed:
                run_(seed_)

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
        train_classifier(args.dataset, settings, seed=args.seed, device=args.device)

if __name__ == '__main__':
    main()