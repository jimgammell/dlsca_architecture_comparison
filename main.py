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

def train_classifier(dataset_names, model_name_):
    for dataset_name in dataset_names:
        for model_name in ['ASCAD-CNN', 'ResNet']:
            for l1_weight_penalty in [1e-7, 1e-5, 1e-3, 1e-1, 1e1]: #weight_decay in [1e-5, 1e-3, 1e-1]:
                try:
                    print('Training model with weight decay {}...'.format(l1_weight_penalty))
                    save_dir = config.results_subdir('test', 'weight_decay_sweep', 'value_%e__model_%s'%(l1_weight_penalty, model_name))
                    trainer = train.classifier.ClassifierTrainer(dataset_name, model_name, l1_weight_penalty=l1_weight_penalty)
                    #trainer.optimizer_kwargs['weight_decay'] = weight_decay
                    trainer.train_model(100, results_save_dir=save_dir)
                    print()
                except:
                    pass

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
        '--architecture', choices=models.get_available_models(), default=models.get_available_models()[0],
        help='Model architecture to use for this trial. Valid options: \'{}\'.'.format('\', \''.join(models.get_available_models()))
    )
    parser.add_argument(
        '--download', default=False, action='store_true',
        help='Download the specified datasets.'
    )
    parser.add_argument(
        '--train-classifier', default=False, action='store_true',
        help='Train the specified combinations of datasets and models.'
    )
    parser.add_argument(
        '-f', '--force', default=False, action='store_true',
        help='Force the command to take effect.'
    )
    args = parser.parse_args()
    
    get_datasets(args)
    if args.download:
        download_dataset(args.dataset, force=args.force)
    if args.run_tests:
        datasets.test.test_datasets()
    if args.train_classifier:
        train_classifier(args.dataset, args.architecture)

if __name__ == '__main__':
    main()