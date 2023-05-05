import time
import argparse

import datasets
import train

def get_datasets(args):
    if 'all' in args.dataset:
        args.dataset = datasets.get_available_datasets()
    if any(x not in datasets.get_available_datasets() for x in args.dataset):
        raise argparse.ArgumentError(
            'Error running trials. The following datasets are not recognized: \'{}\''.format(
                '\', \''.join([x for x in args.dataset if x not in datasets.get_available_datasets()])
            ))

def download_dataset(dataset_args):
    for dataset_name in dataset_args:
        t0 = time.time()
        print('Downloading dataset \'{}\'...'.format(dataset_name))
        datasets.download_dataset(dataset_name, force=args.force)
        print('\tDone. Time taken: {} sec.'.format(time.time()-t0))

def train_model(dataset_args):
    print('Training models...')
    for dataset_name in dataset_args:
        train.train_model(dataset_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=datasets.get_available_datasets() + ['all'], nargs='+', default=[],
        help='Dataset to use for this trial. Valid options: \'{}\'. Pass \'all\' to download all valid options.'.format('\', \''.join(datasets.get_available_datasets()))
    )
    parser.add_argument(
        '--download', default=False, action='store_true',
        help='Download the specified datasets.'
    )
    parser.add_argument(
        '--train', default=False, action='store_true',
        help='Train the specified combinations of datasets and models.'
    )
    parser.add_argument(
        '-f', '--force', default=False, action='store_true',
        help='Force the command to take effect.'
    )
    args = parser.parse_args()
    
    get_datasets(args)
    if args.download:
        download_dataset(args.dataset)
    if args.train:
        train_model(args.dataset)

if __name__ == '__main__':
    main()