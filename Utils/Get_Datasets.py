import argparse, os
from Utils.Dataset import Adult, ExtendedYaleB, MNIST_ROT, MNIST_DIL, MNIST_ROT_VIS
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_datasets(dataset, train_batch_size, test_batch_size, cuda=False, root='Data'):
    print(f'Loading {dataset} dataset...')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    if dataset == 'yaleb':
        Dataset = ExtendedYaleB
        dataset_path = os.path.join(root, 'yaleb')
        dataset = Dataset(dataset_path)
        train_loader = DataLoader(dataset.train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(dataset.test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        print('Done!\n')
        return train_loader, test_loader

    elif dataset == 'adult':
        Dataset = Adult
        dataset_path = os.path.join(root, 'adult')
        dataset = Dataset(dataset_path)
        train_loader = DataLoader(dataset.train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(dataset.val_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(dataset.test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        print('Done!\n')
        return train_loader, val_loader, test_loader

    elif dataset == 'mnist-rot':
        Dataset = MNIST_ROT
        dataset_path = os.path.join(root, 'mnist')
        dataset = Dataset(dataset_path)
        train_loader = DataLoader(dataset.train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(dataset.test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_55_loader = DataLoader(dataset.test_55_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_65_loader = DataLoader(dataset.test_65_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        print('Done!\n')
        return train_loader, test_loader, test_55_loader, test_65_loader

    elif dataset == 'mnist-dil':
        Dataset = MNIST_DIL
        dataset_path = os.path.join(root, 'mnist')
        dataset = Dataset(dataset_path)
        test_erode_2_loader = DataLoader(dataset.test_erode_2_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_dilate_2_loader = DataLoader(dataset.test_dilate_2_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_dilate_3_loader = DataLoader(dataset.test_dilate_3_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_dilate_4_loader = DataLoader(dataset.test_dilate_4_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        print('Done!\n')
        return test_erode_2_loader, test_dilate_2_loader, test_dilate_3_loader, test_dilate_4_loader

    elif dataset == 'mnist-rot-vis':
        Dataset = MNIST_ROT_VIS
        dataset_path = os.path.join(root, 'mnist')
        dataset = Dataset(dataset_path)
        test_loader = DataLoader(dataset.test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        print('Done!\n')
        return test_loader


    else:
        raise ValueError('Dataset not supported')

