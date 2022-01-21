import importlib
import torch.utils.data
from .base_dataset import BaseDataset
import math
from torch.utils.data.sampler import Sampler


class dataset_info():
    def __init__(self):
        self.prefix = [
            './algorithm/DDFA/example/Images',
            'PREFIX-TO-YOUR-DATASET'
        ]
        self.file_list = [
            './algorithm/DDFA/example/file_list.txt',
            'YOUE-FILE-LIST.txt'
        ]

        self.land_mark_list = [
            './algorithm/DDFA/example/realign_lmk',
            'LANDMARKS-OF-FACES-IN-YOUR-DATASET'
        ]

        self.params_dir = [
            './algorithm/DDFA/results',
            '3DFITTING-RESULTS-HOME-DIR'
        ]
        self.dataset_names = {'example': 0, 'YOUR-DATASET': 1}
        self.folder_level = [1, 2]

    def get_dataset(self, opt):
        dataset = opt.dataset.split(',')
        dataset_list = [self.dataset_names[dataset[i].lower()] for i in range(len(dataset))]

        return dataset_list


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "algorithm.Rotate_and_Render.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain
    )
    return dataloader


class MySampler(Sampler):

    def __init__(self, opt, dataset, render_thread=None, rank=None, round_up=True):
        self.dataset = dataset
        self.opt = opt
        self.render_thread = render_thread
        self.rank = rank
        self.round_up = round_up
        self.epoch = 0

        self.common_num = self.opt.batchSize * self.render_thread
        if self.round_up:
            self.total_size = int(math.ceil(len(self.dataset) * 1.0 / self.common_num)) * self.common_num
        else:
            self.total_size = len(self.dataset)
        self.num_samples = int(math.ceil(self.total_size / self.render_thread))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.opt.isTrain:
            indices = list(torch.randperm(len(self.dataset), generator=g))
        else:
            indices = list(torch.arange(len(self.dataset)))

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size, 'indices {} != total_size {}'.format(len(indices), self.total_size)

        # subsample
        # offset = self.num_samples * self.rank
        # indices = indices[offset:offset + self.num_samples]
        indices = indices[self.rank::self.render_thread]
        if self.round_up or (not self.round_up and self.rank == 0):
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def create_dataloader_test(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    samplers = [MySampler(opt, instance, render_thread=opt.render_thread, rank=i, round_up=opt.isTrain) for i in
                range(opt.render_thread)]
    dataloaders = [
        torch.utils.data.DataLoader(
            instance,
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=int(opt.nThreads),
            sampler=samplers[i],
            drop_last=opt.isTrain,
        )
        for i in range(opt.render_thread)
    ]
    return dataloaders
