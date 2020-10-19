import torch
import torchvision.transforms as transforms

from .verification import evaluate

import numpy as np
import bcolz
import io
import os
import random
from PIL import Image


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}/{}_list.npy'.format(path, name, name))
    return carray, issame


def create_pairs(path, number):
    # Get unique pairs and non pairs balanced
    print('CREATING PAIRS')
    pairs = []
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    for i in range(number):
        cnt = 0
        key1 = subfolders[random.randint(0, len(subfolders) - 1)]
        key2 = subfolders[random.randint(0, len(subfolders) - 1)]
        file1 = random.choice(os.listdir(key1))
        while key2 == key1:
            key2 = subfolders[random.randint(1, len(subfolders) - 1)]

        if i % 2 == 0:
            # get a non pair
            file2 = random.choice(os.listdir(key2))
            while [os.path.join(key1, file1), os.path.join(key2, file2), False] in pairs and cnt < 100:
                cnt += 1
                file2 = random.choice(os.listdir(key2))
            pairs.append([os.path.join(key1, file1), os.path.join(key2, file2), False])
        else:
            # get a true pair
            file2 = random.choice(os.listdir(key1))
            while [os.path.join(key1, file1), os.path.join(key1, file2), True] in pairs and cnt < 100:
                cnt += 1
                file2 = random.choice(os.listdir(key1))
            pairs.append([os.path.join(key1, file1), os.path.join(key1, file2), True])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])])
    carray = []
    issame = []
    for i in range(len(pairs)):
        carray.append(np.asarray(transform(Image.open(pairs[i][0]))))
        carray.append(np.asarray(transform(Image.open(pairs[i][1]))))
        issame.append(pairs[i][2])

    carray = np.array(carray)
    issame = np.array(issame)

    return carray, issame


def get_val_data(path, dataset_type):
    if dataset_type in ['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30', 'vgg2_fp']:
        data, pairs = get_val_pair(path, dataset_type)
    else:
        path = os.path.join(path, dataset_type)
        data, pairs = create_pairs(path, 2000)
    return data, pairs


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

    return paras_only_bn, paras_wo_bn


def separate_mobilenet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


# Horizontal flip of images
hflip = transforms.Compose([
    de_preprocess,
    transforms.ToPILImage(),
    transforms.functional.hflip,
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


# Center crop of images
ccrop = transforms.Compose([
    de_preprocess,
    transforms.ToPILImage(),
    transforms.Resize([128, 128]),  # smaller side resized
    transforms.CenterCrop([112, 112]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def ccrop_batch(imgs_tensor):
    ccropped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs


def perform_val(multi_gpu, device, embedding_size, batch_size, backbone, carray, issame, nrof_folds=10, tta=True):
    if multi_gpu:
        backbone = backbone.module  # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval()  # switch to evaluation mode

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.to(device))).cpu()
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                embeddings[idx:] = l2_norm(backbone(ccropped.to(device))).cpu()

    tpr, fpr, acc, best_thresholds = evaluate(embeddings, issame, nrof_folds)

    return acc.mean(), best_thresholds.mean()


def buffer_val(writer, db_name, acc, best_threshold, epoch):
    writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
    writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
