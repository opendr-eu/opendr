import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from models.networks.sync_batchnorm import DataParallelWithCallback
import sys
import numpy as np
import os
import data
from util.iter_counter import IterationCounter
from options.test_options import TestOptions
from models.test_model import TestModel
from util.visualizer import Visualizer
from util import html, util
from torch.multiprocessing import Process, Queue, Pool
from data.data_utils import init_parallel_jobs
from skimage import transform as trans
import cv2
import time
import torch
from models.networks.test_render import TestRender



def create_path(a_path, b_path):
    name_id_path = os.path.join(a_path, b_path)
    if not os.path.exists(name_id_path):
        os.makedirs(name_id_path)
    return name_id_path


def create_paths(save_path, img_path, foldername='orig', folderlevel=2):
    save_rotated_path_name = create_path(save_path, foldername)

    path_split = img_path.split('/')
    rotated_file_savepath = save_rotated_path_name
    for level in range(len(path_split) - folderlevel, len(path_split)):
        file_name = path_split[level]
        rotated_file_savepath = os.path.join(rotated_file_savepath, file_name)
    return rotated_file_savepath

def affine_align(img, landmark=None, **kwargs):
    M = None
    src = np.array([
     [38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041] ], dtype=np.float32 )
    src=src * 224 / 112

    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    warped = cv2.warpAffine(img, M, (224, 224), borderValue = 0.0)
    return warped

def landmark_68_to_5(t68):
    le = t68[36:42, :].mean(axis=0, keepdims=True)
    re = t68[42:48, :].mean(axis=0, keepdims=True)
    no = t68[31:32, :]
    lm = t68[48:49, :]
    rm = t68[54:55, :]
    t5 = np.concatenate([le, re, no, lm, rm], axis=0)
    t5 = t5.reshape(10)
    return t5


def save_img(img, save_path):
    image_numpy = util.tensor2im(img)
    util.save_image(image_numpy, save_path, create_dir=True)
    return image_numpy


if __name__ == '__main__':


    opt = TestOptions().parse()

    data_info = data.dataset_info()
    datanum = data_info.get_dataset(opt)
    folderlevel = data_info.folder_level[datanum]

    dataloaders = data.create_dataloader_test(opt)

    visualizer = Visualizer(opt)
    iter_counter = IterationCounter(opt, len(dataloaders[0]) * opt.render_thread)
    # create a webpage that summarizes the all results

    testing_queue = Queue(10)

    ngpus = opt.device_count

    render_gpu_ids = list(range(ngpus - opt.render_thread, ngpus))
    render_layer_list = []
    for gpu in render_gpu_ids:
        opt.gpu_ids = gpu
        render_layer = TestRender(opt)
        render_layer_list.append(render_layer)

    opt.gpu_ids = list(range(0, ngpus - opt.render_thread))
    print('Testing gpu ', opt.gpu_ids)
    if opt.names is None:
        model = TestModel(opt)
        model.eval()
        model = torch.nn.DataParallel(model.cuda(),
                                      device_ids=opt.gpu_ids,
                                      output_device=opt.gpu_ids[-1],
                                      )
        models = [model]
        names = [opt.name]
        save_path = create_path(create_path(opt.save_path, opt.name), opt.dataset)
        save_paths = [save_path]
        f = [open(
                os.path.join(save_path, opt.dataset + str(opt.list_start) + str(opt.list_end) + '_rotate_lmk.txt'), 'w')]
    else:
        models = []
        names = []
        save_paths = []
        f = []
        for name in opt.names.split(','):
            opt.name = name
            model = TestModel(opt)
            model.eval()
            model = torch.nn.DataParallel(model.cuda(),
                                          device_ids=opt.gpu_ids,
                                          output_device=opt.gpu_ids[-1],
                                          )
            models.append(model)
            names.append(name)
            save_path = create_path(create_path(opt.save_path, opt.name), opt.dataset)
            save_paths.append(save_path)
            f_rotated = open(
                os.path.join(save_path, opt.dataset + str(opt.list_start) + str(opt.list_end) + '_rotate_lmk.txt'), 'w')
            f.append(f_rotated)

    test_tasks = init_parallel_jobs(testing_queue, dataloaders, iter_counter, opt, render_layer_list)
    # test
    landmarks = []

    process_num = opt.list_start
    first_time = time.time()
    try:
        for i, data_i in enumerate(range(len(dataloaders[0]) * opt.render_thread)):
            # if i * opt.batchSize >= opt.how_many:
            #     break
            # data = trainer.get_input(data_i)
            start_time = time.time()
            data = testing_queue.get(block=True)

            current_time = time.time()
            time_per_iter = (current_time - start_time) / opt.batchSize
            message = '(************* each image render time: %.3f *****************) ' % (time_per_iter)
            print(message)

            img_path = data['path']
            rotated_landmarks = data['rotated_landmarks'][:, :, :2].cpu().numpy().astype(np.float)


            generate_rotateds = []
            for model in models:
                generate_rotated = model.forward(data, mode='single')
                generate_rotateds.append(generate_rotated)

            for n, name in enumerate(names):
                opt.name = name
                for b in range(generate_rotateds[n].shape[0]):
                    # get 5 key points
                    rotated_keypoints = landmark_68_to_5(rotated_landmarks[b])
                    # get savepaths
                    rotated_file_savepath = create_paths(save_paths[n], img_path[b], folderlevel=folderlevel)

                    image_numpy = save_img(generate_rotateds[n][b], rotated_file_savepath)
                    rotated_keypoints_str = rotated_file_savepath + ' 1 ' + ' '.join([str(int(n)) for n in rotated_keypoints]) + '\n'
                    print('process image...' + rotated_file_savepath)
                    f[n].write(rotated_keypoints_str)

                    current_time = time.time()
                    if n == 0:
                        process_num += 1
                        print('processed num ' + str(process_num))
                    if opt.align:
                        aligned_file_savepath = create_paths(save_paths[n], img_path[b], 'aligned', folderlevel=folderlevel)
                        warped = affine_align(image_numpy, rotated_keypoints.reshape(5, 2))
                        util.save_image(warped, aligned_file_savepath, create_dir=True)

            current_time = time.time()
            time_per_iter = (current_time - start_time) / opt.batchSize
            message = '(************* each image time total: %.3f *****************) ' % (time_per_iter)
            print(message)

    except KeyboardInterrupt:
        print("Interrupted!")
        for fs in f:
            fs.close()
        pass

    except Exception as e:
        print(e)
        for fs in f:
            fs.close()

    else:
        print('finished')
        for fs in f:
            fs.close()





