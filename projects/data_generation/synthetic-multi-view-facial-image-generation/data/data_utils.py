import traceback
from torch.multiprocessing import Process
import numpy as np

import os
import torch


def get_input(data, render):
    real_image = data['image']
    input_semantics, rotated_mesh, orig_landmarks, rotate_landmarks,  rendered_images_erode, original_angles, \
        Rd_a, rendered_images_rotate_artifacts = render.rotate_render(data['param_path'], real_image, data['M'])
    output = {}
    real_image = real_image * 2 - 1
    input_semantics = input_semantics * 2 - 1
    rotated_mesh = rotated_mesh * 2 - 1
    rendered_images_erode = rendered_images_erode * 2 - 1
    Rd_a = Rd_a * 2 - 1
    rendered_images_rotate_artifacts = rendered_images_rotate_artifacts * 2 - 1
    output['image'] = real_image.cpu()
    output['rendered_images_erode'] = rendered_images_erode.cpu()
    output['mesh'] = input_semantics.cpu()
    output['rotated_mesh'] = rotated_mesh.cpu()
    output['Rd_a'] = Rd_a.cpu()
    output['orig_landmarks'] = orig_landmarks.cpu()
    output['rotated_landmarks'] = rotate_landmarks.cpu()
    output['original_angles'] = original_angles.cpu()
    output['rendered_images_rotate_artifacts'] = rendered_images_rotate_artifacts.cpu()
    output['path'] = data['path']
    return output


def get_test_input(data, render):
    real_image = data['image']
    rotated_mesh, rotate_landmarks, original_angles \
        = render.rotate_render(data['param_path'], real_image, data['M'])
    output = {}
    real_image = real_image * 2 - 1
    rotated_mesh = rotated_mesh * 2 - 1
    output['image'] = real_image.cpu()
    output['rotated_mesh'] = rotated_mesh.cpu()
    output['rotated_landmarks'] = rotate_landmarks.cpu()
    output['original_angles'] = original_angles.cpu()
    output['path'] = data['path']
    return output


def get_multipose_test_input(data, render, yaw_poses, pitch_poses):
    real_image = data['image']
    # num_poses = len(yaw_poses) + len(pitch_poses)
    rotated_meshs = []
    rotated_landmarks_list = []
    original_angles_list = []
    rotated_landmarks_list_106 = []
    paths = []
    real_images = []
    pose_list = []
    for i in range(2):
        prefix = 'yaw' if i == 0 else 'pitch'
        poses = yaw_poses if i == 0 else pitch_poses
        for pose in poses:
            if i == 0:
                rotated_mesh, rotate_landmarks, original_angles, rotate_landmarks_106 \
                    = render.rotate_render(data['param_path'], real_image, data['M'], yaw_pose=pose)
            else:
                rotated_mesh, rotate_landmarks, original_angles, rotate_landmarks_106 \
                    = render.rotate_render(data['param_path'], real_image, data['M'], pitch_pose=pose)
            rotated_meshs.append(rotated_mesh)
            rotated_landmarks_list.append(rotate_landmarks)
            rotated_landmarks_list_106.append(rotate_landmarks_106)
            original_angles_list.append(original_angles)
            paths += data['path']
            pose_list += ['{}_{}'.format(prefix, pose) for i in range(len(data['path']))]
            real_images.append(real_image)
    rotated_meshs = torch.cat(rotated_meshs, 0)
    rotated_landmarks_list = torch.cat(rotated_landmarks_list, 0)
    rotated_landmarks_list_106 = torch.cat(rotated_landmarks_list_106, 0)
    original_angles_list = torch.cat(original_angles_list, 0)
    output = {}
    real_image = real_image * 2 - 1
    rotated_meshs = rotated_meshs * 2 - 1
    output['image'] = real_image.cpu()
    output['rotated_mesh'] = rotated_meshs.cpu()
    output['rotated_landmarks'] = rotated_landmarks_list.cpu()
    output['rotated_landmarks_106'] = rotated_landmarks_list_106.cpu()
    output['original_angles'] = original_angles_list.cpu()
    output['path'] = paths
    output['pose_list'] = pose_list
    return output


class data_prefetcher():
    def __init__(self, loader, opt, render_layer):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.opt = opt
        self.render_layer = render_layer
        self.preload()

    def preload(self):
        try:
            data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        if self.opt.isTrain:
            self.next_input = get_input(data, self.render_layer)
        elif self.opt.yaw_poses is None and self.opt.pitch_poses is None:
            self.next_input = get_test_input(data, self.render_layer)
        else:
            if self.opt.yaw_poses is not None:
                if self.opt.posesrandom:
                    self.opt.yaw_poses = [round(np.random.uniform(-0.5, 0.5, 1)[0], 2) for k in
                                          range(len(self.opt.yaw_poses))]
            else:
                self.opt.yaw_poses = []

            if self.opt.pitch_poses is not None:
                if self.opt.posesrandom:
                    self.opt.pitch_poses = [round(np.random.uniform(-0.5, 0.5, 1)[0], 2) for k in
                                            range(len(self.opt.pitch_poses))]
            else:
                self.opt.pitch_poses = []

            self.next_input = get_multipose_test_input(data, self.render_layer, self.opt.yaw_poses,
                                                       self.opt.pitch_poses)
        with torch.cuda.stream(self.stream):
            for k, v in self.next_input.items():
                if type(v) == torch.Tensor:
                    self.next_input[k] = v.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input

        if input is not None:
            for k in input.keys():
                if type(input[k]) == torch.Tensor:
                    input[k].record_stream(torch.cuda.current_stream())
        self.preload()
        return input


def prefetch_data(queue, dataloader, iter_counter, opt, render_layer):
    print("start prefetching data...")
    np.random.seed(os.getpid())
    for epoch in iter_counter.training_epochs():
        prefetcher = data_prefetcher(dataloader, opt, render_layer)
        input = prefetcher.next()
        while input is not None:
            try:
                queue.put(input)
            except Exception as e:
                traceback.print_exc()
                raise e
            input = prefetcher.next()


def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        for k, v in data.items():
            data[k] = v.pin_memory()

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return


def init_parallel_jobs(queue, dataloader, iter_counter, opt, render_layer):
    if isinstance(dataloader, list):
        tasks = [Process(target=prefetch_data, args=(queue, dataloader[i], iter_counter, opt, render_layer[i])) for i in
                 range(opt.render_thread)]
    else:
        tasks = [Process(target=prefetch_data, args=(queue, dataloader, iter_counter, opt, render_layer)) for i in
                 range(opt.render_thread)]
    # task.daemon = True
    for task in tasks:
        task.start()

    return tasks
