import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from trainers import create_trainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from torch.multiprocessing import Process, Queue, Pool
from data.data_utils import init_parallel_jobs

from models.networks.render import Render


if __name__ == '__main__':

    # parse options
    opt = TrainOptions().parse()

    # print options to help debugging
    print(' '.join(sys.argv))

    # load the dataset
    dataloader = data.create_dataloader_test(opt)

    # create tool for counting iterations


    if type(dataloader) == list:
        data_loader_size = len(dataloader[0]) * opt.render_thread
    else:
        data_loader_size = len(dataloader)
    iter_counter = IterationCounter(opt, data_loader_size)

    ngpus = opt.device_count


    training_queue = Queue(10)

    # render layers

    render_gpu_ids = list(range(ngpus - opt.render_thread, ngpus))
    render_layer_list = []
    for gpu in render_gpu_ids:
        opt.gpu_ids = gpu
        render_layer = Render(opt)
        render_layer_list.append(render_layer)

    training_tasks = init_parallel_jobs(training_queue, dataloader, iter_counter, opt, render_layer_list)

    opt.gpu_ids = list(range(0, ngpus - opt.render_thread))
    print('Training gpu ', opt.gpu_ids)
    # create trainer for our model
    trainer = create_trainer(opt)
    # create tool for visualization
    visualizer = Visualizer(opt)

    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(range(data_loader_size), start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            # data = trainer.get_input(data_i)
            data = training_queue.get(block=True)
            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data)

            # train discriminator
            trainer.run_discriminator_one_step(data)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                visuals = trainer.get_current_visuals(data)
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
           epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

    for training_task in training_tasks:
        training_task.terminate()
    print('Training was successfully finished.')
