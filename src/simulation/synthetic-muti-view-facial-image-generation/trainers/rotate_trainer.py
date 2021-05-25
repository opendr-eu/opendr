import torch
from models.networks.sync_batchnorm import DataParallelWithCallback
from models import create_model
from collections import OrderedDict


class RotateTrainer(object):
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = create_model(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids,
                                                          output_device=opt.gpu_ids[-1],
                                                          chunk_size=opt.chunk_size)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model
        # self.Render = networks.Render(opt, render_size=opt.crop_size)
        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model.forward(data=data, mode='generator')
        if not self.opt.train_rotate:
            with torch.no_grad():
                g_rotate_losses, generated_rotate = self.pix2pix_model.forward(data=data, mode='generator_rotated')

        else:
            g_rotate_losses, generated_rotate = self.pix2pix_model.forward(data=data, mode='generator_rotated')
            g_losses['GAN_rotate'] = g_rotate_losses['GAN']
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        # g_rotate_loss = sum(g_rotate_losses.values()).mean()
        # g_rotate_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        # self.g_rotate_losses = g_rotate_losses
        self.generated = generated
        self.generated_rotate = generated_rotate

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model.forward(data=data, mode='discriminator')
        if self.opt.train_rotate:
            d_rotated_losses = self.pix2pix_model.forward(data=data, mode='discriminator_rotated')
            d_losses['D_rotate_Fake'] = d_rotated_losses['D_Fake']
            d_losses['D_rotate_real'] = d_rotated_losses['D_real']
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_generated(self):
        return self.generated

    def get_latest_generated_rotate(self):
        return self.generated_rotate

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_current_visuals(self, data):
        return OrderedDict([('input_mesh', data['mesh']),
                            ('input_rotated_mesh', data['rotated_mesh']),
                               ('synthesized_image', self.get_latest_generated()),
                               ('synthesized_rotated_image', self.get_latest_generated_rotate()),
                               ('real_image', data['image'])])

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
