from trainers.rotate_trainer import RotateTrainer
from models.rotate_model import RotateModel
from collections import OrderedDict
import torch


class RotateSPADETrainer(RotateTrainer):
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        super(RotateSPADETrainer, self).__init__(opt)

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated, self.input_mesh = self.pix2pix_model.forward(data=data, mode='generator')
        if not self.opt.train_rotate:
            with torch.no_grad():
                g_rotate_losses, generated_rotate, self.input_rotated_mesh = self.pix2pix_model.forward(data=data, mode='generator_rotated')

        else:
            g_rotate_losses, generated_rotate, self.input_rotated_mesh = self.pix2pix_model.forward(data=data, mode='generator_rotated')
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


    def get_current_visuals(self, data):
        return OrderedDict([('input_mesh', self.input_mesh),
                            ('input_rotated_mesh', self.input_rotated_mesh),
                               ('synthesized_image', self.get_latest_generated()),
                               ('synthesized_rotated_image', self.get_latest_generated_rotate()),
                               ('input_images_erode', data['rendered_images_erode']),
                               ('rendered_images_rotate_artifacts', data['rendered_images_rotate_artifacts']),
                               ('Rd_a', data['Rd_a']),
                               ('real_image', data['image'])])

