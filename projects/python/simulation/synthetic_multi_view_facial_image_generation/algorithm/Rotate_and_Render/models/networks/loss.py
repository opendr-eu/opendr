import torch
import torch.nn as nn
import torch.nn.functional as F
from .architecture import VGG19, VGGFace19


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, opt):
        super(VGGLoss, self).__init__()
        if opt.face_vgg:
            self.vgg = VGGFace19(opt).cuda()
        else:
            self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y, layer=0):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            if i >= layer:
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGGwithContrastiveLoss(VGGLoss):
    def __init__(self, opt):
        super(VGGwithContrastiveLoss, self).__init__(opt)
        self.closs = L2ContrastiveLoss(opt.l2_margin)

    def forward(self, x, y, layer=0):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            if i >= layer:
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

            if i == len(x_vgg) - 1:
                x_feature = x_vgg[i].view(x_vgg[i].size(0), -1)
                y_feature = y_vgg[i].view(y_vgg[i].size(0), -1)
                loss + self.closs(x_feature, y_feature.detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class L2ContrastiveLoss(nn.Module):
    """
    Compute L2 contrastive loss
    """

    def __init__(self, margin=1, max_violation=False):
        super(L2ContrastiveLoss, self).__init__()
        self.margin = margin

        self.sim = self.l2_sim

        self.max_violation = max_violation

    def forward(self, feature1, feature2):
        # compute image-sentence score matrix
        feature1 = self.l2_norm(feature1)
        feature2 = self.l2_norm(feature2)
        scores = self.sim(feature1, feature2)
        # diagonal = scores.diag().view(feature1.size(0), 1)
        diagonal_dist = scores.diag()
        # d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin - scores).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask.clone()
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]

        loss = (torch.sum(cost_s ** 2) + torch.sum(diagonal_dist ** 2)) / (2 * feature1.size(0))

        return loss

    def l2_norm(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        return x_norm

    def l2_sim(self, feature1, feature2):
        Feature = feature1.expand(feature1.size(0), feature1.size(0), feature1.size(1)).transpose(0, 1)
        return torch.norm(Feature - feature2, p=2, dim=2)
