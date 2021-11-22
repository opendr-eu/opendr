import torch.nn as nn
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from .base_network import BaseNetwork
from .sync_batchnorm import SynchronizedBatchNorm2d
from .normalization import get_nonspade_norm_layer
from ...util import util
import torch
from torch.utils.checkpoint import checkpoint


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super(MultiscaleDiscriminator, self).__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt):

        super(NLayerDiscriminator, self).__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        if opt.D_input == "concat":
            input_nc = opt.label_nc + opt.output_nc
            if opt.contain_dontcare_label:
                input_nc += 1
            if not opt.no_instance:
                input_nc += 1
        else:
            input_nc = 3
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():

            # intermediate_output = checkpoint(submodel, results[-1])
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class ImageDiscriminator(BaseNetwork):
    """Defines a PatchGAN discriminator"""
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ImageDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        if opt.D_input == "concat":
            input_nc = opt.label_nc + opt.output_nc
        else:
            input_nc = opt.label_nc
        ndf = 64
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class ProjectionDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ndf
        input_nc = 3
        label_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)

        if opt.norm_D.startswith('spectral'):
            use_spectral = True
        else:
            use_spectral = False

        self.enc1 = nn.Sequential(norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1)),
                                  nn.LeakyReLU(0.2, True))

        self.relu = nn.LeakyReLU(0.2, True)
        for i in range(2, 6):
            nf_prev = nf
            nf = min(nf * 2, opt.ndf * 8)
            enconv = nn.Conv2d(nf_prev, nf, kernel_size=3, stride=2, padding=1)
            latconv = nn.Conv2d(nf, opt.ndf * 4, kernel_size=3, stride=2, padding=1)
            if use_spectral:
                enconv = spectral_norm(enconv)
                latconv = spectral_norm(latconv)

            self.add_module('enc' + str(i), enconv)
            self.add_module('lat' + str(i), latconv)
            self.add_module('norm_enc' + str(i), self.get_norm(enconv))
            self.add_module('norm_lat' + str(i), self.get_norm(latconv))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        for i in range(2, 5):
            finalconv = nn.Conv2d(opt.ndf * 4, opt.ndf, kernel_size=3, padding=1)
            if use_spectral:
                finalconv = spectral_norm(finalconv)
            self.add_module('final' + str(i), finalconv)
            self.add_module('norm_final' + str(i), self.get_norm(finalconv))

        # shared True/False layer
        self.tf = nn.Conv2d(opt.ndf, 1, kernel_size=1)
        self.seg = nn.Conv2d(opt.ndf, opt.ndf, kernel_size=1)  # do not need softmax
        self.embedding = nn.Conv2d(label_nc, opt.ndf, kernel_size=1)

    def forward(self, input, segmap):
        # feat11 = self.enc1(input)
        feat11 = checkpoint(self.enc1, input)
        feat12 = self.relu(self.norm_enc2(self.enc2(feat11)))
        feat13 = self.relu(self.norm_enc3(self.enc3(feat12)))
        feat14 = self.relu(self.norm_enc4(self.enc4(feat13)))
        feat15 = self.relu(self.norm_enc5(self.enc5(feat14)))
        feat25 = self.relu(self.norm_lat5(self.lat5(feat15)))
        feat24 = self.up(feat25) + self.relu(self.norm_lat4(self.lat4(feat14)))
        feat23 = self.up(feat24) + self.relu(self.norm_lat3(self.lat3(feat13)))
        feat22 = self.up(feat23) + self.relu(self.norm_lat2(self.lat2(feat12)))
        feat32 = self.norm_final2(self.final2(feat22))
        feat33 = self.norm_final3(self.final3(feat23))
        feat34 = self.norm_final4(self.final4(feat24))

        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)

        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        if self.opt.gan_matching_feats == 'basic':
            feats = [feat12, feat13, feat14, feat15]
        elif self.opt.gan_matching_feats == 'more':
            feats = [feat12, feat13, feat14, feat15, feat25, feat24, feat23, feat22]
        elif self.opt.gan_matching_feats == 'chosen':
            feats = [feat11, feat12, feat13, feat14, feat15]
        else:
            feats = [feat12, feat13, feat14, feat15, feat25, feat24, feat23, feat22, feat32, feat33, feat34]

        # calculate segmentation loss
        # segmentation map embedding
        segemb = self.embedding(segmap)
        # downsample
        segemb2 = F.adaptive_avg_pool2d(segemb, seg2.size(-1))
        segemb3 = F.adaptive_avg_pool2d(segemb, seg3.size(-1))
        segemb4 = F.adaptive_avg_pool2d(segemb, seg4.size(-1))

        # product
        pred2 += torch.mul(segemb2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segemb3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segemb4, seg4).sum(dim=1, keepdim=True)

        results = [pred2, pred3, pred4]

        return feats, results

    def get_out_channel(self, layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def get_norm(self, layer):
        norm_type = self.opt.norm_D
        if norm_type.startswith('spectral'):
            subnorm_type = norm_type[len('spectral'):]
        else:
            subnorm_type = norm_type

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(self.get_out_channel(layer), affine=True)
        elif subnorm_type == 'syncbatch':
            norm_layer = SynchronizedBatchNorm2d(self.get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(self.get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return norm_layer
