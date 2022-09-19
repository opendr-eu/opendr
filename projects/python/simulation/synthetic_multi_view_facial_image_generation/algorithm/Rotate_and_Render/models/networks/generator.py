import torch.nn as nn
from .base_network import BaseNetwork
from .normalization import get_nonspade_norm_layer
from .architecture import ResnetBlock as ResnetBlock
from .architecture import ResnetSPADEBlock
from torch.utils.checkpoint import checkpoint


class Interpolate(nn.Module):
    def __init__(self, scale_factor=2, size=None, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        if self.size is not None:
            x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        else:
            x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class RotateGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4,
                            help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9,
                            help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='spectralsyncbatch')
        return parser

    def __init__(self, opt):
        super(RotateGenerator, self).__init__()
        input_nc = 3

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)
        # initial conv
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                                         norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                                              kernel_size=opt.resnet_initial_kernel_size,
                                                              padding=0)),
                                         activation)
        # downsample
        downsample_model = []

        mult = 1
        for i in range(opt.resnet_n_downsample):
            downsample_model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                                      kernel_size=3, stride=2, padding=1)),
                                 activation]
            mult *= 2

        self.downsample_layers = nn.Sequential(*downsample_model)

        # resnet blocks
        resnet_model = []

        for i in range(opt.resnet_n_blocks):
            resnet_model += [ResnetBlock(opt.ngf * mult,
                                         norm_layer=norm_layer,
                                         activation=activation,
                                         kernel_size=opt.resnet_kernel_size)]

        self.resnet_layers = nn.Sequential(*resnet_model)

        # upsample

        upsample_model = []

        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            upsample_model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                             kernel_size=3, stride=2,
                                                             padding=1, output_padding=1)),
                               activation]
            mult = mult // 2

        self.upsample_layers = nn.Sequential(*upsample_model)

        # final output conv
        self.final_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                                         nn.Tanh())

    def forward(self, input, z=None):
        net = self.first_layer(input)
        net = self.downsample_layers(net)
        net = self.resnet_layers(net)
        net = self.upsample_layers(net)
        net = self.final_layer(net)
        return net


class RotateSPADEGenerator(RotateGenerator):
    def __init__(self, opt):
        super(RotateSPADEGenerator, self).__init__(opt)
        del self.resnet_layers
        self.resnet_n_blocks = opt.resnet_n_blocks
        mult = 1
        for i in range(opt.resnet_n_downsample):
            mult *= 2
        for i in range(opt.resnet_n_blocks):
            self.add_module('resnet_layers' + str(i), ResnetSPADEBlock(opt.ngf * mult, opt.semantic_nc))

    def forward(self, input, seg=None):
        # net = self.first_layer(input)
        net = checkpoint(self.first_layer, input)
        # net = self.downsample_layers(net)
        net = checkpoint(self.downsample_layers, net)
        for i in range(self.resnet_n_blocks):
            # net = self._modules['resnet_layers' + str(i)](net, seg)
            net = checkpoint(self._modules['resnet_layers' + str(i)], net, seg)
        # net = self.upsample_layers(net)
        net = checkpoint(self.upsample_layers, net)
        # net = self.final_layer(net)
        net = checkpoint(self.final_layer, net)
        return net
