from opendr.simulation.human_model_generation.utilities.PIFu.lib.model import ResBlkPIFuNet, HGPIFuNet


def config_vanilla_parameters(opt):
    # Network configuration
    opt.batch_size = 1
    opt.mlp_dim = [257, 1024, 512, 256, 128, 1]
    opt.mlp_dim_color = [513, 1024, 512, 256, 128, 3]
    opt.num_stack = 4
    opt.num_hourglass = 2
    opt.resolution = 256
    opt.hg_down = 'ave_pool'
    opt.norm = 'group'
    opt.norm_color = 'group'
    opt.projection_mode = 'orthogonal'
    return opt


def config_nets(opt, device):
    netG = HGPIFuNet(opt, opt.projection_mode).to(device=device)
    netC = ResBlkPIFuNet(opt).to(device=device)
    return netG, netC
