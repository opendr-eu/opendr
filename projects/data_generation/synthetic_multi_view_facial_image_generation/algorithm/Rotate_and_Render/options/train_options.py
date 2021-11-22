from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=100,
                            help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true',
                            help='if specified, use tensorboard logging. Requires tensorflow installed')
        parser.add_argument('--tensorboard', default=True,
                            help='if specified, use tensorboard logging. Requires tensorflow installed')
        parser.add_argument('--load_pretrain', type=str, default='',
                            help='load the pretrained model from the specified location')
        parser.add_argument('--train_rotate', action='store_true',
                            help='whether train rotated mesh')
        parser.add_argument('--lambda_rotate_D', type=float, default='0.1',
                            help='rotated D loss weight')
        parser.add_argument('--lambda_D', type=float, default='1',
                            help='D loss weight')
        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--noload_D', action='store_true', help='whether to load D when continue training')
        parser.add_argument('--large_pose', action='store_true', help='whether to use large pose training')
        parser.add_argument('--pose_noise', action='store_true', help='whether to use pose noise training')
        parser.add_argument('--load_separately', action='store_true',
                            help='whether to continue train by loading separate models')
        parser.add_argument('--niter', type=int, default=50,
                            help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is '
                                 'niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=1000,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1,
                            help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--D_input', type=str, default='single', help='(concat|single|hinge)')
        parser.add_argument('--gan_matching_feats', type=str, default='more', help='(concat|single|hinge)')
        parser.add_argument('--G_pretrain_path', type=str, default='./checkpoints/100_net_G.pth',
                            help='G pretrain path')
        parser.add_argument('--D_pretrain_path', type=str, default='', help='D pretrain path')
        parser.add_argument('--E_pretrain_path', type=str, default='', help='E pretrain path')
        parser.add_argument('--D_rotate_pretrain_path', type=str, default='', help='D_rotate pretrain path')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_image', type=float, default=1.0, help='weight for image reconstruction')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true',
                            help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true',
                            help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--face_vgg', action='store_true', help='if specified, use VGG feature matching loss')
        parser.add_argument('--vggface_checkpoint', type=str, default='', help='pth to vggface ckpt')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image|projection)')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        self.isTrain = True
        return parser
