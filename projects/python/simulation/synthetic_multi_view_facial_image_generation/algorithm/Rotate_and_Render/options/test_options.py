from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--list_num', type=int, default=0, help='list num')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--list_start', type=int, default=0, help='which num in the list to start')
        parser.add_argument('--list_end', type=int, default=10, help='how many test images to run')
        # parser.add_argument('--save_path', type=str, default='./results/', help='where to save data')
        parser.add_argument('--names', type=str, default='rs_model', help='dataset')
        parser.add_argument('--multi_gpu', action='store_false', help='whether to use multi gpus')
        parser.add_argument('--align', action='store_false', help='whether to save align')
        # parser.add_argument('--yaw_poses', type=str, default='30,40', nargs='+',
        # help='yaw poses list during testing')
        # parser.add_argument('--pitch_poses', type=str, default='10,20', nargs='+',
        # help='pitch poses list during testing')
        parser.add_argument('--posesrandom', action='store_true', help='whether to random the poses')

        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
