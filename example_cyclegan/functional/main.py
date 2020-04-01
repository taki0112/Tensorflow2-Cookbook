from CycleGAN_functional import CycleGAN
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of CycleGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--dataset', type=str, default='horse2zebra', help='dataset_name')
    parser.add_argument('--augment_flag', type=bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')

    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_iter', type=int, default=500000, help='decay start iteration')

    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=100, help='The number of ckpt_save_freq')

    parser.add_argument('--gan_type', type=str, default='lsgan', help='[gan / lsgan / hinge]')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='weight of adversarial loss')
    parser.add_argument('--cyc_weight', type=float, default=10.0, help='weight of cycle loss')
    parser.add_argument('--identity_weight', type=float, default=5.0, help='weight of identity loss')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=6, help='The number of residual blocks')

    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.iteration >= 1
    except:
        print('number of iterations must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():

    args = parse_args()
    # automatic_gpu_usage()

    gan = CycleGAN(args)

    # build graph
    gan.build_model()


    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    else :
        gan.test()
        print(" [*] Test finished!")



if __name__ == '__main__':
    main()
