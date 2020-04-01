from template_model_functional import Network
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

    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of ckpt_save_freq')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')

    parser.add_argument('--img_height', type=int, default=256, help='The height size of image')
    parser.add_argument('--img_width', type=int, default=256, help='The width size of image')
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

    automatic_gpu_usage()

    model = Network(args)

    # build graph
    model.build_model()


    if args.phase == 'train' :
        model.train()
        print(" [*] Training finished!")

    else :
        model.test()
        print(" [*] Test finished!")



if __name__ == '__main__':
    main()