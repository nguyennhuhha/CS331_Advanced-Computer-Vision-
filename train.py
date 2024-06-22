import argparse
import os
from models.TripletEmbedding import TripletNet


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parase_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--photo_root', type=str, default='dataset/photo_train', help='Training photo root')
    parser.add_argument('--sketch_root', type=str, default='dataset/sketch_train',
                        help='Training sketch root')
<<<<<<< HEAD
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch (default :16')
    parser.add_argument('--device', type=str, default='cuda', help='The cuda device to be used (default: 0)') #'cuda' ->'0'
    parser.add_argument('--epochs', type=int, default=2, help='The number of epochs to run (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate of the model')
=======
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch (default :128')
    parser.add_argument('--device', type=str, default='0', help='The cuda device to be used (default: 0)')
    parser.add_argument('--epochs', type=int, default=1, help='The number of epochs to run (default: 5)')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate of the model')
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658
    
    parser.add_argument('--test', type=str2bool, nargs='?', default=True)
    parser.add_argument('--test_f', type=int, default=5, help='The frequency of testing (default: 5)')
    # parser.add_argument('--photo_test', type=str, default='dataset/photo_test', help='Testing photo root')
    # parser.add_argument('--sketch_test', type=str, default='dataset/sketch_test',
    #                     help='Testing sketch root')

    # parser.add_argument('--save_model', type=str2bool, nargs='?', default=True)
    parser.add_argument('--save_dir', type=str, default='model/resnet50',
                        help='The folder to save the model status')
    
    # I/O
    # parser.add_argument('--log_interval', type=int, default=100, metavar='N', 
    #                     help='How many batches to wait before logging training status')

    # parser.add_argument('--vis', type=str2bool, nargs='?', default=True, help='Whether to visualize')
    # parser.add_argument('--env', type=str, default='caffe2torch_tripletloss', help='The visualization environment')

    # parser.add_argument('--fine_tune', type=str2bool, nargs='?', default=False, help='Whether to fine tune')
    # parser.add_argument('--model_root', type=str, default=None, help='The model status files\'s root')

    parser.add_argument('--margin', type=float, default=0.3, help='The margin of the triplet loss')
    parser.add_argument('--p', type=int, default=2, help='The p of the triplet loss')

    parser.add_argument('--net', type=str, default='resnet50', help='The model to be used (vgg16, resnet50)')
    parser.add_argument('--cat', type=str2bool, nargs='?', default=True, help='Whether to use category loss')

    return check_args(parser.parse_args())


def check_args(args):
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_dir = args.save_dir  # Thư mục lưu model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


    try:
        assert args.epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    try:
        assert args.net in ['vgg16', 'resnet50']
    except:
        print('net model must be chose from [\'vgg16\', \'resnet50\']')

    # if args.fine_tune:
    #     try:
    #         assert not args.model_root

    #     except:
    #         print('you should specify the model status file')

    return args


def main():

    args = parase_args()
    if args is None:
        exit()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    tripletNet = TripletNet(args)
    tripletNet.train()


if __name__ == '__main__':

    main()