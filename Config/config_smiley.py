import argparse


def model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_hidden_encoder', type=int, default=3000)
    parser.add_argument('--dim_hidden_decoder', type=int, default=3000)
    parser.add_argument('--dim_embedding', type=int, default=18)
    parser.add_argument('--num_pseudo_data', type=int, default=20)
    parser.add_argument('--sigmoid_temperature', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--need_bn', type=bool, default=False)
    opt = parser.parse_args()

    return opt

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=float, default=1.0, help='type of model')

    parser.add_argument('--model', type=str, default='resnet34', help='type of model')
    parser.add_argument('--max_iter', type=int, default=200000, help='number of iterations to train for')
    parser.add_argument('--warm_up_start_iter', type=int, default=50000, help='number of iterations to train for')
    parser.add_argument('--warm_up_end_iter', type=int, default=60000, help='number of iterations to train for')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--train_step', type=int, default=1)
    parser.add_argument('--train_dis_step', type=int, default=0)
    parser.add_argument('--dis_loss_type', type=str, default='average_entropy',
                        choices=['maximum_entropy', 'average_entropy', 'maximum_K_class'])

    parser.add_argument('--save_dir', type=str, default='logs', help='Directory name to save the checkpoints')
    parser.add_argument('--load_dir', type=str, default='', help='Directory name to load checkpoints')
    parser.add_argument('--load_pth', type=str, default='', help='pth name to load checkpoints')
    parser.add_argument('--pretrained', type=str, default='', help='Directory/pth name to load pretrained checkpoints')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise ratio')
    parser.add_argument('--noise_type', type=str, default='val_split_symm_exc', help='Noise Type')
    parser.add_argument('--ln_neg', type=int, default=1,
                        help='number of negative labels on single image for training (ex. 110 for cifar100)')
    parser.add_argument('--cut', type=float, default=0.5, help='threshold value')
    opt = parser.parse_args()

    return opt


def dataset_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='Data')
    parser.add_argument('--dataset', default='Smiley')

    parser.add_argument('--train_batch_size', type=int, default=500)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--is_RGB', type=bool, default=False)
    parser.add_argument('--dim_data', type=int, default=1024)
    parser.add_argument('--num_target_class', type=int, default=38)
    parser.add_argument('--num_sensitive_class', type=int, default=5)
    opt = parser.parse_args()

    return opt
