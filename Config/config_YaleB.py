import argparse


def dataset_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='Data')
    parser.add_argument('--dataset', default='yaleb')

    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--shape_data', type=int, default=504)
    parser.add_argument('--num_target_class', type=int, default=38)
    parser.add_argument('--num_sensitive_class', type=int, default=5)
    # opt = parser.parse_args()
    opt = parser.parse_known_args()[0]
    return opt


def model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_embedding_clean', type=int, default=100)
    parser.add_argument('--dim_embedding_noise', type=int, default=100)

    parser.add_argument('--reconstruction_weight', type=float, default=1e-3)
    parser.add_argument('--pairwise_kl_clean_weight', type=float, default=0.2)
    parser.add_argument('--pairwise_kl_noise_weight', type=float, default=0.001)

    parser.add_argument('--sparsity_clean', type=float, default=0.1)
    parser.add_argument('--sparsity_noise', type=float, default=0.5)

    # 0.05
    parser.add_argument('--sparse_kl_weight_clean', type=float, default=0.05)
    parser.add_argument('--sparse_kl_weight_noise', type=float, default=0.05)

    # opt = parser.parse_args()
    opt = parser.parse_known_args()[0]
    return opt


def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=3, help='type of model')
    parser.add_argument('--seed', type=int, default=888, help='learning rate')

    parser.add_argument('--model', type=str, default='resnet34', help='type of model')
    parser.add_argument('--max_epoch', type=int, default=3500, help='number of iterations to train for')
    parser.add_argument('--num_samples', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # opt = parser.parse_args()
    opt = parser.parse_known_args()[0]

    return opt
