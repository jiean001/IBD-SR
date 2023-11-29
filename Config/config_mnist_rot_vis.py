import argparse


def dataset_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='Data')
    parser.add_argument('--dataset', default='mnist-rot-vis')

    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--is_RGB', type=bool, default=False)
    parser.add_argument('--shape_data', type=int, default=(1, 28, 28))
    parser.add_argument('--num_target_class', type=int, default=10)
    parser.add_argument('--num_sensitive_class', type=int, default=5)
    opt = parser.parse_known_args()[0]

    return opt


def model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel_hidden_encoder', type=int, default=64)
    parser.add_argument('--channel_hidden_decoder', type=int, default=(256, 128))

    parser.add_argument('--dim_hidden_classifier', type=int, default=128)
    parser.add_argument('--dim_hidden_discriminator', type=int, default=128)

    parser.add_argument('--dim_embedding_clean', type=int, default=10)
    parser.add_argument('--dim_embedding_noise', type=int, default=20)

    parser.add_argument('--reconstruction_weight', type=float, default=0.01)
    parser.add_argument('--pairwise_kl_clean_weight', type=float, default=0.075)
    parser.add_argument('--pairwise_kl_noise_weight', type=float, default=0.01)

    parser.add_argument('--sparsity_clean', type=float, default=0.1)
    parser.add_argument('--sparsity_noise', type=float, default=0.1)

    parser.add_argument('--sparse_kl_weight_clean', type=float, default=0.1)
    parser.add_argument('--sparse_kl_weight_noise', type=float, default=0.01)

    # opt = parser.parse_args()
    opt = parser.parse_known_args()[0]
    return opt


def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='type of model')
    parser.add_argument('--seed', type=int, default=0, help='learning rate')

    parser.add_argument('--model', type=str, default='resnet34', help='type of model')
    parser.add_argument('--max_epoch', type=int, default=60, help='number of iterations to train for')
    parser.add_argument('--num_samples', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    # opt = parser.parse_args()
    opt = parser.parse_known_args()[0]

    return opt
