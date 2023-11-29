model_name = 'fair_single_best.pt'
ema_model_name = 'fair_ema_best.pt'

tsne_model_name = '000%s_%s_tsne_yaleb_single_%03d_%s.png'
tsne_ema_model_name = '000%s_%s_tsne_yaleb_ema_%03d_%s.png'

crt_model_name = model_name
crt_tsne_name = tsne_model_name

root = '../Data'
root_dir = './results_image'

from base_utils.color_util import get_cmap_xkcd

import sys
sys.path.append('../')


import os
import torch
import random
import numpy as np
from torch.autograd import Variable
import Config.config_YaleB as config
from Utils.Get_Datasets import get_datasets
from Models.VIB_YaleB_whole_model import VariationalInformationBottleneck

train_args = config.train_args()
dataset_args = config.dataset_args()
model_args = config.model_args()
use_cuda = True

# 可以在这里修改GPU
train_args.gpu = 3

init_seed = train_args.seed
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)
random.seed(init_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(train_args.gpu)

train_loader, test_loader = get_datasets(
    dataset_args.dataset, dataset_args.train_batch_size, dataset_args.test_batch_size, root=root)

parameter = {
    "reconstruction_weight": model_args.reconstruction_weight,
    "pairwise_kl_clean_weight": model_args.pairwise_kl_clean_weight,
    "pairwise_kl_noise_weight": model_args.pairwise_kl_noise_weight,
    "sparse_kl_weight_clean": model_args.sparse_kl_weight_clean,
    "sparse_kl_weight_noise": model_args.sparse_kl_weight_noise,
    "sparsity_clean": model_args.sparsity_clean,
    "sparsity_noise": model_args.sparsity_noise,

    "num_sensitive_class": dataset_args.num_sensitive_class
}

print(parameter)

path_1 = '%.03f_%.03f_%.03f' % (
model_args.reconstruction_weight, model_args.pairwise_kl_clean_weight, model_args.pairwise_kl_noise_weight)
path_2 = '%.02f_%.02f_%.02f_%.02f' % (
model_args.sparse_kl_weight_clean, model_args.sparse_kl_weight_noise, model_args.sparsity_clean,
model_args.sparsity_noise)
save_model_dir = os.path.join('../saved_model/YaleB/', path_1, path_2, str(train_args.seed))
print(save_model_dir, os.path.exists(save_model_dir))
print(path_1)
print(path_2)

vib_model = VariationalInformationBottleneck(
    dataset_args.shape_data, dataset_args.num_target_class,
    model_args.dim_embedding_clean, model_args.dim_embedding_noise, parameter
)
if use_cuda:
    vib_model = vib_model.cuda()


def evaluation(epoch_index, test_dataloader, is_drawing=True):
    # 设置网络模型的模式
    vib_model.eval()

    valid_embedding_labels = []
    valid_embedding_sensitive_labels = []
    valid_embedding_clean_images = []
    valid_embedding_noise_images = []

    # or iter_index, data in enumerate(test_dataloader):
    for iter_index, (images, labels, sensitive_labels) in enumerate(test_dataloader):
        images = Variable(images.float())
        labels = Variable(labels.long())
        sensitive_labels = Variable(sensitive_labels.long())
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
            sensitive_labels = sensitive_labels.cuda()

        single_embedding, single_embedding_noise, single_classification_prob = vib_model(
            images, labels, num_samples=100, training=False, drawing=True
        )

        single_embedding_mean = single_embedding.mean(1)
        single_embedding_noise_mean = single_embedding_noise.mean(1)

        if is_drawing:
            valid_embedding_labels.extend(np.asarray(labels.detach().cpu().numpy()))
            valid_embedding_sensitive_labels.extend(np.asarray(sensitive_labels.detach().cpu().numpy()))
            valid_embedding_clean_images.extend(np.asarray(single_embedding_mean.detach().cpu().numpy()))
            valid_embedding_noise_images.extend(np.asarray(single_embedding_noise_mean.detach().cpu().numpy()))
    return valid_embedding_labels, valid_embedding_sensitive_labels, valid_embedding_clean_images, valid_embedding_noise_images


def get_data():
    vib_model.load_state_dict(torch.load(os.path.join(save_model_dir, crt_model_name)))
    is_drawing = True
    return evaluation(50, test_loader, is_drawing=is_drawing)

import matplotlib
import numpy as np
# matplotlib.use("Agg")
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
import seaborn


color_key = [color for color in seaborn.colors.xkcd_rgb.keys()]


def manual_legend(ax, x_min, x_max, y_min, y_max, cmap, num):
    delta = (y_max - y_min) / num
    labels, x, y = [], [], []
    for i in range(num):
        labels.append(i)
        x.append(x_max)
        y.append(y_max - i * delta)
    ax.scatter(x, y, c=labels, s=15, cmap=cmap)


LEGEND_TYPE_AUTO = 0
LEGEND_TYPE_SEMI_AUTO = 1
LEGEND_TYPE_MANUAL = 2


def tsne_embedding_without_images(images, labels, save_name=None, legend_type=LEGEND_TYPE_AUTO, legend_recall=None):
    images_scaled = StandardScaler().fit_transform(images)
    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=30.0, early_exaggeration=12.0, random_state=1)
    tsne_result = tsne.fit_transform(images_scaled)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

    fig = plt.figure(figsize=(10, 10))
    for plot_index in range(0, len(labels)):
        ax = fig.add_subplot(1, len(labels), plot_index + 1)
        class_num = len(set(labels[plot_index]))
        if 10 >= class_num:
            cmap = plt.get_cmap('tab10')
        elif 20 >= class_num:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = get_cmap_xkcd()
        scatter = ax.scatter(tsne_result_scaled[:, 0], tsne_result_scaled[:, 1],
                             c=labels[plot_index], s=15, cmap=cmap)
        if legend_type == LEGEND_TYPE_AUTO:
            if len(set(labels[plot_index])) == 10:
                legendClass = ax.legend(*scatter.legend_elements(prop="colors"),
                                        loc="best", prop={'size': 10})
            elif len(set(labels[plot_index])) == 5:
                legendClass = ax.legend(*scatter.legend_elements(prop="colors"),
                                        loc="best", prop={'size': 15})
            else:
                legendClass = ax.legend(*scatter.legend_elements(prop="colors"),
                                        loc="best", prop={'size': 8})
            ax.add_artist(legendClass)
        elif legend_type == LEGEND_TYPE_SEMI_AUTO:
            x_min, x_max, y_min, y_max = plt.axis()
            manual_legend(ax, x_min, x_max, y_min, y_max, cmap=cmap, num=class_num)
        elif legend_type == LEGEND_TYPE_MANUAL:
            assert legend_recall is not None, 'please set the recall function'
            legend_recall(plt=plt, ax=ax, scatter=scatter, cmap=cmap, num=len(set(labels[plot_index])))
        ax.update_datalim(tsne_result_scaled)
        ax.autoscale()

    if save_name:
        plt.savefig(save_name)
        plt.close(fig=fig)
    else:
        plt.show()
        plt.close(fig=fig)

valid_embedding_labels, valid_embedding_sensitive_labels, valid_embedding_clean_images, valid_embedding_noise_images = get_data()

labels = [valid_embedding_labels] # [valid_embedding_sensitive_labels]  # [valid_embedding_labels]


def recall_legend(plt, ax, scatter, cmap, num):
    x_min, x_max, y_min, y_max = plt.axis()

    ax.scatter([x_max+0.1], [y_max/2], c='white', s=15)

    print(type(scatter.legend_elements(prop="colors", num=num)))

    legendClass = ax.legend(*scatter.legend_elements(prop="colors", num=num),
                            loc='upper right', prop={'size': 7.5}, title='person ID',
                            frameon=False)  #  bbox_to_anchor=(0.8, 0.8))
    frame = legendClass.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')
    ax.add_artist(legendClass)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')


tsne_embedding_without_images(images=valid_embedding_clean_images, labels=labels, save_name='123.png',
                              legend_type=LEGEND_TYPE_MANUAL, legend_recall=recall_legend)