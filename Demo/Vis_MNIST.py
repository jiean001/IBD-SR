#!/usr/bin/env python
# coding: utf-8

# In[1]:


model_name = 'fair_single_best.pt'
ema_model_name = 'fair_ema_best.pt'


tsne_model_name = '000_tsne_mnist_single_%s_%s.png'
tsne_ema_model_name = '000_tsne_yaleb_ema_%s_%s.png'

crt_model_name = model_name
crt_tsne_name = tsne_model_name

root = '../Data'
root_dir = './results_image'


# In[2]:


import sys
sys.path.append('../')


# In[3]:


import os
import torch
import random
import numpy as np
from torch.autograd import Variable
from base_utils.dir_util import mkdirs
from Utils.Get_Datasets import get_datasets
from Utils.Visualize import tsne_embedding_without_images
from Utils.Visualize import LEGEND_TYPE_MANUAL


# MNIST_ROT

# In[4]:


import Config.config_mnist_rot_vis as config
from Models.VIB_MNIST_ROT_whole_model import VariationalInformationBottleneck


# In[6]:


train_args = config.train_args()
dataset_args = config.dataset_args()
model_args = config.model_args()
use_cuda = True


# In[7]:


# 可以在这里修改GPU
train_args.gpu = 3


# In[8]:


init_seed = train_args.seed
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)
random.seed(init_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(train_args.gpu)


# In[10]:


test_loader = get_datasets(
    dataset_args.dataset, dataset_args.train_batch_size, dataset_args.test_batch_size, root=root)

# In[11]:


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


# In[14]:


path_1 = '%.03f_%.03f_%.03f' %(model_args.reconstruction_weight, model_args.pairwise_kl_clean_weight, model_args.pairwise_kl_noise_weight)
path_2 = '%.02f_%.02f_%.02f_%.02f' %(model_args.sparse_kl_weight_clean, model_args.sparse_kl_weight_noise, model_args.sparsity_clean, model_args.sparsity_noise)
save_model_dir = os.path.join('../saved_model/Mnist-Rot/', path_1, path_2, str(train_args.seed))
mkdirs(save_model_dir)
print(save_model_dir, os.path.exists(save_model_dir))
print(path_1)
print(path_2)


# In[15]:


vib_model = VariationalInformationBottleneck(
    dataset_args.shape_data, dataset_args.num_target_class,
    model_args.dim_embedding_clean, model_args.dim_embedding_noise,
    model_args.channel_hidden_encoder, model_args.channel_hidden_decoder,
    model_args.dim_hidden_classifier, parameter
)

if use_cuda:
    vib_model = vib_model.cuda()


# In[19]:


def evaluation(epoch_index, test_dataloader):
    # 设置网络模型的模式
    vib_model.eval()
    
    valid_embedding_labels = []
    valid_embedding_sensitive_labels = []
    valid_embedding_clean_images = []
    valid_embedding_noise_images = []
    
    # or iter_index, data in enumerate(test_dataloader):
    for iter_index, data in enumerate(test_dataloader):
        images = data[0]
        labels = data[1]
        sensitive_labels = data[2]

        images = Variable(images.unsqueeze(dim=1).float())
        labels = Variable(labels.long())
        sensitive_labels = Variable(sensitive_labels.long())

        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
            sensitive_labels = sensitive_labels.cuda()
        
        single_embedding, single_embedding_noise, single_classification_prob = vib_model(
            images, labels, num_samples=100, training=False
        )
        
        single_embedding_mean = single_embedding.mean(1)
        single_embedding_noise_mean = single_embedding_noise.mean(1)
        
        valid_embedding_labels.extend(np.asarray(labels.detach().cpu().numpy()))
        valid_embedding_sensitive_labels.extend(np.asarray(sensitive_labels.detach().cpu().numpy()))
        valid_embedding_clean_images.extend(np.asarray(single_embedding_mean.detach().cpu().numpy()))
        valid_embedding_noise_images.extend(np.asarray(single_embedding_noise_mean.detach().cpu().numpy()))
    return valid_embedding_labels, valid_embedding_sensitive_labels, valid_embedding_clean_images, valid_embedding_noise_images


# In[20]:


def get_data():
    vib_model.load_state_dict(torch.load(os.path.join(save_model_dir, crt_model_name)))
    return evaluation(50, test_loader)


# In[21]:


valid_embedding_labels, valid_embedding_sensitive_labels, valid_embedding_clean_images, valid_embedding_noise_images = get_data()


# In[ ]:





# In[16]:


def recall_legend_categorical(plt, ax, scatter, cmap, num):
    x_min, x_max, y_min, y_max = plt.axis()
    # 加一些透明的白点，扩展页面范围
    ax.scatter([x_max+0.1], [y_max/2], c='white', s=15, alpha=0)
    
    legendClass = ax.legend(*scatter.legend_elements(prop="colors", num=num),
                            loc='upper right', prop={'size': 7.5}, title='person ID',
                            frameon=False)
    frame = legendClass.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')
    ax.add_artist(legendClass)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')


# In[17]:


def recall_legend_sensitive(plt, ax, scatter, cmap, num):
    handles, _ = scatter.legend_elements(prop="colors")
    labels = ['upper-left', 'lower-left', 'upper-right', 'lower-right', 'front']
    legendClass = ax.legend(*(handles, labels), loc="best", prop={'size': 7.5}, title='sensitive label',
                           frameon=False)
    frame = legendClass.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')
    ax.add_artist(legendClass)
    
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')


# In[18]:


saved_path = os.path.join(root_dir, crt_tsne_name)


# In[19]:


labels = [valid_embedding_labels]
save_name = saved_path %(path_1, path_2, 50, 'clean')
tsne_embedding_without_images(images=valid_embedding_clean_images, labels=labels, save_name=save_name,
                              legend_type=LEGEND_TYPE_MANUAL, legend_recall=recall_legend_categorical)


# In[20]:


labels = [valid_embedding_sensitive_labels]
save_name = saved_path %(path_1, path_2, 50, 'clean_sensitive')
tsne_embedding_without_images(images=valid_embedding_clean_images, labels=labels, save_name=save_name,
                              legend_type=LEGEND_TYPE_MANUAL, legend_recall=recall_legend_sensitive)


# In[21]:


labels = [valid_embedding_labels]
save_name = saved_path %(path_1, path_2, 50, 'noise')
tsne_embedding_without_images(images=valid_embedding_noise_images, labels=labels, save_name=save_name,
                              legend_type=LEGEND_TYPE_MANUAL, legend_recall=recall_legend_categorical)


# In[22]:


labels = [valid_embedding_sensitive_labels]
save_name = saved_path %(path_1, path_2, 50, 'noise_sensitive')
tsne_embedding_without_images(images=valid_embedding_noise_images, labels=labels, save_name=save_name,
                              legend_type=LEGEND_TYPE_MANUAL, legend_recall=recall_legend_sensitive)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




