import matplotlib
import numpy as np
# matplotlib.use("Agg")
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from base_utils.color_util import get_cmap_xkcd


LEGEND_TYPE_AUTO = 0
LEGEND_TYPE_SEMI_AUTO = 1
LEGEND_TYPE_MANUAL = 2


def visualize_scatter_with_images(X_2d_data, images, labels, save_name, figsize=(45, 45), image_zoom=1.0):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for (index, (xy, i)) in enumerate(zip(X_2d_data, images)):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom, cmap='gray')
        ab_img = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab_img))

    for (index, (xy, i)) in enumerate(zip(X_2d_data, images)):
        x0, y0 = xy
        text = TextArea(str(labels[index]), textprops=dict(size=20, weight='bold'))
        ab_text = AnnotationBbox(text, (x0 - 0.04, y0 + 0.05), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab_text))

    ax.update_datalim(X_2d_data)
    ax.autoscale()

    plt.savefig(save_name)
    plt.close(fig=fig)


def tsne_embedding_with_images(images, labels, label_images, save_name, image_size=(24, 21), ):
    images_scaled = StandardScaler().fit_transform(images)
    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=30.0, early_exaggeration=12.0, random_state=1)
    tsne_result = tsne.fit_transform(images_scaled)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
    visualize_scatter_with_images(tsne_result_scaled, images=[np.reshape(i, image_size) for i in label_images],
                                  labels=labels, image_zoom=3.0, save_name=save_name)


def semi_auto_legend(ax, x_min, x_max, y_min, y_max, cmap, num):
    delta = (y_max - y_min) / num
    labels, x, y = [], [], []
    for i in range(num):
        labels.append(i)
        x.append(x_max)
        y.append(y_max - i * delta)
    ax.scatter(x, y, c=labels, s=15, cmap=cmap)


def tsne_embedding_without_images(images, labels, save_name=None, is_show=False, legend_type=LEGEND_TYPE_AUTO, legend_recall=None):
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
            semi_auto_legend(ax, x_min, x_max, y_min, y_max, cmap=cmap, num=class_num)
        elif legend_type == LEGEND_TYPE_MANUAL:
            assert legend_recall is not None, 'please set the recall function'
            legend_recall(plt=plt, ax=ax, scatter=scatter, cmap=cmap, num=len(set(labels[plot_index])))
        ax.update_datalim(tsne_result_scaled)
        ax.autoscale()

    if save_name:
        plt.savefig(save_name)
        if is_show:
            plt.show()
        plt.close(fig=fig)
    else:
        plt.show()
        plt.close(fig=fig)


def embedding_images(images, clean_embedding, noise_embedding, labels, sensitive_labels, save_name):
    fig = plt.figure(figsize=(20, 10))
    plot_index = 1
    add_embedding = np.asarray(clean_embedding) + np.asarray(noise_embedding)

    for sensitive_index in range(0, 5):
        for class_index in range(0, 5):
            class_label = np.asarray(np.where(np.array(labels) == class_index))
            sensitive_class_label = np.asarray(np.where(np.array(sensitive_labels) == sensitive_index))
            images_index = np.intersect1d(class_label, sensitive_class_label)[0]

            ax1 = fig.add_subplot(5, 20, plot_index)
            show_images = np.reshape(images[images_index], (24, 21))
            ax1.imshow(show_images, cmap='gray')
            ax1.autoscale()
            plot_index = plot_index + 1

            ax2 = fig.add_subplot(5, 20, plot_index)
            show_clean_images = np.reshape(clean_embedding[images_index], (24, 21))
            ax2.imshow(show_clean_images, cmap='gray')
            ax2.autoscale()
            plot_index = plot_index + 1

            ax3 = fig.add_subplot(5, 20, plot_index)
            show_noise_images = np.reshape(noise_embedding[images_index], (24, 21))
            ax3.imshow(show_noise_images, cmap='gray')
            ax3.autoscale()
            plot_index = plot_index + 1

            ax4 = fig.add_subplot(5, 20, plot_index)
            show_add_images = np.reshape(add_embedding[images_index], (24, 21))
            ax4.imshow(show_add_images, cmap='gray')
            ax4.autoscale()
            plot_index = plot_index + 1

            i = 1

    plt.savefig(save_name)
    plt.close(fig=fig)


# def recall_legend_categorical_y(plt, ax, scatter, cmap, num):
#     x_min, x_max, y_min, y_max = plt.axis()
#     # 加一些透明的白点，扩展页面范围
#     ax.scatter([x_max+0.1], [y_max/2], c='white', s=15, alpha=0)
#
#     legendClass = ax.legend(*scatter.legend_elements(prop="colors", num=num),
#                             loc='upper right', prop={'size': 7.5}, title='person ID',
#                             frameon=False)
#     frame = legendClass.get_frame()
#     frame.set_alpha(1)
#     frame.set_facecolor('none')
#     ax.add_artist(legendClass)
#
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')