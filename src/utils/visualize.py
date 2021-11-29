import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import itertools
from sklearn.manifold import TSNE
from modeling.utils import Flattening


def plot_embedding(X, dY, prY, save_name):
    color_list = [
        ["b", "g", "m", "r"],
        ["lightsteelblue", "lightgreen", "thistle", "lightcoral"],
    ]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    dY = np.array(dY)
    prY = np.array_split(prY, len(dY))
    X = np.array_split(X, len(dY))

    plt.figure(figsize=(20, 20))
    # kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)  # Normalize
    kwargs = dict(alpha=0.5, bins=100)
    for i in range(len(dY) - 1):
        plt.hist(prY[i], **kwargs, color=color_list[0][i], label=f"Source_${i}_prY")
        plt.hist(dY[i], **kwargs, color=color_list[1][i], label=f"Source_${i}_Y")
    plt.hist(prY[-1], **kwargs, color=color_list[0][-1], label="Target_prY")
    plt.hist(dY[-1], **kwargs, color=color_list[1][-1], label="Target_Y")

    plt.xlim(0, 1.0)
    plt.ylim(0, 400)
    plt.legend()
    if save_name is not None:
        plt.title(save_name)

    fig_name = str(save_name)
    plt.savefig(fig_name)
    print("{} is saved".format(fig_name))


def Visualize(framework, target_iter, source_iter, save_name):

    # Get source_test samples
    source_X_list = list()
    source_Y_list = list()
    flatten = Flattening()

    for i in range(len(source_iter)):  # i = n of source domain
        source_X = list()
        source_Y = list()
        for _ in range(len(source_iter[i])):
            X, R, y = next(source_iter[i])
            y = y.numpy()
            source_X.append(X)
            source_Y.extend(y)

        source_X_list.append(torch.stack(source_X).view(-1, 43))  # .view(-1, 43, 5))
        source_Y_list.append(source_Y)

    # Get target_test samples
    target_X_list = list()
    target_Y_list = list()
    for _ in range(len(target_iter)):
        X, R, y = next(target_iter)
        y = y.numpy()
        target_X_list.append(X)
        target_Y_list.extend(y)

    target_X_list = torch.stack(target_X_list)
    target_X_list = target_X_list.view(-1, 43)

    # Stack source_list + target_list
    combined_X_list = torch.cat((torch.cat((source_X_list)), target_X_list), 0).cuda()

    print("Extract features to draw T-SNE plot...")
    combined_feature = flatten(framework.DNA_attention(combined_X_list))
    combined_predict = framework.predictor(combined_feature).squeeze()
    combined_predict = combined_predict.detach().cpu().numpy()

    tsne = TSNE(perplexity=30, n_components=1, init="random", n_iter=1000)
    tsne_fit = tsne.fit_transform(combined_feature.detach().cpu().numpy())

    print("Draw plot ...")
    Y_list = source_Y_list + [target_Y_list]
    plot_embedding(tsne_fit, Y_list, combined_predict, save_name)
