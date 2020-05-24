#from word2vec_basic import *
#from word2vec_train import *
from matplotlib import animation
import matplotlib.pyplot as plt
import time
import numpy as np
#import matrix_priors
import seaborn as sns
import random
import os

grid = []
object_labels = []


def pairwise_sim_grid(embeddings, dictionary, object_labels):
    """
    Calculates cosine matrix for use in generating heatmap
    """
    data = np.zeros((len(object_labels), len(object_labels)))


    for obj1 in range(len(object_labels)):
        for obj2 in range(len(object_labels)): 
            if obj1 == obj2: 
                data[obj1][obj2] = 1.
                continue
            if data[obj1][obj2] > 0: continue

            embed1 = embeddings[dictionary.get(object_labels[obj1])]
            embed2 = embeddings[dictionary.get(object_labels[obj2])]
            
            n_embed1 = embed1 / np.linalg.norm(embed1)
            n_embed2 = embed2 / np.linalg.norm(embed2)

            cossim = np.dot(n_embed1, n_embed2)
            data[obj1][obj2] = cossim
            data[obj2][obj1]= cossim
        

    return data

def get_object_labels(X):
    labels = []
    for x in X:
        for y in x:
            if y not in labels: labels.append(y)

    print(labels)
    return labels


def gen_cosine_heatmap(object_labels, data, save_name=None):
    """
    Generates a heatmap of the cosine values
    over a time domain for debugging purposes.

    Parameter object_labels: A list of all of the objects being
    tested on.
    """
    #for label in object_labels:
        

    # create the figure
    fig, ax = plt.subplots()
    img = data[0]
    im = ax.imshow(img)

    ax.set_xticks(np.arange(len(object_labels)))
    ax.set_yticks(np.arange(len(object_labels)))

    ax.set_xticklabels(object_labels)
    ax.set_yticklabels(object_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")



    fig.tight_layout()

    if save_name == None:
        plt.show(block=False)
    else:
        plt.savefig('../heatmaps/'+save_name+'.png')

    # draw some data in loop
    for i in range(len(data)):
        # wait for a second
        time.sleep(0.05)
        # replace the image contents
        im.set_array(data[i])
        
        # Loop over data dimensions and create text annotations.
        """
        for j in range(len(object_labels)):
            for k in range(len(object_labels)):
                text = ax.text(k, j, round(data[i][j][k], 2),
                    ha="center", va="center", color="w")
        """
        # redraw the figure
        fig.canvas.draw()



