"""
SCRIPT TO CREATE INTERACTIVE CLUSTER VISUALIZATION
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import datetime

cluster_sentence = pd.read_csv("/Users/axelahdritz/coding_projects/Unsupervised-Emotion-Clustering/cluster_sentence.csv").to_numpy()

def create_interactive_plot(cluster_sentence):

    fig, ax = plt.subplots()
    fig.suptitle('PCA K-means Clusters on Feature Matrix', fontsize=16)
    scatter = plt.scatter(
        x = cluster_sentence[:,-1],
        y = cluster_sentence[:,-2],
        c = cluster_sentence[:,1],
        cmap = ListedColormap(['indigo', 'magenta', 'brown', 'olive', 'turquoise'], 'indexed')
    )

    annotation = ax.annotate(
        text='',
        xy=(0,0),
        xytext=(15,15),
        textcoords='offset points',
        bbox={'boxstyle':'round','fc':'w'},
        arrowprops={'arrowstyle':'->'}
    )

    annotation.set_visible(False)

    def motion_hover(event):
        annotation_visibility = annotation.get_visible()
        if event.inaxes == ax:
            is_contained, annotation_index = scatter.contains(event)
            if is_contained:
                data_point_location = scatter.get_offsets()[annotation_index['ind'][0]]
                tx = cluster_sentence[:,2][annotation_index['ind'][0]]
                audio = cluster_sentence[:,3:6][annotation_index['ind'][0]]
                audio_start = str(datetime.timedelta(seconds=cluster_sentence[:,3][annotation_index['ind'][0]]))
                print(tx)
                print(audio)
                print(audio_start)
                annotation.xy = data_point_location
                #text_label = '{} ,({0:.2f}, {0:.2f})'.format(tx, data_point_location[0], data_point_location[1])
                #annotation.set_text(text_label)
                #annotation.set_visible(True)
                #fig.canvas.draw_idle()
            else:
                if annotation_visibility:
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()
                
    fig.canvas.mpl_connect('motion_notify_event', motion_hover)
    plt.show()

create_interactive_plot(cluster_sentence)