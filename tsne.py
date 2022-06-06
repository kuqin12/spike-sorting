from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from collections import defaultdict

def scatter(waves, labels, svm):
    model = TSNE(
            perplexity=100,
            learning_rate=10,
            init="pca",
            n_iter=5000,
            n_iter_without_progress=200,
            angle=0.2)

    waveforms_embedded = model.fit_transform(waves)

    wave_d = defaultdict(list)
    for w, l in zip(waveforms_embedded, labels):
        wave_d[l].append(w)

    cs = []
    for l in wave_d:
        cs.append(np.mean(wave_d[l], axis=0))

    fig, ax = plt.subplots(1, 1)
    voronoi_plot_2d(Voronoi(np.array(cs)), ax)
    ax.scatter(*waveforms_embedded.T, c=labels, cmap="Set1")

    plt.show()
