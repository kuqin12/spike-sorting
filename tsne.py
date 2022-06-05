from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def scatter(waves, labels):
    model = TSNE(
            perplexity=100,
            learning_rate=10,
            init="pca",
            n_iter=5000,
            n_iter_without_progress=200,
            angle=0.2)

    waveforms_embedded = model.fit_transform(waves)

    plt.scatter(*waveforms_embedded.T, c=labels, cmap="Set1")
    plt.show()
