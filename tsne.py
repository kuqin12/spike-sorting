from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def scatter(waves, labels, classifier):
    model = TSNE(
            perplexity=100,
            learning_rate=10,
            init="pca",
            n_iter=5000,
            n_iter_without_progress=200,
            angle=0.2)

    boundary_waves = []
    for j in range(10):
        for k in range(j+1,10):
            for each in classifier.svms:
                # w0*x + w1*y + w2*z + ... + b = 0, kind of?
                (w,b) = classifier.svms[each]
                sample = np.zeros(10)
                sample[j] = 1
                sample[k] = (-b-w[j])/w[k]
                boundary_waves.append (sample)

    waves = np.vstack((waves, boundary_waves))
    labels += [-1] * len(boundary_waves)
    waveforms_embedded = model.fit_transform(waves)

    print (len(boundary_waves))
    print (waveforms_embedded.shape)
    print (waves.shape)
    x_max = np.max(waveforms_embedded.T[0][:-len(boundary_waves)], axis=0)
    y_max = np.max(waveforms_embedded.T[1][:-len(boundary_waves)], axis=0)
    x_min = np.min(waveforms_embedded.T[0][:-len(boundary_waves)], axis=0)
    y_min = np.min(waveforms_embedded.T[1][:-len(boundary_waves)], axis=0)
    dur_x = x_max - x_min
    dur_y = y_max - y_min
    plt.xlim([x_min - dur_x * 0.1, x_max + dur_x * 0.1])
    plt.ylim([y_min - dur_y * 0.1, y_max + dur_y * 0.1])

    plt.scatter(*waveforms_embedded.T, c=labels, cmap="Set1")
    plt.show()
