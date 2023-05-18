import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import matplotlib.pyplot as plt


# rastrigin function in 2D
def rastrigin(x, y):
    ras = (20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) * 0.001

    # around (4, 4) makse a hole
    hole = np.exp(-100 * ((x - 2)**2 + (y - 2)**2))
    ras = ras + hole
    return ras


# cross entropy method to find the best samples
max_iter = 100
num_samples = 100
elite_ratio = 0.5

samples = np.random.uniform(-4, 4, size=(100, 2))
samples_history = []
samples_history.append(samples)
for i in range(max_iter):
    # evaluate samples
    values = rastrigin(samples[:, 0], samples[:, 1])

    # sort samples descending
    elite_idx = np.argsort(values)[::-1][:int(len(samples) * elite_ratio)]
    elite_samples = samples[elite_idx]

    # fit gaussian and sample
    gmm = GaussianMixture(
        n_components=len(elite_samples),
        weights_init=np.ones(len(elite_samples)) / len(elite_samples),
        reg_covar=1e-2,)
    gmm.fit(elite_samples)
    samples = gmm.sample(n_samples=100)[0]

    # bgmm = BayesianGaussianMixture(
    #     n_components=len(elite_samples),
    #     reg_covar=1e-2,)
    # bgmm.fit(elite_samples)
    # samples = bgmm.sample(n_samples=10)[0]

    # append samples
    samples_history.append(samples)


# visualize
x = np.linspace(-5.12, 5.12, 1000)
y = np.linspace(-5.12, 5.12, 1000)
X, Y = np.meshgrid(x, y)
Z = rastrigin(X, Y)

# show with imshow
plt.figure()
plt.imshow(Z, extent=[-5.12, 5.12, -5.12, 5.12])
plt.colorbar()

# show collected samples
for i, samples in enumerate(samples_history):
    plt.scatter(samples[:, 0], samples[:, 1], c='k', alpha=1.0 - i / len(samples_history))

plt.show()
