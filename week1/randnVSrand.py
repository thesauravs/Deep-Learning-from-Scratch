import numpy as np
import matplotlib.pyplot as plt

sample_size = 100000
uniform = np.random.rand(sample_size)
normal = np.random.randn(sample_size)

pdf, bins, patches = plt.hist(uniform, bins=50, range=(0, 1), density=True)
plt.title('rand: uniform')
plt.show()

pdf, bins, patches = plt.hist(normal, bins=50, range=(-4, 4), density=True)
plt.title('randn: normal')
plt.show()