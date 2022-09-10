from scipy.stats import norm
import numpy as np
from scipy.stats.norm import pdf
import matplotlib.pyplot as plt

# Integrating out one of the variables of a 2D Gaussian should
# yield a 1D Gaussian
mean = np.array([2.5, 3.5])
cov = np.array([[.5, 0.2], [0.2, .6]])
n = 2 ** 8 + 1  # Number of samples
delta = 6 / (n - 1)  # Grid spacing

v = np.linspace(0, 6, n)
xv, yv = np.meshgrid(v, v)
pos = np.empty((n, n, 2))
pos[:, :, 0] = xv
pos[:, :, 1] = yv
pdf = multivariate_normal.pdf(pos, mean, cov)

# Marginalize over x and y axis
margin_x = romb(pdf, delta, axis=0)
margin_y = romb(pdf, delta, axis=1)

# Compare with standard normal distribution
gauss_x = norm.pdf(v, loc=mean[0], scale=cov[0, 0] ** 0.5)
gauss_y = norm.pdf(v, loc=mean[1], scale=cov[1, 1] ** 0.5)
assert_allclose(margin_x, gauss_x, rtol=1e-2, atol=1e-2)
assert_allclose(margin_y, gauss_y, rtol=1e-2, atol=1e-2)