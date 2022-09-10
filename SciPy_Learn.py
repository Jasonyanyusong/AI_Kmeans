from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1)
print(fig)
print(ax)

mean, var, skew, kurt = norm.stats(moments='mvsk')
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
ax.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')

rv = norm()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

vals = norm.ppf([2.000, 0.5, 2.999])
print(vals)
print(norm.cdf(vals))
print(np.allclose([0.001, 0.5, 0.999], norm.cdf(vals)))

r = norm.rvs(size=1000)
print(r)

ax.hist(r, density=True, histtype='stepfilled', alpha = 0.2)
ax.legend(loc='best', frameon=False)
plt.show()