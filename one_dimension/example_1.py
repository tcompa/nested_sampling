'''
program: example_1.py
author: tc
created: 2016-02-17 -- 17 CEST

-------------------------------------------------------------------------------

Nested-sampling computation of the integral
    Z = \int_{-1}^{1} P(theta) * prior(theta) dtheta,
where
    P(theta) = exp(- beta * x ** 2 / 2.0),
and
    prior(theta) = 1/2.

Several runs (each one with different choices of t's) are performed, to
accumulate statistics on log(Z).

Loosely based on
    Skilling, J. (2006). Nested sampling for general Bayesian computation.
    Bayesian analysis, 1(4), 833-859.

-------------------------------------------------------------------------------

'''

import numpy
import random
import math
import matplotlib
import matplotlib.pyplot as plt

from lib_nested_sampling import nested_sampling_run

matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['figure.titlesize'] = 20

beta = 200.0


def P(x):
    '''
    Non-normalized probability distribution
    '''
    return math.exp(- beta * x ** 2 / 2.0)


def sample_theta(P_value):
    '''
    Samples a value of theta from a uniform distribution in (-1,1), with the
    constraint that P(theta) > P_value.
    '''
    assert P_value <= 1.0
    try:
        assert P_value > 0.0
        theta_val = math.sqrt(- (2.0 / beta) * math.log(P_value))
    except AssertionError:
        theta_val = 1.0
    return random.uniform(- theta_val, theta_val)


# collect input
n_samples = 50
f = 2e-3
n_runs = 1000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# perform runs
list_logZ = []
for i_run in xrange(n_runs):
    Z, X_i, P_i = nested_sampling_run(P, sample_theta, n_samples=n_samples,
                                      f=f)
    list_logZ.append(math.log(Z))
    if i_run < 10:
        ax2.plot(X_i, P_i)

# Gaussian with same mean/variance as data
list_logZ = numpy.array(list_logZ)
av = list_logZ.mean()
var = (list_logZ ** 2).mean() - av ** 2
x = numpy.linspace(list_logZ.min(), list_logZ.max(), 1000)
y = numpy.exp(- (x - av) ** 2 / (2.0 * var)) / math.sqrt(2.0 * math.pi * var)

ax1.hist(list_logZ, bins=30, alpha=0.75, normed=True)
ax1.plot(x, y, 'r-', lw=2, zorder=11)
ax1.grid()
ax1.set_xlabel('$\log(Z)$')
ax1.set_ylabel('frequency')

ax2.set_yscale('log')
ax2.set_xlabel('$X$')
ax2.set_ylabel('$P(X)$')

plt.suptitle('Nested sampling in 1D ' +
             '($N_\mathrm{samples}=%i$,' % n_samples +
             ' $f=%s$, $N_\mathrm{runs}=%i$)' % (f, n_runs), y=1.03)
plt.tight_layout()
plt.savefig('fig_example_1.png', bbox_inches='tight')
