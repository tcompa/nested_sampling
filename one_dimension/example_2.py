'''
program: example_2.py
author: tc
created: 2016-02-09 -- 13 CEST
notes: pure python, can be exectuted via pypy

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

import sys
import random
import math

from lib_nested_sampling import nested_sampling_run


beta = 100.0


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
if len(sys.argv) != 4 or '-h' in sys.argv or '--help' in sys.argv:
    sys.exit('Required arguments: n_samples, f, n_runs\n' +
             '(f: small control parameter which fixes run duration)')
n_samples = int(sys.argv[1])
f = float(sys.argv[2])
n_runs = int(sys.argv[3])
ID = 'N%07i_f%s_runs%s' % (n_samples, f, n_runs)

# perform runs
list_logZ = []
for run in xrange(n_runs):
    Z, _, _ = nested_sampling_run(P, sample_theta, n_samples=n_samples, f=f)
    list_logZ.append(math.log(Z))

# exact results
Z_exact = math.sqrt(math.pi / (beta * 2.0)) * math.erf(math.sqrt(beta / 2.0))
logZ_exact = math.log(Z_exact)

# statistic of logZ over runs
logZ_mean = sum(list_logZ) / float(n_runs)
logZ_sq_mean = sum(x ** 2 for x in list_logZ) / float(n_runs)
var_logZ = logZ_sq_mean - logZ_mean ** 2
err_logZ = (var_logZ / float(n_runs - 1.0)) ** 0.5

# store output
out = open('data_%s.dat' % ID, 'w')
out.write('%08i %.8f %08i ' % (n_samples, f, n_runs))
out.write('%.12f ' % logZ_exact)
out.write('%.12f %.12f %.6f ' % (logZ_mean, err_logZ,
                                 err_logZ / abs(logZ_mean)))
out.write(' %.3f\n' % (abs(logZ_mean - logZ_exact) / err_logZ))
out.close()
