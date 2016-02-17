'''
program: lib_nested_sampling.py
author: tc
created: 2016-02-09 -- 13 CEST

notes: loosely based on
           Skilling, J. (2006). Nested sampling for general Bayesian
           computation. Bayesian analysis, 1(4), 833-859.
'''

import sys
import random

__all__ = ['nested_sampling_run']


def _argmin_list(x):
    '''
    Finds the argmin in a list, without numpy.
    '''
    x_min = min(x)
    try:
        assert x.count(x_min) == 1
        return x.index(x_min)
    except AssertionError:
        sys.exit('ERROR: more than one minimum value of P(x)\n' +
                 ' (this could be avoided by adding an additional variable,' +
                 ' but I am keeping the program as simple as possible)')


def nested_sampling_run(P, sample_theta, n_samples=32, f=1e-3):
    '''
    Performs one nested-sampling run.

    args:
        *P(theta)*            : function returning the non-normalized
                                probability distribution P(theta)
        *sample_theta(P_val)* : function drawing one sample theta from the
                                prior, under the constraint that P(theta)>P_val

    kwargs:
        *n_samples*           : size of the population
        *f*                   : small parameter to determine when to stop

    output:
        *Z*                   : estimate of the partition function
        *X_i*                 : list of X
        *P_i*                 : list of P
    '''

    # initialize run
    theta_samples = [sample_theta(0.0) for _ in xrange(n_samples)]
    P_samples = [P(theta) for theta in theta_samples]
    P_i = [0.0]
    X_i = [1.0]
    Z = 0.0
    n_steps = 0

    # main loop
    while True:
        n_steps += 1
        ind_worst = _argmin_list(P_samples)
        P_i.append(P_samples.pop(ind_worst))

        # sample t, to get X_{i}=t*X_{t-1}
        z = random.uniform(0.0, 1.0)
        t = z ** (1.0 / n_samples)
        X_i_sampled = X_i[-1] * t
        X_i.append(X_i_sampled)

        # update Z
        w_i = X_i[- 2] - X_i[-1]
        dZ = P_i[-1] * w_i
        Z += dZ

        # replace worst theta
        theta_samples.pop(ind_worst)
        new_theta = sample_theta(P_i[-1])
        new_P = P(new_theta)
        theta_samples.append(new_theta)
        P_samples.append(new_P)

        # check end condition
        if max(P_samples) * X_i[-1] < f * Z:
            break

    Z += X_i[- 1] * P_i[-1]

    return Z, X_i, P_i
