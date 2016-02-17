##Description
We consider the problem of evaluating the simple one-dimensional integral

    Z = \int_{-1}^{1} P(theta) * prior(theta) dtheta,
where `P()` and `prior()` are defined as

    P(theta) = exp(- beta * x ** 2 / 2.0),
    prior(theta) = 1/2,
and we solve this task with nested sampling.
The notation is loosely based on [J. Skilling, Nested sampling for general Bayesian computation, Bayesian analysis, **1** (2006), 833-859].

Note that, for this problem, `X(P_value)` can be computed analytically.
However, we use its stochastic version

    X_i = t * X_{i-1}
where t is drawn from

    prob(t) = N * t^(N - 1),
to remain in a general case.


##Files

####lib_nested_sampling.py
General-purpose library that performs nested sampling.
The docstring of the main function reads
```
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
```

####example_1.py
The nested-sampling algorithm is run `n_runs=1000` times, and a histogram of `log(Z)` and some of the `(X_i,P_i)` curves are shown.

Example of output:
![fig_example_1.png](fig_example_1.png?raw=true)

####example_2.py
Pure-python implementation (so that it can be run in  [pypy](http://pypy.org)).
The nested-sampling algorithm is run `n_runs` times, and statistics of `log(Z)` is computed and stored.

Example of output (ns="nested sampling", lines sorted for decreasing relative error on the average):
```
n_samples           f   n_runs         Z_exact            Z_ns       err_Z_ns  err_Z_ns/Z_ns  sigmas
 00000010  0.00100000 00000010 -2.076793740349 -1.923032254057 0.192579697542       0.100144   0.798
 00000100  0.00100000 00000100 -2.076793740349 -2.080812733187 0.018332780121       0.008810   0.219
 00001000  0.00100000 00000100 -2.076793740349 -2.080708626043 0.005190043216       0.002494   0.754
 00000500  0.00010000 00001000 -2.076793740349 -2.080376942734 0.002500306914       0.001202   1.433
 00001000  0.00050000 00001000 -2.076793740349 -2.075796881962 0.001802503274       0.000868   0.553
 00001000  0.00060000 00004000 -2.076793740349 -2.076713393667 0.000914755084       0.000440   0.088
```
