Underlying Software
==================

Most of the arrays are implemented in ``Jax`` [1_]
which enables differentiable algorithms for the Bayesian inference with
``NIFTy.re`` [2_].
For integrating ordinary differential equations, one can choose between
``solveivp`` from ``scipy.integrate`` [3_] and ``diffrax`` [4_].

References
----------
.. [1] https://docs.jax.dev/
.. [2] http://ift.pages.mpcdf.de/nifty/
.. [3] https://scipy.org/
.. [4] https://docs.kidger.site/diffrax/
