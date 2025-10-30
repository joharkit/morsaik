``infer``-Module
============

The ``infer`` module is responsible for inferring variables from data.
Different models are implemented here that allow different inferences.
Typical inference functions use Bayesian inference (using NIFTy.re),
but there are also infer-functions that naively compute one variable from the
other in a deterministic, non-statistical fashion,
as well as functions that simulate dynamics,
such as motif dynamics,
given the parameters of those dynamics.
