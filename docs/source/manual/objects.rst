``morsaik``-Objects
==============
Functions in ``morsaik`` typically return ``morsaik``-objects typically of
three categories.
We define vectors that contain values at a given time point.
A set of lined up vectors is called trajectories.
If multiple trajectories are computed from the same parameters,
they can be collected in an ensemble.
Example objects are

- ``MotifVector``
- ``MotifBreakageVector``
- ``MotifProductionVector``,

with corresponding

- ``MotifTrajectory``
- ``MotifBreakageTrajectory``
- ``MotifProductionTrajectory``

and

- ``MotifTrajectoryEnsembles``
- ``MotifBreakageTrajectoryEnsemble``
- ``MotifProductionTrajectoryEnsemble``.
