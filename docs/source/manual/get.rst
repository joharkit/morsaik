``get``-Module
==========

The ``get`` submodule is a convenience submodule combining the ``read``- and the
``infer``-submodules.
Giving the id of the object, one wants to get, it looks, whether it can read
an already computed version from disk and give you the corresponding version.
If there is no saved version, it infers (or simulates) the object asked
for from the information it has.
For each id, a yaml file specifies this information.
The yaml files are saved in the directory ``./config/`` .
