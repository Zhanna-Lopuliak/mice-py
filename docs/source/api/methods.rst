Imputation Methods
==================

Each method can be called on its own: ``f(y, id_obs, x, id_mis=None, ...,
rng=None)`` returns the imputations of the cells in ``id_mis``. Through
:meth:`imputation.MICE.impute` their parameters are passed with the method
name as prefix (``pmm_donors=3``). See :doc:`../user_guide/imputation_methods`.

pmm
---

.. automodule:: imputation.PMM
   :members:

midas
-----

.. automodule:: imputation.midas
   :members:

cart
----

.. automodule:: imputation.cart
   :members:

rf
--

.. automodule:: imputation.rf
   :members:

sample
------

.. automodule:: imputation.sample
   :members:
