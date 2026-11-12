Imputation Methods
==================

Each method can be called on its own: ``f(y, id_obs, x, id_mis=None, ...,
rng=None)`` returns the imputations of the cells in ``id_mis``. Through
:meth:`imputation.MICE.impute` their parameters are passed with a prefix:
the method name with every dot replaced by an underscore (``pmm_donors=3``,
``norm_boot_ridge=1e-4`` for parameter ``ridge`` of ``norm.boot``,
``polr_log_fallback=True``). See :doc:`../user_guide/imputation_methods`.

Donor methods
-------------

pmm
~~~

.. automodule:: imputation.PMM
   :members:

midas
~~~~~

.. automodule:: imputation.midas
   :members:

cart
~~~~

.. automodule:: imputation.cart
   :members:

rf
~~

.. automodule:: imputation.rf
   :members:

sample
~~~~~~

.. automodule:: imputation.sample
   :members:

Model-based methods
-------------------

norm, norm.nob, norm.boot, norm.predict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: imputation.norm
   :members:

mean
~~~~

.. automodule:: imputation.mean
   :members:

logreg, logreg.boot
~~~~~~~~~~~~~~~~~~~

.. automodule:: imputation.logreg
   :no-members:

.. autofunction:: imputation.logreg.logreg

.. autofunction:: imputation.logreg.logreg_boot

.. autofunction:: imputation.logreg.target_categories

.. autoclass:: imputation.logreg.Categories
   :no-members:

.. autofunction:: imputation.logreg.augment

.. autoclass:: imputation.logreg.Augmented
   :no-members:

.. autodata:: imputation.logreg.MAX_CATEGORIES

polyreg
~~~~~~~

.. automodule:: imputation.polyreg
   :members:

polr
~~~~

.. automodule:: imputation.polr
   :members:

Model fits
----------

The fits behind ``logreg``, ``logreg.boot``, ``polyreg`` and ``polr``, as R
computes them.

.. automodule:: imputation.fitting
   :no-members:

.. autofunction:: imputation.fitting.logit_fit

.. autoclass:: imputation.fitting.LogitFit
   :no-members:

.. autofunction:: imputation.fitting.vmmin

.. autoclass:: imputation.fitting.Minimum
   :no-members:

.. autofunction:: imputation.fitting.multinom_fit

.. autoclass:: imputation.fitting.MultinomFit
   :no-members:

.. automethod:: imputation.fitting.MultinomFit.predict

.. autofunction:: imputation.fitting.polr_fit

.. autoclass:: imputation.fitting.PolrFit
   :no-members:

.. automethod:: imputation.fitting.PolrFit.predict

.. autoexception:: imputation.fitting.PolrError
