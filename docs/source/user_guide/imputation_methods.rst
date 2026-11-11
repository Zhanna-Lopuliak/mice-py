Imputation Methods
==================

mice-py has five of R mice's univariate imputation methods. Every method
fills a missing cell with a value **observed** in the same column, so
imputations stay within the range and the type of the data. The examples use ``df`` from the
:ref:`example data <example-data>`.

.. list-table::
   :header-rows: 1
   :widths: 13 17 35 35

   * - Method
     - R name
     - Model
     - Target types
   * - ``"pmm"``
     - ``pmm``
     - Bayesian linear regression, then the nearest observed values
     - numeric, boolean, categorical, object, string
   * - ``"midas"``
     - ``midastouch``
     - bootstrap-weighted regression, donors weighted by distance
     - numeric, boolean
   * - ``"cart"``
     - ``cart``
     - one classification or regression tree
     - all
   * - ``"rf"``
     - ``rf``
     - random forest
     - all
   * - ``"sample"``
     - ``sample``
     - none: draws from the observed values
     - all

Choose a method per column with a dict; columns left out get ``"pmm"``.
Method parameters are passed to ``impute`` with the method name as prefix:

.. code-block:: python

   from imputation import MICE

   mice = MICE(df)
   mice.impute(
       method={"bmi": "pmm", "chl": "midas", "sex": "cart"},
       pmm_donors=3,
       cart_min_samples_leaf=10,
       seed=1,
   )
   print(mice.method)

Each method is also a plain function, ``f(y, id_obs, x, id_mis=None, ...,
rng=None)``, that returns the imputations of the cells in ``id_mis``; see
:doc:`../api/methods`.

pmm: predictive mean matching
-----------------------------

R's default for numeric data, and mice-py's default for every column.

1. Fit a linear regression of the target on the predictors in the observed
   rows (with a small ridge penalty) and draw the coefficients from their
   posterior distribution (Rubin 1987).
2. Predict the target for the observed and the missing rows.
3. For each missing row, find the ``donors`` observed rows whose predictions
   are closest, pick one at random (ties are broken at random) and impute
   its observed value.

With ``matchtype=1`` (default) the observed rows are predicted with the
estimated coefficients and the missing rows with the drawn ones; ``0`` uses
the estimated coefficients for both, ``2`` the drawn ones for both. For a
categorical target the categories are first replaced by optimal scores
(``quantify``, as R), and the imputations are categories.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter (in ``impute``)
     - Default
     - Meaning
   * - ``pmm_donors``
     - 5
     - number of candidate donors; clamped to the number of observed values
   * - ``pmm_matchtype``
     - 1
     - 0, 1 or 2, see above
   * - ``pmm_quantify``
     - ``True``
     - optimal scaling of a categorical target; ``False`` uses its codes
   * - ``pmm_ridge``
     - 1e-5
     - ridge penalty of the regression
   * - ``pmm_ls_meth``
     - ``"qr"``
     - least-squares method: ``"qr"``, ``"ridge"`` or ``"svd"``

midas: distance-aided donor selection
-------------------------------------

R's ``midastouch`` (Gaffert, Meinfelder and Bosch 2018), a variant of
predictive mean matching that represents the uncertainty of the regression
by a bootstrap instead of drawn coefficients, and weights the donors by their
distance.

1. Draw bootstrap weights for the observed rows.
2. Fit a weighted linear regression on the observed rows with these weights.
3. With ``outout=True`` (default), predict each observed row from a fit that
   leaves that row out.
4. Give each observed row a probability proportional to its bootstrap weight
   times its distance to the missing row's prediction raised to the power
   ``-kappa``, and draw one donor per missing cell.

``kappa`` is estimated from the R² of the fit unless it is given (Siddique and
Belin 2008 find 3 sensible). midas imputes numeric and boolean targets only:
a categorical or text target raises ``ValueError``, as R's midastouch cannot
impute factors either.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter (in ``impute``)
     - Default
     - Meaning
   * - ``midas_kappa``
     - ``None``
     - sharpness of the distance weighting; ``None`` estimates it
   * - ``midas_outout``
     - ``True``
     - leave-one-out predictions of the donors
   * - ``midas_ridge``
     - 1e-5
     - ridge penalty of the regression

cart: classification and regression trees
-----------------------------------------

R's ``cart`` (Breiman et al. 1984; Doove, van Buuren and Dusseldorp 2014),
which grows the tree with rpart;
mice-py uses scikit-learn with rpart's settings.

1. Fit a regression tree (numeric target) or a classification tree (any other
   target) on the observed rows.
2. For each missing row, find its leaf and draw one of the observed values in
   that leaf.

A numeric target's tree is pruned by cost complexity with
``ccp_alpha = cp × var(y_obs)``, which is rpart's relative ``cp``. A
classification tree is not pruned unless ``cart_ccp_alpha`` is given.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter (in ``impute``)
     - Default
     - Meaning
   * - ``cart_min_samples_leaf``
     - 5
     - minimum observations in a leaf (rpart's ``minbucket``)
   * - ``cart_cp``
     - 1e-4
     - complexity parameter, relative to the variance of the target
   * - ``cart_min_samples_split``
     - ``None`` (3 × ``min_samples_leaf``)
     - minimum observations in a node to try a split (rpart's ``minsplit``)
   * - ``cart_max_depth``
     - 30
     - maximum depth (rpart's ``maxdepth``)
   * - ``cart_ccp_alpha``
     - ``None``
     - pruning in scikit-learn's absolute units; replaces ``cp``
   * - ``cart_bootstrap``
     - ``False``
     - fit the tree on a bootstrap sample (mice-py 0.1's behavior; not in R)
   * - ``cart_tree_params``
     - ``None``
     - further arguments of scikit-learn's tree, e.g.
       ``{"max_leaf_nodes": 20}``

rf: random forest
-----------------

R's ``rf`` (Doove, van Buuren and Dusseldorp 2014), which grows the forest
with ranger; mice-py uses scikit-learn with ranger-like settings.

1. Fit ``n_estimators`` trees, each on a bootstrap sample of the observed
   rows, with the square root of the number of predictors tried per split.
2. For each missing row, collect, over all trees, the observed values that
   share its leaf, and draw one of them.

With categorical predictors the forests of scikit-learn and ranger differ
noticeably; see :ref:`rf-categorical`.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter (in ``impute``)
     - Default
     - Meaning
   * - ``rf_n_estimators``
     - 10
     - number of trees (R's ``ntree``)
   * - ``rf_max_depth``
     - ``None``
     - maximum depth; ``None`` grows the trees fully
   * - ``rf_min_samples_leaf``
     - 1
     - minimum observations in a leaf
   * - ``rf_min_samples_split``
     - ``None`` (5 for a numeric target, 2 otherwise)
     - minimum observations in a node to try a split
   * - ``rf_forest_params``
     - ``None``
     - further arguments of scikit-learn's forest, e.g. ``{"n_jobs": -1}``
       to grow the trees in parallel

sample: random draws
--------------------

Draws each missing value from the observed values of the column, ignoring
the predictors. It keeps the marginal distribution but not the relations
with other variables. It has no parameters.

Choosing a method
-----------------

- **pmm** is R's default for numeric data and a reasonable first choice;
  its imputations are observed values.
- **midas** is an alternative to pmm for numeric targets, proposed to make
  predictive mean matching proper (Gaffert et al. 2018).
- **cart** and **rf** model interactions and non-linear relations without
  specifying them (Doove et al. 2014).
- **sample** ignores all relations; useful as a baseline.
- For categorical columns R uses logistic, multinomial or ordered logistic
  regression, which mice-py does not have; pmm, cart, rf and sample impute
  them.

Compare methods on your data with the plots of :doc:`diagnostic_plots`. The
simulation study of :doc:`../r_mice/validation` compares mice-py with R, not
the methods with each other.

Next steps
----------

- :doc:`predictor_matrices`
- :doc:`convergence_diagnostics`
