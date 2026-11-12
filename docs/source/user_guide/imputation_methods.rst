Imputation Methods
==================

mice-py has fourteen of R mice's univariate imputation methods. The five
**donor methods** fill a missing cell with a value **observed** in the same
column, so imputations stay within the range and the type of the data. The
nine **model-based methods** draw from a fitted model instead: the ``norm``
methods and ``mean`` impute new numbers in a numeric column, ``logreg``,
``polyreg`` and ``polr`` impute the labels of a categorical column. The
examples use ``df`` from the :ref:`example data <example-data>`.

.. list-table::
   :header-rows: 1
   :widths: 15 15 38 32

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
   * - ``"norm"``
     - ``norm``
     - Bayesian linear regression, new values
     - numeric
   * - ``"norm.boot"``
     - ``norm.boot``
     - linear regression on a bootstrap sample, new values
     - numeric
   * - ``"norm.nob"``
     - ``norm.nob``
     - linear regression without parameter uncertainty, new values
     - numeric
   * - ``"norm.predict"``
     - ``norm.predict``
     - the predicted value of a linear regression
     - numeric
   * - ``"mean"``
     - ``mean``
     - the mean of the observed values
     - numeric
   * - ``"logreg"``
     - ``logreg``
     - Bayesian logistic regression on augmented data
     - two categories: boolean, categorical, object, string, or numeric
       with two values
   * - ``"logreg.boot"``
     - ``logreg.boot``
     - logistic regression on a bootstrap sample
     - as ``logreg``
   * - ``"polyreg"``
     - ``polyreg``
     - multinomial logistic regression on augmented data
     - categorical (ordered or not), object, string; at most 50 categories
   * - ``"polr"``
     - ``polr``
     - proportional-odds logistic regression on augmented data
     - ordered categorical; at most 50 categories

Default methods
---------------

As in R, ``impute()`` without ``method`` gives each incomplete column the
method of its type (``imputation.make_method``, R's ``make.method``):

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Incomplete column
     - Default method
   * - numeric (also ``Int64``, ``Float64``)
     - ``"pmm"``
   * - boolean, or with two categories
     - ``"logreg"``
   * - unordered categorical, object or string, more than two categories
     - ``"polyreg"``
   * - ordered categorical, more than two categories
     - ``"polr"``
   * - anything else (for example a categorical column with one category)
     - ``"pmm"``

The categories of a ``category`` column are its declared categories, unused
ones included; those of an object or string column are its distinct
observed values. In ``df``, ``sex`` gets ``"logreg"`` and ``bmi`` and
``chl`` get ``"pmm"``. ``default_method`` replaces the four methods, in
this order (R's ``defaultMethod``); ``default_method=("pmm",) * 4`` gives
the default of mice-py 0.2, pmm for every column.

``polyreg`` and ``polr``, like R, model at most 50 categories. A text column
with more than 50 distinct values, such as names or identifiers, gets
``polyreg`` by default and makes ``impute`` raise a ``ValueError`` before
the chains start; give it a donor method (``method={"name": "pmm"}``) or
leave it out of the data.

Choose a method per column with a dict; columns left out get the default of
their type. Method parameters are passed to ``impute`` with the method name
as prefix, a dot in the name becoming an underscore (``norm_boot_ridge``
for parameter ``ridge`` of ``norm.boot``):

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

Before the chains start, ``impute`` checks each method against the type of
its column: a ``norm`` method or ``mean`` on a boolean, categorical or text
column, ``logreg`` or ``logreg.boot`` on more than two distinct values, and
a categorical method on more than 50 distinct values raise ``ValueError``.
A categorical method on a numeric column warns with R's text "Type mismatch
for variable(s)" and imputes the column's own values.

pmm: predictive mean matching
-----------------------------

R's default for numeric columns, and mice-py's.

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

norm: Bayesian linear regression
--------------------------------

R's ``norm`` (Rubin 1987, p. 167). A numeric column only; the imputations
are new values, not observed ones.

1. Fit a linear regression of the target on the predictors in the observed
   rows.
2. Draw the residual variance and then the coefficients from their posterior
   distribution.
3. Impute the prediction with the drawn coefficients plus a normal residual
   with the drawn standard deviation.

Use it for a numeric variable that is roughly normal given its predictors.
Unlike pmm it can impute values outside the observed range, and values a
variable cannot take (negative weights, fractional counts). A nullable
integer column (``Int64``, ...) comes back as ``Float64``.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter (in ``impute``)
     - Default
     - Meaning
   * - ``norm_ls_meth``
     - ``"qr"``
     - least-squares method: ``"qr"``, ``"ridge"`` or ``"svd"``
   * - ``norm_ridge``
     - 1e-5
     - ridge penalty of the regression

.. code-block:: python

   mice = MICE(df)
   mice.impute(method={"bmi": "norm", "chl": "norm"}, seed=1)

norm.boot: linear regression on a bootstrap sample
--------------------------------------------------

R's ``norm.boot``. The observed rows are resampled with replacement, the
least-squares fit on that sample gives the coefficients and the residual
standard deviation, and each imputation is the prediction plus a normal
residual. The bootstrap stands in for the posterior draw of ``norm``; both
are proper. Parameters ``norm_boot_ls_meth`` and ``norm_boot_ridge``, as for
``norm``.

.. code-block:: python

   mice.impute(method={"bmi": "norm.boot", "chl": "norm.boot"}, norm_boot_ridge=1e-4, seed=1)

The residual standard deviation divides the residual sum of squares by
``n_obs - q - 1`` (``q`` coefficients, intercept included), one less than
usual, as R does; with at most ``q + 1`` observed values ``impute`` raises
``ValueError`` (R imputes infinite values there).

norm.nob: linear regression without parameter uncertainty
---------------------------------------------------------

R's ``norm.nob``: the least-squares prediction plus a normal residual with
the estimated residual standard deviation (computed as for ``norm.boot``).
The coefficients are not drawn, so the method is not proper in Rubin's
sense: as R's help page warns, it understates the variability of the
imputations in small samples. R offers it mainly to compare proper and
improper imputation, and for large data, where the difference is small.
Parameters ``norm_nob_ls_meth`` and ``norm_nob_ridge``.

.. code-block:: python

   mice.impute(method={"chl": "norm.nob"}, seed=1)

norm.predict: regression imputation
-----------------------------------

R's ``norm.predict``: the least-squares prediction, without any random
component. It imputes the same value in every chain (given the predictors),
understates the variance and inflates the correlations of the completed
data. R's help page warns against using it for data analysis; in the
simulation study (scenario S4) the 95% intervals for the coefficient of the
imputed variable covered the truth in 35% of the replications, in R and in
mice-py. Parameters ``norm_predict_ls_meth`` and ``norm_predict_ridge``.

.. code-block:: python

   mice.impute(method={"chl": "norm.predict"}, seed=1)

mean: the observed mean
-----------------------

R's ``mean``: every missing cell gets the mean of the observed values of
the column, whatever the predictors. It has no parameters and draws no
random numbers. As R's help page warns, mean imputation is almost never
appropriate (Little and Rubin 2002, pp. 61-62): it understates the
variance and weakens the relations with other variables.

.. code-block:: python

   mice.impute(method={"bmi": "mean"}, seed=1)

logreg: logistic regression
---------------------------

R's ``logreg``, R's default for a column with two categories: boolean,
categorical, text, or numeric with two distinct values. The imputations are
labels of the column.

1. Augment the data with a few weighted rows (White, Daniel and Royston
   2010): for each predictor two rows at its mean ± half its standard
   deviation per category. They keep the fit finite when the predictors
   separate the categories.
2. Fit the logistic regression of the target on the predictors to the
   observed and the added rows (as R's ``glm.fit``).
3. Draw the coefficients from a normal distribution around the estimates,
   and give each missing cell the second category with the probability
   these coefficients predict.

The second category is the second of the sorted labels (``True``, ``"m"``
in ``df``), or the second declared category of an ordered categorical. The
method has no parameters; ``sex`` in ``df`` gets it by default.

.. code-block:: python

   mice.impute(method={"sex": "logreg"}, seed=1)

logreg.boot: logistic regression on a bootstrap sample
------------------------------------------------------

R's ``logreg.boot``: the logistic regression is fitted to a bootstrap
sample of the observed rows, without augmentation and without a draw of the
coefficients, and each cell gets the second category with the predicted
probability. No parameters. A bootstrap sample in which the predictors
separate the categories is fitted anyway, as in R, and gives probabilities
close to 0 or 1.

.. code-block:: python

   mice.impute(method={"sex": "logreg.boot"}, seed=1)

polyreg: multinomial logistic regression
----------------------------------------

R's ``polyreg``, R's default for an unordered categorical, object or string
column with more than two categories (it takes ordered ones too). The data
are augmented as for ``logreg``, the multinomial logistic regression is
fitted as R's ``nnet::multinom`` fits it, and each missing cell gets a
category drawn with the probabilities the fit predicts. There is no draw of
the coefficients: as in R, the augmentation is the method's only allowance
for the uncertainty of the fit. The fit stops where nnet's stops (at most
``maxit`` iterations), not at the exact maximum of the likelihood.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter (in ``impute``)
     - Default
     - Meaning
   * - ``polyreg_maxit``
     - 100
     - maximal number of iterations of the fit (R's ``nnet.maxit``)
   * - ``polyreg_max_weights``
     - 1500
     - maximal number of weights of the model (R's ``nnet.MaxNWts``);
       more raise ``ValueError``

The example data have no categorical column with more than two categories;
the example replaces ``bmi`` by its tertiles, an ordered categorical:

.. code-block:: python

   import pandas as pd

   df_groups = df.drop(columns="bmi").assign(
       bmi_group=pd.qcut(df["bmi"], 3, labels=["low", "middle", "high"])
   )
   mice_groups = MICE(df_groups)
   mice_groups.impute(method={"bmi_group": "polyreg"}, polyreg_maxit=200, seed=1)

At most 50 categories can be modelled, as in R.

polr: proportional-odds logistic regression
-------------------------------------------

R's ``polr``, R's default for an ordered categorical column with more than
two categories. The data are augmented as for ``logreg``, the
proportional-odds model ``P(y <= k) = logistic(zeta_k - x beta)`` is fitted
as R's ``MASS::polr`` fits it, and each missing cell gets a category drawn
with the probabilities the fit predicts; no draw of the coefficients, as in
``polyreg``. Where ``MASS::polr`` cannot be fitted (a boolean or numeric
column, two categories, start values that cannot be found), the
multinomial fit of ``polyreg`` is used instead, as in R.

The order of the categories is the declared one; a text or unordered
categorical column is taken as ordered by its sorted labels. Every declared
category counts, also one that does not occur: the model gives it a small
positive probability, so it can be imputed.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter (in ``impute``)
     - Default
     - Meaning
   * - ``polr_maxit``
     - 100
     - maximal number of iterations of the multinomial fallback (R's
       ``nnet.maxit``); the proportional-odds fit stops after 100, as in R
   * - ``polr_max_weights``
     - 1500
     - maximal number of weights of the fallback (R's ``nnet.MaxNWts``)
   * - ``polr_log_fallback``
     - ``False``
     - record "polr falls back to multinom" in ``logged_events`` whenever
       the fallback is used (R's ``polr.to.loggedEvents``)

.. code-block:: python

   mice_groups.impute(polr_log_fallback=True, seed=1)   # bmi_group is ordered: polr by default
   print(mice_groups.method)
   print(mice_groups.logged_events)                    # empty: polr could be fitted

Choosing a method
-----------------

- **pmm** is R's default for numeric data and a reasonable first choice;
  its imputations are observed values.
- **norm** and **norm.boot** suit numeric variables that are roughly normal
  given their predictors; they impute new values, also outside the observed
  range.
- **logreg**, **polyreg** and **polr** are R's defaults for categorical
  columns with two, more unordered and more ordered categories. pmm, cart,
  rf and sample impute categorical columns too.
- **midas** is an alternative to pmm for numeric targets, proposed to make
  predictive mean matching proper (Gaffert et al. 2018).
- **cart** and **rf** model interactions and non-linear relations without
  specifying them (Doove et al. 2014).
- **sample** ignores all relations; useful as a baseline.
- **norm.nob**, **norm.predict** and **mean** are improper: they ignore the
  uncertainty of the model (``norm.predict`` and ``mean`` also the residual
  noise), so standard errors after pooling are too small. Use them for
  comparisons, not for analyses.

Compare methods on your data with the plots of :doc:`diagnostic_plots`. The
simulation study of :doc:`../r_mice/validation` compares mice-py with R, not
the methods with each other.

Next steps
----------

- :doc:`predictor_matrices`
- :doc:`convergence_diagnostics`
