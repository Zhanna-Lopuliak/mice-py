References
==========

Multiple imputation and MICE
----------------------------

**Rubin, D. B. (1976)**
   Inference and missing data. *Biometrika*, 63(3), 581-592.

**Rubin, D. B. (1987)**
   *Multiple Imputation for Nonresponse in Surveys*. New York: John Wiley &
   Sons. (p. 167: the Bayesian linear regression draw of ``pmm`` and
   ``norm``.)

**Barnard, J., & Rubin, D. B. (1999)**
   Small-sample degrees of freedom with multiple imputation. *Biometrika*,
   86(4), 948-955.

**Van Buuren, S., & Groothuis-Oudshoorn, K. (2011)**
   mice: Multivariate Imputation by Chained Equations in R. *Journal of
   Statistical Software*, 45(3), 1-67.

**Van Buuren, S. (2018)**
   *Flexible Imputation of Missing Data* (2nd ed.). Chapman & Hall/CRC Press.
   Free online at https://stefvanbuuren.name/fimd/

**White, I. R., Royston, P., & Wood, A. M. (2011)**
   Multiple imputation using chained equations: Issues and guidance for
   practice. *Statistics in Medicine*, 30(4), 377-399.

**Little, R. J. A., & Rubin, D. B. (2019)**
   *Statistical Analysis with Missing Data* (3rd ed.). Hoboken, NJ: John
   Wiley & Sons. (The 2nd edition, 2002, pp. 61-62, is the reference of R's
   ``mean`` method.)

Imputation methods
------------------

**Gaffert, P., Meinfelder, F., & Bosch, V. (2018)**
   Towards an MI-proper predictive mean matching. Discussion paper, JSM 2018.
   (R's ``midastouch``, mice-py's ``midas``.)

**Siddique, J., & Belin, T. R. (2008)**
   Multiple imputation using an iterative hot-deck with distance-based donor
   selection. *Statistics in Medicine*, 27(1), 83-102.

**Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984)**
   *Classification and Regression Trees*. Monterey, CA: Wadsworth.
   (``cart``)

**Doove, L. L., van Buuren, S., & Dusseldorp, E. (2014)**
   Recursive partitioning for missing data imputation in the presence of
   interaction effects. *Computational Statistics & Data Analysis*, 72,
   92-104. (``cart`` and ``rf``)

**White, I. R., Daniel, R., & Royston, P. (2010)**
   Avoiding bias due to perfect prediction in multiple imputation of
   incomplete categorical variables. *Computational Statistics & Data
   Analysis*, 54, 2267-2275. (The augmentation of ``logreg``, ``polyreg``
   and ``polr``.)

**Venables, W. N., & Ripley, B. D. (2002)**
   *Modern Applied Statistics with S* (4th ed.). New York: Springer. (The
   book of the R packages MASS and nnet, whose ``polr`` and ``multinom``
   are the fits of ``polr`` and ``polyreg``.)

**Nash, J. C. (1990)**
   *Compact Numerical Methods for Computers: Linear Algebra and Function
   Minimisation* (2nd ed.). Adam Hilger. (The variable-metric
   minimiser of R's ``optim``, ``imputation.fitting.vmmin``.)

Citing mice-py
--------------

.. code-block:: bibtex

   @software{mice-py,
     title = {mice-py: Multiple Imputation by Chained Equations in Python},
     author = {Lopuliak, Zhanna and Vu, The Anh and Haensch, Anna-Carolina},
     year = {2025},
     url = {https://github.com/Zhanna-Lopuliak/mice-py},
     version = {0.3.0}
   }

and R mice, which mice-py ports:

.. code-block:: bibtex

   @article{vanbuuren2011mice,
     title = {mice: Multivariate Imputation by Chained Equations in R},
     author = {Van Buuren, Stef and Groothuis-Oudshoorn, Karin},
     journal = {Journal of Statistical Software},
     volume = {45},
     number = {3},
     pages = {1--67},
     year = {2011}
   }
