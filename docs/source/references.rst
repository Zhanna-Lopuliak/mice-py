References
==========

Key references for multiple imputation, MICE methodology, and missing data analysis.

Foundational Works
------------------

**Rubin, D. B. (1976)**
   Inference and missing data. *Biometrika*, 63(3), 581-592.
   
   *Introduced the concepts of MCAR, MAR, and MNAR.*

**Rubin, D. B. (1987)**
   *Multiple Imputation for Nonresponse in Surveys*. New York: John Wiley & Sons.
   
   *Established the theoretical framework for multiple imputation and Rubin's rules.*

**Little, R. J., & Rubin, D. B. (2019)**
   *Statistical Analysis with Missing Data* (3rd ed.). Hoboken, NJ: John Wiley & Sons.
   
   *Comprehensive textbook on missing data theory and methods.*

MICE Methodology
----------------

**Van Buuren, S., & Groothuis-Oudshoorn, K. (2011)**
   mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1-67.
   
   *Original MICE implementation and comprehensive methodology description.*

**Van Buuren, S. (2018)**
   *Flexible Imputation of Missing Data* (2nd ed.). Boca Raton, FL: Chapman & Hall/CRC Press.
   
   *Detailed book on MICE with practical examples and theory.*

**Azur, M. J., Stuart, E. A., Frangakis, C., & Leaf, P. J. (2011)**
   Multiple imputation by chained equations: What is it and how does it work? 
   *International Journal of Methods in Psychiatric Research*, 20(1), 40-49.
   
   *Accessible introduction to MICE for practitioners.*

Pooling and Inference
---------------------

**Barnard, J., & Rubin, D. B. (1999)**
   Small-sample degrees of freedom with multiple imputation. *Biometrika*, 86(4), 948-955.
   
   *Improved degrees of freedom adjustment for small samples.*

**Marshall, A., Altman, D. G., Holder, R. L., & Royston, P. (2009)**
   Combining estimates of interest in prognostic modelling studies after multiple 
   imputation: Current practice and guidelines. *BMC Medical Research Methodology*, 9(1), 57.
   
   *Practical guidance on pooling results.*

Imputation Methods
------------------

**Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984)**
   *Classification and Regression Trees*. Boca Raton, FL: CRC Press.
   
   *Foundational work on CART methodology.*

**Breiman, L. (2001)**
   Random forests. *Machine Learning*, 45(1), 5-32.
   
   *Original random forest paper.*

**Schenker, N., & Taylor, J. M. (1996)**
   Partially parametric techniques for multiple imputation. *Computational Statistics 
   & Data Analysis*, 22(4), 425-446.
   
   *Theory behind predictive mean matching.*

**Little, R. J. (1988)**
   Missing-data adjustments in large surveys. *Journal of Business & Economic Statistics*, 
   6(3), 287-296.
   
   *Early work on hot-deck and donor-based imputation.*

Number of Imputations
---------------------

**White, I. R., Royston, P., & Wood, A. M. (2011)**
   Multiple imputation using chained equations: Issues and guidance for practice. 
   *Statistics in Medicine*, 30(4), 377-399.
   
   *Practical guidance including how many imputations to use.*

**Von Hippel, P. T. (2020)**
   How many imputations do you need? A two-stage calculation using a quadratic rule. 
   *Sociological Methods & Research*, 49(3), 699-718.
   
   *Modern recommendations for number of imputations.*

Missing Data Mechanisms
-----------------------

**Enders, C. K. (2010)**
   *Applied Missing Data Analysis*. New York: Guilford Press.
   
   *Comprehensive practical guide to missing data analysis.*

**Graham, J. W. (2009)**
   Missing data analysis: Making it work in the real world. *Annual Review of Psychology*, 
   60, 549-576.
   
   *Review of missing data methods and practical advice.*

Validation and Diagnostics
---------------------------

**Abayomi, K., Gelman, A., & Levy, M. (2008)**
   Diagnostics for multivariate imputations. *Journal of the Royal Statistical Society: 
   Series C (Applied Statistics)*, 57(3), 273-291.
   
   *Methods for checking imputation quality.*

**Van Buuren, S., & Oudshoorn, C. G. (2000)**
   *Multivariate Imputation by Chained Equations: MICE V1.0 User's Manual*. 
   TNO Prevention and Health, Leiden.
   
   *Original MICE documentation with diagnostic procedures.*

Congeniality and Compatibility
-------------------------------

**Meng, X. L. (1994)**
   Multiple-imputation inferences with uncongenial sources of input. *Statistical Science*, 
   9(4), 538-558.
   
   *Theory on compatibility between imputation and analysis models.*

**Bartlett, J. W., Seaman, S. R., White, I. R., & Carpenter, J. R. (2015)**
   Multiple imputation of covariates by fully conditional specification: Accommodating 
   the substantive model. *Statistical Methods in Medical Research*, 24(4), 462-487.
   
   *Making imputation model compatible with analysis model.*

Software and Implementation
---------------------------

**Pedregosa, F., et al. (2011)**
   Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 
   12, 2825-2830.
   
   *Scikit-learn used for CART and Random Forest implementation.*

**Seabold, S., & Perktold, J. (2010)**
   Statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 
   9th Python in Science Conference*, 57, 10-25080.
   
   *Statsmodels used for statistical modeling and pooling.*

Applications and Case Studies
------------------------------

**Sterne, J. A., et al. (2009)**
   Multiple imputation for missing data in epidemiological and clinical research: 
   Potential and pitfalls. *BMJ*, 338, b2393.
   
   *Practical guidance for applied researchers.*

**Hardt, J., Herke, M., Brian, T., & Laubach, W. (2013)**
   Multiple imputation of missing data: A simulation study on a binary response. 
   *Open Journal of Statistics*, 3(05), 370.
   
   *Simulation study comparing imputation methods.*

Online Resources
----------------

**MICE Website**
   https://stefvanbuuren.name/mice/
   
   *Official website for the R mice package with extensive documentation.*

**Stef van Buuren's Blog**
   https://www.gerkovink.com/miceVignettes/
   
   *Practical vignettes and examples for MICE.*

**Missing Data Resources**
   https://missingdata.org/
   
   *Comprehensive resource on missing data methods.*

Citing mice-py
--------------

If you use mice-py in your research, please cite:

.. code-block:: bibtex

   @software{mice-py,
     title = {mice-py: Multiple Imputation by Chained Equations in Python},
     author = {Haensch, Anna-Carolina and Vu, The Anh and Lopuliak, Zhanna},
     year = {2025},
     url = {https://github.com/Zhanna-Lopuliak/mice-py},
     version = {0.1.0}
   }

And consider citing the original MICE paper:

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

Additional Reading
------------------

For more resources on missing data and multiple imputation:

- :doc:`theory/index` - Theoretical background in these docs
- :doc:`user_guide/index` - Practical usage guides
- `mice package documentation <https://stefvanbuuren.name/mice/>`_
- `Missing Data Book <https://stefvanbuuren.name/fimd/>`_ by Stef van Buuren (free online)

