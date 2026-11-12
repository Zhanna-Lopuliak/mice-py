"""Names of the supported imputation methods, initial imputations and visit sequences."""

from __future__ import annotations

from enum import Enum


class ImputationMethod(Enum):
    """
    Enumeration of supported imputation methods for MICE.

    The values are R's method names, dots included. The first five are
    donor methods: they impute observed values. The next five are for
    numeric targets and impute new values (floats). ``logreg`` and
    ``logreg.boot`` are for targets with two categories, ``polyreg`` for
    categorical targets with any number of categories, ``polr`` for
    ordered categorical targets; they impute labels. As in R, the default
    method of a column depends on its type (`get_default_methods`).

    Attributes
    ----------
    PMM
        Predictive mean matching, R's ``pmm``.
    MIDAS
        R's ``midastouch``: imputation by predictive mean matching with
        distance aided donor selection.
    CART
        Classification and regression trees, R's ``cart``.
    SAMPLE
        Random draws from the observed values, R's ``sample``.
    RF
        Random forest, R's ``rf``.
    NORM
        Bayesian linear regression, R's ``norm``.
    NORM_NOB
        Linear regression without parameter uncertainty, R's ``norm.nob``.
    NORM_BOOT
        Linear regression on a bootstrap sample, R's ``norm.boot``.
    NORM_PREDICT
        The predicted value of a linear regression, R's ``norm.predict``.
    MEAN
        The mean of the observed values, R's ``mean``.
    LOGREG
        Bayesian logistic regression on augmented data, R's ``logreg``.
    LOGREG_BOOT
        Logistic regression on a bootstrap sample, R's ``logreg.boot``.
    POLYREG
        Multinomial logistic regression on augmented data, R's ``polyreg``.
    POLR
        Proportional-odds logistic regression on augmented data, R's
        ``polr``; multinomial logistic regression where it cannot be fitted.
    """

    PMM = "pmm"
    MIDAS = "midas"
    CART = "cart"
    SAMPLE = "sample"
    RF = "rf"
    NORM = "norm"
    NORM_NOB = "norm.nob"
    NORM_BOOT = "norm.boot"
    NORM_PREDICT = "norm.predict"
    MEAN = "mean"
    LOGREG = "logreg"
    LOGREG_BOOT = "logreg.boot"
    POLYREG = "polyreg"
    POLR = "polr"

    @classmethod
    def get_all_methods(cls) -> list[str]:
        """
        Get all supported method names as a list of strings.

        Returns
        -------
        list of str
            List of all supported method names
        """
        return [method.value for method in cls]

    @classmethod
    def is_valid_method(cls, method: str) -> bool:
        """
        Check if a method string is valid.

        Parameters
        ----------
        method : str
            Method name to validate

        Returns
        -------
        bool
            True if method is valid, False otherwise
        """
        return method in cls.get_all_methods()

    @classmethod
    def get_default_method(cls) -> str:
        """
        Get the default imputation method of a numeric column.

        Returns
        -------
        str
            ``"pmm"``, as in R's ``mice``. Columns of other types get other
            defaults (see `get_default_methods`).
        """
        return cls.PMM.value

    @classmethod
    def get_default_methods(cls) -> tuple[str, str, str, str]:
        """
        Get the default imputation method of each column type, R's ``defaultMethod``.

        Returns
        -------
        tuple of str
            ``("pmm", "logreg", "polyreg", "polr")``: the methods of numeric
            columns, of columns with two categories (and boolean ones), of
            unordered categorical (and text) columns with more than two
            categories, and of ordered categorical columns with more than
            two categories, in this order. See `imputation.make_method`.
        """
        return (cls.PMM.value, cls.LOGREG.value, cls.POLYREG.value, cls.POLR.value)


class InitialMethod(Enum):
    """
    Enumeration of supported initial imputation methods for MICE.

    Attributes
    ----------
    SAMPLE
        Random draws from the observed values (R's default).
    MEANOBS
        The observed value closest to the mean of a numeric column, and the
        most frequent observed value of any other column (not in R).
    """

    SAMPLE = "sample"
    MEANOBS = "meanobs"

    @classmethod
    def get_all_methods(cls) -> list[str]:
        """
        Get all supported initial method names as a list of strings.

        Returns
        -------
        list of str
            List of all supported initial method names
        """
        return [method.value for method in cls]

    @classmethod
    def is_valid_method(cls, method: str) -> bool:
        """
        Check if an initial method string is valid.

        Parameters
        ----------
        method : str
            Initial method name to validate

        Returns
        -------
        bool
            True if method is valid, False otherwise
        """
        return method in cls.get_all_methods()

    @classmethod
    def get_default_method(cls) -> str:
        """
        Get the default initial imputation method.

        Returns
        -------
        str
            Default initial method name
        """
        return cls.SAMPLE.value


class VisitSequence(Enum):
    """
    Enumeration of supported visit sequence types for MICE.

    The keywords of R's ``mice``, plus ``random``. Only incomplete columns
    that are imputed are visited.

    Attributes
    ----------
    ROMAN
        Columns in data order (R's default).
    ARABIC
        Columns in reverse data order.
    MONOTONE
        Columns by increasing number of missing values; ties in data order.
    REVMONOTONE
        The reverse of MONOTONE.
    RANDOM
        A random order, drawn once per call of ``impute``.
    """

    ROMAN = "roman"
    ARABIC = "arabic"
    MONOTONE = "monotone"
    REVMONOTONE = "revmonotone"
    RANDOM = "random"

    @classmethod
    def get_all_sequences(cls) -> list[str]:
        """
        Get all supported visit sequence names as a list of strings.

        Returns
        -------
        list of str
            List of all supported visit sequence names
        """
        return [sequence.value for sequence in cls]

    @classmethod
    def is_valid_sequence(cls, sequence: str) -> bool:
        """
        Check if a visit sequence string is valid.

        Parameters
        ----------
        sequence : str
            Visit sequence name to validate

        Returns
        -------
        bool
            True if sequence is valid, False otherwise
        """
        return sequence in cls.get_all_sequences()

    @classmethod
    def get_default_sequence(cls) -> str:
        """
        Get the default visit sequence.

        Returns
        -------
        str
            Default visit sequence name (``"roman"``, as in R)
        """
        return cls.ROMAN.value


#: The names of the supported imputation methods.
SUPPORTED_METHODS = ImputationMethod.get_all_methods()
#: The default method of a numeric column, ``"pmm"``.
DEFAULT_METHOD = ImputationMethod.get_default_method()
#: The default methods by column type, R's ``defaultMethod``: numeric, two
#: categories (or boolean), unordered with more, ordered with more.
DEFAULT_METHODS = ImputationMethod.get_default_methods()
#: The names of the supported initial imputations.
SUPPORTED_INITIAL_METHODS = InitialMethod.get_all_methods()
#: The default initial imputation, ``"sample"``.
DEFAULT_INITIAL_METHOD = InitialMethod.get_default_method()
#: The keywords of the supported visit sequences.
SUPPORTED_VISIT_SEQUENCES = VisitSequence.get_all_sequences()
#: The default visit sequence, ``"roman"``.
DEFAULT_VISIT_SEQUENCE = VisitSequence.get_default_sequence()
