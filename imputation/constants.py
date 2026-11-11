"""Names of the supported imputation methods, initial imputations and visit sequences."""

from __future__ import annotations

from enum import Enum


class ImputationMethod(Enum):
    """
    Enumeration of supported imputation methods for MICE.

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
    """

    PMM = "pmm"
    MIDAS = "midas"
    CART = "cart"
    SAMPLE = "sample"
    RF = "rf"

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
        Get the default imputation method.

        Returns
        -------
        str
            Default method name, ``"pmm"`` as in R's ``mice`` for numeric
            columns. R uses logreg, polyreg and polr for factors; this
            package has none of those and uses pmm for every column type.
        """
        return cls.PMM.value


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


SUPPORTED_METHODS = ImputationMethod.get_all_methods()
DEFAULT_METHOD = ImputationMethod.get_default_method()
SUPPORTED_INITIAL_METHODS = InitialMethod.get_all_methods()
DEFAULT_INITIAL_METHOD = InitialMethod.get_default_method()
SUPPORTED_VISIT_SEQUENCES = VisitSequence.get_all_sequences()
DEFAULT_VISIT_SEQUENCE = VisitSequence.get_default_sequence()
