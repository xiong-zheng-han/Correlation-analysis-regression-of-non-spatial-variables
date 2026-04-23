"""Regression model definitions for curve fitting."""

from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.metrics import r2_score


@dataclass
class RegressionResult:
    """Result of a regression model fit."""
    model_name: str
    r_squared: float
    formula: str
    coefficients: dict
    success: bool


class RegressionModels:
    """Container for 11 regression models."""

    # Model names in Chinese and English
    MODEL_LINEAR = "线性"
    MODEL_LOGARITHMIC = "对数"
    MODEL_INVERSE = "逆"
    MODEL_QUADRATIC = "二次"
    MODEL_CUBIC = "三次"
    MODEL_POWER = "功率"
    MODEL_COMPOUND = "复合"
    MODEL_S_CURVE = "S曲线"
    MODEL_LOGISTIC = "Logistic"
    MODEL_GROWTH = "增长"
    MODEL_EXPONENTIAL = "指数"

    @staticmethod
    def _linear_func(t: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """Linear: Y = b0 + b1*t"""
        return b0 + b1 * t

    @staticmethod
    def _logarithmic_func(t: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """Logarithmic: Y = b0 + b1*ln(t)"""
        return b0 + b1 * np.log(t)

    @staticmethod
    def _inverse_func(t: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """Inverse: Y = b0 + b1/t"""
        return b0 + b1 / t

    @staticmethod
    def _quadratic_func(t: np.ndarray, b0: float, b1: float, b2: float) -> np.ndarray:
        """Quadratic: Y = b0 + b1*t + b2*t^2"""
        return b0 + b1 * t + b2 * t**2

    @staticmethod
    def _cubic_func(t: np.ndarray, b0: float, b1: float, b2: float, b3: float) -> np.ndarray:
        """Cubic: Y = b0 + b1*t + b2*t^2 + b3*t^3"""
        return b0 + b1 * t + b2 * t**2 + b3 * t**3

    @staticmethod
    def _power_func(t: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """Power: Y = b0 * t^b1"""
        return b0 * np.power(t, b1)

    @staticmethod
    def _compound_func(t: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """Compound: Y = b0 * b1^t"""
        return b0 * np.power(b1, t)

    @staticmethod
    def _s_curve_func(t: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """S-Curve: Y = exp(b0 + b1/t)"""
        return np.exp(b0 + b1 / t)

    @staticmethod
    def _logistic_func(t: np.ndarray, b0: float, b1: float, u: float) -> np.ndarray:
        """Logistic: Y = 1 / (1/u + b0*b1^t)"""
        return 1 / (1/u + b0 * np.power(b1, t))

    @staticmethod
    def _growth_func(t: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """Growth: Y = exp(b0 + b1*t)"""
        return np.exp(b0 + b1 * t)

    @staticmethod
    def _exponential_func(t: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """Exponential: Y = b0 * exp(b1*t)"""
        return b0 * np.exp(b1 * t)

    @staticmethod
    def _format_coefficient(value: float, decimals: int = 4) -> str:
        """Format a coefficient for display in formula."""
        if value >= 0:
            return f"{value:.{decimals}f}"
        else:
            return f"({value:.{decimals}f})"

    @classmethod
    def _linear_formula(cls, b0: float, b1: float) -> str:
        """Generate formula string for linear model."""
        return f"Y = {cls._format_coefficient(b0)} + {cls._format_coefficient(b1)}*t"

    @classmethod
    def _logarithmic_formula(cls, b0: float, b1: float) -> str:
        """Generate formula string for logarithmic model."""
        return f"Y = {cls._format_coefficient(b0)} + {cls._format_coefficient(b1)}*ln(t)"

    @classmethod
    def _inverse_formula(cls, b0: float, b1: float) -> str:
        """Generate formula string for inverse model."""
        return f"Y = {cls._format_coefficient(b0)} + {cls._format_coefficient(b1)}/t"

    @classmethod
    def _quadratic_formula(cls, b0: float, b1: float, b2: float) -> str:
        """Generate formula string for quadratic model."""
        return f"Y = {cls._format_coefficient(b0)} + {cls._format_coefficient(b1)}*t + {cls._format_coefficient(b2)}*t²"

    @classmethod
    def _cubic_formula(cls, b0: float, b1: float, b2: float, b3: float) -> str:
        """Generate formula string for cubic model."""
        return f"Y = {cls._format_coefficient(b0)} + {cls._format_coefficient(b1)}*t + {cls._format_coefficient(b2)}*t² + {cls._format_coefficient(b3)}*t³"

    @classmethod
    def _power_formula(cls, b0: float, b1: float) -> str:
        """Generate formula string for power model."""
        return f"Y = {cls._format_coefficient(b0)} * t^{cls._format_coefficient(b1)}"

    @classmethod
    def _compound_formula(cls, b0: float, b1: float) -> str:
        """Generate formula string for compound model."""
        return f"Y = {cls._format_coefficient(b0)} * {cls._format_coefficient(b1)}^t"

    @classmethod
    def _s_curve_formula(cls, b0: float, b1: float) -> str:
        """Generate formula string for S-curve model."""
        return f"Y = exp({cls._format_coefficient(b0)} + {cls._format_coefficient(b1)}/t)"

    @classmethod
    def _logistic_formula(cls, b0: float, b1: float, u: float) -> str:
        """Generate formula string for logistic model."""
        return f"Y = 1 / (1/{cls._format_coefficient(u)} + {cls._format_coefficient(b0)}*{cls._format_coefficient(b1)}^t)"

    @classmethod
    def _growth_formula(cls, b0: float, b1: float) -> str:
        """Generate formula string for growth model."""
        return f"Y = exp({cls._format_coefficient(b0)} + {cls._format_coefficient(b1)}*t)"

    @classmethod
    def _exponential_formula(cls, b0: float, b1: float) -> str:
        """Generate formula string for exponential model."""
        return f"Y = {cls._format_coefficient(b0)} * exp({cls._format_coefficient(b1)}*t)"

    @classmethod
    def fit_model(cls, model_name: str,
                  x: np.ndarray,
                  y: np.ndarray) -> Optional[RegressionResult]:
        """
        Fit a specific regression model.

        Args:
            model_name: Name of the model to fit
            x: Independent variable values
            y: Dependent variable values

        Returns:
            RegressionResult if successful, None if failed
        """
        try:
            # Ensure x and y are numpy arrays
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)

            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) < 2:
                return None

            result = None

            if model_name == cls.MODEL_LINEAR:
                result = cls._fit_linear(x_clean, y_clean)
            elif model_name == cls.MODEL_LOGARITHMIC:
                result = cls._fit_logarithmic(x_clean, y_clean)
            elif model_name == cls.MODEL_INVERSE:
                result = cls._fit_inverse(x_clean, y_clean)
            elif model_name == cls.MODEL_QUADRATIC:
                result = cls._fit_quadratic(x_clean, y_clean)
            elif model_name == cls.MODEL_CUBIC:
                result = cls._fit_cubic(x_clean, y_clean)
            elif model_name == cls.MODEL_POWER:
                result = cls._fit_power(x_clean, y_clean)
            elif model_name == cls.MODEL_COMPOUND:
                result = cls._fit_compound(x_clean, y_clean)
            elif model_name == cls.MODEL_S_CURVE:
                result = cls._fit_s_curve(x_clean, y_clean)
            elif model_name == cls.MODEL_LOGISTIC:
                result = cls._fit_logistic(x_clean, y_clean)
            elif model_name == cls.MODEL_GROWTH:
                result = cls._fit_growth(x_clean, y_clean)
            elif model_name == cls.MODEL_EXPONENTIAL:
                result = cls._fit_exponential(x_clean, y_clean)

            return result

        except Exception:
            return None

    @classmethod
    def _fit_linear(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit linear model: Y = b0 + b1*t"""
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            r_squared = r_value ** 2
            formula = cls._linear_formula(intercept, slope)
            return RegressionResult(
                model_name=cls.MODEL_LINEAR,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": intercept, "b1": slope},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def _fit_logarithmic(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit logarithmic model: Y = b0 + b1*ln(t)"""
        try:
            if np.any(x <= 0):
                return None
            x_transformed = np.log(x)
            slope, intercept, r_value, p_value, std_err = linregress(x_transformed, y)
            r_squared = r_value ** 2
            formula = cls._logarithmic_formula(intercept, slope)
            return RegressionResult(
                model_name=cls.MODEL_LOGARITHMIC,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": intercept, "b1": slope},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def _fit_inverse(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit inverse model: Y = b0 + b1/t"""
        try:
            if np.any(x == 0):
                return None
            x_transformed = 1 / x
            slope, intercept, r_value, p_value, std_err = linregress(x_transformed, y)
            r_squared = r_value ** 2
            formula = cls._inverse_formula(intercept, slope)
            return RegressionResult(
                model_name=cls.MODEL_INVERSE,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": intercept, "b1": slope},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def _fit_quadratic(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit quadratic model: Y = b0 + b1*t + b2*t^2"""
        try:
            if len(x) < 3:
                return None
            # Create polynomial features: t and t^2
            X = np.column_stack([x, x**2])
            # Add intercept column
            X = np.column_stack([np.ones(len(x)), X])

            # Solve using least squares
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

            b0, b1, b2 = coeffs
            y_pred = b0 + b1 * x + b2 * x**2
            r_squared = r2_score(y, y_pred)
            formula = cls._quadratic_formula(b0, b1, b2)
            return RegressionResult(
                model_name=cls.MODEL_QUADRATIC,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": b0, "b1": b1, "b2": b2},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def _fit_cubic(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit cubic model: Y = b0 + b1*t + b2*t^2 + b3*t^3"""
        try:
            if len(x) < 4:
                return None
            # Create polynomial features: t, t^2, t^3
            X = np.column_stack([x, x**2, x**3])
            # Add intercept column
            X = np.column_stack([np.ones(len(x)), X])

            # Solve using least squares
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

            b0, b1, b2, b3 = coeffs
            y_pred = b0 + b1 * x + b2 * x**2 + b3 * x**3
            r_squared = r2_score(y, y_pred)
            formula = cls._cubic_formula(b0, b1, b2, b3)
            return RegressionResult(
                model_name=cls.MODEL_CUBIC,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": b0, "b1": b1, "b2": b2, "b3": b3},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def _fit_power(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit power model: Y = b0 * t^b1"""
        try:
            if np.any(x <= 0) or np.any(y <= 0):
                return None

            # Take log of both sides: ln(Y) = ln(b0) + b1*ln(t)
            log_x = np.log(x)
            log_y = np.log(y)

            slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

            b0 = np.exp(intercept)
            b1 = slope
            r_squared = r_value ** 2
            formula = cls._power_formula(b0, b1)

            return RegressionResult(
                model_name=cls.MODEL_POWER,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": b0, "b1": b1},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def _fit_compound(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit compound model: Y = b0 * b1^t"""
        try:
            if np.any(y <= 0):
                return None

            # Take log: ln(Y) = ln(b0) + t*ln(b1)
            # This is linear in t
            log_y = np.log(y)

            slope, intercept, r_value, p_value, std_err = linregress(x, log_y)

            b0 = np.exp(intercept)
            b1 = np.exp(slope)
            r_squared = r_value ** 2
            formula = cls._compound_formula(b0, b1)

            return RegressionResult(
                model_name=cls.MODEL_COMPOUND,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": b0, "b1": b1},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def _fit_s_curve(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit S-curve model: Y = exp(b0 + b1/t)"""
        try:
            if np.any(x == 0) or np.any(y <= 0):
                return None

            # Take log: ln(Y) = b0 + b1/t
            log_y = np.log(y)
            x_transformed = 1 / x

            slope, intercept, r_value, p_value, std_err = linregress(x_transformed, log_y)

            b0 = intercept
            b1 = slope
            r_squared = r_value ** 2
            formula = cls._s_curve_formula(b0, b1)

            return RegressionResult(
                model_name=cls.MODEL_S_CURVE,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": b0, "b1": b1},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def _fit_logistic(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit logistic model: Y = 1 / (1/u + b0*b1^t)"""
        try:
            if np.any(y <= 0):
                return None

            # Calculate upper bound u
            y_max = np.max(y)
            y_min = np.min(y)
            u = y_max + (y_max - y_min) * 0.1

            # Transform: ln(1/y - 1/u) = ln(b0) + t*ln(b1)
            # This requires 1/y - 1/u > 0, which means y < u

            if u <= y_max:
                return None

            transformed_y = 1 / y - 1 / u

            if np.any(transformed_y <= 0):
                return None

            log_transformed_y = np.log(transformed_y)

            slope, intercept, r_value, p_value, std_err = linregress(x, log_transformed_y)

            b0 = np.exp(intercept)
            b1 = np.exp(slope)
            r_squared = r_value ** 2
            formula = cls._logistic_formula(b0, b1, u)

            return RegressionResult(
                model_name=cls.MODEL_LOGISTIC,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": b0, "b1": b1, "u": u},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def _fit_growth(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit growth model: Y = exp(b0 + b1*t)"""
        try:
            if np.any(y <= 0):
                return None

            # Take log: ln(Y) = b0 + b1*t
            log_y = np.log(y)

            slope, intercept, r_value, p_value, std_err = linregress(x, log_y)

            b0 = intercept
            b1 = slope
            r_squared = r_value ** 2
            formula = cls._growth_formula(b0, b1)

            return RegressionResult(
                model_name=cls.MODEL_GROWTH,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": b0, "b1": b1},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def _fit_exponential(cls, x: np.ndarray, y: np.ndarray) -> Optional[RegressionResult]:
        """Fit exponential model: Y = b0 * exp(b1*t)"""
        try:
            if np.any(y <= 0):
                return None

            # Take log: ln(Y) = ln(b0) + b1*t
            log_y = np.log(y)

            slope, intercept, r_value, p_value, std_err = linregress(x, log_y)

            b0 = np.exp(intercept)
            b1 = slope
            r_squared = r_value ** 2
            formula = cls._exponential_formula(b0, b1)

            return RegressionResult(
                model_name=cls.MODEL_EXPONENTIAL,
                r_squared=r_squared,
                formula=formula,
                coefficients={"b0": b0, "b1": b1},
                success=True
            )
        except Exception:
            return None

    @classmethod
    def get_all_model_names(cls) -> list[str]:
        """Get list of all model names."""
        return [
            cls.MODEL_LINEAR,
            cls.MODEL_LOGARITHMIC,
            cls.MODEL_INVERSE,
            cls.MODEL_QUADRATIC,
            cls.MODEL_CUBIC,
            cls.MODEL_POWER,
            cls.MODEL_COMPOUND,
            cls.MODEL_S_CURVE,
            cls.MODEL_LOGISTIC,
            cls.MODEL_GROWTH,
            cls.MODEL_EXPONENTIAL
        ]
