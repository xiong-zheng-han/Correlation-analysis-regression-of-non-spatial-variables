"""Regression analysis module with R² calculation."""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from ..models.regression_models import RegressionModels
from .failure_tracker import FailureTracker


class RegressionResult:
    """Result of regression analysis for a single dependent variable."""

    def __init__(self, dependent_var: str):
        self.dependent_var = dependent_var
        self.model_results: Dict[str, pd.DataFrame] = {}  # {model_name: DataFrame with results}
        self.summary_df: pd.DataFrame = None  # Summary of R² values


def fit_regression_models(dependent_var_name: str,
                         dependent_data: pd.Series,
                         independent_df: pd.DataFrame,
                         independent_var_list: List[str],
                         failure_tracker: FailureTracker) -> RegressionResult:
    """
    Fit all 11 regression models for each independent variable.

    Args:
        dependent_var_name: Name of dependent variable
        dependent_data: Dependent variable data
        independent_df: DataFrame with all independent variables
        independent_var_list: List of independent variable names to analyze
        failure_tracker: FailureTracker to record failed fits

    Returns:
        RegressionResult with all model results
    """
    result = RegressionResult(dependent_var_name)
    y_values = dependent_data.values

    # Get all model names
    model_names = RegressionModels.get_all_model_names()

    # Initialize data structures for each model
    model_data = {model: [] for model in model_names}

    # Fit each independent variable with all models
    for ind_var in independent_var_list:
        if ind_var not in independent_df.columns:
            continue

        x_values = independent_df[ind_var].values

        # Try to fit each model
        for model_name in model_names:
            fit_result = RegressionModels.fit_model(model_name, x_values, y_values)

            if fit_result is None or not fit_result.success:
                # Record failure
                failure_tracker.add_failure(
                    independent_var=ind_var,
                    dependent_var=dependent_var_name,
                    model_name=model_name,
                    reason="拟合失败或数据不满足模型要求"
                )
                model_data[model_name].append({
                    "自变量名称": ind_var,
                    "回归计算R方": None,
                    "具体回归函数": None
                })
            else:
                model_data[model_name].append({
                    "自变量名称": ind_var,
                    "回归计算R方": fit_result.r_squared,
                    "具体回归函数": fit_result.formula
                })

    # Create DataFrames for each model
    for model_name in model_names:
        df = pd.DataFrame(model_data[model_name])
        result.model_results[model_name] = df

    # Create summary DataFrame
    summary_data = []
    for ind_var in independent_var_list:
        row = {"自变量名称": ind_var}
        for model_name in model_names:
            df = result.model_results[model_name]
            if ind_var in df["自变量名称"].values:
                r2 = df[df["自变量名称"] == ind_var]["回归计算R方"].values[0]
                row[model_name] = r2
            else:
                row[model_name] = None
        summary_data.append(row)

    result.summary_df = pd.DataFrame(summary_data)

    return result


def save_regression_results(result: RegressionResult,
                           output_dir: Path,
                           dependent_var_name: str) -> Path:
    """
    Save regression results to Excel with one sheet per model.

    Args:
        result: RegressionResult containing all analysis results
        output_dir: Output directory
        dependent_var_name: Name for filename

    Returns:
        Path to saved Excel file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dependent_var_name}_回归.xlsx"
    filepath = output_dir / filename

    # Save with one sheet per model plus a summary sheet
    with pd.ExcelWriter(filepath, engine='openpyxl', mode='w') as writer:
        # Save each model's results
        for model_name, df in result.model_results.items():
            df.to_excel(writer, sheet_name=model_name, index=False)

        # Save summary sheet
        result.summary_df.to_excel(writer, sheet_name="R方汇总", index=False)

    return filepath


def get_best_r2_for_each_independent(result: RegressionResult) -> Dict[str, float]:
    """
    Get the best R² value for each independent variable across all models.

    Args:
        result: RegressionResult

    Returns:
        Dictionary mapping independent variable name to best R² value
    """
    best_r2 = {}

    for ind_var in result.summary_df["自变量名称"]:
        # Get all R² values for this independent variable
        r2_values = []
        for model_name in RegressionModels.get_all_model_names():
            if model_name in result.summary_df.columns:
                r2 = result.summary_df[result.summary_df["自变量名称"] == ind_var][model_name].values
                if len(r2) > 0 and pd.notna(r2[0]):
                    r2_values.append(r2[0])

        if r2_values:
            best_r2[ind_var] = max(r2_values)
        else:
            best_r2[ind_var] = 0.0

    return best_r2


def get_regression_summary(result: RegressionResult) -> str:
    """
    Get a summary string of regression results.

    Args:
        result: RegressionResult

    Returns:
        Summary string
    """
    total_vars = len(result.summary_df)
    summary_lines = [f"对 {total_vars} 个自变量进行了回归分析"]

    # Count successful fits per model
    model_names = RegressionModels.get_all_model_names()
    for model_name in model_names:
        if model_name in result.model_results:
            df = result.model_results[model_name]
            success_count = df["回归计算R方"].notna().sum()
            summary_lines.append(f"  - {model_name}: {success_count}/{total_vars} 成功拟合")

    return "\n".join(summary_lines)
