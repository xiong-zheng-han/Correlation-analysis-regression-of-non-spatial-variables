"""Spearman correlation analysis module."""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


class SpearmanResult:
    """Result of Spearman correlation analysis."""

    def __init__(self, full_results: pd.DataFrame, filtered_results: pd.DataFrame):
        self.full_results = full_results  # All correlations
        self.filtered_results = filtered_results  # P-value < 0.1 only


def spearman_correlation(dependent_var_name: str,
                         dependent_data: pd.Series,
                         independent_df: pd.DataFrame) -> SpearmanResult:
    """
    Calculate Spearman correlation between dependent variable and all independent variables.

    Args:
        dependent_var_name: Name of the dependent variable
        dependent_data: Dependent variable data (Series)
        independent_df: DataFrame containing all independent variables

    Returns:
        SpearmanResult containing full and filtered results
    """
    results = []
    dependent_values = dependent_data.values

    for ind_col in independent_df.columns:
        ind_values = independent_df[ind_col].values

        # Remove NaN values
        mask = ~(np.isnan(dependent_values) | np.isnan(ind_values))
        dep_clean = dependent_values[mask]
        ind_clean = ind_values[mask]

        if len(dep_clean) < 3:  # Need at least 3 points for correlation
            results.append({
                "自变量名称": ind_col,
                "相关性数值": np.nan,
                "P值": np.nan
            })
            continue

        try:
            correlation, p_value = spearmanr(ind_clean, dep_clean)
            results.append({
                "自变量名称": ind_col,
                "相关性数值": correlation if not np.isnan(correlation) else np.nan,
                "P值": p_value if not np.isnan(p_value) else np.nan
            })
        except Exception:
            results.append({
                "自变量名称": ind_col,
                "相关性数值": np.nan,
                "P值": np.nan
            })

    full_df = pd.DataFrame(results)

    # Filter by P-value < 0.1
    filtered_df = full_df[full_df["P值"] < 0.1].copy()

    return SpearmanResult(full_df, filtered_df)


def save_spearman_results(result: SpearmanResult,
                          output_dir: Path,
                          dependent_var_name: str) -> Path:
    """
    Save Spearman correlation results to Excel.

    Args:
        result: SpearmanResult containing analysis results
        output_dir: Output directory
        dependent_var_name: Name of dependent variable for filename

    Returns:
        Path to saved Excel file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dependent_var_name}_相关性.xlsx"
    filepath = output_dir / filename

    # Save with two sheets: full results and filtered results
    with pd.ExcelWriter(filepath, engine='openpyxl', mode='w') as writer:
        result.full_results.to_excel(writer, sheet_name="Sheet1", index=False)
        result.filtered_results.to_excel(writer, sheet_name="Sheet2", index=False)

    return filepath


def get_correlation_summary(result: SpearmanResult) -> str:
    """
    Get a summary string of correlation results.

    Args:
        result: SpearmanResult

    Returns:
        Summary string
    """
    total = len(result.full_results)
    significant = len(result.filtered_results)

    return f"共分析 {total} 个自变量，其中 {significant} 个变量与因变量显著相关 (P < 0.1)"


def get_independent_variables_for_regression(result: SpearmanResult) -> list[str]:
    """
    Get list of independent variables that should be used for regression analysis.

    Args:
        result: SpearmanResult

    Returns:
        List of independent variable names with P-value < 0.1
    """
    return result.filtered_results["自变量名称"].tolist()
