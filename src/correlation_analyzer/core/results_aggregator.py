"""Results aggregation and ranking module."""

from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

from .spearman_analysis import SpearmanResult
from .regression_analysis import RegressionResult


def aggregate_results(spearman_result: SpearmanResult,
                     regression_result: RegressionResult) -> pd.DataFrame:
    """
    Aggregate Spearman correlation and regression R² values.

    Calculates: 累加值 = |斯皮尔曼相关系数| + 所有模型R²之和

    New logic: Instead of using only the best R² value, this function now sums
    all R² values from the 11 regression models for each independent variable.

    Args:
        spearman_result: Spearman correlation results
        regression_result: Regression analysis results

    Returns:
        DataFrame with aggregated results, sorted by 累加值 in descending order
    """
    # Get the filtered Spearman results (P < 0.1)
    spearman_df = spearman_result.filtered_results.copy()

    # Get sum of all R² for each independent variable
    total_r2 = get_total_r2_for_each_independent(regression_result)

    # Build aggregation data
    agg_data = []

    for _, row in spearman_df.iterrows():
        ind_var = row["自变量名称"]
        spearman_corr = row["相关性数值"]
        spearman_abs = abs(spearman_corr) if pd.notna(spearman_corr) else 0.0

        # Get sum of all R² for this independent variable
        r2_sum = total_r2.get(ind_var, 0.0)

        # Calculate accumulated value
        accumulated = spearman_abs + r2_sum

        agg_data.append({
            "自变量名称": ind_var,
            "累加值": accumulated,
            "斯皮尔曼相关系数": spearman_corr,
            "斯皮尔曼相关系数_绝对值": spearman_abs,
            "R方总和": r2_sum
        })

    # Create DataFrame
    agg_df = pd.DataFrame(agg_data)

    # Sort by accumulated value in descending order
    agg_df = agg_df.sort_values("累加值", ascending=False).reset_index(drop=True)

    return agg_df


def get_total_r2_for_each_independent(regression_result: RegressionResult) -> Dict[str, float]:
    """
    Get the sum of all R² values for each independent variable.

    Args:
        regression_result: Regression analysis results

    Returns:
        Dictionary mapping independent variable name to sum of all R² values
    """
    total_r2 = {}

    if regression_result.summary_df is None:
        return total_r2

    for _, row in regression_result.summary_df.iterrows():
        ind_var = row["自变量名称"]

        # Get all R² values for this independent variable and sum them
        r2_sum = 0.0
        for col in regression_result.summary_df.columns:
            if col != "自变量名称" and pd.notna(row[col]):
                r2_sum += row[col]

        total_r2[ind_var] = r2_sum

    return total_r2


def save_aggregated_results(agg_df: pd.DataFrame,
                           regression_result: RegressionResult,
                           output_dir: Path,
                           dependent_var_name: str) -> Path:
    """
    Save aggregated results to Excel.

    Args:
        agg_df: Aggregated results DataFrame
        regression_result: Regression results (for getting all R² values)
        output_dir: Output directory
        dependent_var_name: Name for filename

    Returns:
        Path to saved Excel file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dependent_var_name}_汇总.xlsx"
    filepath = output_dir / filename

    # Prepare full data with all model R² values
    full_data = []

    for _, row in agg_df.iterrows():
        ind_var = row["自变量名称"]

        data_row = {
            "自变量名称": ind_var,
            "累加值": row["累加值"],
            "斯皮尔曼相关系数": row["斯皮尔曼相关系数"],
            "斯皮尔曼相关系数_绝对值": row["斯皮尔曼相关系数_绝对值"],
            "R方总和": row["R方总和"]
        }

        # Add R² values for each model
        if regression_result.summary_df is not None:
            ind_row = regression_result.summary_df[
                regression_result.summary_df["自变量名称"] == ind_var
            ]

            if len(ind_row) > 0:
                for col in regression_result.summary_df.columns:
                    if col != "自变量名称":
                        data_row[col] = ind_row[col].values[0] if len(ind_row) > 0 else None

        full_data.append(data_row)

    full_df = pd.DataFrame(full_data)

    # Save to Excel
    full_df.to_excel(filepath, index=False, engine='openpyxl')

    return filepath


def get_aggregation_summary(agg_df: pd.DataFrame) -> str:
    """
    Get a summary string of aggregated results.

    Args:
        agg_df: Aggregated results DataFrame

    Returns:
        Summary string
    """
    if len(agg_df) == 0:
        return "没有符合条件的自变量"

    top = agg_df.iloc[0]
    return (
        f"共 {len(agg_df)} 个自变量进入最终排名\n"
        f"排名第一: {top['自变量名称']} (累加值: {top['累加值']:.4f})"
    )
