"""Data preprocessing module for right shift check."""

from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import numpy as np


class PreprocessResult:
    """Result of preprocessing a DataFrame."""

    def __init__(self, original_df: pd.DataFrame, processed_df: pd.DataFrame,
                 shift_info: Dict[str, float]):
        self.original_df = original_df
        self.processed_df = processed_df
        self.shift_info = shift_info  # {column_name: shift_value}


def right_shift_check(df: pd.DataFrame, df_name: str = "数据表") -> PreprocessResult:
    """
    Perform right shift check on DataFrame columns.

    For each column:
    - If minimum value > 0: no shift
    - If minimum value == 0: shift by 1
    - If minimum value < 0: shift by |min| + 1

    Args:
        df: Input DataFrame
        df_name: Name of the DataFrame for logging

    Returns:
        PreprocessResult containing processed data and shift information
    """
    processed_df = df.copy()
    shift_info = {}

    for col in df.columns:
        min_val = df[col].min()

        if min_val > 0:
            # No shift needed
            shift_info[col] = 0.0
        elif min_val == 0:
            # Shift by 1
            processed_df[col] = df[col] + 1
            shift_info[col] = 1.0
        else:  # min_val < 0
            # Shift by |min| + 1
            shift = abs(min_val) + 1
            processed_df[col] = df[col] + shift
            shift_info[col] = shift

    return PreprocessResult(df, processed_df, shift_info)


def preprocess_both_files(independent_df: pd.DataFrame,
                         dependent_df: pd.DataFrame) -> Tuple[PreprocessResult, PreprocessResult]:
    """
    Preprocess both independent and dependent variable DataFrames.

    Args:
        independent_df: Independent variables DataFrame
        dependent_df: Dependent variables DataFrame

    Returns:
        Tuple of (independent_result, dependent_result)
    """
    independent_result = right_shift_check(independent_df, "自变量表格")
    dependent_result = right_shift_check(dependent_df, "因变量表格")

    return independent_result, dependent_result


def save_preprocessed_data(result: PreprocessResult,
                          output_dir: Path,
                          filename: str) -> Path:
    """
    Save preprocessed data to Excel file.

    Args:
        result: PreprocessResult containing the processed data
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Create DataFrame with variable names as first row
    df_to_save = result.processed_df.copy()

    # Save to Excel
    df_to_save.to_excel(output_path, index=False, engine='openpyxl')

    return output_path


def get_shift_summary(result: PreprocessResult) -> str:
    """
    Get a summary string of shift operations.

    Args:
        result: PreprocessResult

    Returns:
        Summary string
    """
    lines = []
    shifted_cols = {k: v for k, v in result.shift_info.items() if v > 0}

    if not shifted_cols:
        return "所有变量无需偏移处理"

    lines.append(f"共{len(shifted_cols)}个变量进行了偏移处理:")
    for col, shift in shifted_cols.items():
        lines.append(f"  - {col}: 偏移 {shift}")

    return "\n".join(lines)
