"""Excel data loader for the correlation analyzer."""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

from ..utils.validators import (
    validate_excel_file,
    validate_dataframe,
    validate_numeric_data,
    ValidationError
)


class LoadResult:
    """Result of loading an Excel file."""

    def __init__(self, filepath: Path, df: pd.DataFrame, variable_count: int):
        self.filepath = filepath
        self.df = df
        self.variable_count = variable_count
        self.variable_names = df.columns.tolist()


def load_excel_file(filepath: str | Path, file_type: str = "数据表") -> LoadResult:
    """
    Load an Excel file and validate its contents.

    The Excel file should have variable names in the first row (headers)
    and numeric data in subsequent rows.

    Args:
        filepath: Path to the Excel file
        file_type: Type of file for error messages (e.g., "自变量表格")

    Returns:
        LoadResult containing the loaded data

    Raises:
        ValidationError: If file is invalid
    """
    # Validate file path
    path = validate_excel_file(filepath)

    try:
        # Read Excel file - first row is used as column names (headers)
        df = pd.read_excel(path, engine='openpyxl')
    except Exception as e:
        raise ValidationError(f"读取Excel文件失败: {path}\n错误: {e}")

    # Validate DataFrame structure
    validate_dataframe(df, file_type)

    # Validate numeric data
    validate_numeric_data(df, file_type)

    # Convert all data to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for NaN values after conversion
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValidationError(
            f"{file_type}包含无法转换为数字的数据，请检查以下列: {', '.join(nan_cols)}"
        )

    variable_count = len(df.columns)

    return LoadResult(path, df, variable_count)


def load_both_files(independent_path: str | Path,
                   dependent_path: str | Path) -> Tuple[LoadResult, LoadResult]:
    """
    Load both independent and dependent variable files.

    Args:
        independent_path: Path to independent variables Excel file
        dependent_path: Path to dependent variables Excel file

    Returns:
        Tuple of (independent_result, dependent_result)

    Raises:
        ValidationError: If either file is invalid
    """
    independent_result = load_excel_file(independent_path, "自变量表格")
    dependent_result = load_excel_file(dependent_path, "因变量表格")

    return independent_result, dependent_result
