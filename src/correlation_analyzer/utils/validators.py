"""Data validation utilities for the correlation analyzer."""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def validate_excel_file(filepath: str | Path) -> Path:
    """
    Validate that the file exists and is an Excel file.

    Args:
        filepath: Path to the Excel file

    Returns:
        Path object if valid

    Raises:
        ValidationError: If file doesn't exist or is not an Excel file
    """
    path = Path(filepath)

    if not path.exists():
        raise ValidationError(f"文件不存在: {filepath}")

    if not path.is_file():
        raise ValidationError(f"路径不是文件: {filepath}")

    suffix = path.suffix.lower()
    if suffix not in ['.xlsx', '.xls']:
        raise ValidationError(f"文件格式不支持，请选择Excel文件 (.xlsx 或 .xls): {filepath}")

    return path


def validate_dataframe(df: pd.DataFrame, file_type: str = "数据表") -> None:
    """
    Validate that the DataFrame meets the requirements.

    Args:
        df: DataFrame to validate
        file_type: Type of file for error messages (e.g., "自变量表格")

    Raises:
        ValidationError: If DataFrame doesn't meet requirements
    """
    if df is None or df.empty:
        raise ValidationError(f"{file_type}为空，请确保包含数据")

    if len(df) < 1:
        raise ValidationError(f"{file_type}至少需要1行数据")

    if len(df.columns) < 1:
        raise ValidationError(f"{file_type}至少需要1列变量")

    # Check column names (variable names)
    if df.columns.isna().any():
        raise ValidationError(f"{file_type}列名（变量名）不能为空")


def validate_numeric_data(df: pd.DataFrame, file_type: str = "数据表") -> None:
    """
    Validate that all data is numeric.

    Args:
        df: DataFrame to validate
        file_type: Type of file for error messages

    Raises:
        ValidationError: If non-numeric data is found
    """
    # Check each column for non-numeric data
    for col_idx, col_name in enumerate(df.columns):
        try:
            # Convert to numeric, this will raise errors if conversion fails
            pd.to_numeric(df[col_name], errors='raise')
        except (ValueError, TypeError):
            # Get first non-numeric value for error message
            non_numeric_mask = pd.to_numeric(df[col_name], errors='coerce').isna()
            if non_numeric_mask.any():
                non_numeric_idx = non_numeric_mask.idxmax()
                non_numeric_value = df.loc[non_numeric_idx, col_name]
                raise ValidationError(
                    f"{file_type}第{col_idx + 1}列 ('{col_name}') 包含非数字数据: '{non_numeric_value}'"
                )


def validate_workspace(workspace: Path) -> Path:
    """
    Validate and prepare workspace directory.

    Args:
        workspace: Path to workspace directory

    Returns:
        Path object

    Raises:
        ValidationError: If workspace is invalid
    """
    if workspace.exists() and not workspace.is_dir():
        raise ValidationError(f"工作空间路径不是目录: {workspace}")

    # Create workspace if it doesn't exist
    workspace.mkdir(parents=True, exist_ok=True)

    return workspace


def get_variable_info(df: pd.DataFrame) -> Tuple[int, list[str]]:
    """
    Get information about variables in the DataFrame.

    Args:
        df: DataFrame with first row as variable names

    Returns:
        Tuple of (number of variables, list of variable names)
    """
    variable_names = df.columns.tolist()
    return len(variable_names), variable_names


def calculate_total_rounds(independent_count: int, dependent_count: int) -> int:
    """
    Calculate total number of analysis rounds.

    Args:
        independent_count: Number of independent variables
        dependent_count: Number of dependent variables

    Returns:
        Total number of analysis rounds
    """
    return independent_count * dependent_count
