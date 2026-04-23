"""Failure tracking for regression analysis."""

from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class FailureRecord:
    """Record of a failed regression model."""
    independent_var: str
    dependent_var: str
    model_name: str
    reason: str


class FailureTracker:
    """Track and report failed regression model fits."""

    def __init__(self):
        self.failures: List[FailureRecord] = []

    def add_failure(self,
                   independent_var: str,
                   dependent_var: str,
                   model_name: str,
                   reason: str) -> None:
        """
        Record a failed regression model fit.

        Args:
            independent_var: Name of independent variable
            dependent_var: Name of dependent variable
            model_name: Name of the regression model
            reason: Reason for failure
        """
        record = FailureRecord(
            independent_var=independent_var,
            dependent_var=dependent_var,
            model_name=model_name,
            reason=reason
        )
        self.failures.append(record)

    def has_failures(self) -> bool:
        """Check if any failures have been recorded."""
        return len(self.failures) > 0

    def get_failure_count(self) -> int:
        """Get total number of failures."""
        return len(self.failures)

    def get_failures_by_model(self, model_name: str) -> List[FailureRecord]:
        """Get all failures for a specific model."""
        return [f for f in self.failures if f.model_name == model_name]

    def get_failures_by_variables(self,
                                  independent_var: str,
                                  dependent_var: str) -> List[FailureRecord]:
        """Get all failures for a specific variable pair."""
        return [
            f for f in self.failures
            if f.independent_var == independent_var and f.dependent_var == dependent_var
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert failures to a DataFrame for export."""
        if not self.failures:
            return pd.DataFrame(columns=["自变量", "因变量", "模型名称", "失败原因"])

        data = {
            "自变量": [f.independent_var for f in self.failures],
            "因变量": [f.dependent_var for f in self.failures],
            "模型名称": [f.model_name for f in self.failures],
            "失败原因": [f.reason for f in self.failures]
        }
        return pd.DataFrame(data)

    def save_to_excel(self, filepath: Path) -> None:
        """
        Save failure records to an Excel file.

        Args:
            filepath: Path to save the Excel file
        """
        if not self.has_failures():
            # Create empty file with headers
            df = pd.DataFrame(columns=["自变量", "因变量", "模型名称", "失败原因"])
            df.to_excel(filepath, index=False, engine='openpyxl')
            return

        df = self.to_dataframe()
        df.to_excel(filepath, index=False, engine='openpyxl')

    def get_summary(self) -> str:
        """Get a summary string of failures."""
        if not self.has_failures():
            return "所有回归模型拟合成功"

        # Count failures by model
        model_counts = {}
        for f in self.failures:
            model_counts[f.model_name] = model_counts.get(f.model_name, 0) + 1

        lines = [f"共记录 {len(self.failures)} 次拟合失败:"]
        for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {model}: {count} 次")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all failure records."""
        self.failures.clear()
