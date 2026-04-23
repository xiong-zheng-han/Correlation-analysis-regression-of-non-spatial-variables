"""PyQt6 main window for the correlation analyzer."""

from pathlib import Path
from typing import Optional
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox,
    QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import pandas as pd

from ..core.data_loader import LoadResult, load_excel_file
from ..core.preprocessor import PreprocessResult, preprocess_both_files, save_preprocessed_data
from ..core.spearman_analysis import spearman_correlation, save_spearman_results
from ..core.regression_analysis import fit_regression_models, save_regression_results
from ..core.results_aggregator import aggregate_results, save_aggregated_results
from ..core.failure_tracker import FailureTracker
from ..utils.validators import ValidationError


class AnalysisWorker(QThread):
    """Worker thread for running analysis in background."""

    # Signals
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    def __init__(self, independent_result: LoadResult,
                 dependent_result: LoadResult,
                 workspace: Path,
                 preprocess_results: tuple[PreprocessResult, PreprocessResult]):
        super().__init__()
        self.independent_result = independent_result
        self.dependent_result = dependent_result
        self.workspace = workspace
        self.preprocess_results = preprocess_results
        self.failure_tracker = FailureTracker()

    def run(self):
        """Run the complete analysis pipeline."""
        try:
            ind_preprocess, dep_preprocess = self.preprocess_results

            # Create output directories
            spearman_dir = self.workspace / "斯皮尔曼相关性分析结果"
            regression_dir = self.workspace / "回归分析结果"
            summary_dir = self.workspace / "系数结果排名汇总"

            independent_df = ind_preprocess.processed_df
            dependent_df = dep_preprocess.processed_df

            # Process each dependent variable
            total_vars = len(dependent_df.columns)

            for idx, dep_var in enumerate(dependent_df.columns, 1):
                self.log_message.emit(f"{'='*60}")
                self.progress_update.emit(
                    f"正在处理第{idx}/{total_vars}个因变量: {dep_var}"
                )

                dep_data = dependent_df[dep_var]

                # Step 1: Spearman correlation
                self.log_message.emit(f"开始斯皮尔曼相关性分析...")
                spearman_result = spearman_correlation(dep_var, dep_data, independent_df)
                save_spearman_results(spearman_result, spearman_dir, dep_var)
                self.log_message.emit(f"斯皮尔曼相关性分析完成: {spearman_result.filtered_results.shape[0]} 个显著相关变量")

                # Get independent variables for regression (P < 0.1)
                ind_vars_for_regression = spearman_result.filtered_results["自变量名称"].tolist()

                if not ind_vars_for_regression:
                    self.log_message.emit(f"没有显著相关的自变量 (P < 0.1)，跳过回归分析")
                    # Create empty summary
                    empty_summary = pd.DataFrame({"自变量名称": [], "累加值": []})
                    save_aggregated_results(empty_summary, None, summary_dir, dep_var)
                    continue

                # Step 2: Regression analysis
                self.log_message.emit(f"开始回归分析 ({len(ind_vars_for_regression)} 个自变量)...")
                regression_result = fit_regression_models(
                    dep_var, dep_data, independent_df, ind_vars_for_regression, self.failure_tracker
                )
                save_regression_results(regression_result, regression_dir, dep_var)
                self.log_message.emit(f"回归分析完成")

                # Step 3: Aggregate results
                self.log_message.emit(f"汇总结果...")
                agg_result = aggregate_results(spearman_result, regression_result)
                save_aggregated_results(agg_result, regression_result, summary_dir, dep_var)

                if len(agg_result) > 0:
                    top = agg_result.iloc[0]
                    self.log_message.emit(
                        f"最佳匹配: {top['自变量名称']} (累加值: {top['累加值']:.4f})"
                    )

            # Save failure report
            failure_path = self.workspace / "拟合失败记录.xlsx"
            self.failure_tracker.save_to_excel(failure_path)

            if self.failure_tracker.has_failures():
                self.log_message.emit(f"\n警告: {self.failure_tracker.get_failure_count()} 次拟合失败已记录")

            self.log_message.emit(f"{'='*60}")
            self.log_message.emit(f"所有分析完成!")

            self.finished.emit(True, "分析完成!")

        except Exception as e:
            self.log_message.emit(f"错误: {e}")
            self.finished.emit(False, str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.independent_result: Optional[LoadResult] = None
        self.dependent_result: Optional[LoadResult] = None
        self.workspace: Optional[Path] = None
        self.preprocess_results: Optional[tuple[PreprocessResult, PreprocessResult]] = None
        self.worker: Optional[AnalysisWorker] = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("空间要素与鱼塘要素相关性分析工具")
        self.setGeometry(100, 100, 800, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        layout = QVBoxLayout(central)
        layout.setSpacing(15)

        # Title
        title = QLabel("空间要素与鱼塘要素相关性分析工具")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # File selection group
        file_group = self.create_file_selection_group()
        layout.addWidget(file_group)

        # Info display
        info_group = self.create_info_group()
        layout.addWidget(info_group)

        # Action buttons
        action_group = self.create_action_group()
        layout.addWidget(action_group)

        # Log output
        log_group = self.create_log_group()
        layout.addWidget(log_group)

    def create_file_selection_group(self) -> QGroupBox:
        """Create file selection group."""
        group = QGroupBox("文件选择")
        layout = QVBoxLayout()

        # Independent variable file
        ind_layout = QHBoxLayout()
        self.ind_label = QLabel("自变量表格: 未选择")
        self.ind_button = QPushButton("选择自变量表格")
        self.ind_button.clicked.connect(self.select_independent_file)
        ind_layout.addWidget(self.ind_label, 1)
        ind_layout.addWidget(self.ind_button)
        layout.addLayout(ind_layout)

        # Dependent variable file
        dep_layout = QHBoxLayout()
        self.dep_label = QLabel("因变量表格: 未选择")
        self.dep_button = QPushButton("选择因变量表格")
        self.dep_button.clicked.connect(self.select_dependent_file)
        dep_layout.addWidget(self.dep_label, 1)
        dep_layout.addWidget(self.dep_button)
        layout.addLayout(dep_layout)

        # Workspace
        ws_layout = QHBoxLayout()
        self.ws_label = QLabel("工作空间: 未选择")
        self.ws_button = QPushButton("选择工作空间")
        self.ws_button.clicked.connect(self.select_workspace)
        ws_layout.addWidget(self.ws_label, 1)
        ws_layout.addWidget(self.ws_button)
        layout.addLayout(ws_layout)

        group.setLayout(layout)
        return group

    def create_info_group(self) -> QGroupBox:
        """Create information display group."""
        group = QGroupBox("信息显示")
        layout = QHBoxLayout()

        self.ind_count_label = QLabel("自变量数: 0")
        self.dep_count_label = QLabel("因变量数: 0")
        self.total_rounds_label = QLabel("操作共执行轮数: 0")

        layout.addWidget(self.ind_count_label)
        layout.addWidget(self.dep_count_label)
        layout.addWidget(self.total_rounds_label)

        group.setLayout(layout)
        return group

    def create_action_group(self) -> QGroupBox:
        """Create action buttons group."""
        group = QGroupBox("操作")
        layout = QHBoxLayout()

        self.check_button = QPushButton("右偏检查")
        self.check_button.clicked.connect(self.run_preprocess_check)
        self.check_button.setEnabled(False)

        self.run_button = QPushButton("开始执行")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setEnabled(False)

        layout.addWidget(self.check_button)
        layout.addWidget(self.run_button)

        group.setLayout(layout)
        return group

    def create_log_group(self) -> QGroupBox:
        """Create log output group."""
        group = QGroupBox("运行日志")
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(300)

        # Progress label
        self.progress_label = QLabel("就绪")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.progress_label)
        layout.addWidget(self.log_text)

        group.setLayout(layout)
        return group

    def select_independent_file(self):
        """Select independent variable Excel file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择自变量表格", "", "Excel Files (*.xlsx *.xls)"
        )
        if filepath:
            try:
                result = load_excel_file(filepath, "自变量表格")
                self.independent_result = result
                self.ind_label.setText(f"自变量表格: {Path(filepath).name}")
                self.ind_count_label.setText(f"自变量数: {result.variable_count}")
                self.log(f"已加载自变量表格: {filepath}")
                self.check_ready_state()
            except ValidationError as e:
                QMessageBox.warning(self, "加载失败", e.message)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载文件时出错: {e}")

    def select_dependent_file(self):
        """Select dependent variable Excel file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择因变量表格", "", "Excel Files (*.xlsx *.xls)"
        )
        if filepath:
            try:
                result = load_excel_file(filepath, "因变量表格")
                self.dependent_result = result
                self.dep_label.setText(f"因变量表格: {Path(filepath).name}")
                self.dep_count_label.setText(f"因变量数: {result.variable_count}")
                self.log(f"已加载因变量表格: {filepath}")
                self.check_ready_state()
            except ValidationError as e:
                QMessageBox.warning(self, "加载失败", e.message)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载文件时出错: {e}")

    def select_workspace(self):
        """Select workspace directory."""
        dirpath = QFileDialog.getExistingDirectory(
            self, "选择工作空间", ""
        )
        if dirpath:
            self.workspace = Path(dirpath)
            self.ws_label.setText(f"工作空间: {dirpath}")
            self.log(f"已设置工作空间: {dirpath}")
            self.check_ready_state()

    def check_ready_state(self):
        """Check if all files are loaded and enable buttons."""
        has_ind = self.independent_result is not None
        has_dep = self.dependent_result is not None
        has_ws = self.workspace is not None

        self.check_button.setEnabled(has_ind and has_dep and has_ws)
        self.run_button.setEnabled(has_ind and has_dep and has_ws and self.preprocess_results is not None)

        if has_ind and has_dep and has_ws:
            total_rounds = self.independent_result.variable_count * self.dependent_result.variable_count
            self.total_rounds_label.setText(f"操作共执行轮数: {total_rounds}")

    def run_preprocess_check(self):
        """Run preprocessing (right shift check)."""
        if not (self.independent_result and self.dependent_result and self.workspace):
            QMessageBox.warning(self, "警告", "请先选择所有文件和工作空间")
            return

        try:
            self.log("="*60)
            self.log("开始数据预处理...")

            # Run preprocessing
            self.preprocess_results = preprocess_both_files(
                self.independent_result.df,
                self.dependent_result.df
            )

            # Create preprocess directory
            preprocess_dir = self.workspace / "数据预处理"
            preprocess_dir.mkdir(parents=True, exist_ok=True)

            # Save preprocessed data
            ind_path = save_preprocessed_data(
                self.preprocess_results[0], preprocess_dir, "自变量_预处理.xlsx"
            )
            self.log(f"自变量预处理完成: {ind_path}")
            self.log(f"  {self.preprocess_results[0].shift_info}")

            dep_path = save_preprocessed_data(
                self.preprocess_results[1], preprocess_dir, "因变量_预处理.xlsx"
            )
            self.log(f"因变量预处理完成: {dep_path}")
            self.log(f"  {self.preprocess_results[1].shift_info}")

            self.log("数据预处理完成!")
            self.log("="*60)

            QMessageBox.information(
                self, "完成",
                "数据预处理完成!\n预处理结果已保存到工作空间的'数据预处理'文件夹。"
            )

            self.check_ready_state()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"预处理失败: {e}")
            self.log(f"错误: {e}")

    def run_analysis(self):
        """Run the complete analysis."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "警告", "分析正在进行中，请稍候...")
            return

        if not self.preprocess_results:
            QMessageBox.warning(self, "警告", "请先进行右偏检查")
            return

        # Clear log
        self.log_text.clear()
        self.log("开始分析...")
        self.log("="*60)

        # Disable buttons
        self.run_button.setEnabled(False)
        self.check_button.setEnabled(False)

        # Create and start worker
        self.worker = AnalysisWorker(
            self.independent_result,
            self.dependent_result,
            self.workspace,
            self.preprocess_results
        )
        self.worker.progress_update.connect(self.update_progress)
        self.worker.log_message.connect(self.log)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()

    def update_progress(self, message: str):
        """Update progress label."""
        self.progress_label.setText(message)

    def analysis_finished(self, success: bool, message: str):
        """Handle analysis completion."""
        # Re-enable buttons
        self.run_button.setEnabled(True)
        self.check_button.setEnabled(True)

        if success:
            QMessageBox.information(
                self, "完成",
                f"分析完成!\n\n结果已保存到工作空间。\n{self.worker.failure_tracker.get_summary()}"
            )
        else:
            QMessageBox.critical(self, "错误", f"分析失败: {message}")

    def log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
