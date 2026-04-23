"""End-to-end test of the correlation analyzer."""

from pathlib import Path
import sys

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from correlation_analyzer.core.data_loader import load_excel_file
from correlation_analyzer.core.preprocessor import preprocess_both_files, save_preprocessed_data
from correlation_analyzer.core.spearman_analysis import spearman_correlation, save_spearman_results
from correlation_analyzer.core.regression_analysis import fit_regression_models, save_regression_results
from correlation_analyzer.core.results_aggregator import aggregate_results, save_aggregated_results
from correlation_analyzer.core.failure_tracker import FailureTracker

def test_full_pipeline():
    """Test the complete analysis pipeline."""
    print("="*60)
    print("开始端到端测试（因变量为中心模式）...")
    print("="*60)

    # Test data paths
    ind_path = Path("test_data/自变量测试数据.xlsx")
    dep_path = Path("test_data/因变量测试数据.xlsx")
    workspace = Path("test_data/workspace")

    # Step 1: Load data
    print("\n1. 加载Excel文件...")
    try:
        ind_result = load_excel_file(ind_path, "自变量表格")
        print(f"   自变量数量: {ind_result.variable_count}")
        print(f"   自变量名称: {ind_result.variable_names}")

        dep_result = load_excel_file(dep_path, "因变量表格")
        print(f"   因变量数量: {dep_result.variable_count}")
        print(f"   因变量名称: {dep_result.variable_names}")
    except Exception as e:
        print(f"   错误: {e}")
        return False

    # Step 2: Preprocess
    print("\n2. 数据预处理...")
    try:
        preprocess_results = preprocess_both_files(ind_result.df, dep_result.df)
        ind_preprocess, dep_preprocess = preprocess_results

        print(f"   自变量偏移信息: {ind_preprocess.shift_info}")
        print(f"   因变量偏移信息: {dep_preprocess.shift_info}")

        # Save preprocessed data
        preprocess_dir = workspace / "数据预处理"
        save_preprocessed_data(ind_preprocess, preprocess_dir, "自变量_预处理.xlsx")
        save_preprocessed_data(dep_preprocess, preprocess_dir, "因变量_预处理.xlsx")
        print(f"   预处理结果已保存")
    except Exception as e:
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Spearman correlation and regression (dependent variable centered)
    print("\n3. 斯皮尔曼相关性分析（因变量为中心）...")
    try:
        failure_tracker = FailureTracker()
        spearman_dir = workspace / "斯皮尔曼相关性分析结果"
        regression_dir = workspace / "回归分析结果"
        summary_dir = workspace / "系数结果排名汇总"

        ind_df = ind_preprocess.processed_df
        dep_df = dep_preprocess.processed_df

        # Process each DEPENDENT variable
        total_vars = len(dep_df.columns)
        print(f"   将处理 {total_vars} 个因变量")

        for idx, dep_var in enumerate(dep_df.columns, 1):
            print(f"\n   处理因变量 {idx}/{total_vars}: {dep_var}")

            dep_data = dep_df[dep_var]

            # Spearman correlation (dependent var with all independent vars)
            spearman_result = spearman_correlation(dep_var, dep_data, ind_df)
            save_spearman_results(spearman_result, spearman_dir, dep_var)
            print(f"     相关性分析完成: {len(spearman_result.filtered_results)} 个显著相关自变量")

            # Get independent variables for regression (P < 0.1)
            ind_vars_for_regression = spearman_result.filtered_results["自变量名称"].tolist()

            if not ind_vars_for_regression:
                print(f"     没有显著相关的自变量，跳过回归分析")
                continue

            # Regression analysis (dependent var with each independent var)
            regression_result = fit_regression_models(
                dep_var, dep_data, ind_df, ind_vars_for_regression, failure_tracker
            )
            save_regression_results(regression_result, regression_dir, dep_var)
            print(f"     回归分析完成")

            # Aggregate results
            agg_result = aggregate_results(spearman_result, regression_result)
            save_aggregated_results(agg_result, regression_result, summary_dir, dep_var)

            if len(agg_result) > 0:
                top = agg_result.iloc[0]
                print(f"     最佳匹配: {top['自变量名称']} (累加值: {top['累加值']:.4f})")

    except Exception as e:
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Save failure report
    print("\n4. 保存失败报告...")
    failure_path = workspace / "拟合失败记录.xlsx"
    failure_tracker.save_to_excel(failure_path)
    print(f"   失败报告已保存: {failure_tracker.get_failure_count()} 次失败")

    print("\n" + "="*60)
    print("测试完成！所有功能正常工作。")
    print(f"结果保存在: {workspace}")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
