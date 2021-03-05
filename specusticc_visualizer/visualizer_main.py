from specusticc_visualizer.results_analyzer import ResultsAnalyzer

"""
Shortcut paths to visualize
    results_path = 'examples/regular_test'
    results_path = 'examples/test_no_context_data'
    results_path = 'examples/test_correlated_data'
    results_path = 'examples/test_non_correlated_data'
    results_path = 'examples/past_future_test'
    results_path = 'examples/horizon_5'
    results_path = 'examples/horizon_10'
    results_path = 'examples/horizon_15'
    results_path = 'examples/horizon_20'
    results_path = 'examples/horizon_25'
    results_path = 'examples/horizon_30'
"""

if __name__ == "__main__":
    results_path = "examples/test_non_correlated_data"
    analyzer = ResultsAnalyzer(results_path)
    analyzer.analyze_predictions()
