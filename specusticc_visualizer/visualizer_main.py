from specusticc_visualizer.output_analyzer import OutputAnalyzer


if __name__ == "__main__":
    # output_path = 'output/test1_no_additional_data'
    # output_path = 'output/test1_correlated'
    # output_path = 'output/test1_non_correlated'
    output_path = 'output/test2'
    # output_path = 'output/test3_5'
    # output_path = 'output/test3_10'
    # output_path = 'output/test3_15'
    # output_path = 'output/test3_20'
    # output_path = 'output/test3_25'
    # output_path = 'output/test3_30'
    # output_path = '../specusticc/output/test'
    analyzer = OutputAnalyzer(output_path)
    analyzer.analyze_predictions()