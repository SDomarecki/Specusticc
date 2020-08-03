from specusticc_visualizer.output_analyzer import OutputAnalyzer


if __name__ == "__main__":
    output_path = 'output/test3_10'
    analyzer = OutputAnalyzer(output_path)
    analyzer.analyze_predictions()