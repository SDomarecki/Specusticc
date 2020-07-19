from specusticc_visualizer.output_analyzer import OutputAnalyzer


if __name__ == "__main__":
    output_path = 'output/2020-07-18_18-50-41'
    analyzer = OutputAnalyzer(output_path)
    analyzer.analyze_predictions()