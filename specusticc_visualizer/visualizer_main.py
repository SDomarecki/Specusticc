from specusticc_visualizer.output_analyzer import OutputAnalyzer


if __name__ == "__main__":
    output_path = 'output/2020-07-13_14-22-17'
    analyzer = OutputAnalyzer(output_path)
    analyzer.run()