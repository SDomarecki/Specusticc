from specusticc_visualizer.output_analyzer import OutputAnalyzer


if __name__ == "__main__":
    output_path = 'output/2020-07-09_19-13-13'
    analyzer = OutputAnalyzer(output_path)
    analyzer.run()