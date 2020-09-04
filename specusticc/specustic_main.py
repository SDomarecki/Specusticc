import sys

from market.market import Market

if __name__ == "__main__":
    example_directory = sys.argv[1]
    model_names = sys.argv[2:]

    example_path = f'./examples/{example_directory}'
    market = Market(example_path=example_path, models=model_names)
    market.run()
