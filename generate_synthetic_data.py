import argparse
import random
import yaml

from lorem_text import lorem
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("-config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    df = pd.DataFrame.from_dict(
        {
            "sentences": [lorem.sentence() for _ in range(args.n)],
            "labels": [
                random.randint(0, config["num_labels"] - 1)
                for _ in range(args.n)
            ]
        }
    )

    df.to_csv("synthetic_data.csv", index=False)


if __name__ == "__main__":
    main()
