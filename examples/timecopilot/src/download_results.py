import pandas as pd

from .evaluate_model import tasks


def download_results():
    summaries = []
    for task in tasks():
        csv_path = f"s3://timecopilot-fev/results/{task.dataset_config}.csv"
        df = pd.read_csv(csv_path)
        summaries.append(df)
    # Show and save the results
    df = pd.concat(summaries)
    print(df)
    df.to_csv("timecopilot.csv", index=False)


if __name__ == "__main__":
    download_results()
