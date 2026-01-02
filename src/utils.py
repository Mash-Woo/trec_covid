import os
import sys
import yaml
from datetime import datetime

import pandas as pd

__root__ = os.getcwd()
sys.path.insert(0, __root__)

def load_config() -> dict:
    CONFIG_PATH = os.path.join(__root__, "config.yaml")
    with open(CONFIG_PATH, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def read_file(filepath: str):
    """
    Reads a file and returns its contents. Supports CSV and TXT formats.
    Args:
        filepath (str): The path to the file to be read.
    Returns:
        pd.DataFrame or list: The contents of the file as a DataFrame (for CSV) or a list (for TXT).
    """
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        return df
    elif filepath.endswith(".txt"):
        with open(filepath, "r") as file:
            content = file.readlines()
        return [line.strip() for line in content]
    else:
        raise ValueError("Unsupported file format.")


def load_topic_file() -> pd.DataFrame:
    """
    Load topic file from the given filepath.
    Args:
        
    Returns:
        pd.DataFrame: DataFrame containing the topics with columns 'topic-id' and 'query'.
    """
    topic_file_path = os.path.join(__root__, "data", "raw", "CORD_19", "topics-rnd3.csv")
    return pd.read_csv(topic_file_path)


def generate_submission(results: pd.DataFrame) -> None:
    """
    Generate a TREC-formatted submission file from the topics file.
    Args:
        results (pd.DataFrame): DataFrame containing the results with columns
        ```
        topic-id,cord-id
        1,fi35kidw
        1,h3zvzr6w
        ...
        ```
    """
    output_dir = os.path.join(__root__, "output", "submissions", datetime.now().strftime("%Y%m%d_%H%M%S"), "submission.csv")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    results.to_csv(output_dir, index=False)
    print(f"Submission file generated at: {output_dir}")

    return None