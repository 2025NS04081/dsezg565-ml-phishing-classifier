"""
Data Preparation - Phishing Website Classification
Done by: Aravinth
BITS ID: 2025NS04081
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Place the raw Kaggle download here, keep the original filename as-is
SOURCE_CSV_PATH = PROJECT_ROOT / "data" / "dataset_phishing.csv"
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw_phishing_websites.csv"

ID_COLUMN = "url"
TARGET_COLUMN = "status"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV_PATH)
    # status column has text labels, converting to 0/1 so sklearn can use it
    df["Result"] = df[TARGET_COLUMN].map({"legitimate": 0, "phishing": 1})
    df = df.drop(columns=[TARGET_COLUMN])
    return df


def main():
    df = load_dataset()

    print("Dataset shape:", df.shape)
    print("\nClass balance (Result):")
    print(df["Result"].value_counts(normalize=True))
    print("\nMissing values:", df.isnull().sum().sum())

    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"\nSaved full raw dataset ({len(df)} rows) to {RAW_CSV_PATH.relative_to(PROJECT_ROOT)}")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["Result"]
    )

    train_df.to_csv(PROJECT_ROOT / "model" / "train_data.csv", index=False)
    test_df.to_csv(PROJECT_ROOT / "test_data.csv", index=False)

    print(f"Saved {len(train_df)} rows to model/train_data.csv")
    print(f"Saved {len(test_df)} rows to test_data.csv")


if __name__ == "__main__":
    main()

