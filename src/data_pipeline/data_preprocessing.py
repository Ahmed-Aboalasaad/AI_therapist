import os
import re
import pandas as pd
from datasets import load_dataset
from src.helpers.config import get_settings
settings = get_settings()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(re.compile(r'[\s\r\n\t]+'), ' ', text)
    return text.strip()

def fetch_and_prepare_data() -> pd.DataFrame:
    print("Loading dataset from Hugging Face Hub...")
    dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    return pd.DataFrame(dataset)

def process_and_group_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Running cleaning rules and GroupBy logic...")
    
    df["cleaned_Context"] = df["Context"].apply(clean_text)
    df["cleaned_Response"] = df["Response"].apply(clean_text)
    
    df = df[df['cleaned_Response'].str.len() > 10].reset_index(drop=True)
    
    grouped_df = df.groupby("cleaned_Context")["cleaned_Response"].apply(
        lambda responses: "\n\n---\n\n".join(responses)
    ).reset_index()
    
    grouped_df.rename(columns={"cleaned_Context": "Context", "cleaned_Response": "Response"}, inplace=True)
    return grouped_df

def run_preprocessing():
    print("Checking Data Preprocessing Status...")

    CLEANED_DATA_PATH = settings.CLEANED_DATA_PATH

    if os.path.exists(CLEANED_DATA_PATH):
        print(f"Cleaned dataset already exists at: {CLEANED_DATA_PATH}. Skipping Pipeline.")
        return
    
    print(f"Cleaned data not found. Triggering Full Pipeline...")

    try:
        os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)

        raw_df = fetch_and_prepare_data()
        final_df = process_and_group_data(raw_df)

        final_df.to_csv(CLEANED_DATA_PATH, index=False, encoding="utf-8")
        print(f"Success! Cleaned and Grouped dataset saved locally as: '{CLEANED_DATA_PATH}'")
        print(f"Total unique QA pairs saved: {len(final_df)}")

    except Exception as e:
        raise RuntimeError(f"Failed during data preprocessing pipeline: {str(e)}")
    

if __name__ == "__main__":
    run_preprocessing()