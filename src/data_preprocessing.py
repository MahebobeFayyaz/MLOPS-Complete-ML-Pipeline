import os
import logging

import string
import pandas as pd
import nltk

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ==========================================================
# Ensure logs directory exists
# ==========================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================================
# Configure Logger
# ==========================================================
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

# Prevent duplicate logs
logger.handlers.clear()

# ==========================================================
# Console Handler
# ==========================================================
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# ==========================================================
# File Handler
# ==========================================================
log_file = os.path.join(LOG_DIR, "data_preprocessing.log")

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# ==========================================================
# Formatter
# ==========================================================
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# ==========================================================
# Add Handlers to Logger
# ==========================================================
logger.addHandler(console_handler)
logger.addHandler(file_handler)




def transform_text(text):
    """
    Perform text preprocessing:
    1. Lowercase
    2. Tokenization
    3. Remove special characters
    4. Remove stopwords and punctuation
    5. Apply stemming
    """

    text = text.lower()
    text = nltk.word_tokenize(text)

    tokens = []

    # Keep only alphanumeric words
    for word in text:
        if word.isalnum():
            tokens.append(word)

    text = tokens[:]
    tokens.clear()

    # Remove stopwords and punctuation
    for word in text:
        if word not in stopwords.words("english") and word not in string.punctuation:
            tokens.append(word)

    text = tokens[:]
    tokens.clear()

    # Apply stemming
    for word in text:
        tokens.append(ps.stem(word))

    return " ".join(tokens)


def preprocess_data(train_df, test_df):
    
    try:
        logger.info("Starting data preprocessing...")
    except Exception as e:
        logger.exception(f"Failed to start preprocessing: {e}")
        raise

    # ======================================================
    # Label Encoding
    # ======================================================
    try:
        logger.info("Applying Label Encoding...")

        encoder = LabelEncoder()
        train_df["target"] = encoder.fit_transform(train_df["target"])
        test_df["target"] = encoder.transform(test_df["target"])

        logger.info("Label Encoding completed successfully.")

    except Exception as e:
        logger.exception(f"Error during Label Encoding: {e}")
        raise

    # ======================================================
    # Remove Duplicates
    # ======================================================
    try:
        logger.info("Removing duplicate records...")

        train_df = train_df.drop_duplicates(keep="first")
        test_df = test_df.drop_duplicates(keep="first")

        logger.info("Duplicate removal completed successfully.")

    except Exception as e:
        logger.exception(f"Error while removing duplicates: {e}")
        raise

    # ======================================================
    # Text Preprocessing
    # ======================================================
    try:
        logger.info("Applying text preprocessing...")

        train_df["transformed_text"] = train_df["text"].apply(transform_text)
        test_df["transformed_text"] = test_df["text"].apply(transform_text)

        logger.info("Text preprocessing completed successfully.")

    except Exception as e:
        logger.exception(f"Error during text preprocessing: {e}")
        raise

    try:
        logger.info("Data preprocessing completed successfully.")
        return train_df, test_df

    except Exception as e:
        logger.exception(f"Unexpected error while returning data: {e}")
        raise
    
    
def save_preprocessed_data(train_df, test_df):
    """
    Save preprocessed train and test datasets.
    """

    try:
        logger.info("Saving preprocessed datasets...")

        output_dir = os.path.join("data", "Interim")
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, "train_preprocessed.csv")
        test_path = os.path.join(output_dir, "test_preprocessed.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Train data saved to {train_path}")
        logger.info(f"Test data saved to {test_path}")

    except Exception as e:
        logger.exception(f"Error while saving preprocessed data: {e}")
        raise


def main():

    try:
        logger.info("Starting data preprocessing pipeline...")

        train_path = os.path.join("data", "raw", "train.csv")
        test_path = os.path.join("data", "raw", "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info("Train and test datasets loaded successfully.")

        train_df, test_df = preprocess_data(train_df, test_df)

        save_preprocessed_data(train_df, test_df)

        logger.info("Data preprocessing pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()