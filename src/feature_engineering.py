import os
import logging

import string
import pandas as pd
import nltk

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import yaml

from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================================
# Ensure logs directory exists
# ==========================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================================
# Configure Logger
# ==========================================================
logger = logging.getLogger("feature_engineering")
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
log_file = os.path.join(LOG_DIR, "feature_engineering.log")

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




def load_params(params_path="params.yaml"):
    """
    Load parameters from params.yaml.
    """

    try:
        logger.info("Loading parameters from params.yaml...")

        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.info("Parameters loaded successfully.")

        return params

    except Exception as e:
        logger.exception(f"Error loading parameters: {e}")
        raise
    
    
    


def feature_engineering(train_df, test_df, max_features):
    
    """
    Apply TF-IDF feature engineering.
    """
    train_df["transformed_text"] = train_df["transformed_text"].fillna("")

    try:
        logger.info(
            f"Applying TF-IDF Vectorization with max_features={max_features}..."
        )

        tfidf = TfidfVectorizer(max_features=max_features)

        X_train = tfidf.fit_transform(train_df["transformed_text"]).toarray()
        X_test = tfidf.transform(test_df["transformed_text"]).toarray()

        logger.info("TF-IDF feature engineering completed successfully.")

        return X_train, X_test, tfidf

    except Exception as e:
        logger.exception(f"Error during feature engineering: {e}")
        raise    
    
def save_transformed_data(X_train, X_test, train_target, test_target):
    
    try:
        logger.info("Saving transformed feature vectors...")

        output_dir = os.path.join("data", "raw", "transformed")
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, "train_vectors.csv")
        test_path = os.path.join(output_dir, "test_vectors.csv")

        train_vectors = pd.DataFrame(X_train)
        train_vectors["target"] = train_target.values

        test_vectors = pd.DataFrame(X_test)
        test_vectors["target"] = test_target.values

        train_vectors.to_csv(train_path, index=False)
        test_vectors.to_csv(test_path, index=False)

        logger.info(f"Train vectors saved to {train_path}")
        logger.info(f"Test vectors saved to {test_path}")

    except Exception as e:
        logger.exception(f"Error while saving transformed data: {e}")
        raise
    
def main():
    
    try:
        logger.info("Starting feature engineering pipeline...")

        params = load_params()
        max_features = params["feature_engineering"]["max_features"]

        logger.info(f"Loaded max_features={max_features}")

        train_df = pd.read_csv("data/Interim/train_preprocessed.csv")
        test_df = pd.read_csv("data/Interim/test_preprocessed.csv")

        logger.info("Preprocessed datasets loaded successfully.")

        X_train, X_test, tfidf = feature_engineering(
            train_df,
            test_df,
            max_features,
        )

        save_transformed_data(
                             X_train,
                             X_test,
                             train_df["target"],
                             test_df["target"]
                            )

        logger.info("Feature engineering pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()    
    
    