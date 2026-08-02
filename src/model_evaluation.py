import os
import logging

import string
import pandas as pd

import yaml

import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import json

# ==========================================================
# Ensure logs directory exists
# ==========================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================================
# Configure Logger
# ==========================================================
logger = logging.getLogger("model_evaluation")
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
log_file = os.path.join(LOG_DIR, "model_evaluation.log")

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

# ==========================================================
# Load Test Data
# ==========================================================

def load_data(test_file_path):

    try:
        logger.info("Loading test dataset...")

        test_df = pd.read_csv(test_file_path)

        logger.info("Test dataset loaded successfully.")

        return test_df

    except Exception as e:
        logger.exception(f"Error loading test data: {e}")
        raise
    
    

# ==========================================================
# Load Model
# ==========================================================

def load_model(model_path):

    try:
        logger.info("Loading trained model...")

        model = joblib.load(model_path)

        logger.info("Model loaded successfully.")

        return model

    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        raise

def evaluate_model(model, test_df):
    
    try:
        logger.info("Starting model evaluation...")


        X_test = test_df.drop("target", axis=1)
        y_test = test_df["target"]


        logger.info("Splitting features and target completed.")


        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:,1]


        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )


        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc_score": roc_auc
        }


        logger.info(f"Evaluation metrics: {metrics}")

        return metrics


    except Exception as e:
        logger.exception(f"Error during model evaluation: {e}")
        raise


# ==========================================================
# Save Metrics
# ==========================================================

def save_metrics(metrics, report_path):

    try:
        logger.info("Saving evaluation metrics...")


        os.makedirs(
            os.path.dirname(report_path),
            exist_ok=True
        )


        with open(report_path, "w") as file:

            json.dump(
                metrics,
                file,
                indent=4
            )


        logger.info(
            f"Metrics saved successfully at {report_path}"
        )


    except Exception as e:
        logger.exception(f"Error saving metrics: {e}")
        raise



# ==========================================================
# Main Pipeline
# ==========================================================

def main():

    try:
        logger.info("Starting model evaluation pipeline...")


        # File paths are defined here

        test_file_path = os.path.join(
            "data",
            "transformed",
            "test_vectors.csv"
        )


        model_path = os.path.join(
            "model",
            "randomforest_model.pkl"
        )


        report_path = os.path.join(
            "report",
            "metrics.json"
        )


        # Load data

        test_df = load_data(
            test_file_path
        )


        # Load model

        model = load_model(
            model_path
        )


        # Evaluate

        metrics = evaluate_model(
            model,
            test_df
        )


        # Save results

        save_metrics(
            metrics,
            report_path
        )


        logger.info(
            "Model evaluation pipeline completed successfully."
        )


    except Exception as e:
        logger.exception(
            f"Pipeline failed: {e}"
        )
        raise



if __name__ == "__main__":
    main()


