import os
import logging

import string
import pandas as pd

import yaml

from sklearn.ensemble import RandomForestClassifier

import joblib



# ==========================================================
# Ensure logs directory exists
# ==========================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================================
# Configure Logger
# ==========================================================
logger = logging.getLogger("model_building")
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
log_file = os.path.join(LOG_DIR, "model_building.log")

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




def train_model(X_train, y_train, n_estimators, random_state):

    try:
        logger.info("Training Random Forest model...")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        logger.info("Random Forest model trained successfully.")

        return model

    except Exception as e:
        logger.exception(f"Error while training model: {e}")
        raise
    


def save_model(model):

    try:
        logger.info("Saving trained model...")

        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "randomforest_model.pkl")

        joblib.dump(model, model_path)

        logger.info(f"Model saved successfully at {model_path}")

    except Exception as e:
        logger.exception(f"Error while saving model: {e}")
        raise
    
    
    
def main():
    
    try:
        logger.info("Starting model building pipeline...")

        # Load parameters
        params = load_params()

        n_estimators = params["model_building"]["n_estimators"]
        random_state = params["model_building"]["random_state"]

        logger.info(
            f"Loaded Parameters -> n_estimators={n_estimators}, random_state={random_state}"
        )

        # Load feature vectors
        train_df = pd.read_csv(
            os.path.join("data", "transformed", "train_vectors.csv")
        )

        logger.info("Training vectors loaded successfully.")

        # Split features and target
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]

        logger.info("Features and target separated successfully.")

        # Train model
        model = train_model(
            X_train,
            y_train,
            n_estimators,
            random_state,
        )

        # Save model
        save_model(model)

        logger.info("Model building pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise
    
    

if __name__ == "__main__":
    main()    
    
    