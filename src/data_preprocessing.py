import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
import os
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    logger.info("Fetching California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    # Save raw data (important for DVC)
    os.makedirs("data/raw", exist_ok=True)
    raw_data_path = "data/raw/california_housing.csv"
    pd.concat([X, y.rename("target")], axis=1).to_csv(raw_data_path, index=False)
    logger.info(f"Raw data saved to {raw_data_path}")

    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optional: Power Transform (more Gaussian-like for some models)
    logger.info("Applying PowerTransformer (Yeo-Johnson)...")
    transformer = PowerTransformer(method='yeo-johnson')
    X_train = pd.DataFrame(transformer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(transformer.transform(X_test), columns=X.columns)

    logger.info("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    os.makedirs("data/processed", exist_ok=True)
    X_train_scaled_df.to_csv("data/processed/train_features.csv", index=False)
    y_train.to_csv("data/processed/train_target.csv", index=False)
    X_test_scaled_df.to_csv("data/processed/test_features.csv", index=False)
    y_test.to_csv("data/processed/test_target.csv", index=False)
    logger.info("Processed data saved to data/processed/")

    # Save scalers for inference
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(transformer, 'power_transformer.pkl')
    logger.info("Scalers saved for future use.")

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data()
