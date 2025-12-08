import numpy as np
import pickle
import os
from typing import Dict, List
from features import extract_features, prepare_data, TRAJECTORY_POINTS, EXPECTED_FEATURE_SIZE

TEST_DATA_PATH = "../../data/test/parsed.npz"
MODEL_FILE = "svm_model.pkl"
SCALER_FILE = "scaler.pkl"


# ECLAUTE

def evaluate_test_data():
    print("\nSTARTING PREDICTION ON TEST FILES")

    # 1. LOAD MODEL AND SCALER
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print(f"ERROR: Required classifier files ('{MODEL_FILE}' and '{SCALER_FILE}') not found. Run training first.")
        return

    try:
        with open(MODEL_FILE, 'rb') as f:
            svm_model = pickle.load(f)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        print("Trained classifier and scaler loaded.")
    except Exception as e:
        print(f"ERROR loading classifier/scaler: {e}")
        return

    # LOAD TEST DATA
    try:
        data_test = np.load(TEST_DATA_PATH, allow_pickle=True)
        parsed_test_dict = dict(data_test)
        print(f"Test data loaded from '{TEST_DATA_PATH}'. Total files/samples: {len(parsed_test_dict)}")
    except FileNotFoundError:
        print(f"ERROR: Test data file not found at {TEST_DATA_PATH}.")
        return
    except Exception as e:
        print(f"An error occurred during test data loading: {e}")
        return

    # PREPARE TEST DATA
    def prepare_evaluation_data(parsed_data_dict: Dict[str, np.ndarray]) -> tuple[np.ndarray, List[str]]:
        X_features: List[np.ndarray] = []
        original_keys: List[str] = []

        for file_name, trajectory_array in parsed_data_dict.items():
            try:
                sequence_arr = np.array(trajectory_array, dtype=np.float32)
            except Exception:
                continue

            if sequence_arr.ndim == 2 and sequence_arr.shape[1] == 3 and sequence_arr.shape[0] >= 2:
                # Uses the imported function
                features = extract_features(sequence_arr)
                if features.size == EXPECTED_FEATURE_SIZE:
                    X_features.append(features)
                    original_keys.append(file_name)

        X = np.array(X_features)
        return X, original_keys

    X_test, original_keys = prepare_evaluation_data(parsed_test_dict)
    total_samples = X_test.shape[0]

    if total_samples == 0:
        print("No valid data found in the test file.")
        return

    X_test_scaled = scaler.transform(X_test)

    # PREDICT
    y_pred = svm_model.predict(X_test_scaled)

    print("\n" + "=" * 60)
    print("FILE NAME CLASSIFICATION RESULTS")
    print("=" * 60)

    max_key_len = max(len(key) for key in original_keys) if original_keys else 15
    max_pred_len = max(len(label) for label in y_pred) if y_pred.size > 0 else 10

    header = f"{'File Name':<{max_key_len + 5}}{'Predicted Class':<{max_pred_len + 5}}"
    print(header)
    print("-" * (len(header) + 5))

    # Iterate and print results
    for i in range(total_samples):
        test_key = original_keys[i]
        predicted_class = y_pred[i]

        print(f"{test_key:<{max_key_len + 5}}{predicted_class:<{max_pred_len + 5}}")

    print("=" * 60)


if __name__ == '__main__':
    evaluate_test_data()