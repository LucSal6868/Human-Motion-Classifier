import numpy as np
import pickle
import os
from typing import Dict, List
from scipy.interpolate import interp1d

TEST_DATA_PATH = "../../data/test/parsed.npz"
MODEL_FILE = "svm_model.pkl"
SCALER_FILE = "scaler.pkl"

# TRAJECTORY SAMPLING (Must match training file)
TRAJECTORY_POINTS = 50
EXPECTED_FEATURE_SIZE = 125

def extract_features(sequence: np.ndarray) -> np.ndarray:
    N = sequence.shape[0]
    if N < 2:
        return np.zeros(EXPECTED_FEATURE_SIZE, dtype=np.float32)

    #  BASIC STATS
    mean_features = np.mean(sequence, axis=0)
    std_features = np.std(sequence, axis=0)
    min_features = np.min(sequence, axis=0)
    max_features = np.max(sequence, axis=0)

    basic_stats = np.concatenate([mean_features, std_features, min_features, max_features])

    # DISPLACEMENT
    xy_sequence = sequence[:, :2]
    start_point = xy_sequence[0, :]
    end_point = xy_sequence[-1, :]
    delta_xy = end_point - start_point
    displacement_features = np.concatenate([start_point, end_point, delta_xy])

    #  CORRELATION
    corr_matrix = np.corrcoef(sequence[:, 0], sequence[:, 1])
    xy_correlation = corr_matrix[0, 1] if corr_matrix.ndim > 1 else 0.0
    xy_correlation_feature = np.array([xy_correlation])

    #  DIRECTIONAL VELOCITY & CURVATURE
    deltas = sequence[1:] - sequence[:-1]
    step_magnitudes = np.linalg.norm(deltas[:, :2], axis=1)
    mean_magnitude = np.mean(step_magnitudes) if len(step_magnitudes) > 0 else 0.0
    std_magnitude = np.std(step_magnitudes) if len(step_magnitudes) > 0 else 0.0

    angle_changes = []
    if N >= 3:
        for i in range(len(deltas) - 1):
            v1 = deltas[i, :2]
            v2 = deltas[i + 1, :2]
            dot_product = np.dot(v1, v2)
            magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            if magnitude_product > 1e-6:
                cosine_angle = np.clip(dot_product / magnitude_product, -1.0, 1.0)
                angle_rad = np.arccos(cosine_angle)
                angle_changes.append(angle_rad)

    mean_angle_change = np.mean(angle_changes) if angle_changes else 0.0
    std_angle_change = np.std(angle_changes) if angle_changes else 0.0

    directional_features = np.array([mean_magnitude, std_magnitude, mean_angle_change, std_angle_change])

    # 5. CENTROID (2 features)
    centroid = sequence.mean(axis=0)
    distances_from_centroid = np.linalg.norm(sequence - centroid, axis=1)
    mean_dist_centroid = np.mean(distances_from_centroid)
    std_dist_centroid = np.std(distances_from_centroid)
    centroid_distance_features = np.array([mean_dist_centroid, std_dist_centroid])

    # 6. TRAJECTORY (100 features)
    normalized_path_full = sequence[:, :2] - sequence[0, :2]
    movement_mask = np.logical_not(np.all(normalized_path_full[1:] == normalized_path_full[:-1], axis=1))
    indices_to_keep = np.concatenate([[True], movement_mask])
    unique_path = normalized_path_full[indices_to_keep, :]

    if unique_path.shape[0] < 2:
        trajectory_features = np.zeros(TRAJECTORY_POINTS * 2, dtype=np.float32)
    else:
        deltas_unique = unique_path[1:] - unique_path[:-1]
        path_distances_unique = np.linalg.norm(deltas_unique, axis=1)
        total_length = np.sum(path_distances_unique)

        scale_factor = 1.0 / (total_length + 1e-6)
        scaled_unique_path = unique_path * scale_factor

        cumulative_length_unique = np.cumsum(path_distances_unique)
        time_vector = np.concatenate([[0], cumulative_length_unique])
        scaled_time_vector = time_vector * scale_factor

        # Resampling
        target_time = np.linspace(scaled_time_vector[0], scaled_time_vector[-1], TRAJECTORY_POINTS)
        interp_x = interp1d(scaled_time_vector, scaled_unique_path[:, 0], kind='linear')
        interp_y = interp1d(scaled_time_vector, scaled_unique_path[:, 1], kind='linear')

        resampled_x = interp_x(target_time)
        resampled_y = interp_y(target_time)

        trajectory_features = np.concatenate([resampled_x, resampled_y])

    # CONCATENATE ALL 125 FEATURES
    features = np.concatenate([
        basic_stats,
        displacement_features,
        xy_correlation_feature,
        directional_features,
        centroid_distance_features,
        trajectory_features
    ])

    # Ensure the feature vector size is exactly 125
    if features.size != EXPECTED_FEATURE_SIZE:
        return np.zeros(EXPECTED_FEATURE_SIZE, dtype=np.float32)

    return features.astype(np.float32)


def prepare_data(parsed_data_dict: Dict[str, np.ndarray]) -> tuple[np.ndarray, List[str]]:
    X_features: List[np.ndarray] = []
    original_keys: List[str] = []

    for file_name, trajectory_array in parsed_data_dict.items():
        try:
            sequence_arr = np.array(trajectory_array, dtype=np.float32)
        except Exception:
            continue

        if sequence_arr.ndim == 2 and sequence_arr.shape[1] == 3 and sequence_arr.shape[0] >= 2:
            features = extract_features(sequence_arr)
            if features.size == EXPECTED_FEATURE_SIZE:
                X_features.append(features)
                original_keys.append(file_name)

    X = np.array(X_features)
    return X, original_keys


# ECLAUTE

def evaluate_test_data():
    print("\n--- STARTING PREDICTION ON TEST FILES ---")

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

    # PREPARE TEST DATA (Feature Extraction)
    X_test, original_keys = prepare_data(parsed_test_dict)
    total_samples = X_test.shape[0]

    if total_samples == 0:
        print("No valid data found in the test file.")
        return


    X_test_scaled = scaler.transform(X_test)

    #PREDICT
    y_pred = svm_model.predict(X_test_scaled)

    print("\n" + "=" * 60)
    print("FILE NAME CLASSIFICATION RESULTS")
    print("=" * 60)

    # Determine column width for formatting
    max_key_len = max(len(key) for key in original_keys) if original_keys else 15
    max_pred_len = max(len(label) for label in y_pred) if y_pred.size > 0 else 10

    # Header format: File Name | Predicted Class
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