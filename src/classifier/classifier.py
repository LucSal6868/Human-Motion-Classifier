import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
from scipy.interpolate import interp1d

# --- NEW IMPORTS FOR PLOTTING ---
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# -------------------------------

# Define file paths
PARSED_DATA_PATH = "../../data/train/augmented.npz"
TEST_DATA_PATH = "../../data/test/parsed.npz"
MODEL_FILE = "svm_model.pkl"
SCALER_FILE = "scaler.pkl"

# Constant for trajectory resampling
TRAJECTORY_POINTS = 50

# EXTRACTS 125 FEATURES FROM 3D ARRAY
def extract_features(sequence: np.ndarray) -> np.ndarray:
    N = sequence.shape[0]
    if N < 2:
        return np.zeros(125, dtype=np.float32)

    # GET X AND Y COORDS
    # WE EXCLUDE Z SO THE MODEL CAN SEE A CIRCLE IN ANY ROTATION OR SHAPE
    # Z IS USED LATER IN OTHER FEATURES
    xy_sequence = sequence[:, :2]

    # BASIC STATS
    mean_features = np.mean(sequence, axis=0)
    std_features = np.std(sequence, axis=0)
    min_features = np.min(sequence, axis=0)
    max_features = np.max(sequence, axis=0)

    basic_stats = np.concatenate([
        mean_features,
        std_features,
        min_features,
        max_features
    ])

    # DISPLACEMENT
    start_point = xy_sequence[0, :]
    end_point = xy_sequence[-1, :]
    delta_xy = end_point - start_point

    displacement_features = np.concatenate([
        start_point,
        end_point,
        delta_xy
    ])

    # CORRELATION
    corr_matrix = np.corrcoef(sequence[:, 0], sequence[:, 1])
    xy_correlation = corr_matrix[0, 1] if corr_matrix.ndim > 1 else 0.0
    xy_correlation_feature = np.array([xy_correlation])

    # DIRECTIONAL VELOCITY
    deltas = sequence[1:] - sequence[:-1]
    step_magnitudes = np.linalg.norm(deltas[:, :2], axis=1)
    mean_magnitude = np.mean(step_magnitudes) if len(step_magnitudes) > 0 else 0.0
    std_magnitude = np.std(step_magnitudes) if len(step_magnitudes) > 0 else 0.0

    # CURVATURE
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


    # ADD ALL TO DIRECTIONAL FEATURES

    directional_features = np.array([
        mean_magnitude,
        std_magnitude,
        mean_angle_change,
        std_angle_change,
    ])

    # CENTROID
    centroid = sequence.mean(axis=0)
    distances_from_centroid = np.linalg.norm(sequence - centroid, axis=1)
    mean_dist_centroid = np.mean(distances_from_centroid)
    std_dist_centroid = np.std(distances_from_centroid)
    centroid_distance_features = np.array([mean_dist_centroid, std_dist_centroid])

    # TRAJECTORY
    normalized_path_full = xy_sequence - xy_sequence[0, :]
    movement_mask = np.logical_not(np.all(normalized_path_full[1:] == normalized_path_full[:-1], axis=1))
    indices_to_keep = np.concatenate([[True], movement_mask])
    unique_path = normalized_path_full[indices_to_keep, :]

    if unique_path.shape[0] < 2:
        trajectory_features = np.zeros(100, dtype=np.float32)
    else:
        deltas_unique = unique_path[1:] - unique_path[:-1]
        path_distances_unique = np.linalg.norm(deltas_unique, axis=1)
        total_length = np.sum(path_distances_unique)

        scale_factor = 1.0 / (total_length + 1e-6)
        scaled_unique_path = unique_path * scale_factor

        cumulative_length_unique = np.cumsum(path_distances_unique)
        time_vector = np.concatenate([[0], cumulative_length_unique])

        scaled_time_vector = time_vector * scale_factor
        target_time = np.linspace(scaled_time_vector[0], scaled_time_vector[-1], TRAJECTORY_POINTS)

        interp_x = interp1d(scaled_time_vector, scaled_unique_path[:, 0], kind='linear')
        interp_y = interp1d(scaled_time_vector, scaled_unique_path[:, 1], kind='linear')

        resampled_x = interp_x(target_time)
        resampled_y = interp_y(target_time)

        trajectory_features = np.concatenate([resampled_x, resampled_y])

    # CONCATONATE 125 FEATURES

    features = np.concatenate([
        basic_stats,
        displacement_features,
        xy_correlation_feature,
        directional_features,
        centroid_distance_features,
        trajectory_features
    ])

    return features.astype(np.float32)

# convert data to x and y arrays
def prepare_data(parsed_data_dict: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X_features: List[np.ndarray] = []
    y_labels: List[str] = []

    EXPECTED_FEATURE_SIZE = 125  # Updated to 125

    for class_name, data_list in parsed_data_dict.items():
        for sequence in data_list:
            try:
                sequence_arr = np.array(sequence, dtype=np.float32)
            except Exception:
                continue

            if sequence_arr.ndim == 2 and sequence_arr.shape[1] == 3 and sequence_arr.shape[0] >= 2:
                features = extract_features(sequence_arr)
                if features.size == EXPECTED_FEATURE_SIZE:
                    X_features.append(features)
                    y_labels.append(class_name)
            else:
                pass

    X = np.array(X_features)
    y = np.array(y_labels)

    return X, y

# TRAIN CLASSIFIER
def train_svm_classifier():
    print("--- STARTING TRAINING PROCESS ---")
    try:
        # LOAD DATA
        data = np.load(PARSED_DATA_PATH, allow_pickle=True)
        parsed_data_dict = dict(data)
        print(f"Data loaded with classes: {list(parsed_data_dict.keys())}")

    except FileNotFoundError:
        print(f"ERROR: Training data file not found at {PARSED_DATA_PATH}.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # PREPARE DATA
    X, y = prepare_data(parsed_data_dict)

    if X.shape[0] == 0:
        print("No valid data found for training.")
        return

    # CREATE VALIDATION SET
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"\nTotal training samples after split: {len(X_train)}")

    # SCALE FEATURES (NORMALIZE)
    # Mean is Zero
    # standard deviation is 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ----------------------------------------------------
    # --- START: New PCA and Plotting Code Insertion ---
    # ----------------------------------------------------

    # 3D Visualization using PCA
    print("\nPerforming PCA for 3D visualization of the training data...")
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_train_scaled)

    # Convert labels to numerical for coloring the scatter plot
    unique_labels = np.unique(y_train)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y_train])

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(
        X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
        c=y_numeric,
        cmap='viridis', # 'viridis' is a perceptually uniform colormap
        marker='o'
    )

    # Add labels and title
    ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    ax.set_zlabel(f'PCA Component 3 ({pca.explained_variance_ratio_[2]*100:.2f}%)')
    ax.set_title('3D Visualization of Scaled Features (PCA Reduced)')

    # Create a legend with class names
    legend1 = ax.legend(
        *scatter.legend_elements(),
        title="Classes",
        loc="upper right",
        labels=unique_labels
    )
    ax.add_artist(legend1)

    plt.tight_layout()
    plt.savefig('pca_3d_visualization.png')
    print(f"3D PCA visualization saved as 'pca_3d_visualization.png' in the current directory.")

    # ----------------------------------------------------
    # --- END: New PCA and Plotting Code Insertion ---
    # ----------------------------------------------------


    # TRAIN FINAL SVM MODEL
    print("\nInitializing SVM Classifier with fixed parameters (C=1.0, gamma='scale', kernel='rbf')...")
    svm_model = SVC(C=1.0, gamma='scale', kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    print("Training complete.")

    # SAVE
    try:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(svm_model, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"\nModel and Scaler saved as '{MODEL_FILE}' and '{SCALER_FILE}'.")
    except Exception as e:
        print(f"ERROR: Could not save classifier/scaler: {e}")
        return

# LOAD MODEL AND SCALER
def evaluate_test_data():
    print("\n--- STARTING EVALUATION ON SEPARATE TEST FILE ---")

    # LOAD MODEL AND SCALER
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
        print(f"Test data loaded with classes: {list(parsed_test_dict.keys())}")
    except FileNotFoundError:
        print(f"ERROR: Test data file not found at {TEST_DATA_PATH}.")
        return
    except Exception as e:
        print(f"An error occurred during test data loading: {e}")
        return

    # PREPARE TEST DATA (Feature Extraction)
    X_test, y_test = prepare_data(parsed_test_dict)

    if X_test.shape[0] == 0:
        print("No valid data found in the test file.")
        return

    # SCALE TEST FEATURES
    X_test_scaled = scaler.transform(X_test)

    # 5. PREDICT AND EVALUATE
    y_pred = svm_model.predict(X_test_scaled)

    print("\n" + "=" * 60)
    print(f"CLASSIFICATION REPORT ON UNSEEN TEST FILE: {TEST_DATA_PATH}")
    print("=" * 60)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    train_svm_classifier()
    evaluate_test_data()