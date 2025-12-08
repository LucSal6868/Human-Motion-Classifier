import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report  # Import classification_report
from typing import Dict, List
from scipy.interpolate import interp1d

# PLOTTING IMPORTS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PATHS
PARSED_DATA_PATH = "../../data/train/augmented.npz"
MODEL_FILE = "svm_model.pkl"
SCALER_FILE = "scaler.pkl"

# TRAJECTORY SAMPLING
TRAJECTORY_POINTS = 50
EXPECTED_FEATURE_SIZE = 125
EVALUATION_SIZE = 0.05


# EXTRACTY

def extract_features(sequence: np.ndarray) -> np.ndarray:
    """Extracts 125 handcrafted features from a 3D sequence array."""
    N = sequence.shape[0]
    if N < 2:
        return np.zeros(EXPECTED_FEATURE_SIZE, dtype=np.float32)

    # 1. BASIC STATS
    mean_features = np.mean(sequence, axis=0)
    std_features = np.std(sequence, axis=0)
    min_features = np.min(sequence, axis=0)
    max_features = np.max(sequence, axis=0)

    basic_stats = np.concatenate([mean_features, std_features, min_features, max_features])

    # 2. DISPLACEMENT
    xy_sequence = sequence[:, :2]
    start_point = xy_sequence[0, :]
    end_point = xy_sequence[-1, :]
    delta_xy = end_point - start_point
    displacement_features = np.concatenate([start_point, end_point, delta_xy])

    # 3. CORRELATION
    corr_matrix = np.corrcoef(sequence[:, 0], sequence[:, 1])
    xy_correlation = corr_matrix[0, 1] if corr_matrix.ndim > 1 else 0.0
    xy_correlation_feature = np.array([xy_correlation])

    # 4. DIRECTIONAL VELOCITY & CURVATURE
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

    # 5. CENTROID
    centroid = sequence.mean(axis=0)
    distances_from_centroid = np.linalg.norm(sequence - centroid, axis=1)
    mean_dist_centroid = np.mean(distances_from_centroid)
    std_dist_centroid = np.std(distances_from_centroid)
    centroid_distance_features = np.array([mean_dist_centroid, std_dist_centroid])

    # 6. TRAJECTORY (Resampled path)
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

    return features.astype(np.float32)


def prepare_data(parsed_data_dict: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X_features: List[np.ndarray] = []
    y_labels: List[str] = []

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

    X = np.array(X_features)
    y = np.array(y_labels)

    return X, y


# TRAIN

def train_svm_classifier():
    print("--- STARTING SVM TRAINING PROCESS ---")
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

    #  PREPARE DATA
    X, y = prepare_data(parsed_data_dict)

    if X.shape[0] == 0:
        print("No valid data found for training.")
        return

    # CREATE VALIDATION SET
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=EVALUATION_SIZE, random_state=42, stratify=y
    )
    print(f"\nTotal training samples after split: {len(X_train)}")

    #SCALE FEATURES
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Scale validation data using the training scaler
    X_val_scaled = scaler.transform(X_val)

    # PCA FOR VISUALIZATION (Plotting code remains the same)
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
    scatter = ax.scatter(
        X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
        c=y_numeric,
        cmap='viridis',
        marker='o'
    )

    # Add labels and title
    total_explained_variance = np.sum(pca.explained_variance_ratio_[:3])
    ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
    ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
    ax.set_zlabel(f'PCA Component 3 ({pca.explained_variance_ratio_[2] * 100:.2f}%)')
    ax.set_title(f'3D Visualization of Scaled Features (Total Variance: {total_explained_variance * 100:.2f}%)')

    legend1 = ax.legend(
        *scatter.legend_elements(),
        title="Classes",
        loc="upper right",
        labels=unique_labels
    )
    ax.add_artist(legend1)

    plt.tight_layout()
    plt.savefig('pca_3d_visualization.png')
    print(f"3D PCA visualization saved as 'pca_3d_visualization.png'.")

    # TRAIN FINAL SVM MODEL
    print("\nInitializing and Training SVM Classifier...")
    svm_model = SVC(C=1.0, gamma='scale', kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    print("Training complete.")

    # EVALUATE ON VALIDATION SET
    print("\n" + "=" * 50)
    print("EVALUATION ON VALIDATION SET")
    print("=" * 50)

    # Predict on the scaled validation data
    y_val_pred = svm_model.predict(X_val_scaled)

    # Print the classification report
    print(classification_report(y_val, y_val_pred))
    print("=" * 50)

    #  SAVE MODEL AND SCALER (Updated step number)
    try:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(svm_model, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"\nModel and Scaler successfully saved as '{MODEL_FILE}' and '{SCALER_FILE}'.")
    except Exception as e:
        print(f"ERROR: Could not save classifier/scaler: {e}")
        return


if __name__ == '__main__':
    train_svm_classifier()