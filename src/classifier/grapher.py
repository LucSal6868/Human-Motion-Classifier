import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from typing import Dict, List
from scipy.interpolate import interp1d

# --- 1. CONFIGURATION AND FILE PATHS ---

PARSED_DATA_PATH = "../../data/train/augmented.npz"
TEST_DATA_PATH = "../../data/test/parsed.npz"

# Artifacts for the OPTIMAL ACCURACY model (Full 125D features)
MODEL_FILE_125D = "svm_model_125D_tuned.pkl"
SCALER_FILE_125D = "scaler_125D.pkl"

# Artifacts for the 3D/2D VISUALIZATION model (LDA-based for best separability)
MODEL_FILE_2D_LDA = "svm_model_lda_2d_viz.pkl"
SCALER_FILE_LDA_VIZ = "scaler_lda_viz.pkl"
LDA_FILE_3D = "lda_3D_viz.pkl"

TRAINING_DATA_CACHE = "training_data_cache_raw.npz"

# Constants
TRAJECTORY_POINTS = 50
EXPECTED_FEATURE_SIZE = 125  # Your custom 125 features
FINAL_VIZ_DIMENSIONS_2D = 2
FINAL_VIZ_DIMENSIONS_3D = 3


# --- 2. FEATURE EXTRACTION (Your 125-Feature Version) ---

def extract_features(sequence: np.ndarray) -> np.ndarray:
    N = sequence.shape[0]
    if N < 2: return np.zeros(EXPECTED_FEATURE_SIZE, dtype=np.float32)

    xy_sequence = sequence[:, :2]

    # BASIC STATS (12 features)
    mean_features = np.mean(sequence, axis=0)
    std_features = np.std(sequence, axis=0)
    min_features = np.min(sequence, axis=0)
    max_features = np.max(sequence, axis=0)
    basic_stats = np.concatenate([mean_features, std_features, min_features, max_features])

    # DISPLACEMENT (6 features)
    start_point = xy_sequence[0, :]
    end_point = xy_sequence[-1, :]
    delta_xy = end_point - start_point
    displacement_features = np.concatenate([start_point, end_point, delta_xy])

    # CORRELATION (1 feature)
    corr_matrix = np.corrcoef(sequence[:, 0], sequence[:, 1])
    xy_correlation = corr_matrix[0, 1] if corr_matrix.ndim > 1 else 0.0
    xy_correlation_feature = np.array([xy_correlation])

    # DIRECTIONAL / CURVATURE (4 features)
    deltas = sequence[1:] - sequence[:-1]
    step_magnitudes = np.linalg.norm(deltas[:, :2], axis=1)
    mean_magnitude = np.mean(step_magnitudes) if len(step_magnitudes) > 0 else 0.0
    std_magnitude = np.std(step_magnitudes) if len(step_magnitudes) > 0 else 0.0
    angle_changes = []
    if N >= 3:
        for i in range(len(deltas) - 1):
            v1 = deltas[i, :2];
            v2 = deltas[i + 1, :2]
            dot_product = np.dot(v1, v2)
            magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            if magnitude_product > 1e-6:
                cosine_angle = np.clip(dot_product / magnitude_product, -1.0, 1.0)
                angle_rad = np.arccos(cosine_angle);
                angle_changes.append(angle_rad)
    mean_angle_change = np.mean(angle_changes) if angle_changes else 0.0
    std_angle_change = np.std(angle_changes) if angle_changes else 0.0
    directional_features = np.array([mean_magnitude, std_magnitude, mean_angle_change, std_angle_change])

    # CENTROID (2 features)
    centroid = sequence.mean(axis=0)
    distances_from_centroid = np.linalg.norm(sequence - centroid, axis=1)
    mean_dist_centroid = np.mean(distances_from_centroid)
    std_dist_centroid = np.std(distances_from_centroid)
    centroid_distance_features = np.array([mean_dist_centroid, std_dist_centroid])

    # TRAJECTORY (100 features)
    normalized_path_full = xy_sequence - xy_sequence[0, :]
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
        target_time = np.linspace(scaled_time_vector[0], scaled_time_vector[-1], TRAJECTORY_POINTS)

        interp_x = interp1d(scaled_time_vector, scaled_unique_path[:, 0], kind='linear')
        interp_y = interp1d(scaled_time_vector, scaled_unique_path[:, 1], kind='linear')
        resampled_x = interp_x(target_time)
        resampled_y = interp_y(target_time)
        trajectory_features = np.concatenate([resampled_x, resampled_y])

    # CONCATENATE ALL 125 FEATURES
    features = np.concatenate([
        basic_stats, displacement_features, xy_correlation_feature,
        directional_features, centroid_distance_features, trajectory_features
    ])
    return features.astype(np.float32)


# --- Data Preparation (Unchanged) ---
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
                    X_features.append(features);
                    y_labels.append(class_name)
    return np.array(X_features), np.array(y_labels)


# --- 3. TRAINING AND EVALUATION FUNCTIONS ---

def tune_and_train_svm(X_train_scaled, y_train, scaler):
    """Performs Grid Search Cross-Validation to find best SVM hyperparameters for 125D features."""
    print("\n--- STARTING GRID SEARCH FOR OPTIMAL HYPERPARAMETERS (125D Features) ---")

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf']
    }

    svm = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        verbose=2,
        n_jobs=-1,
        scoring='accuracy'
    )

    grid_search.fit(X_train_scaled, y_train)
    svm_model = grid_search.best_estimator_

    print("\nGrid Search Complete.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Cross-Validation Accuracy with Best Model: {grid_search.best_score_:.4f}")

    # SAVE THE BEST MODEL
    try:
        with open(MODEL_FILE_125D, 'wb') as f:
            pickle.dump(svm_model, f)
        with open(SCALER_FILE_125D, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Optimal 125D Model and Scaler saved: {MODEL_FILE_125D}")
    except Exception as e:
        print(f"ERROR: Could not save optimal 125D model: {e}")


def train_lda_viz_models(X_train: np.ndarray, y_train: np.ndarray):
    """Trains 3D LDA model for plotting/analysis and 2D LDA model for margin visualization."""
    print("\n--- TRAINING LDA VISUALIZATION MODELS ---")

    # Use a separate scaler for the LDA model
    scaler_lda = StandardScaler()
    X_train_scaled_lda = scaler_lda.fit_transform(X_train)

    # 1. 3D LDA Model (For 3D data extraction)
    lda_3d = LDA(n_components=FINAL_VIZ_DIMENSIONS_3D)
    lda_3d.fit(X_train_scaled_lda, y_train)

    try:
        with open(LDA_FILE_3D, 'wb') as f:
            pickle.dump(lda_3d, f)
        with open(SCALER_FILE_LDA_VIZ, 'wb') as f:
            pickle.dump(scaler_lda, f)
        print(f"✅ 3D LDA model and Scaler saved for plotting: {LDA_FILE_3D}")
    except Exception as e:
        print(f"ERROR saving 3D LDA model: {e}")

    # 2. 2D LDA Model (For 2D margin plot)
    print("Training 2D LDA SVM Model (for margin plot)...")

    lda_2d = LDA(n_components=FINAL_VIZ_DIMENSIONS_2D)
    X_train_lda_2d = lda_2d.fit_transform(X_train_scaled_lda, y_train)

    svm_model_2d = SVC(C=1.0, gamma='scale', kernel='rbf', random_state=42)
    svm_model_2d.fit(X_train_lda_2d, y_train)

    try:
        with open(MODEL_FILE_2D_LDA, 'wb') as f:
            pickle.dump(svm_model_2d, f)
        print(f"✅ 2D LDA SVM Model saved for margin plot: {MODEL_FILE_2D_LDA}")
    except Exception as e:
        print(f"ERROR saving 2D LDA SVM model: {e}")


def train_full_process():
    print("\n" + "=" * 70)
    print("--- STARTING DUAL TRAINING (125D Tuned & 3D LDA Viz) ---")
    print("=" * 70)

    try:
        data = np.load(PARSED_DATA_PATH, allow_pickle=True);
        parsed_data_dict = dict(data)
    except FileNotFoundError:
        print(f"ERROR: Training data file not found at {PARSED_DATA_PATH}.");
        return

    X, y = prepare_data(parsed_data_dict)
    if X.shape[0] == 0: print("No valid data found for training."); return

    # CACHE RAW DATA
    try:
        np.savez_compressed(TRAINING_DATA_CACHE, X=X, y=y)
        print(f"Raw feature data ({X.shape[1]}D) cached to '{TRAINING_DATA_CACHE}'.")
    except Exception as e:
        print(f"WARNING: Could not save raw feature data cache: {e}")

    # Use the full training data for training artifacts
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    print(f"Total training samples after split: {len(X_train)}")

    # 1. Train and Tune the 125D ACCURACY Model
    scaler_125d = StandardScaler()
    X_train_scaled_125d = scaler_125d.fit_transform(X_train)
    tune_and_train_svm(X_train_scaled_125d, y_train, scaler_125d)

    # 2. Train the LDA VISUALIZATION Models
    train_lda_viz_models(X_train, y_train)


def evaluate_test_data():
    print("\n" + "=" * 60)
    print(f"--- STARTING EVALUATION ON OPTIMAL 125D MODEL ---")
    print("=" * 60)

    required_files = [MODEL_FILE_125D, SCALER_FILE_125D]
    for file in required_files:
        if not os.path.exists(file):
            print(f"ERROR: Required optimal file not found: '{file}'. Run training first.");
            return

    try:
        with open(MODEL_FILE_125D, 'rb') as f:
            svm_model = pickle.load(f)
        with open(SCALER_FILE_125D, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"ERROR loading optimal artifacts: {e}");
        return

    try:
        data_test = np.load(TEST_DATA_PATH, allow_pickle=True);
        parsed_test_dict = dict(data_test)
    except FileNotFoundError:
        print(f"ERROR: Test data file not found at {TEST_DATA_PATH}.");
        return

    X_test, y_test = prepare_data(parsed_test_dict)
    if X_test.shape[0] == 0: print("No valid data found in the test file."); return

    # Transform using 125D scaler
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm_model.predict(X_test_scaled)

    print(f"\nCLASSIFICATION REPORT ON UNSEEN TEST FILE (Full 125 Features):")
    print(classification_report(y_test, y_pred))


# --- 4. VISUALIZATION FUNCTIONS ---

def get_plot_artifacts():
    """Helper to load common plotting artifacts."""
    required_files = [MODEL_FILE_2D_LDA, SCALER_FILE_LDA_VIZ, LDA_FILE_3D, TRAINING_DATA_CACHE]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Skipping plot: Required visualization file '{file}' not found.");
            return None, None, None, None, None

    try:
        with open(MODEL_FILE_2D_LDA, 'rb') as f:
            svm_model: SVC = pickle.load(f)
        with open(SCALER_FILE_LDA_VIZ, 'rb') as f:
            scaler: StandardScaler = pickle.load(f)
        with open(LDA_FILE_3D, 'rb') as f:
            lda: LDA = pickle.load(f)
        data_cache = np.load(TRAINING_DATA_CACHE, allow_pickle=True)
        X_raw = data_cache['X'];
        y_raw = data_cache['y']
        return svm_model, scaler, lda, X_raw, y_raw
    except Exception as e:
        print(f"Error loading LDA plot artifacts: {e}");
        return None, None, None, None, None


def visualize_training_data_2d():
    """Plots the 2D LDA decision boundary with training data points and support vectors."""
    print("\n--- Generating 2D LDA SVM Training Data Plot (Optimal Separability) ---")
    plt.switch_backend('Agg')

    svm_model, scaler, lda, X_raw, y = get_plot_artifacts()
    if svm_model is None: return

    # Transform training data using LDA artifacts
    X_scaled = scaler.transform(X_raw)
    X_2d_lda = lda.transform(X_scaled)[:, :2]  # Use 3D LDA object, but take only LD1 and LD2

    unique_labels = np.unique(y);
    n_classes = len(unique_labels)
    colors = plt.cm.get_cmap('Dark2', n_classes)
    label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}

    # Define plot bounds based on training data
    x_min, x_max = X_2d_lda[:, 0].min() - 0.5, X_2d_lda[:, 0].max() + 0.5
    y_min, y_max = X_2d_lda[:, 1].min() - 0.5, X_2d_lda[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict on mesh using the 2D SVM model
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_color_idx = np.array([label_to_color_idx[label] for label in Z])
    Z_color_idx = Z_color_idx.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(12, 9))

    # 1. Plot Decision Boundary Background
    ax.contourf(xx, yy, Z_color_idx, cmap=colors, alpha=0.2)

    # 2. Plot Training Data Points (Circles)
    for i, label in enumerate(unique_labels):
        indices = y == label
        ax.scatter(X_2d_lda[indices, 0], X_2d_lda[indices, 1],
                   color=colors(i), label=f'Train: {label}', alpha=0.8, edgecolors='black', s=100)

    # 3. Highlight Support Vectors
    ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
               s=300, facecolors='none', edgecolors='yellow', linewidth=1.5, label='Support Vectors')

    ax.set_xlabel(f'Linear Discriminant 1 (LD1)', fontsize=14)
    ax.set_ylabel(f'Linear Discriminant 2 (LD2)', fontsize=14)
    ax.set_title(f'Optimal 2D Separability (LDA) with Training Data', fontsize=16)
    ax.legend(title='Class & Support', loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    output_filename = 'decision_boundary_plot_lda_2d_train.png'
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ 2D Training Plot saved to {output_filename}")


def visualize_test_data_on_2d_lda():
    """Plots the test data points in the 2D LDA space, highlighting misclassifications."""
    print("\n--- Generating 2D LDA Test Data Plot (Misclassified Points) ---")
    plt.switch_backend('Agg')

    svm_model_2d, scaler, lda, X_raw_train, y_train_raw = get_plot_artifacts()
    if svm_model_2d is None: return

    # Load and Prepare Test Data
    try:
        data_test = np.load(TEST_DATA_PATH, allow_pickle=True);
        parsed_test_dict = dict(data_test)
        X_test, y_test = prepare_data(parsed_test_dict)
    except FileNotFoundError:
        print(f"ERROR: Test data file not found at {TEST_DATA_PATH}.");
        return
    except Exception as e:
        print(f"Error preparing test data: {e}");
        return

    if X_test.shape[0] == 0:
        print("No valid test data to plot.");
        return

    # --- 1. Predict and Transform Test Data ---
    X_test_scaled = scaler.transform(X_test)
    X_test_2d_lda = lda.transform(X_test_scaled)[:, :2]

    # Predict using the 2D SVM model
    y_pred_2d = svm_model_2d.predict(X_test_2d_lda)
    is_correct = y_test == y_pred_2d

    # --- 2. Plot Setup (Boundary) ---
    unique_labels = np.unique(y_train_raw);
    n_classes = len(unique_labels)
    colors = plt.cm.get_cmap('Dark2', n_classes)
    label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 9))

    # Calculate and plot the background decision boundary based on TRAINING data limits for scale
    X_train_scaled = scaler.transform(X_raw_train)
    X_train_2d_lda = lda.transform(X_train_scaled)[:, :2]
    x_min, x_max = X_train_2d_lda[:, 0].min() - 0.5, X_train_2d_lda[:, 0].max() + 0.5
    y_min, y_max = X_train_2d_lda[:, 1].min() - 0.5, X_train_2d_lda[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm_model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_color_idx = np.array([label_to_color_idx[label] for label in Z])
    Z_color_idx = Z_color_idx.reshape(xx.shape)
    ax.contourf(xx, yy, Z_color_idx, cmap=colors, alpha=0.2)

    # --- 3. Plot Test Data Points (Crosses) ---

    # We plot all points in one loop, differentiating marker size/outline based on classification status
    # This prevents the need for a separate legend entry for misclassified points.
    plotted_labels = set()

    for i, label in enumerate(unique_labels):
        indices_label = (y_test == label)

        # Correctly classified points
        indices_correct = indices_label & is_correct
        if indices_correct.any():
            # Only label the first set of correct points for the legend
            label_text = f'Test: {label}' if label not in plotted_labels else None
            ax.scatter(X_test_2d_lda[indices_correct, 0], X_test_2d_lda[indices_correct, 1],
                       marker='x', s=150, linewidths=2, alpha=0.8,
                       color=colors(i), label=label_text)
            plotted_labels.add(label)

        # Misclassified points (thicker/larger version of the true class color)
        indices_misclassified = indices_label & (~is_correct)
        if indices_misclassified.any():
            # Misclassified points are plotted over the correct points for visibility
            # Use marker='X' (thicker cross) for emphasis
            ax.scatter(X_test_2d_lda[indices_misclassified, 0], X_test_2d_lda[indices_misclassified, 1],
                       marker='X',
                       s=250,  # Larger size
                       linewidths=3,  # Thicker lines
                       alpha=1.0,
                       color=colors(i),  # Same color as the true class
                       edgecolors='black',  # Add a black edge for contrast
                       label=None)  # No label here, handled by the custom legend entry below

    # Add a separate, generic legend entry for misclassified points
    if (~is_correct).any():
        ax.scatter([], [],
                   marker='X', s=250, linewidths=3, alpha=1.0,
                   color='gray', edgecolors='black',
                   label='Test: Misclassified (Thicker Marker)')

    # Final Touches
    ax.set_xlabel(f'Linear Discriminant 1 (LD1)', fontsize=14)
    ax.set_ylabel(f'Linear Discriminant 2 (LD2)', fontsize=14)
    ax.set_title(f'Test Data Performance in 2D LDA Space', fontsize=16)
    ax.legend(title='Test Classification', loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    output_filename = 'test_data_misclassification_lda_2d.png'
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"✅ 2D Test Data Plot saved to {output_filename}")


def export_3d_lda_data():
    """Exports 3D LDA data for external 3D plotting."""
    print("\n--- Exporting 3D LDA Data for External Plotting ---")

    required_files = [SCALER_FILE_LDA_VIZ, LDA_FILE_3D, TRAINING_DATA_CACHE]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Skipping export: Required file '{file}' not found.");
            return

    try:
        with open(SCALER_FILE_LDA_VIZ, 'rb') as f:
            scaler: StandardScaler = pickle.load(f)
        with open(LDA_FILE_3D, 'rb') as f:
            lda: LDA = pickle.load(f)
        data_cache = np.load(TRAINING_DATA_CACHE, allow_pickle=True)
        X_raw = data_cache['X'];
        y = data_cache['y']
    except Exception as e:
        print(f"Error loading 3D LDA artifacts: {e}");
        return

    # Transform 125D data to 3D LDA space
    X_scaled = scaler.transform(X_raw)
    X_3d_lda = lda.transform(X_scaled)

    output_filename = 'lda_3d_projection_data.npz'
    np.savez_compressed(output_filename, X_3d_lda=X_3d_lda, y_labels=y)
    print(f"✅ 3D LDA data saved to {output_filename} (LD1, LD2, LD3) for external visualization.")


# --- 5. MAIN EXECUTION ---

if __name__ == '__main__':
    train_full_process()
    evaluate_test_data()
    # Outputs the graph with training data points (circles) and support vectors
    visualize_training_data_2d()
    # Outputs the graph with test data points (crosses), highlighting misclassified points (thick/large 'X')
    visualize_test_data_on_2d_lda()
    export_3d_lda_data()