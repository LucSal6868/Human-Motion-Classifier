import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List

# CONSTANTS (Must be defined once and imported)
TRAJECTORY_POINTS = 50
EXPECTED_FEATURE_SIZE = 125


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

    # DISPLACEMENT
    xy_sequence = sequence[:, :2]
    start_point = xy_sequence[0, :]
    end_point = xy_sequence[-1, :]
    delta_xy = end_point - start_point
    displacement_features = np.concatenate([start_point, end_point, delta_xy])

    # CORRELATION
    corr_matrix = np.corrcoef(sequence[:, 0], sequence[:, 1])
    xy_correlation = corr_matrix[0, 1] if corr_matrix.ndim > 1 else 0.0
    xy_correlation_feature = np.array([xy_correlation])

    # DIRECTIONAL VELOCITY & CURVATURE
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

    # CENTROID
    centroid = sequence.mean(axis=0)
    distances_from_centroid = np.linalg.norm(sequence - centroid, axis=1)
    mean_dist_centroid = np.mean(distances_from_centroid)
    std_dist_centroid = np.std(distances_from_centroid)
    centroid_distance_features = np.array([mean_dist_centroid, std_dist_centroid])

    #  TRAJECTORY (Resampled path)
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

    if features.size != EXPECTED_FEATURE_SIZE:
        return np.zeros(EXPECTED_FEATURE_SIZE, dtype=np.float32)

    return features.astype(np.float32)


def prepare_data(parsed_data_dict: Dict[str, np.ndarray], is_training: bool) -> tuple:
    X_features: List[np.ndarray] = []
    identifiers: List[str] = []

    for class_name, data_list in parsed_data_dict.items():
        # If not training, assume file mode where key is the file name
        iterable_data = data_list if is_training else [data_list]

        for sequence in iterable_data:
            try:
                sequence_arr = np.array(sequence, dtype=np.float32)
            except Exception:
                continue

            if sequence_arr.ndim == 2 and sequence_arr.shape[1] == 3 and sequence_arr.shape[0] >= 2:
                features = extract_features(sequence_arr)
                if features.size == EXPECTED_FEATURE_SIZE:
                    X_features.append(features)
                    identifiers.append(class_name if is_training else class_name)

    X = np.array(X_features)

    if is_training:
        return X, np.array(identifiers)
    else:
        return X, identifiers