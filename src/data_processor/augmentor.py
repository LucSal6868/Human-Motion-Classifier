import numpy as np

NUM_AUGMENTED_SAMPLES = 100

NOISE_RANGE = (0.00, 0.01)
SKEW_RANGE = (0.05, 0.25)
ROTATION_RANGE = (-np.pi / 16, np.pi / 16)
TWIST_RANGE = (-np.pi / 12, np.pi / 12)
WOBBLE_RANGE = (0.05, 1)
SCALE_RANGE = (0.7, 1.3)


#######################################################################

def augment(input_file: str, output_file: str):
    data = np.load(input_file, allow_pickle=True)
    augmented_data = {key: value for key, value in data.items()}

    for key, dataset in data.items():
        print(key)
        dataset: np.ndarray
        augmented_dataset: list[np.ndarray] = []

        for array in dataset:
            array: np.ndarray

            # ORIGINAL
            augmented_dataset.append(array)

            # AUGMENTED
            for n in range(NUM_AUGMENTED_SAMPLES):
                current_array = array.copy()

                # NOISE
                noise_mag = np.random.uniform(*NOISE_RANGE)
                current_array = add_noise(current_array, noise_mag)

                # SKEW
                skew_mag = np.random.uniform(*SKEW_RANGE)
                current_array = skew(current_array, skew_mag)

                # ROTATE
                rotation_mag = np.random.uniform(*ROTATION_RANGE)
                current_array = rotate(current_array, rotation_mag)

                # TWIST
                twist_mag = np.random.uniform(*TWIST_RANGE)
                current_array = twist(current_array, twist_mag)

                # WOBBLE
                wobble_mag = np.random.uniform(*WOBBLE_RANGE)
                current_array = wobble(current_array, wobble_mag)

                # SCALE
                scale_factor = np.random.uniform(*SCALE_RANGE)
                current_array = scale(current_array, scale_factor)

                # FINISH
                augmented_dataset.append(current_array)

        augmented_data[key] = np.array(augmented_dataset, dtype=object)

    np.savez_compressed(output_file, **augmented_data)


#######################################################################
def add_noise(array: np.ndarray, magnitude: float) -> np.ndarray:
    if array.size == 0:
        return array
    scale = np.std(array, axis=0) * magnitude
    noise = np.random.normal(loc=0.0, scale=scale, size=array.shape)
    return array + noise


def skew(array: np.ndarray, magnitude: float) -> np.ndarray:
    skew_xy = np.random.uniform(-magnitude, magnitude)
    skew_xz = np.random.uniform(-magnitude, magnitude)
    skew_yz = np.random.uniform(-magnitude, magnitude)

    skew_matrix = np.array([
        [1, skew_xy, skew_xz],
        [0, 1, skew_yz],
        [0, 0, 1]
    ])

    skewed_array = array @ skew_matrix.T
    return skewed_array


def rotate(array: np.ndarray, magnitude: float) -> np.ndarray:
    cos_theta = np.cos(magnitude)
    sin_theta = np.sin(magnitude)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    rotated_array = array @ rotation_matrix.T
    return rotated_array


def twist(array: np.ndarray, magnitude: float) -> np.ndarray:
    if array.size == 0:
        return array

    z_coords = array[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    if z_max == z_min:
        return array.copy()

    normalized_z = (z_coords - z_min) / (z_max - z_min)

    twist_angles = normalized_z * magnitude

    x = array[:, 0]
    y = array[:, 1]
    new_x = x * np.cos(twist_angles) - y * np.sin(twist_angles)
    new_y = x * np.sin(twist_angles) + y * np.cos(twist_angles)

    twisted_array = array.copy()
    twisted_array[:, 0] = new_x
    twisted_array[:, 1] = new_y
    return twisted_array


def wobble(array: np.ndarray, magnitude: float) -> np.ndarray:
    if array.size == 0:
        return array

    wobbled_array = array.copy().astype(np.float64)

    z_coords = wobbled_array[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    if z_max == z_min:
        return wobbled_array

    normalized_z = (z_coords - z_min) / (z_max - z_min)

    frequency_x = np.random.uniform(1.0, 3.0)
    frequency_y = np.random.uniform(1.0, 3.0)
    phase_x = np.random.uniform(0, 2 * np.pi)
    phase_y = np.random.uniform(0, 2 * np.pi)

    offset_x = magnitude * np.sin(2 * np.pi * frequency_x * normalized_z + phase_x)
    offset_y = magnitude * np.cos(2 * np.pi * frequency_y * normalized_z + phase_y)

    wobbled_array[:, 0] += offset_x
    wobbled_array[:, 1] += offset_y

    return wobbled_array


def scale(array: np.ndarray, magnitude: float) -> np.ndarray:
    if array.size == 0:
        return array

    scaled_array = array * magnitude
    return scaled_array