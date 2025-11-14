import numpy as np
from src.paths import PATHS

#######################################################################

def augment():
    data = np.load(PATHS.PARSED_DATA.get_path(), allow_pickle=True)

    # Convert NpzFile to normal dict so we can modify it
    augmented_data = {key: value for key, value in data.items()}

    for key, dataset in data.items():
        dataset: np.ndarray
        augmented_dataset: list[np.ndarray] = [] # dataset.tolist()  # convert to list

        for array in dataset:
            array: np.ndarray
            augmented_array = array

            # IM INSERTING THEM NEXT TO EACH OTHER TO COMPARE
            augmented_dataset.append(array)
            augmented_dataset.append(skew(array, max_skew=0.25))

        augmented_data[key] = np.array(augmented_dataset, dtype=object)

    np.savez_compressed(PATHS.AUGMENTED_DATA.get_path(), **augmented_data)


#######################################################################

def up_sample(array : np.ndarray) -> np.ndarray:
    pass

def down_sample(array : np.ndarray) -> np.ndarray:
    pass

def add_noise(array : np.ndarray) -> np.ndarray:
    pass


def skew(array: np.ndarray, max_skew: float = 0.2) -> np.ndarray:
    skew_xy = np.random.uniform(-max_skew, max_skew) # LEAN X TOWARDS Y
    skew_xz = np.random.uniform(-max_skew, max_skew) # LEAN X TOWARDS Z
    skew_yz = np.random.uniform(-max_skew, max_skew) # LEAN Y TOWARDS Z

    skew_matrix = np.array([
        [1, skew_xy, skew_xz],
        [0, 1, skew_yz],
        [0, 0, 1]
    ])

    skewed_array = array @ skew_matrix.T
    return skewed_array

def scale(array : np.ndarray) -> np.ndarray:
    pass

def bend(array : np.ndarray) -> np.ndarray:
    pass