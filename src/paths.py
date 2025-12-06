from enum import Enum
import os


class PATHS(Enum):

    # This class manages any folder or file paths
    # EXAMPLE : PATHS.PARSED_DATA.get_path()

    # FOLDERS
    DATA_FOLDER = "data"
    RAW_TRAIN_DATA_FOLDER = "data/train/raw"
    RAW_TEST_DATA_FOLDER = "data/test/raw"

    # FILES
    PARSED_TRAIN_DATA = "data/train/parsed.npz"
    PARSED_TEST_DATA = "data/test/parsed.npz"
    AUGMENTED_TRAIN_DATA = "data/train/augmented.npz"
    AUGMENTED_TEST_DATA = "data/test/augmented.npz"

    # RELATIVE PATH FINDER
    def get_path(self) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, self.value)

    #test 1isuqdfhsqfgqsdjkfqsduk