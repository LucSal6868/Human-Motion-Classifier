from parser import parse
from augmentor import augment
from src.paths import PATHS

print("PROCESSING RAW DATA")

print("\tparsing train data...")
parse(PATHS.RAW_TRAIN_DATA_FOLDER.get_path(), PATHS.PARSED_TRAIN_DATA.get_path())
print("\taugmenting...")
augment(PATHS.PARSED_TRAIN_DATA.get_path(), PATHS.AUGMENTED_TRAIN_DATA.get_path())

print("\tparsing test data...")
parse(PATHS.RAW_TEST_DATA_FOLDER.get_path(), PATHS.PARSED_TEST_DATA.get_path())
print("\taugmenting test data...")
augment(PATHS.PARSED_TEST_DATA.get_path(), PATHS.PARSED_TEST_DATA.get_path())


print("DONE")