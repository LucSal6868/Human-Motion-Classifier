import os
import numpy as np
from src.paths import PATHS
from typing import List, Tuple

#######################################################################
## HELPER FUNCTIONS (Defined first)
#######################################################################

def get_subfolders(folder: str) -> List[str]:
    subfolders = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_dir():
                    subfolders.append(entry.name)

        if len(subfolders) <= 0:
            raise Exception("RAW DATA HAS NO SUBFOLDERS")

    except Exception as e:
        print("CANT GET SUBFOLDERS IN " + folder)
        return []
    return subfolders


def get_data_from_file(file_path: str) -> Tuple[np.ndarray, str]:
    """
    Parses a raw file to get 3D point data.
    Returns: (np.ndarray data, str base_filename_key)
    """
    lines: List[str] = []
    # Extract the base filename without extension (e.g., 'test_1')
    file_key = os.path.splitext(os.path.basename(file_path))[0]

    # OPEN FILE
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"\tERROR READING {file_path}: {e}")
        return np.array([], dtype=int), file_key

    result: List[List[int]] = []

    # PARSE LINE BY LINE
    for line in lines:
        row: List[str] = line.strip().split(",")

        # Check for 'r' and ensure index 6 exists
        if row and row[0] == "r" and len(row) > 6:
            vector3_str: List[str] = row[6].split("/")
            try:
                vector3_i = [int(s) for s in vector3_str]
                result.append(vector3_i)
            except ValueError as ve:
                print(f"\tWARNING: Could not parse vector in {file_path}: {row[6]} - {ve}")
                continue

        elif row and row[0] == "s":
            pass

    return np.array(result, dtype=int), file_key

#######################################################################
## MAIN PARSING FUNCTIONS
#######################################################################

def parse(input_folder : str, output_file : str):
    """
    Parses data from subfolders, combining all files within a subfolder into one array,
    and saves to a single .npz file (e.g., {subfolder_name: data}).
    """
    # GET SUBFOLDERS
    subfolders: List[str] = get_subfolders(input_folder)

    parsed_data = {}

    # PARSE EACH SUBFOLDER
    for sf_name in subfolders:
        sf_path = os.path.join(input_folder, sf_name)
        sf_data_list: List[np.ndarray] = []

        try:
            with os.scandir(sf_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        file_path = os.path.join(sf_path, entry.name)
                        # We only need the data (index 0) from the helper function
                        file_data: np.ndarray = get_data_from_file(file_path)[0]
                        sf_data_list.append(file_data)

        except Exception as e:
            print(f"\tERROR GETTING DATA FROM {sf_name}: {e}")
            continue

        sf_data = np.array(sf_data_list, dtype=object)
        parsed_data[sf_name] = sf_data

    # SAVE PARSED DATA TO FILE
    np.savez_compressed(output_file, **parsed_data)

#######################################################################

def parse_test_data(input_folder: str, output_file: str):
    """
    Parses all raw test data files directly in the input_folder and consolidates
    the resulting data into a single compressed NumPy archive (.npz).

    Each file's base name (e.g., 'test_1') becomes a key in the output .npz file.
    """
    all_test_data = {}

    print(f"Starting to parse test files from: {input_folder}")

    try:
        with os.scandir(input_folder) as entries:
            for entry in entries:
                if entry.is_file():
                    file_path = os.path.join(input_folder, entry.name)

                    # Get data and the filename key (e.g., 'test_1')
                    file_data, file_key = get_data_from_file(file_path)

                    # Store the data with the filename as the key
                    all_test_data[file_key] = file_data
                    print(f"\tProcessed: {file_key}")

    except Exception as e:
        print(f"ERROR PARSING TEST DATA FROM {input_folder}: {e}")
        return

    if not all_test_data:
        print("No test files were parsed. Output file will not be created.")
        return

    # SAVE ALL PARSED DATA TO THE SINGLE OUTPUT FILE
    print(f"Saving all parsed test data to single file: {output_file}")
    np.savez_compressed(output_file, **all_test_data)
    print("Test data parsing complete.")

#######################################################################