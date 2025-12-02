import os
import numpy as np
from src.paths import PATHS

#######################################################################

def parse(input_folder : str, output_file : str):
    # GET SUBFOLDERS
    subfolders: list[str] = get_subfolders(input_folder)

    parsed_data = {}

    # PARSE EACH SUBFOLDER
    for sf_name in subfolders:
        sf_path = os.path.join(input_folder, sf_name)
        sf_data_list: list[np.ndarray] = []

        try:
            with os.scandir(sf_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        file_path = os.path.join(sf_path, entry.name)
                        file_data: np.ndarray = get_data_from_file(file_path)
                        sf_data_list.append(file_data)

        except Exception as e:
            print(f"\tERROR GETTING DATA FROM {sf_name}: {e}")
            continue

        sf_data = np.array(sf_data_list, dtype=object)
        parsed_data[sf_name] = sf_data

    # SAVE PARSED DATA TO FILE
    np.savez_compressed(output_file, **parsed_data)


#######################################################################

# GETS SUBFOLDERS IN A FOLDER
def get_subfolders(folder: str) -> list[str]:
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


# GETS 3D POINT DATA FROM A RAW FILE
def get_data_from_file(file_path: str) -> np.ndarray:
    lines: list[str] = []

    # OPEN FILE
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"\tERROR READING {file_path}: {e}")

    result: list[list[int]] = []

    # PARSE LINE BY LINE
    for line in lines:
        row: list[str] = line.strip().split(",")

        if row[0] == "r":
            vector3_str: list[str] = row[6].split("/") # GET THE 6TH INDEX
            vector3_i = [int(s) for s in vector3_str]
            result.append(vector3_i)
        elif row[0] == "s":
            pass  # ignore

    return np.array(result, dtype=int)

#######################################################################
