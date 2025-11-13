import os
import numpy as np
import csv

#######################################################################

def parse(data_folder: str, target_folder: str):
    # GET SUBFOLDERS
    subfolders: list[str] = get_subfolders(data_folder)

    # CREATE PARSED FOLDER
    os.makedirs(target_folder, exist_ok=True)

    parsed_data = {}

    # PARSE EACH SUBFOLDER
    for sf_name in subfolders:
        sf_path = os.path.join(data_folder, sf_name)

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

    target_path = os.path.join(target_folder, "parsed.npz")
    np.savez_compressed(target_path, **parsed_data)


#######################################################################

def get_subfolders(folder: str) -> list[str]:
    subfolders = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_dir():
                    subfolders.append(entry.name)
    except Exception as e:
        print("CANT GET SUBFOLDERS IN " + folder)
        return []
    return subfolders


def get_data_from_file(file_path: str) -> np.ndarray:
    lines: list[str] = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"\tERROR READING {file_path}: {e}")

    result: list[list[int]] = []

    for line in lines:
        row: list[str] = line.strip().split(",")

        if row[0] == "r":
            vector3_str: list[str] = row[6].split("/")
            try:
                vector3_i = [int(s) for s in vector3_str]
                result.append(vector3_i)
            except ValueError:
                print(f"\tERROR PARSING VECTOR: {row[6]} in {file_path}")
        elif row[0] == "s":
            pass  # ignore

    return np.array(result, dtype=int)

#######################################################################
