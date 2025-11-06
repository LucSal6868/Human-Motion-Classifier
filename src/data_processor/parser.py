import os
import csv

#######################################################################

def parse(data_folder : str, target_folder : str):

    # GET SUBFOLDERS IN RAW
    subfolders : list[str] = get_subfolders(data_folder)

    # CREATE PARSED FOLDER IF IT DOES NOT EXIST
    try:
        os.makedirs(target_folder, exist_ok=True)
    except Exception as e:
        print(e)

    # PARSE EACH SUBFOLDER
    for sf_name in subfolders:
        sf_path = os.path.join(data_folder, sf_name)
        try:
            with os.scandir(sf_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        file_name = entry.name
                        file_path = os.path.join(sf_path, file_name)

                        print(get_data_from_file(file_path))
                        print("\n")

        except Exception as e:
            print(f"\tFAILED TO PARSE {sf_name}: {e}")
            continue



#######################################################################

def get_subfolders(folder : str) -> list[str]:
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


def get_data_from_file(file_path: str) -> list[list[int]]:
    lines : list[str] = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"\tERROR READING {file_path}: {e}")

    vector3_array : list[list[int]] = []

    for line in lines:
        row : list[str] = line.split(",")
        if len(row) >= 7:
            vector3_str: list[str] = row[6].split("/")
            vector3_i = [int(s) for s in vector3_str]
            vector3_array.append(vector3_i)

    return vector3_array

