import os
import numpy as np

#######################################################################

def parse(input_folder: str, output_file: str, mode: str = 'subfolder'):
    if mode not in ['subfolder', 'file']:
        raise ValueError("Invalid mode. Must be 'subfolder' or 'file'.")

    parsed_data = {}

    if mode == 'subfolder':
        print(f"PARSING SUBFOLDER")
        subfolders: list[str] = get_subfolders(input_folder)

        if not subfolders:
            print("No valid subfolders found. Exiting.")
            return

        for sf_name in subfolders:
            sf_path = os.path.join(input_folder, sf_name)
            sf_data_list: list[np.ndarray] = []

            try:
                for entry in os.scandir(sf_path):
                    if entry.is_file():
                        file_data: np.ndarray = get_data_from_file(entry.path)
                        if file_data.size > 0:
                            sf_data_list.append(file_data)

            except Exception as e:
                print(f"\tERROR GETTING DATA FROM {sf_name}: {e}")
                continue

            if sf_data_list:
                parsed_data[sf_name] = np.array(sf_data_list, dtype=object)
                print(f"\tProcessed {len(sf_data_list)} files for class: {sf_name}")
            else:
                print(f"\tWarning: No valid data found in subfolder: {sf_name}")

    elif mode == 'file':
        print(f"PARSING FILEs")

        try:
            for entry in os.scandir(input_folder):
                if entry.is_file():
                    file_name_key = os.path.splitext(entry.name)[0]
                    file_path = entry.path

                    file_data: np.ndarray = get_data_from_file(file_path)

                    if file_data.size > 0:
                        parsed_data[file_name_key] = file_data
                        print(f"\tProcessed file: {entry.name} -> Key: {file_name_key}")
                    else:
                        print(f"\tWarning: File {entry.name} contained no valid data.")

        except Exception as e:
            print(f"ERROR processing files in {input_folder}: {e}")
            return

    # SAVE PARSED DATA TO FILE
    if parsed_data:
        np.savez_compressed(output_file, **parsed_data)


#######################################################################

# GETS SUBFOLDERS IN A FOLDER
def get_subfolders(folder: str) -> list[str]:
    subfolders = []
    try:
        for entry in os.scandir(folder):
            if entry.is_dir():
                subfolders.append(entry.name)

        if not subfolders:
            print("Warning: RAW DATA has no subfolders.")

    except Exception as e:
        print("CANT GET SUBFOLDERS IN " + folder)
        print(f"Error: {e}")
        return []
    return subfolders


# GETS 3D POINT DATA FROM A RAW FILE
def get_data_from_file(file_path: str) -> np.ndarray:
    result: list[list[int]] = []

    # OPEN FILE
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                row: list[str] = line.strip().split(",")
                if row and row[0] == "r" and len(row) > 6:
                    # row[6] contains the "x/y/z" string
                    vector3_str: list[str] = row[6].split("/")

                    try:
                        vector3_i = [int(s) for s in vector3_str]
                        if len(vector3_i) == 3:
                            result.append(vector3_i)
                    except ValueError:

                        continue
                elif row and row[0] == "s":
                    pass  # ignore 's' lines

    except Exception as e:
        pass

    return np.array(result, dtype=int)

