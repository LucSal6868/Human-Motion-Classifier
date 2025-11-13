import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
from tkinter import Tk, filedialog

DATA_FOLDER = '../../data/'

def visualize_npz(npz_path: str):
    # LOAD FILE
    data = np.load(npz_path, allow_pickle=True)

    # LIST DATASETS
    keys = list(data.keys())
    if not keys:
        print("No datasets found in file.")
        return
    else:
        print(f"\nLoaded file: {npz_path}")
        print(f"Datasets: {keys}")

    # SUBFOLDER
    while True:
        chosen = input(f"\nEnter dataset name to view ({', '.join(keys)}): ").strip()
        if chosen in keys:
            break
        else:
            print("Invalid name")

    sf_data_list = data[chosen]
    print(f"\nDataset \"{chosen}\" loaded. Contains {len(sf_data_list)} arrays.")

    # ARRAY SUMMARY
    for i, arr in enumerate(sf_data_list):
        if isinstance(arr, np.ndarray):
            print(f"  [{i}] shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  [{i}] Not a NumPy array (type={type(arr)})")

    # ARRAY
    while True:
        try:
            idx = int(input(f"\nEnter index [0–{len(sf_data_list)-1}] to view in 3D: "))
            if 0 <= idx < len(sf_data_list):
                break
            else:
                print("Invalid index, try again.")
        except ValueError:
            print("Please enter a valid integer index.")

    points = sf_data_list[idx]
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        print("Selected entry is not a valid Nx3 array.")
        return

    # PLOT IN 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x, y, z, c=z, cmap='viridis', s=20)

    ax.set_title(f"{chosen} — Entry {idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FOLDER)
    path = filedialog.askopenfilename(
        title="Select parsed .npz File",
        initialdir=script_dir,
        filetypes=[("NumPy Compressed Files", "*.npz"), ("All Files", "*.*")]
    )

    if path:
        visualize_npz(path)
    else:
        print("NO FILE SELECTED")
