import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from tkinter import Tk, filedialog
from src.paths import PATHS


class NPZVisualizer:
    def __init__(self, npz_path):
        self.data = np.load(npz_path, allow_pickle=True)
        self.keys = list(self.data.keys())
        if not self.keys:
            raise ValueError("No datasets found in file.")

        self.dataset_idx = 0
        self.array_idx = 0

        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.subplots_adjust(bottom=0.2)

        # BUTTONS
        axprev_data = plt.axes([0.1, 0.05, 0.1, 0.075])
        axnext_data = plt.axes([0.21, 0.05, 0.1, 0.075])
        axprev_array = plt.axes([0.6, 0.05, 0.1, 0.075])
        axnext_array = plt.axes([0.71, 0.05, 0.1, 0.075])

        self.btn_prev_data = Button(axprev_data, 'Prev Dataset')
        self.btn_next_data = Button(axnext_data, 'Next Dataset')
        self.btn_prev_array = Button(axprev_array, 'Prev Array')
        self.btn_next_array = Button(axnext_array, 'Next Array')

        self.btn_prev_data.on_clicked(self.prev_dataset)
        self.btn_next_data.on_clicked(self.next_dataset)
        self.btn_prev_array.on_clicked(self.prev_array)
        self.btn_next_array.on_clicked(self.next_array)

        self.plot_current()

    def get_current_array(self):
        key = self.keys[self.dataset_idx]
        arr_list = self.data[key]
        if len(arr_list) == 0:
            return None
        return arr_list[self.array_idx]

    def plot_current(self):
        self.ax.clear()
        key = self.keys[self.dataset_idx]
        arr_list = self.data[key]

        if len(arr_list) == 0:
            self.ax.set_title(f"{key} — No arrays")
            plt.draw()
            return

        array = arr_list[self.array_idx]
        if isinstance(array, np.ndarray) and array.ndim == 2 and array.shape[1] == 3:
            x, y, z = array[:, 0], array[:, 1], array[:, 2]
            self.ax.scatter(x, y, z, c=z, cmap='viridis', s=20)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            num_points = array.shape[0]
            self.ax.set_title(f"{key} — Array {self.array_idx} / {len(arr_list) - 1} — {num_points} points")
        else:
            self.ax.set_title(f"{key} — Array {self.array_idx} not Nx3")

        plt.draw()

    def prev_dataset(self, event):
        self.dataset_idx = (self.dataset_idx - 1) % len(self.keys)
        self.array_idx = 0
        self.plot_current()

    def next_dataset(self, event):
        self.dataset_idx = (self.dataset_idx + 1) % len(self.keys)
        self.array_idx = 0
        self.plot_current()

    def prev_array(self, event):
        arr_list = self.data[self.keys[self.dataset_idx]]
        if len(arr_list) == 0:
            return
        self.array_idx = (self.array_idx - 1) % len(arr_list)
        self.plot_current()

    def next_array(self, event):
        arr_list = self.data[self.keys[self.dataset_idx]]
        if len(arr_list) == 0:
            return
        self.array_idx = (self.array_idx + 1) % len(arr_list)
        self.plot_current()


if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select parsed .npz File",
        initialdir=PATHS.DATA_FOLDER.get_path(),
        filetypes=[("NumPy Compressed Files", "*.npz"), ("All Files", "*.*")]
    )

    if path:
        visualizer = NPZVisualizer(path)
        plt.show()
    else:
        print("NO FILE SELECTED")
