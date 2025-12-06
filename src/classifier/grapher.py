import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button  # Import interactive widgets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

# Define file paths (must match those in trajectory_classifier.py)
SCALER_FILE = "scaler.pkl"
TRAINING_DATA_CACHE = "training_data_cache.npz"


def visualize_pca_interactive():
    """
    Loads cached training data and the fitted scaler, then performs PCA
    to visualize the separation of features in 2D using interactive sliders
    to select the Principal Components (PC).
    """
    print("\n--- STARTING INTERACTIVE PCA Visualization ---")

    # 1. Check for required files
    if not os.path.exists(TRAINING_DATA_CACHE):
        print(f"ERROR: Raw feature cache not found at '{TRAINING_DATA_CACHE}'. Run trajectory_classifier.py first.")
        return
    if not os.path.exists(SCALER_FILE):
        print(f"ERROR: Scaler file not found at '{SCALER_FILE}'. Run trajectory_classifier.py first.")
        return

    # 2. Load cached raw features (X) and labels (y)
    try:
        data_cache = np.load(TRAINING_DATA_CACHE, allow_pickle=True)
        X = data_cache['X']
        y = data_cache['y']
        print(f"Raw feature data loaded successfully. Total samples: {len(X)}")
    except Exception as e:
        print(f"ERROR loading raw feature cache: {e}")
        return

    # 3. Load the fitted scaler
    try:
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        print("Fitted StandardScaler loaded.")
    except Exception as e:
        print(f"ERROR loading scaler: {e}")
        return

    # 4. Scale the data and perform PCA
    X_scaled = scaler.transform(X)
    n_components = X_scaled.shape[1]

    # PCA to calculate all components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Initial components to plot
    initial_pc_x = 1
    initial_pc_y = 2

    # Setup Figure and main plot axis, adjusting layout to make space for sliders
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    unique_labels = np.unique(y)
    colors = plt.cm.get_cmap('Spectral', len(unique_labels))

    # --- PLOTTING FUNCTION ---
    def update_scatter(pc_x, pc_y):
        # 0-based indexing
        idx_x = int(pc_x) - 1
        idx_y = int(pc_y) - 1

        # Input validation
        if idx_x >= n_components or idx_y >= n_components:
            ax.set_title(f"Error: Component index out of bounds (Max PC: {n_components})", color='red')
            return

        if idx_x == idx_y:
            ax.set_title("PCA Visualization (Select two different components)", color='red')
            return

        x_data = X_pca[:, idx_x]
        y_data = X_pca[:, idx_y]

        ax.cla()  # Clear current axes content

        # Re-plot all classes
        for i, label in enumerate(unique_labels):
            indices = y == label

            ax.scatter(x_data[indices], y_data[indices],
                       label=label,
                       alpha=0.7,
                       edgecolors='k',
                       s=70,
                       color=colors(i))

        # Calculate variance information
        variance_x = pca.explained_variance_ratio_[idx_x]
        variance_y = pca.explained_variance_ratio_[idx_y]

        # Determine maximum PC for cumulative variance calculation
        max_pc = max(pc_x, pc_y)
        total_variance_top_n = np.sum(pca.explained_variance_ratio_[:max_pc]) * 100

        # Update axis labels and title
        ax.set_xlabel(f'Principal Component {pc_x} ({variance_x * 100:.1f}% Variance)', fontsize=12)
        ax.set_ylabel(f'Principal Component {pc_y} ({variance_y * 100:.1f}% Variance)', fontsize=12)
        ax.set_title(f'PCA Visualization (PC{pc_x} vs PC{pc_y}) | Cumulative Top {max_pc}: {total_variance_top_n:.1f}%',
                     fontsize=12)

        ax.legend(title='Gesture Class', loc='upper right', frameon=True, shadow=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)

        fig.canvas.draw_idle()  # Redraw the figure

    # --- SLIDER SETUP ---

    # Define slider positions (left, bottom, width, height)
    ax_x = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor='#f0f0f0')
    ax_y = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='#f0f0f0')

    # Create the Sliders
    # Slider min, max, initial value, step
    slider_x = Slider(
        ax=ax_x,
        label='PC X-Axis (1-based index)',
        valmin=1,
        valmax=n_components,
        valinit=initial_pc_x,
        valstep=1
    )

    slider_y = Slider(
        ax=ax_y,
        label='PC Y-Axis (1-based index)',
        valmin=1,
        valmax=n_components,
        valinit=initial_pc_y,
        valstep=1
    )

    # Function called when the slider value changes
    def update(val):
        pc_x = slider_x.val
        pc_y = slider_y.val
        update_scatter(pc_x, pc_y)

    # Connect sliders to update function
    slider_x.on_changed(update)
    slider_y.on_changed(update)

    # Initial plot call
    update_scatter(initial_pc_x, initial_pc_y)

    # Display the plot window
    plt.show()


if __name__ == '__main__':
    visualize_pca_interactive()