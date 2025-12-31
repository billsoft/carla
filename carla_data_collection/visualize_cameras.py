
import os
import glob
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import sys

# Constants based on the camera configuration
CAMERA_LAYOUT = {
    'cam_front_main': {'pos': (1, 1), 'title': 'Front Main'},
    'cam_front_wide': {'pos': (0, 1), 'title': 'Front Wide'},
    'cam_front_narrow': {'pos': (2, 1), 'title': 'Front Narrow'},
    'cam_left_pillar': {'pos': (0, 0), 'title': 'Left Pillar'},
    'cam_right_pillar': {'pos': (2, 0), 'title': 'Right Pillar'},
    'cam_left_repeater': {'pos': (0, 2), 'title': 'Left Repeater'},
    'cam_right_repeater': {'pos': (2, 2), 'title': 'Right Repeater'},
    'cam_rear': {'pos': (1, 2), 'title': 'Rear'}
}

# Grid layout (Rows, Cols)
# Row 0: Front Wide, Front Main, Front Narrow
# Row 1: Left Pillar, BEV (Car), Right Pillar
# Row 2: Left Repeater, Rear, Right Repeater
GRID_MAPPING = {
    'cam_front_wide': (0, 0),
    'cam_front_main': (0, 1),
    'cam_front_narrow': (0, 2),
    'cam_left_pillar': (1, 0),
    'cam_right_pillar': (1, 2),
    'cam_left_repeater': (2, 0),
    'cam_rear': (2, 1),
    'cam_right_repeater': (2, 2)
}

class DatasetViewer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.cameras = list(GRID_MAPPING.keys())
        self.frames = self._scan_frames()
        self.current_frame_idx = 0
        
        self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 10))
        self.fig.canvas.manager.set_window_title('CARLA Dataset Viewer')
        
        # Setup axes
        self.image_axes = {}
        for cam_name, (row, col) in GRID_MAPPING.items():
            ax = self.axes[row, col]
            self.image_axes[cam_name] = ax
            ax.set_title(CAMERA_LAYOUT[cam_name]['title'])
            ax.axis('off')
            
        # Setup Central BEV axis
        self.bev_ax = self.axes[1, 1]
        self.bev_ax.set_title("Ego Vehicle (BEV)")
        self.bev_ax.axis('off')
        self._draw_bev_car()

        # Buttons
        ax_prev = plt.axes([0.3, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.6, 0.02, 0.1, 0.05])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        
        # Initial Render
        self.update_view()
        plt.show()

    def _scan_frames(self):
        # Scan one camera folder to get frame counts
        cam_path = os.path.join(self.dataset_path, self.cameras[0])
        files = sorted(glob.glob(os.path.join(cam_path, "*.dng")))
        frames = [os.path.splitext(os.path.basename(f))[0] for f in files]
        print(f"Found {len(frames)} frames.")
        return frames

    def _draw_bev_car(self):
        # Simple rectangle representing the car
        rect = plt.Rectangle((0.4, 0.3), 0.2, 0.4, color='blue', alpha=0.7)
        self.bev_ax.add_patch(rect)
        self.bev_ax.set_xlim(0, 1)
        self.bev_ax.set_ylim(0, 1)
        # Add arrow for direction
        self.bev_ax.arrow(0.5, 0.7, 0, 0.1, head_width=0.05, head_length=0.05, fc='red', ec='red')
        self.bev_ax.text(0.5, 0.5, "Hero Car", ha='center', va='center', color='white')

    def load_dng(self, path):
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess()
            return rgb
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((100, 100, 3), dtype=np.uint8)

    def update_view(self):
        if not self.frames:
            return
            
        frame_id = self.frames[self.current_frame_idx]
        self.fig.suptitle(f"Frame: {frame_id} ({self.current_frame_idx + 1}/{len(self.frames)})", fontsize=16)
        
        for cam_name, ax in self.image_axes.items():
            img_path = os.path.join(self.dataset_path, cam_name, f"{frame_id}.dng")
            
            if os.path.exists(img_path):
                img = self.load_dng(img_path)
                ax.imshow(img)
            else:
                # Try loading PNG if DNG not found (fallback)
                png_path = img_path.replace('.dng', '.png')
                if os.path.exists(png_path):
                    img = Image.open(png_path)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                    
        self.fig.canvas.draw_idle()

    def prev_frame(self, event):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_view()

    def next_frame(self, event):
        if self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            self.update_view()

if __name__ == "__main__":
    # Default path
    default_path = r"d:\code\carla\dataset_10k\cameras"
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = default_path
        
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
    else:
        viewer = DatasetViewer(dataset_path)
