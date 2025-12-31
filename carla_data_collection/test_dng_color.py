
import rawpy
import numpy as np
import os
import cv2

# Path to a sample DNG
dng_path = r"d:\code\carla\dataset_10k_fix\cameras\cam_front_main\000000.dng"

if not os.path.exists(dng_path):
    print(f"File not found: {dng_path}")
else:
    try:
        with rawpy.imread(dng_path) as raw:
            print(f"Raw type: {raw.raw_type}")
            print(f"Color description: {raw.color_desc}")
            print(f"Sizes: {raw.sizes}")
            print(f"Pattern: {raw.raw_pattern}")
            
            rgb = raw.postprocess(no_auto_bright=True, use_camera_wb=False, user_wb=[1,1,1,1])
            print(f"Processed shape: {rgb.shape}")
            
            # Check if it's grayscale
            is_gray = np.allclose(rgb[:,:,0], rgb[:,:,1]) and np.allclose(rgb[:,:,1], rgb[:,:,2])
            print(f"Is grayscale? {is_gray}")
            
            if is_gray:
                print("Rawpy treated it as grayscale. Trying manual demosaic...")
                # Access raw data directly
                raw_data = raw.raw_image
                print(f"Raw data shape: {raw_data.shape}, dtype: {raw_data.dtype}")
                
                # Manual demosaic using OpenCV
                # CARLA is usually RGGB.
                # Note: rawpy reads into numpy. 
                # OpenCV expects uint8 or uint16.
                
                # Simple demosaic
                # We need to know the pattern. Assuming RGGB.
                # cv2.COLOR_BayerRG2RGB
                
                rgb_manual = cv2.cvtColor(raw_data, cv2.COLOR_BayerRG2RGB)
                print(f"Manual demosaic shape: {rgb_manual.shape}")
                
                # Check variance in color channels to confirm color
                std_r = np.std(rgb_manual[:,:,0])
                std_g = np.std(rgb_manual[:,:,1])
                std_b = np.std(rgb_manual[:,:,2])
                print(f"Channel std devs: R={std_r:.2f}, G={std_g:.2f}, B={std_b:.2f}")

    except Exception as e:
        print(f"Error: {e}")
