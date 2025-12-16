
import sys
import os
from pathlib import Path

# Add project root
project_root = Path(r"d:\code\carla\carla_data_collection")
sys.path.insert(0, str(project_root))

try:
    from data.occupancy_generator import OccupancyGenerator
    print("Import successful")
    gen = OccupancyGenerator()
    print("Initialization successful")
except Exception as e:
    print(f"Error: {e}")
