# Comprehensive Voxel Occupancy Collection Plan

## 1. Complete Static Object Mapping (Missing Types)
Based on the analysis of `CityObjectLabel` and your requirements, I will map all missing static environment objects to the standard 18-class occupancy labels. This ensures that parking vehicles, sidewalks, and terrain are no longer "air".

| CARLA CityObjectLabel | Target Occupancy Label | Note |
| :--- | :--- | :--- |
| `Car`, `Truck`, `Bus`, `Motorcycle`, `Bicycle` | `17` (general_object) or specific vehicle IDs | Map static vehicles to `general_object` or corresponding vehicle classes if possible (e.g., Car->4, Truck->10). **Decision**: Map to `general_object` (17) to distinguish from dynamic actors, or mapped to specific classes but with static ID. I will map them to their **semantic classes** (Car->4, Truck->10) for better realism. |
| `Sidewalks` | `13` (sidewalk) | Currently ignored, causes holes near roads. |
| `Roads`, `RoadLines` | `11` (driveable_surface) | To ensure road surface continuity. |
| `Terrain` | `14` (terrain) | For off-road areas. |
| `Ground` | `12` (other_flat) | Generic ground. |
| `Water` | `12` (other_flat) | Treated as flat surface. |
| `Bridge` | `2` (construction) or `11` (driveable) | Map to `construction` (2) as per standard. |

## 2. Distance Threshold Adjustment
*   **Current Issue**: Hardcoded `dist > 60.0` check in `ground_truth_voxel_generator.py`.
*   **Fix**: Increase this threshold to **100.0 meters** (matching the voxel grid range). This ensures that objects at the edge of the perception range (like distant towers or buildings) are included.

## 3. "Conservative Rasterization" Refinement
*   **Current Status**: Implemented padding `resolution * 0.6`.
*   **Refinement**: Ensure this logic applies to **all** relevant static objects (not just poles), but be careful with large objects to avoid performance penalty. I will apply it selectively to objects with small bounding box dimensions.

## 4. LiDAR Visibility Filter Optimization (Phase 2)
After verifying completeness in "God Mode" (Phase 1), I will re-enable the filter with the optimized parameters:
*   **LiDAR**: 256 channels, 3.5M points/sec (Proven stable).
*   **Thresholds**:
    *   Large objects (Buildings, etc.): 100 points.
    *   Small/Thin objects (Poles, Signs, Fences, distant Vehicles): **1-2 points**.

## 5. Verification Steps
1.  **Code Modification**: Apply mapping and distance changes.
2.  **God Mode Run**: Run `collect_panorama.py` (Frames=1) with filter disabled.
3.  **Visual Check**: Use `occupancy_viewer` to confirm:
    *   Parking lots are full (static cars).
    *   Sidewalks are present.
    *   Distant objects (up to 100m) are visible.
4.  **Filter Re-enable**: Turn on visibility filter and verify small objects are retained.
