# 对齐 Actor 类型与 17 类语义标签计划

为了确保所有 CARLA actor 都能正确映射到自动驾驶通用的 17 类语义标签 (nuScenes 标准)，我们需要进行一次全面的“对齐”操作。

## 第一步：全面盘点 Actor 类型 (List All)

**目标**：获取 CARLA 蓝图库中所有可能的 Actor 类型 ID，以及当前场景中实际存在的静态对象标签。

*   **现状**：
    *   `scripts/list_actor_types.py` 可以列出所有蓝图。
    *   `scripts/query_all_actors.py` 可以列出当前场景中的对象。
*   **行动**：
    1.  运行 `scripts/list_actor_types.py`，获取所有**可生成**的 Actor (Vehicle, Walker, Prop)。
    2.  运行 `scripts/query_all_actors.py` (需先运行 `main_data_collection.py` 生成一些 NPC)，获取**实际场景**中的对象类型。
    3.  整理这两个列表，得到一份“待映射清单”。

## 第二步：审查与对齐映射表 (Align Mapping)

**目标**：检查并修正 `config/actor_occupancy_mapping.py`，确保所有“待映射清单”中的 Actor 都有归宿。

*   **参照标准**：
    *   `nuScenes` 17 类标准 (0-17)。
    *   `体素分类建议.md` 中的定义。
*   **检查重点**：
    *   **Car vs Truck/Bus**：确保大型车辆 (如 `carlacola`, `firetruck`) 映射为 Truck (10) 或 Bus (3)，而不是默认的 Car (4)。
    *   **Cyclist vs Bicycle**：确保 `vehicle.bh.crossbike` 等映射为 Bicycle (2)。
    *   **Motorcycle**：确保 `yamaha`, `harley` 等映射为 Motorcycle (6)。
    *   **Props**：确保 `trafficcone` 映射为 8，`barrier` 映射为 1。
    *   **Static Objects**：检查 `CityObjectLabel` 映射是否正确 (如 Pole -> Manmade)。

## 第三步：验证映射结果 (Validate)

**目标**：通过脚本验证映射逻辑是否覆盖了所有类型。

*   **行动**：
    1.  编写/运行 `scripts/test_actor_mapping.py`。
    2.  该脚本将遍历蓝图库中的每一个 ID，调用 `get_occupancy_label_from_type_id`，检查返回值。
    3.  如果发现映射为 `general_object (17)` 且本应属于具体类别的（如漏掉的卡车），则视为 **Failure**。
    4.  **修正**：根据测试结果，反向修改 `config/actor_occupancy_mapping.py` 直到所有关键类型都正确归类。

## 第四步：最终确认

*   **行动**：
    1.  再次运行 `main_data_collection.py` (God Mode)。
    2.  在 Viewer 中肉眼检查颜色是否合理（卡车是蓝色，小车是绿色，自行车是深黄）。

***

**当前立即执行：第一步 (盘点)**
我将运行现有脚本来获取全量列表。
