# Dense Occupancy Collection - Actor映射配置说明

## 完成的工作

### 1. 创建了完整的Actor映射配置文件
**文件**: `dense_occupancy_collection/config/actor_occupancy_mapping.py`

这是一个**可维护的1对多关系配置文件**，包含：

#### 17类Occupancy标签定义
基于业界标准（nuScenes）：
- 0: free - 自由空间
- 1: barrier - 隔离栏/护栏
- 2: bicycle - 自行车
- 3: bus - 公交车
- 4: car - 小汽车
- 5: construction_vehicle - 工程车
- 6: motorcycle - 摩托车
- 7: pedestrian - 行人
- 8: traffic_cone - 交通标识
- 9: trailer - 拖车
- 10: truck - 卡车
- 11: driveable_surface - 可行驶路面
- 12: other_flat - 其他平坦表面
- 13: sidewalk - 人行道
- 14: terrain - 地形
- 15: manmade - 人造物体
- 16: vegetation - 植被
- 17: general_object - 通用障碍物/其他

#### 1对多映射表（可维护）
```python
# 示例：Truck类型 (10) 对应多个CARLA vehicle type_id
VEHICLE_MAPPING = {
    10: [
        'vehicle.carlamotors.carlacola',
        'vehicle.carlamotors.firetruck',
        'vehicle.ford.ambulance',
        'vehicle.mercedes.sprinter',
        'vehicle.tesla.cybertruck',
    ],
}
```

### 2. 映射配置的三个部分

#### A. 车辆映射 (VEHICLE_MAPPING)
- Bus (3): VW T2面包车, Mitsubishi Fuso
- Truck (10): 货车、消防车、救护车、Cybertruck, Sprinter, Carlacola
- Bicycle (2): Crossbike, Diamondback, Gazelle
- Motorcycle (6): Harley, Kawasaki, Yamaha, Vespa
- Car (4): 其他所有轿车（默认）

#### B. 行人映射 (WALKER_MAPPING)
- Pedestrian (7): 所有 walker.pedestrian.* 类型

#### C. Props映射 (PROP_MAPPING)
- Traffic Cone (8): 交通锥桶
- Barrier (1): 隔离栏、护栏、施工警告
- Manmade (15): 长椅、喷泉、路牌、ATM、广告牌、自动售货机、电话亭
- Vegetation (16): 灌木、植物、花盆
- General Object (17): 垃圾桶、箱子、碎片、可乐罐、头盔

#### D. CityObjectLabel映射 (CITY_OBJECT_MAPPING)
- 静态环境对象：建筑、道路、植被等

### 3. 统一的映射函数

```python
from dense_occupancy_collection.config.actor_occupancy_mapping import (
    get_occupancy_label_from_actor
)

# 使用
occ_label = get_occupancy_label_from_actor(actor)
```

**映射优先级**：
1. type_id 精确匹配（最准确）
2. semantic_tags 兜底
3. 默认为 general_object (17)

### 4. 更新的文件

#### 核心文件
- ✅ `dense_occupancy_collection/config/actor_occupancy_mapping.py` - **新建**
  - 完整的17类映射配置
  - 1对多关系表（易于维护）
  - 统一的映射函数

- ✅ `dense_occupancy_collection/processing/ground_truth_voxel_generator.py` - **更新**
  - 使用新的映射配置
  - 导入 `get_occupancy_label_from_actor`
  - 第518行调用映射函数

- ✅ `dense_occupancy_collection/scripts/collect_panorama.py` - **已修复**
  - 修复了CARLA API导入（使用UE5.5 CARLA 0.10.0）

- ✅ `occupancy_viewer/viewer.js` - **已更新**
  - 使用优化的颜色配置

#### 保持不变
- ✅ `dense_occupancy_collection/config/occupancy_config.py` - 未修改
- ✅ 其他所有传感器和处理模块

## 如何维护映射配置

### 添加新的actor类型
编辑 `actor_occupancy_mapping.py`，在对应的映射字典中添加：

```python
# 示例：添加新的卡车类型
VEHICLE_MAPPING = {
    10: [
        'vehicle.carlamotors.carlacola',
        'vehicle.carlamotors.firetruck',
        'vehicle.ford.ambulance',
        'vehicle.mercedes.sprinter',
        'vehicle.tesla.cybertruck',
        'vehicle.new.truck',  # 新增
    ],
}
```

### 添加新的分类
如果需要添加新的Occupancy类别：
1. 更新 `OCCUPANCY_LABELS` 字典
2. 更新 `OCCUPANCY_COLORS` 字典
3. 在对应的映射表中添加条目
4. 更新 viewer.js 的颜色配置

## 查询工具

### 查询场景中的所有actor类型
```bash
conda activate carla
python dense_occupancy_collection/scripts/query_all_actors.py
```

输出：
- 控制台：所有不重复的actor类型和CityObjectLabel
- 文件：`dense_occupancy_collection/config/actor_types_query_result.json`

## 使用方法

### 运行数据采集
```bash
conda activate carla
python dense_occupancy_collection/scripts/collect_panorama.py --frames 5
```

### 查看结果
```bash
python occupancy_viewer/run_viewer.py
# 浏览器打开 http://localhost:8000
```

## 技术细节

### 映射流程
```
Actor对象
    ↓
1. 检查 type_id（精确匹配）
    ├─ VEHICLE_MAPPING
    ├─ WALKER_MAPPING
    └─ PROP_MAPPING
    ↓
2. type_id 模糊匹配
    ├─ vehicle.* → car (4)
    └─ static.prop* → general_object (17)
    ↓
3. semantic_tags 兜底
    ↓
4. 默认 general_object (17)
```

### 优势
1. **可维护性**: 1对多关系清晰，易于添加新类型
2. **准确性**: 优先使用type_id精确匹配
3. **鲁棒性**: 多层兜底机制，不会遗漏任何actor
4. **标准化**: 基于nuScenes业界标准

## 参考文档

### 业界标准
- [nuScenes Dataset](https://www.nuscenes.org/nuscenes#lidarseg) - 17类分类体系
- [Occ3D Benchmark](https://arxiv.org/abs/2304.14365) - 3D Occupancy预测基准
- [OpenOccupancy](https://opendrivelab.com/OpenOccupancy.html) - 大规模Occupancy数据集

### CARLA文档
- [CARLA CityObjectLabel](https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera) - 语义标签定义
- [CARLA Actor Blueprints](https://carla.readthedocs.io/en/latest/bp_library/) - 所有Actor类型

## 故障排除

### Q: 某个actor类型没有被正确识别？
A:
1. 运行 `query_all_actors.py` 查看该actor的type_id
2. 在 `actor_occupancy_mapping.py` 的对应映射表中添加该type_id
3. 重新运行数据采集

### Q: 如何验证映射是否正确？
A:
1. 查看 `voxel_mapping.log` 文件
2. 运行viewer查看体素颜色是否符合预期
3. 检查 `actor_types_query_result.json` 中的类型列表

### Q: 颜色配置在哪里？
A:
- Python后端: `actor_occupancy_mapping.py` 的 `OCCUPANCY_COLORS`
- Viewer前端: `occupancy_viewer/viewer.js` 的 `OCCUPANCY_COLORS`
- 两者需要保持一致

## 版本历史

### v1.0 - 2024年
- 创建完整的17类映射配置
- 基于nuScenes标准
- 支持CARLA UE5.5 / 0.10.0
- 1对多关系表，易于维护

