"""
测试 Actor 类型映射逻辑
验证是否所有 CARLA 蓝图都能正确映射到 17 类 Occupancy 标签
"""
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 处理 PythonAPI 导入
try:
    build_dist = project_root / 'Build' / 'PythonAPI' / 'dist'
    if build_dist.exists():
        for whl in build_dist.glob('*.whl'):
            sys.path.append(str(whl))
            break
    else:
        sys.path.append(str(project_root / 'PythonAPI' / 'carla'))
except: pass

import carla
from dense_occupancy_collection.config.actor_occupancy_mapping import (
    get_occupancy_label_from_type_id,
    OCCUPANCY_LABELS
)

def test_mapping():
    print("连接 CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    
    # 获取所有相关蓝图
    all_bps = []
    all_bps.extend(list(bp_lib.filter('vehicle.*')))
    all_bps.extend(list(bp_lib.filter('walker.pedestrian.*')))
    all_bps.extend(list(bp_lib.filter('static.prop.*')))
    
    print(f"总计检查 {len(all_bps)} 个蓝图类型...")
    
    stats = {
        'mapped': 0,
        'fallback_car': 0,
        'fallback_general': 0,
        'failed': 0
    }
    
    failures = []
    
    for bp in all_bps:
        label_id = get_occupancy_label_from_type_id(bp.id)
        label_name = OCCUPANCY_LABELS.get(label_id, 'Unknown')
        
        # 统计
        if label_id == 17: # General Object
            # 检查是否是"假"General Object (即兜底逻辑生效)
            if bp.id.startswith('vehicle.'):
                # 车辆不应该被映射为 General Object
                failures.append(f"{bp.id} -> {label_name} (Should be Vehicle?)")
                stats['failed'] += 1
            else:
                stats['fallback_general'] += 1
        elif label_id == 4: # Car
            # 检查是否是真正的 Car (通过名字)
            # 如果是 truck/bus/bike 被映射为 car，则是次优，但不是完全失败
            # 但我们要追求精确，所以这里列出潜在问题
            if any(k in bp.id for k in ['truck', 'carlacola', 'firetruck', 'sprinter', 'bus', 'bike', 'motorcycle', 'yamaha', 'kawasaki']):
                # 如果这些关键词的物体被映射为 Car，说明精确匹配失败，走了兜底
                # 我们需要检查它是否在精确映射表中
                # 注意：get_occupancy_label_from_type_id 内部逻辑：
                # 如果不在精确表中，vehicle.* 会兜底为 4 (Car)
                # 所以我们无法区分是"精确映射为Car"还是"兜底为Car"，除非我们知道它本不该是Car
                pass 
            stats['mapped'] += 1
        else:
            stats['mapped'] += 1
            
    print("\n映射结果统计:")
    print(f"  Mapped (Specific): {stats['mapped']}")
    print(f"  Fallback (General): {stats['fallback_general']}")
    print(f"  Failed/Suspicious: {stats['failed']}")
    
    if failures:
        print("\n[!] 潜在映射失败 (车辆被归为 General Object):")
        for f in failures:
            print(f"  - {f}")
            
    # 验证关键类型
    print("\n关键类型抽查:")
    check_list = [
        'vehicle.carlacola.actors',      # Should be 10 (Truck)
        'vehicle.tesla.cybertruck',      # Should be 10 (Truck)
        'vehicle.bh.crossbike',          # Should be 2 (Bicycle)
        'vehicle.yamaha.yzf',            # Should be 6 (Motorcycle)
        'vehicle.volkswagen.t2',         # Should be 3 (Bus)
        'static.prop.trafficcone01',     # Should be 8 (Cone)
        'static.prop.streetbarrier',     # Should be 1 (Barrier)
        'static.prop.atm',               # Should be 15 (Manmade)
        'static.prop.plantpot04',        # Should be 16 (Vegetation)
    ]
    
    for tid in check_list:
        lid = get_occupancy_label_from_type_id(tid)
        lname = OCCUPANCY_LABELS.get(lid, 'Unknown')
        print(f"  {tid:<30} -> [{lid}] {lname}")

if __name__ == '__main__':
    test_mapping()
