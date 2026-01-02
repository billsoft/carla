import carla
import argparse
import sys
import os
import json
from collections import defaultdict

# 添加项目根目录到 sys.path 以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # script -> dense_occupancy -> carla
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入映射模块
try:
    from dense_occupancy_collection.config.actor_occupancy_mapping import (
        get_occupancy_label_from_actor,
        get_occupancy_name,
        OCCUPANCY_LABELS
    )
    from dense_occupancy_collection.config.occupancy_config import CARLA_TO_OCCUPANCY_MAPPING
except ImportError as e:
    print(f"Error importing mapping module: {e}")
    print("Please run this script from the project root or ensure PYTHONPATH is set correctly.")
    sys.exit(1)

def get_label_name(label_id):
    if label_id is None:
        return "None"
    return OCCUPANCY_LABELS.get(label_id, f"Unknown({label_id})")

def audit_actors(world):
    print("\n=== Auditing Actors (Dynamic & Props) ===")
    actors = world.get_actors()
    
    # 统计: type_id -> {count, semantic_tags, mapped_label}
    stats = defaultdict(lambda: {"count": 0, "semantic_tags": set(), "mapped_label": None})
    
    for actor in actors:
        type_id = actor.type_id
        # 跳过传感器和控制器
        if type_id.startswith('sensor.') or type_id.startswith('controller.'):
            continue
            
        stats[type_id]["count"] += 1
        if hasattr(actor, 'semantic_tags'):
            stats[type_id]["semantic_tags"].update(actor.semantic_tags)
        
        # 获取当前映射
        label_id = get_occupancy_label_from_actor(actor)
        stats[type_id]["mapped_label"] = label_id

    # 打印结果
    print(f"{'Type ID':<50} | {'Count':<6} | {'SemTags':<10} | {'Mapped Label'}")
    print("-" * 100)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    results = []
    
    for type_id, data in sorted_stats:
        sem_tags = list(data["semantic_tags"])
        sem_str = str(sem_tags) if sem_tags else "[]"
        label_id = data["mapped_label"]
        label_name = get_label_name(label_id)
        
        print(f"{type_id:<50} | {data['count']:<6} | {sem_str:<10} | {label_id} ({label_name})")
        
        results.append({
            "type": "Actor",
            "type_id": type_id,
            "count": data["count"],
            "semantic_tags": sem_tags,
            "mapped_label_id": label_id,
            "mapped_label_name": label_name
        })
        
    return results

def audit_environment_objects(world):
    print("\n=== Auditing Environment Objects (Static) ===")
    # 获取所有环境对象
    env_objs = world.get_environment_objects(carla.CityObjectLabel.Any)
    
    # 统计: type_id (name) -> {count, type (CityObjectLabel), mapped_label}
    # EnvironmentObject 没有 type_id，只有 name, id, type (CityObjectLabel)
    # 我们按 CityObjectLabel 分组统计
    
    stats = defaultdict(lambda: {"count": 0, "examples": set(), "mapped_label": None})
    
    for obj in env_objs:
        label_type = obj.type
        # 获取映射
        # 注意：这里需要模拟 actor_occupancy_mapping 中对 CityObjectLabel 的处理逻辑
        # 但 actor_occupancy_mapping 主要针对 Actor。
        # 对于 Environment Object，通常使用 semantic tag (即 CityObjectLabel 的值)
        
        # 在 occupancy_config.py 中有 CARLA_TO_OCCUPANCY_MAPPING
        mapped_label = CARLA_TO_OCCUPANCY_MAPPING.get(int(label_type), 17) # 默认 17
        
        stats[label_type]["count"] += 1
        if len(stats[label_type]["examples"]) < 3:
            stats[label_type]["examples"].add(obj.name)
        stats[label_type]["mapped_label"] = mapped_label

    print(f"{'CityObjectLabel':<30} | {'Count':<8} | {'Mapped Label'}")
    print("-" * 80)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    results = []
    
    for label_type, data in sorted_stats:
        label_str = str(label_type).replace("CityObjectLabel.", "")
        mapped_id = data["mapped_label"]
        mapped_name = get_label_name(mapped_id)
        
        print(f"{label_str:<30} | {data['count']:<8} | {mapped_id} ({mapped_name})")
        
        results.append({
            "type": "EnvironmentObject",
            "city_object_label": str(label_type),
            "count": data["count"],
            "mapped_label_id": mapped_id,
            "mapped_label_name": mapped_name,
            "examples": list(data["examples"])
        })
        
    return results

def main():
    argparser = argparse.ArgumentParser(description='Audit CARLA Actors and Mapping')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--map', metavar='M', default=None, help='Map to load (optional)')
    argparser.add_argument('--output', metavar='O', default='actor_audit_report.json', help='Output JSON file')
    
    args = argparser.parse_args()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        
        if args.map:
            print(f"Loading map: {args.map}")
            world = client.load_world(args.map)
        else:
            world = client.get_world()
            print(f"Using current map: {world.get_map().name}")

        actor_results = audit_actors(world)
        env_results = audit_environment_objects(world)
        
        full_report = {
            "map": world.get_map().name,
            "actors": actor_results,
            "environment_objects": env_results
        }
        
        with open(args.output, 'w') as f:
            json.dump(full_report, f, indent=4)
        
        print(f"\nAudit complete. Report saved to {args.output}")
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == '__main__':
    main()
