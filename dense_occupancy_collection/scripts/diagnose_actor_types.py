"""
诊断CARLA场景中所有Actor的类型和occupancy映射
"""
import carla
import sys

def get_occupancy_label(actor):
    """根据actor类型返回occupancy标签"""
    type_id = actor.type_id.lower()

    # 车辆类别映射
    if 'vehicle' in type_id:
        if 'bike' in type_id or 'bicycle' in type_id:
            return 2, 'bicycle'
        elif 'motor' in type_id or 'vespa' in type_id or 'harley' in type_id or 'kawasaki' in type_id or 'yamaha' in type_id:
            return 6, 'motorcycle'
        elif 'bus' in type_id or 'fusorosa' in type_id:
            return 3, 'bus'
        elif 'truck' in type_id or 'firetruck' in type_id or 'ambulance' in type_id or 'carlacola' in type_id or 'hgv' in type_id:
            return 10, 'truck'
        elif 'trailer' in type_id:
            return 9, 'trailer'
        else:
            return 4, 'car'

    # 行人
    elif 'walker.pedestrian' in type_id:
        return 7, 'pedestrian'

    # 其他
    else:
        return 17, 'general_object'

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    print("\n" + "="*80)
    print("CARLA场景中所有Actor的类型诊断")
    print("="*80 + "\n")

    actors = world.get_actors()

    # 按类型分组
    vehicles = actors.filter('vehicle.*')
    walkers = actors.filter('walker.pedestrian.*')
    props = actors.filter('static.*')

    print(f"总Actor数: {len(actors)}")
    print(f"  车辆: {len(vehicles)}")
    print(f"  行人: {len(walkers)}")
    print(f"  静态物体: {len(props)}")

    # 统计每个occupancy类别
    occupancy_counts = {}

    print("\n" + "="*80)
    print("车辆列表 (Vehicle)")
    print("="*80)
    print(f"{'ID':<6} {'Type ID':<50} {'Occupancy':<15} {'BBox'}")
    print("-"*80)

    for v in vehicles:
        occ_id, occ_name = get_occupancy_label(v)
        bbox = v.bounding_box.extent
        print(f"{v.id:<6} {v.type_id:<50} [{occ_id}] {occ_name:<12} ({bbox.x:.1f}, {bbox.y:.1f}, {bbox.z:.1f})")

        if occ_name not in occupancy_counts:
            occupancy_counts[occ_name] = 0
        occupancy_counts[occ_name] += 1

    print("\n" + "="*80)
    print("行人列表 (Walker)")
    print("="*80)
    print(f"{'ID':<6} {'Type ID':<50} {'Occupancy':<15} {'BBox'}")
    print("-"*80)

    for w in walkers:
        occ_id, occ_name = get_occupancy_label(w)
        bbox = w.bounding_box.extent
        print(f"{w.id:<6} {w.type_id:<50} [{occ_id}] {occ_name:<12} ({bbox.x:.1f}, {bbox.y:.1f}, {bbox.z:.1f})")

        if occ_name not in occupancy_counts:
            occupancy_counts[occ_name] = 0
        occupancy_counts[occ_name] += 1

    print("\n" + "="*80)
    print("Occupancy类别统计")
    print("="*80)
    for occ_name, count in sorted(occupancy_counts.items(), key=lambda x: -x[1]):
        print(f"  {occ_name:<20}: {count} 个")

    print()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
