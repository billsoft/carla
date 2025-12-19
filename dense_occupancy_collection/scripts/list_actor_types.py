"""
列出CARLA中所有可用的Actor类型
"""
import carla

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    print("\n" + "="*60)
    print("车辆类型 (Vehicles)")
    print("="*60)
    vehicles = bp_lib.filter('vehicle.*')
    vehicle_types = {}

    for v in vehicles:
        # 提取类型：vehicle.category.model
        parts = v.id.split('.')
        if len(parts) >= 3:
            category = parts[1]  # audi, tesla, bmw, truck, etc.
            if category not in vehicle_types:
                vehicle_types[category] = []
            vehicle_types[category].append(v.id)

    for category in sorted(vehicle_types.keys()):
        print(f"\n{category.upper()}:")
        for vid in sorted(vehicle_types[category]):
            print(f"  {vid}")

    print("\n" + "="*60)
    print("行人类型 (Pedestrians)")
    print("="*60)
    walkers = bp_lib.filter('walker.pedestrian.*')
    for w in sorted(walkers, key=lambda x: x.id):
        print(f"  {w.id}")

    print("\n" + "="*60)
    print("其他Actor类型")
    print("="*60)

    bicycles = list(bp_lib.filter('vehicle.bh.crossbike')) + list(bp_lib.filter('vehicle.diamondback.*')) + list(bp_lib.filter('vehicle.gazelle.*'))
    motorcycles = list(bp_lib.filter('vehicle.harley*')) + list(bp_lib.filter('vehicle.kawasaki.*')) + list(bp_lib.filter('vehicle.yamaha.*')) + list(bp_lib.filter('vehicle.vespa.*'))
    
    print("  (Bicycles)")
    for bp in bicycles:
        print(f"  {bp.id}")
        
    print("\n  (Motorcycles)")
    for bp in motorcycles:
        print(f"  {bp.id}")

    # 3. 静态物体/Props (部分)
    print("\n" + "="*60)
    print("Props (部分示例)")
    print("="*60)
    props = bp_lib.filter('static.prop.*')
    # 只列出前20个作为示例
    for bp in list(props)[:20]:
        print(f"  {bp.id}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
