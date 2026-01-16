
import carla
import sys
import os

# Ensure we can find the config
sys.path.append(os.path.join(os.getcwd(), 'occnetv3_data_generator'))

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
        return

    # 1. Get all vehicle blueprints
    vehicles = [bp.id for bp in bp_lib.filter('vehicle.*')]
    walkers = [bp.id for bp in bp_lib.filter('walker.*')]
    
    print(f"Found {len(vehicles)} vehicle blueprints and {len(walkers)} walker blueprints.")
    
    # 2. Categorize based on keywords (Heuristic)
    categories = {
        'Truck (10)': [],
        'Bus (3)': [],
        'Motorcycle (6)': [],
        'Bicycle (2)': [],
        'Car (4)': [], # Default
        'Unknown/Check': []
    }
    
    # Heuristic Rules
    truck_keywords = ['truck', 'carlacola', 'ambulance', 'firetruck', 'sprinter', 'van', 'pickup']
    bus_keywords = ['bus', 'fusorosa']
    moto_keywords = ['harley', 'kawasaki', 'yamaha', 'vespa', 'motorcycle']
    bike_keywords = ['bike', 'gazelle', 'diamondback', 'bicycle', 'crossbike']
    
    for vid in sorted(vehicles):
        vid_lower = vid.lower()
        
        if any(k in vid_lower for k in bus_keywords):
            categories['Bus (3)'].append(vid)
        elif any(k in vid_lower for k in truck_keywords):
             categories['Truck (10)'].append(vid)
        elif any(k in vid_lower for k in moto_keywords):
             categories['Motorcycle (6)'].append(vid)
        elif any(k in vid_lower for k in bike_keywords):
             categories['Bicycle (2)'].append(vid)
        else:
             categories['Car (4)'].append(vid)

    # 3. Output formatted for python config
    print("\n" + "="*50)
    print("SUGGESTED MAPPING UPDATE (Copy relevant parts to actor_occupancy_mapping.py)")
    print("="*50 + "\n")
    
    print("# Bus (3)")
    print("3: [")
    for v in categories['Bus (3)']: print(f"    '{v}',")
    print("],")
    
    print("\n# Truck (10)")
    print("10: [")
    for v in categories['Truck (10)']: print(f"    '{v}',")
    print("],")
    
    print("\n# Motorcycle (6)")
    print("6: [")
    for v in categories['Motorcycle (6)']: print(f"    '{v}',")
    print("],")
    
    print("\n# Bicycle (2)")
    print("2: [")
    for v in categories['Bicycle (2)']: print(f"    '{v}',")
    print("],")
    
    print("\n# Car (4) - (These are defaulted, usually don't need explicit mapping unless overriding)")
    print("# " + str(categories['Car (4)']))

if __name__ == '__main__':
    main()
