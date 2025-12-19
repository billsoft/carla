
import carla
import argparse
import time
import numpy as np

def main():
    argparser = argparse.ArgumentParser(description="Debug Environment Objects")
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int, help='TCP port')
    args = argparser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # 1. 获取 EnvironmentObjects (Buildings)
    env_objs = world.get_environment_objects(carla.CityObjectLabel.Buildings)
    print(f"Found {len(env_objs)} EnvironmentObjects (Buildings)")
    
    # 2. 获取 Level BBs (Buildings)
    level_bbs = world.get_level_bbs(carla.CityObjectLabel.Buildings)
    print(f"Found {len(level_bbs)} Level BBs (Buildings)")
    
    if not env_objs:
        print("No buildings found!")
        return

    # Check first 5 objects
    print("\n--- Inspecting first 5 EnvironmentObjects ---")
    for i in range(min(5, len(env_objs))):
        obj = env_objs[i]
        print(f"\nObject ID: {obj.id}, Name: {obj.name}")
        print(f"  Type: {obj.type}")
        print(f"  Transform: {obj.transform}")
        print(f"  BBox Location: {obj.bounding_box.location}")
        print(f"  BBox Extent: {obj.bounding_box.extent}")
        
        # Test vertices transformation logic
        # My previous logic was: bb.get_world_vertices(obj.transform)
        # Let's see what that produces
        try:
            verts = obj.bounding_box.get_world_vertices(obj.transform)
            z_values = [v.z for v in verts]
            print(f"  [Transformed Vertices] Min Z: {min(z_values):.2f}, Max Z: {max(z_values):.2f}")
            
            # Check if this matches BBox location
            # If BBox is local, applying transform moves it to world.
            # If BBox is ALREADY world, applying transform moves it TWICE!
            
            # Hypothesis: In CARLA 0.9.10+, EnvironmentObject.bounding_box IS ALREADY IN WORLD COORDINATES?
            # OR is it local to World Origin (which is effectively world coords)?
            # Let's check distance between transform and bbox
            dist = obj.transform.location.distance(obj.bounding_box.location)
            print(f"  Distance (Transform <-> BBox): {dist:.2f}")
            
        except Exception as e:
            print(f"  Error calculating vertices: {e}")

    print("\n--- Inspecting first 5 Level BBs ---")
    for i in range(min(5, len(level_bbs))):
        bb = level_bbs[i]
        print(f"\nLevel BB #{i}")
        print(f"  Location: {bb.location}")
        print(f"  Extent: {bb.extent}")
        # Level BBs are usually world aligned AABBs
        print(f"  Min Z: {bb.location.z - bb.extent.z:.2f}")
        print(f"  Max Z: {bb.location.z + bb.extent.z:.2f}")

if __name__ == '__main__':
    main()
