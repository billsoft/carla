
import carla
import time

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Clean vehicles
    vehicles = world.get_actors().filter('vehicle.*')
    print(f"Found {len(vehicles)} vehicles. Destroying...")
    for v in vehicles:
        try:
            v.destroy()
        except: pass

    # Clean walkers
    walkers = world.get_actors().filter('walker.*')
    print(f"Found {len(walkers)} walkers. Destroying...")
    for w in walkers:
        try:
            w.destroy()
        except: pass
        
    # Clean sensors
    sensors = world.get_actors().filter('sensor.*')
    print(f"Found {len(sensors)} sensors. Destroying...")
    for s in sensors:
        try:
            s.destroy()
        except: pass

    print("Cleanup done.")

if __name__ == '__main__':
    main()
