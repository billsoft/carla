
import carla

try:
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    
    try:
        bp = bp_lib.find('sensor.camera.fisheye')
        print("Found sensor.camera.fisheye!")
        for attr in bp:
            print(f"  - {attr.id}")
    except:
        print("sensor.camera.fisheye NOT found.")
        
    try:
        bp = bp_lib.find('sensor.camera.wide_angle')
        print("Found sensor.camera.wide_angle!")
    except:
        print("sensor.camera.wide_angle NOT found.")

except Exception as e:
    print(f"Error: {e}")
