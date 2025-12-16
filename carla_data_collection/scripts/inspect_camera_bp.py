
import carla

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    
    bp = bp_lib.find('sensor.camera.rgb')
    print(f"Blueprint: {bp.id}")
    for attr in bp:
        print(f"  - {attr.id} ({attr.type})")
        
except Exception as e:
    print(f"Error: {e}")
