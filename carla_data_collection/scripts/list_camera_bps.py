
import carla

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    
    print("Camera Blueprints:")
    for bp in bp_lib.filter('sensor.camera.*'):
        print(f"- {bp.id}")
        print("  Attributes:")
        for attr in bp:
            print(f"    - {attr.id} ({attr.type}): {attr.recommended_values if attr.recommended_values else 'No recommended values'}")
            
except Exception as e:
    print(f"Error: {e}")
