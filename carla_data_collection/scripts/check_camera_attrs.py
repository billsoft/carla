
import carla

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        
        bp = bp_lib.find('sensor.camera.rgb')
        print(f"--- Attributes for {bp.id} ---")
        for attr in bp:
            print(f"{attr.id}: {attr.type} (Default: {attr.recommended_values})")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
