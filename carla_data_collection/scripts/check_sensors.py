
import carla

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        
        print("Available Camera Sensors:")
        for bp in bp_lib.filter('sensor.camera.*'):
            print(f"- {bp.id}")
            if bp.id == 'sensor.camera.rgb':
                print("  Attributes:")
                for attr in bp:
                    print(f"    - {attr.id}: {attr.type}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
