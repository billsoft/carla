import carla
import time

def main():
    print("========================================")
    print("   CARLA CONNECTION TEST")
    print("========================================")
    
    try:
        print("[*] Attempting to connect to localhost:2000...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        
        print("[*] Retrieving world...")
        world = client.get_world()
        
        print("[*] Connection Successful!")
        print(f"    - Server Version: {client.get_client_version()}")
        print(f"    - Map Name: {world.get_map().name}")
        
        print("\n✓ 所有测试通过！CARLA 服务器运行正常。")
        
    except RuntimeError as e:
        print("\n[!] Connection Failed!")
        print(f"    Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Unreal Editor is running.")
        print("2. Make sure you pressed the 'Play' button in the editor.")
        print("3. Check if port 2000 is blocked by firewall.")
        
    except Exception as e:
        print(f"\n[!] Unexpected Error: {e}")

if __name__ == '__main__':
    main()
