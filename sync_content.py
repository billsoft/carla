"""
CARLA 资源完整同步脚本
从 Unreal/CarlaUnreal 同步所有资源到 server/Windows/CarlaUnreal
"""
import shutil
from pathlib import Path

def sync_content():
    print("开始同步 CARLA 资源...")

    project_root = Path(__file__).parent
    source = project_root / 'Unreal' / 'CarlaUnreal' / 'Content' / 'Carla'
    target = project_root / 'server' / 'Windows' / 'CarlaUnreal' / 'Content' / 'Carla'

    if not source.exists():
        print(f"❌ 源目录不存在: {source}")
        return False

    if not target.parent.exists():
        print(f"❌ 目标目录不存在: {target.parent}")
        return False

    print(f"源目录: {source}")
    print(f"目标目录: {target}")
    print(f"源目录大小: {sum(f.stat().st_size for f in source.rglob('*') if f.is_file()) / (1024**3):.2f} GB")

    # 备份旧目录
    if target.exists():
        backup = target.parent / 'Carla_backup'
        if backup.exists():
            shutil.rmtree(backup)
        print(f"备份旧资源到: {backup}")
        shutil.move(str(target), str(backup))

    # 完整复制
    print("正在复制资源...")
    shutil.copytree(source, target, dirs_exist_ok=True)

    print(f"✅ 同步完成!")
    print(f"目标目录大小: {sum(f.stat().st_size for f in target.rglob('*') if f.is_file()) / (1024**3):.2f} GB")

    # 验证关键目录
    critical_dirs = [
        'Blueprints/Weather',
        'Blueprints/Vehicles',
        'Blueprints/Walkers',
        'Blueprints/Game',
        'Maps'
    ]

    print("\n验证关键目录:")
    for dir_path in critical_dirs:
        check_dir = target / dir_path
        if check_dir.exists():
            file_count = sum(1 for _ in check_dir.rglob('*') if _.is_file())
            print(f"  ✅ {dir_path}: {file_count} 个文件")
        else:
            print(f"  ❌ {dir_path}: 不存在")

    return True

if __name__ == '__main__':
    sync_content()
