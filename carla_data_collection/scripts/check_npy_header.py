import numpy as np
import io

# 创建一个 dummy 数据并保存为 npz，模拟生成过程
data = np.zeros((200, 200, 16), dtype=np.uint8)
buffer = io.BytesIO()
np.savez_compressed(buffer, occupancy=data)
buffer.seek(0)

# 解压并查看 header
import zipfile
with zipfile.ZipFile(buffer) as z:
    with z.open('occupancy.npy') as f:
        # 读取前 256 字节
        header_bytes = f.read(256)
        print(f"Header bytes: {header_bytes}")
        print(f"Header str: {header_bytes.decode('ascii', errors='ignore')}")
