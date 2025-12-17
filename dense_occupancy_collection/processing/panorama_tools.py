
import numpy as np
import cv2

class PanoramaTools:
    """全景图处理工具：拼接与反投影"""

    def __init__(self, pano_w, pano_h, cube_size):
        self.pano_w = pano_w
        self.pano_h = pano_h
        self.cube_size = cube_size
        self.map_x, self.map_y = self._init_remap_tables()

    def _init_remap_tables(self):
        """
        预计算从 Equirectangular 到 Cubemap 的映射表
        用于 cv2.remap 高效拼接

        使用与 unproject_panorama 一致的坐标系统
        """
        # 1. 生成全景图的像素坐标 (u, v)
        uv = np.indices((self.pano_h, self.pano_w), dtype=np.float32)
        u, v = uv[1], uv[0]

        # 2. 像素 → 球面角度
        # θ: 水平角度 [0, 2π] (Longitude)
        # φ: 垂直角度 [π/2, -π/2] (Latitude)
        theta = (u / self.pano_w) * 2 * np.pi
        phi = (0.5 - v / self.pano_h) * np.pi

        # 3. 球面坐标 (θ, φ) -> 3D 方向向量
        # 车辆坐标系: X前, Y左, Z上 (CARLA惯例)
        # Front(0,0,0) -> X+
        # Right(0,90,0) -> Y+ (但实际CARLA中Y是左，这里先按标准CubeMap处理)
        # Back(0,180,0) -> X-
        # Left(0,-90,0) -> Y-
        # Up(90,0,0) -> Z+
        # Down(-90,0,0) -> Z-

        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)

        # 4. 3D 向量 -> Cube Face (face_idx, u_face, v_face)
        # 这是一个复杂的向量化操作
        # 为了性能，我们使用 numpy 掩码
        
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        
        is_x_pos = (x > 0)
        is_y_pos = (y > 0)
        is_z_pos = (z > 0)

        # 判断每个像素属于哪个面
        # Major Axis Selection
        max_axis = np.maximum(np.maximum(abs_x, abs_y), abs_z)
        
        # 面索引定义 (与 panorama_config.py 一致)
        # 0:Front, 1:Right, 2:Back, 3:Left, 4:Up, 5:Down
        
        face_id = np.zeros_like(x, dtype=np.int8)
        u_cube = np.zeros_like(x)
        v_cube = np.zeros_like(x)
        
        # Front (X+)
        mask = (abs_x == max_axis) & is_x_pos
        face_id[mask] = 0
        u_cube[mask] = y[mask] / x[mask]
        v_cube[mask] = -z[mask] / x[mask]
        
        # Back (X-)
        mask = (abs_x == max_axis) & ~is_x_pos
        face_id[mask] = 2
        u_cube[mask] = -y[mask] / -x[mask] # 注意镜像
        v_cube[mask] = -z[mask] / -x[mask]

        # Right (Y+)
        mask = (abs_y == max_axis) & is_y_pos
        face_id[mask] = 1
        u_cube[mask] = -x[mask] / y[mask]
        v_cube[mask] = -z[mask] / y[mask]
        
        # Left (Y-)
        mask = (abs_y == max_axis) & ~is_y_pos
        face_id[mask] = 3
        u_cube[mask] = x[mask] / -y[mask]
        v_cube[mask] = -z[mask] / -y[mask]

        # Up (Z+)
        mask = (abs_z == max_axis) & is_z_pos
        face_id[mask] = 4
        u_cube[mask] = y[mask] / z[mask]
        v_cube[mask] = x[mask] / z[mask] # Rotated
        
        # Down (Z-)
        mask = (abs_z == max_axis) & ~is_z_pos
        face_id[mask] = 5
        u_cube[mask] = y[mask] / -z[mask]
        v_cube[mask] = -x[mask] / -z[mask]

        # 5. 归一化坐标 [-1, 1] -> 像素坐标 [0, CUBE_SIZE]
        # u = (u_norm + 1) / 2 * size
        map_x = (u_cube + 1.0) * 0.5 * (self.cube_size - 1)
        map_y = (v_cube + 1.0) * 0.5 * (self.cube_size - 1)
        
        # 6. 将 Face ID 编码进 Map X (为了把6张图拼成一张长图处理)
        # 我们假设输入的 6 张图是水平拼接的: [Front, Right, Back, Left, Up, Down]
        # 宽度 = CUBE_SIZE * 6
        map_x += face_id.astype(np.float32) * self.cube_size
        
        return map_x.astype(np.float32), map_y.astype(np.float32)

    def stitch(self, cube_faces):
        """
        将6个面的图像拼接成全景图
        
        Args:
            cube_faces: list of 6 images (Front, Right, Back, Left, Up, Down)
                        Images can be (H, W, C) or (H, W)
        
        Returns:
            panorama: (PANO_H, PANO_W, C) or (PANO_H, PANO_W)
        """
        # 水平拼接所有面
        # 确保输入是 numpy array
        if isinstance(cube_faces, list):
            # 假设所有面尺寸一致
            atlas = np.concatenate(cube_faces, axis=1)
        else:
            atlas = cube_faces # 已经是拼接好的

        # 使用 remap 进行映射
        # INTER_NEAREST 对于语义分割图是必须的，防止插值产生不存在的类别
        # INTER_LINEAR 对于深度图和RGB图更好
        if atlas.ndim == 2 or (atlas.ndim == 3 and atlas.shape[2] == 1):
            # 单通道（深度/语义）
            interp = cv2.INTER_NEAREST
        else:
            # RGB
            interp = cv2.INTER_LINEAR
            
        pano = cv2.remap(atlas, self.map_x, self.map_y, interpolation=interp, borderMode=cv2.BORDER_CONSTANT)
        return pano

    def unproject_panorama(self, depth_pano, semantic_pano, max_depth=100.0):
        """
        全景深度图反投影为点云

        使用标准 Equirectangular 投影反投影算法
        参考文档第5.2节：基于360°全景深度图的稠密3D体素生成指南

        Args:
            depth_pano: (H, W) 深度图 (米)
            semantic_pano: (H, W) 语义图
            max_depth: 最大有效深度 (米)

        Returns:
            points: (N, 3) 点云，车辆坐标系 [X前, Y左, Z上]
            labels: (N,) 标签
        """
        H, W = depth_pano.shape

        # ========================================
        # 步骤1: 生成像素网格
        # ========================================
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)

        # ========================================
        # 步骤2: 像素 → 球面角度
        # ========================================
        # θ: 水平角度 [0, 2π] (经度)
        # φ: 垂直角度 [π/2, -π/2] (纬度，从上到下)
        theta = (u / W) * 2 * np.pi
        phi = (0.5 - v / H) * np.pi

        # ========================================
        # 步骤3: 球面角度 → 3D方向向量
        # ========================================
        # 车辆坐标系: X前, Y左, Z上 (CARLA惯例)
        # 全景图: θ=0 对应前方 (+X)
        dir_x = np.cos(phi) * np.cos(theta)   # 前
        dir_y = np.cos(phi) * np.sin(theta)   # 左
        dir_z = np.sin(phi)                   # 上

        # ========================================
        # 步骤4: 方向 × 深度 = 3D点
        # ========================================
        depth = depth_pano

        x = dir_x * depth
        y = dir_y * depth
        z = dir_z * depth

        # ========================================
        # 步骤5: 过滤无效点
        # ========================================
        valid = (depth > 0.1) & (depth < max_depth)

        points = np.stack([x[valid], y[valid], z[valid]], axis=-1)  # (N, 3)
        labels = semantic_pano[valid]                                # (N,)

        return points, labels
