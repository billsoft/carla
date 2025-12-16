// CARLA Occupancy 3D 体素查看器
// 使用 Three.js 渲染体素网格

// Occupancy 类别颜色映射 (CityScapes 配色方案)
const OCCUPANCY_COLORS = [
    0x000000,  // 0:  empty (黑色 - 不渲染)
    0xFF0000,  // 1:  car (红色)
    0xFF6600,  // 2:  truck (橙红)
    0xFFAA00,  // 3:  trailer (橙黄)
    0xFFFF00,  // 4:  bus (黄色)
    0xAAFF00,  // 5:  construction_vehicle (黄绿)
    0x00FF00,  // 6:  pedestrian (绿色)
    0x00FFAA,  // 7:  motorcycle (青绿)
    0x00FFFF,  // 8:  bicycle (青色)
    0x0088FF,  // 9:  road (蓝色)
    0x8888FF,  // 10: sidewalk (浅蓝)
    0xAA00FF,  // 11: traffic_cone (紫色)
    0xFF00FF,  // 12: vegetation (品红)
    0xFF0088,  // 13: terrain (粉红)
    0x888888,  // 14: building (灰色)
    0xCCCCCC,  // 15: barrier (浅灰)
    0xFFFFFF,  // 16: traffic_sign (白色)
    0xFFCC88,  // 17: other (浅橙)
];

const OCCUPANCY_NAMES = [
    'empty', 'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'road', 'sidewalk',
    'traffic_cone', 'vegetation', 'terrain', 'building', 'barrier',
    'traffic_sign', 'other'
];

class OccupancyViewer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.voxelGroup = null;
        this.currentData = null;
        this.frames = [];

        this.init();
    }

    init() {
        const container = document.getElementById('viewer');
        const loading = document.getElementById('loading');

        // 创建场景
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        // 创建相机
        this.camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(80, 80, 80);

        // 创建渲染器
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        // 添加轨道控制器
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 10;
        this.controls.maxDistance = 500;
        this.controls.maxPolarAngle = Math.PI;

        // 添加光源
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 100, 50);
        this.scene.add(directionalLight);

        // 添加网格辅助线
        const gridHelper = new THREE.GridHelper(200, 40, 0x444444, 0x222222);
        this.scene.add(gridHelper);

        // 添加坐标轴
        const axesHelper = new THREE.AxesHelper(60);
        this.scene.add(axesHelper);

        // 创建体素组
        this.voxelGroup = new THREE.Group();
        this.scene.add(this.voxelGroup);

        // 窗口大小调整
        window.addEventListener('resize', () => this.onWindowResize());

        // 开始动画
        loading.style.display = 'none';
        this.animate();

        // 初始化图例
        this.initLegend();

        // 尝试加载默认数据集
        this.loadDefaultDataset();

        console.log('✓ Occupancy Viewer initialized');
    }

    async loadDefaultDataset() {
        try {
            console.log('Fetching file list from /api/list...');
            const response = await fetch('/api/list');
            if (!response.ok) {
                console.log('API not available (static mode?)');
                return;
            }
            
            const files = await response.json();
            if (files.length > 0) {
                console.log(`Auto-loading ${files.length} frames from server`);
                this.frames = files; // 此时 frames 是文件名字符串数组
                this.updateFrameList();
                await this.loadFrame(0);
            }
        } catch (e) {
            console.warn('Auto-load failed:', e);
        }
    }

    initLegend() {
        const legendContainer = document.getElementById('legend');
        legendContainer.innerHTML = '';

        for (let i = 1; i < OCCUPANCY_COLORS.length; i++) { // 跳过 empty
            const item = document.createElement('div');
            item.className = 'legend-item';

            const colorBox = document.createElement('div');
            colorBox.className = 'legend-color';
            colorBox.style.background = `#${OCCUPANCY_COLORS[i].toString(16).padStart(6, '0')}`;

            const label = document.createElement('span');
            label.textContent = `[${i}] ${OCCUPANCY_NAMES[i]}`;

            item.appendChild(colorBox);
            item.appendChild(label);
            legendContainer.appendChild(item);
        }
    }

    onWindowResize() {
        const container = document.getElementById('viewer');
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    async loadDataset(files) {
        const loading = document.getElementById('loading');
        loading.style.display = 'block';
        loading.textContent = '正在加载数据集...';

        try {
            // 筛选 .npz 文件
            this.frames = Array.from(files).filter(f => f.name.endsWith('.npz'));

            if (this.frames.length === 0) {
                alert('未找到 .npz 文件!\n请选择包含 occupancy/*.npz 的目录');
                loading.style.display = 'none';
                return;
            }

            // 按文件名排序
            this.frames.sort((a, b) => a.name.localeCompare(b.name));

            console.log(`Found ${this.frames.length} frames`);

            // 更新帧列表
            this.updateFrameList();

            // 加载第一帧
            await this.loadFrame(0);

            loading.style.display = 'none';
        } catch (error) {
            console.error('Error loading dataset:', error);
            alert('加载失败: ' + error.message);
            loading.style.display = 'none';
        }
    }

    updateFrameList() {
        const frameList = document.getElementById('frameList');
        frameList.innerHTML = '';

        this.frames.forEach((file, index) => {
            const item = document.createElement('div');
            item.className = 'frame-item';
            // 处理 File 对象或字符串
            const name = (typeof file === 'string') ? file : file.name;
            item.textContent = `Frame ${index}: ${name}`;
            item.onclick = () => this.loadFrame(index);
            frameList.appendChild(item);
        });

        // 默认选中第一帧
        if (frameList.children.length > 0) {
            frameList.children[0].classList.add('active');
        }
    }

    async loadFrame(index) {
        const loading = document.getElementById('loading');
        loading.style.display = 'block';
        loading.textContent = `正在加载 Frame ${index}...`;

        try {
            const fileOrName = this.frames[index];
            let arrayBuffer;

            if (typeof fileOrName === 'string') {
                // 从服务器加载
                const url = `/data/${fileOrName}`;
                console.log(`Fetching ${url}...`);
                const response = await fetch(url);
                if (!response.ok) throw new Error(`Fetch failed: ${response.statusText}`);
                arrayBuffer = await response.arrayBuffer();
            } else {
                // 从本地文件对象加载
                arrayBuffer = await fileOrName.arrayBuffer();
            }

            // 使用 fflate 解压 .npz
            const data = await this.parseNPZ(arrayBuffer);

            console.log('Loaded NPZ data:', Object.keys(data));

            // 提取 occupancy 和 mask
            const occupancy = data['occupancy'];
            const mask = data['mask'];

            if (!occupancy) {
                throw new Error('occupancy 数据不存在!');
            }

            this.currentData = {
                occupancy: occupancy,
                mask: mask,
                x_range: Array.from(data['x_range']?.data || [-50, 50]),
                y_range: Array.from(data['y_range']?.data || [-50, 50]),
                z_range: Array.from(data['z_range']?.data || [-4, 4]),
                resolution: data['resolution']?.data?.[0] || 0.5,
                grid_size: Array.from(data['grid_size']?.data || [200, 200, 16])
            };

            console.log('Occupancy shape:', this.currentData.grid_size);
            console.log('Resolution:', this.currentData.resolution);

            // 渲染体素
            this.renderVoxels();

            // 更新统计
            this.updateStats();

            // 更新选中状态
            const frameList = document.getElementById('frameList');
            Array.from(frameList.children).forEach((item, i) => {
                item.classList.toggle('active', i === index);
            });

            loading.style.display = 'none';
        } catch (error) {
            console.error('Error loading frame:', error);
            alert('加载帧失败: ' + error.message);
            loading.style.display = 'none';
        }
    }

    async parseNPZ(arrayBuffer) {
        // 使用 fflate 解压 NPZ (ZIP 格式)
        const uint8Array = new Uint8Array(arrayBuffer);
        const decompressed = fflate.unzipSync(uint8Array);

        const result = {};

        for (const [filename, data] of Object.entries(decompressed)) {
            if (filename.endsWith('.npy')) {
                const name = filename.replace('.npy', '');
                result[name] = this.parseNPY(data);
            }
        }

        return result;
    }

    parseNPY(data) {
        // 简化的 NPY 解析器
        const view = new DataView(data.buffer);

        // NPY 格式规范:
        // Magic (6) + Version (2) + HeaderLen (2) + HeaderString
        
        // 跳过 magic (6) + version (2) = 8 bytes
        let offset = 8;

        // 读取 header length (2 bytes, little-endian)
        const headerLen = view.getUint16(offset, true);
        offset += 2;

        // 读取 header (Python dict string)
        const headerBytes = data.slice(offset, offset + headerLen);
        const headerStr = new TextDecoder().decode(headerBytes).trim();
        offset += headerLen;

        console.log('NPY Header:', headerStr);

        // 解析 shape 和 dtype
        // 增强正则以支持空 shape (), 单元素元组 (10,)
        const shapeMatch = headerStr.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/);
        const dtypeMatch = headerStr.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/);

        if (!shapeMatch || !dtypeMatch) {
            throw new Error(`无法解析 NPY header: ${headerStr}`);
        }

        const shapeStr = shapeMatch[1].trim();
        let shape;
        if (shapeStr === '' || shapeStr === ',') {
            shape = []; // 标量
        } else {
            shape = shapeStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        }
        
        const dtype = dtypeMatch[1];

        // 读取数据
        const dataBytes = data.slice(offset);
        let array;

        if (dtype.includes('u1')) {
            array = new Uint8Array(dataBytes);
        } else if (dtype.includes('f4') || dtype.includes('<f4')) {
            array = new Float32Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 4);
        } else if (dtype.includes('f8') || dtype.includes('<f8')) {
            array = new Float64Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 8);
        } else if (dtype.includes('i4') || dtype.includes('<i4')) {
            array = new Int32Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 4);
        } else if (dtype === '|b1') {
            array = new Uint8Array(dataBytes);
        } else {
            console.warn('Unsupported dtype:', dtype, '- defaulting to Uint8Array');
            array = new Uint8Array(dataBytes);
        }

        return { data: array, shape: shape, dtype: dtype };
    }

    renderVoxels() {
        // 清空现有体素
        while (this.voxelGroup.children.length > 0) {
            const child = this.voxelGroup.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
            this.voxelGroup.remove(child);
        }

        const { occupancy, grid_size, resolution } = this.currentData;
        const [gridX, gridY, gridZ] = grid_size;

        console.log(`Rendering voxels: ${gridX} × ${gridY} × ${gridZ}`);

        // 创建实例化网格 (性能优化)
        const voxelSize = resolution;
        const geometry = new THREE.BoxGeometry(voxelSize, voxelSize, voxelSize);

        // 按类别分组渲染
        const instancesByClass = {};

        for (let label = 1; label < OCCUPANCY_COLORS.length; label++) {
            instancesByClass[label] = [];
        }

        // 遍历体素网格
        let totalVoxels = 0;
        for (let x = 0; x < gridX; x++) {
            for (let y = 0; y < gridY; y++) {
                for (let z = 0; z < gridZ; z++) {
                    const idx = x * (gridY * gridZ) + y * gridZ + z;
                    const label = occupancy.data[idx];

                    if (label > 0) {
                        // 计算世界坐标 (以车辆为中心)
                        const worldX = (x - gridX/2) * voxelSize;
                        const worldY = (y - gridY/2) * voxelSize;
                        const worldZ = (z - gridZ/2) * voxelSize;

                        if (label < OCCUPANCY_COLORS.length) {
                            instancesByClass[label].push({
                                position: new THREE.Vector3(worldX, worldZ, worldY) // 注意: Y-up
                            });
                        }

                        totalVoxels++;
                    }
                }
            }
        }

        console.log(`Non-empty voxels: ${totalVoxels}`);

        // 为每个类别创建实例化网格
        for (const [label, instances] of Object.entries(instancesByClass)) {
            if (instances.length === 0) continue;

            const color = OCCUPANCY_COLORS[parseInt(label)];
            const material = new THREE.MeshLambertMaterial({ color: color });
            const instancedMesh = new THREE.InstancedMesh(geometry, material, instances.length);

            const matrix = new THREE.Matrix4();
            instances.forEach((inst, i) => {
                matrix.setPosition(inst.position);
                instancedMesh.setMatrixAt(i, matrix);
            });

            instancedMesh.instanceMatrix.needsUpdate = true;
            this.voxelGroup.add(instancedMesh);

            console.log(`Class ${label} (${OCCUPANCY_NAMES[label]}): ${instances.length} voxels`);
        }

        // 居中相机
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    updateStats() {
        const { occupancy, mask, grid_size } = this.currentData;
        const totalVoxels = grid_size[0] * grid_size[1] * grid_size[2];

        let nonEmpty = 0;
        let observed = 0;

        for (let i = 0; i < occupancy.data.length; i++) {
            if (occupancy.data[i] > 0) nonEmpty++;
            if (mask && mask.data[i] > 0) observed++;
        }

        const stats = document.getElementById('stats');
        stats.innerHTML = `
            <div>总体素数: ${totalVoxels.toLocaleString()}</div>
            <div>非空体素: ${nonEmpty.toLocaleString()} (${(nonEmpty/totalVoxels*100).toFixed(2)}%)</div>
            <div>有效观测: ${observed.toLocaleString()} (${(observed/totalVoxels*100).toFixed(2)}%)</div>
            <div>网格尺寸: ${grid_size.join(' × ')}</div>
            <div>分辨率: ${this.currentData.resolution.toFixed(2)} m</div>
        `;
    }

    setView(viewName) {
        const distance = 120;

        switch (viewName) {
            case 'top': // 俯视图
                this.camera.position.set(0, distance, 0);
                this.camera.lookAt(0, 0, 0);
                this.controls.target.set(0, 0, 0);
                break;
            case 'front': // 前视图
                this.camera.position.set(0, distance * 0.3, distance);
                this.camera.lookAt(0, 0, 0);
                this.controls.target.set(0, 0, 0);
                break;
            case 'side': // 侧视图
                this.camera.position.set(distance, distance * 0.3, 0);
                this.camera.lookAt(0, 0, 0);
                this.controls.target.set(0, 0, 0);
                break;
            case 'free': // 自由视角
                this.camera.position.set(80, 80, 80);
                this.camera.lookAt(0, 0, 0);
                this.controls.target.set(0, 0, 0);
                break;
        }

        this.controls.update();
    }
}

// 初始化查看器
const viewer = new OccupancyViewer();

// 文件选择器事件
document.getElementById('folderInput').addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        viewer.loadDataset(files);
    }
});

console.log('✓ Occupancy Viewer loaded');
