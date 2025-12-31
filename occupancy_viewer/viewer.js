// CARLA Occupancy 3D 体素查看器
// 使用 Three.js 渲染体素网格
// 更新时间: 2025-12-29
// 新功能: 自动播放、颜色对齐 actor_occupancy_mapping.py

// Occupancy 类别颜色映射 (Hex)
// 严格匹配 dense_occupancy_collection/config/actor_occupancy_mapping.py 中的 OCCUPANCY_COLORS
const OCCUPANCY_COLORS = [
    0x000000,  // 0: free (0, 0, 0)
    0x708090,  // 1: barrier (112, 128, 144) - 灰蓝色
    0xFF3D63,  // 2: bicycle (255, 61, 99) - 粉红色
    0xDC143C,  // 3: bus (220, 20, 60) - 深红色
    0xFF9E00,  // 4: car (255, 158, 0) - 橙色
    0xE99646,  // 5: construction_vehicle (233, 150, 70) - 土黄色
    0xFF00FF,  // 6: motorcycle (255, 0, 255) - 品红色
    0x1E90FF,  // 7: pedestrian (30, 144, 255) - 道奇蓝
    0xFF7F50,  // 8: traffic_cone (255, 127, 80) - 珊瑚橙
    0xFF8C00,  // 9: trailer (255, 140, 0) - 暗橙色
    0xB4A5B4,  // 10: truck (180, 165, 180) - 紫灰色
    0x804080,  // 11: driveable_surface (128, 64, 128) - 深紫色
    0xF423E8,  // 12: other_flat (244, 35, 232) - 洋红色
    0x6B8E23,  // 13: sidewalk (107, 142, 35) - 橄榄绿
    0x98FB98,  // 14: terrain (152, 251, 152) - 淡绿色
    0x464646,  // 15: manmade (70, 70, 70) - 深灰色
    0x00FF00,  // 16: vegetation (0, 255, 0) - 绿色
    0xFFFFFF,  // 17: general_object (255, 255, 255) - 白色
];

const OCCUPANCY_NAMES = [
    'free', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation', 'general_object'
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

        // 自动播放状态
        this.isPlaying = false;
        this.currentFrameIndex = 0;
        this.playInterval = null;
        this.playSpeed = 1000; // 默认 1 秒/帧

        // 类别可见性
        this.classVisibility = new Array(18).fill(true);

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

        // 监听网格尺寸配置变化
        const gridProfile = document.getElementById('gridProfile');
        if (gridProfile) {
            gridProfile.addEventListener('change', () => {
                console.log('Grid profile changed, reloading current frame...');
                if (this.currentData) {
                    this.loadFrame(this.currentFrameIndex);
                }
            });
        }

        // 开始动画
        loading.style.display = 'none';
        this.animate();

        // 初始化图例
        this.initLegend();

        // 初始化播放控制
        this.initPlayControls();

        // 绑定文件输入事件
        const folderInput = document.getElementById('folderInput');
        if (folderInput) {
            folderInput.addEventListener('change', (event) => {
                console.log('📂 File input changed:', event.target.files.length, 'files');
                this.loadDataset(event.target.files);
            });
            console.log('✓ File input listener attached');
        }

        // 尝试加载默认数据集
        this.loadDefaultDataset();

        console.log('✓ Occupancy Viewer initialized');
    }

    initPlayControls() {
        console.log('🎮 Initializing play controls...');

        // 创建播放控制按钮
        const playBtn = document.getElementById('playBtn');
        const speedSlider = document.getElementById('speedSlider');
        const speedValue = document.getElementById('speedValue');

        console.log('playBtn found:', !!playBtn);
        console.log('speedSlider found:', !!speedSlider);
        console.log('speedValue found:', !!speedValue);

        if (playBtn) {
            playBtn.addEventListener('click', () => {
                console.log('🖱️ Play button clicked!');
                this.togglePlay();
            });
            console.log('✓ Play button event listener attached');
        } else {
            console.error('❌ playBtn element not found!');
        }

        if (speedSlider) {
            speedSlider.addEventListener('input', (e) => {
                this.playSpeed = parseInt(e.target.value);
                if (speedValue) {
                    speedValue.textContent = `${this.playSpeed}ms`;
                }

                // 如果正在播放，重启播放器以应用新速度
                if (this.isPlaying) {
                    this.stopPlay();
                    this.startPlay();
                }
            });
            console.log('✓ Speed slider event listener attached');
        }
    }

    togglePlay() {
        console.log('🔄 togglePlay called, isPlaying:', this.isPlaying);
        console.log('   frames.length:', this.frames.length);
        if (this.isPlaying) {
            this.stopPlay();
        } else {
            this.startPlay();
        }
    }

    startPlay() {
        console.log('▶️ startPlay called');
        console.log('   frames.length:', this.frames.length);

        if (this.frames.length === 0) {
            console.warn('⚠️ No frames loaded!');
            alert('请先加载数据集！');
            return;
        }

        this.isPlaying = true;
        const playBtn = document.getElementById('playBtn');
        if (playBtn) {
            playBtn.textContent = '⏸ 暂停播放';
            playBtn.style.background = '#e74c3c';
            console.log('✓ Button updated to pause state');
        }

        // 从当前帧开始播放
        this.playInterval = setInterval(() => {
            this.currentFrameIndex = (this.currentFrameIndex + 1) % this.frames.length;
            console.log(`  Playing frame ${this.currentFrameIndex}/${this.frames.length}`);
            this.loadFrame(this.currentFrameIndex);
        }, this.playSpeed);

        console.log(`✅ 开始自动播放 (速度: ${this.playSpeed}ms/帧)`);
    }

    stopPlay() {
        this.isPlaying = false;
        if (this.playInterval) {
            clearInterval(this.playInterval);
            this.playInterval = null;
        }

        const playBtn = document.getElementById('playBtn');
        if (playBtn) {
            playBtn.textContent = '▶ 自动播放';
            playBtn.style.background = '#27ae60';
        }

        console.log('停止自动播放');
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
                this.frames = files;
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

        const title = document.createElement('div');
        title.textContent = '图例 (点击隐藏/显示)';
        title.style.fontWeight = 'bold';
        title.style.marginBottom = '10px';
        title.style.textAlign = 'center';
        legendContainer.appendChild(title);

        for (let i = 1; i < OCCUPANCY_COLORS.length; i++) {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.style.cursor = 'pointer';
            item.style.userSelect = 'none';
            if (!this.classVisibility[i]) {
                item.style.opacity = '0.3';
            }

            const colorBox = document.createElement('div');
            colorBox.className = 'legend-color';
            colorBox.style.background = `#${OCCUPANCY_COLORS[i].toString(16).padStart(6, '0')}`;

            const label = document.createElement('span');
            label.textContent = `[${i}] ${OCCUPANCY_NAMES[i]}`;

            item.appendChild(colorBox);
            item.appendChild(label);
            
            // 点击切换可见性
            item.onclick = () => {
                this.classVisibility[i] = !this.classVisibility[i];
                item.style.opacity = this.classVisibility[i] ? '1.0' : '0.3';
                if (this.currentData) {
                    this.renderVoxels();
                }
            };

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
            this.frames = Array.from(files).filter(f => f.name.endsWith('.npz'));

            if (this.frames.length === 0) {
                alert('未找到 .npz 文件!\n请选择包含 occupancy/*.npz 的目录');
                loading.style.display = 'none';
                return;
            }

            this.frames.sort((a, b) => a.name.localeCompare(b.name));
            console.log(`Found ${this.frames.length} frames`);

            this.updateFrameList();
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
            const name = (typeof file === 'string') ? file : file.name;
            item.textContent = `Frame ${index}: ${name}`;
            item.onclick = () => {
                this.currentFrameIndex = index;
                this.loadFrame(index);
            };
            frameList.appendChild(item);
        });

        if (frameList.children.length > 0) {
            frameList.children[0].classList.add('active');
        }
    }

    async loadFrame(index) {
        // 自动播放模式下不显示加载提示（避免影响观察）
        const loading = document.getElementById('loading');
        if (!this.isPlaying) {
            loading.style.display = 'block';
            loading.textContent = `正在加载 Frame ${index}...`;
        }

        try {
            const fileOrName = this.frames[index];
            let arrayBuffer;

            if (typeof fileOrName === 'string') {
                const url = `/data/${fileOrName}`;
                const response = await fetch(url);
                if (!response.ok) throw new Error(`Fetch failed: ${response.statusText}`);
                arrayBuffer = await response.arrayBuffer();
            } else {
                arrayBuffer = await fileOrName.arrayBuffer();
            }

            const data = await this.parseNPZ(arrayBuffer);

            const occupancy = data['occupancy'];
            const actor_ids = data['actor_ids']; // 可能为 undefined
            const mask = data['mask'];

            if (!occupancy) {
                throw new Error('occupancy 数据不存在!');
            }

            // 获取用户选择的默认配置
            const profileSelect = document.getElementById('gridProfile');
            const profileValue = profileSelect ? profileSelect.value : '40';
            const defaultGrid = profileValue === '16' ? [200, 200, 16] : [500, 500, 40];
            const defaultRes = profileValue === '16' ? 0.5 : 0.2;

            // 1. 优先使用保存的 grid_size
            let gridSize = data['grid_size']?.data;

            // 2. 否则尝试从 occupancy shape 获取 (仅当它是3D数组时)
            if (!gridSize && occupancy.shape && occupancy.shape.length === 3) {
                gridSize = occupancy.shape;
                console.log('Using occupancy shape for grid_size:', gridSize);
            }

            // 3. 最后使用用户选择的默认值
            if (!gridSize) {
                gridSize = defaultGrid;
                console.log('Using user-selected fallback grid_size:', gridSize);
            }
            gridSize = Array.from(gridSize);

            // 如果实际网格尺寸与下拉列表不一致，尝试同步更新下拉列表
            if (gridSize.length === 3 && profileSelect) {
                // 判断当前是哪种配置
                let detectedProfile = null;
                if (gridSize[0] === 200 && gridSize[1] === 200 && gridSize[2] === 16) {
                    detectedProfile = '16';
                } else if (gridSize[0] === 500 && gridSize[1] === 500 && gridSize[2] === 40) {
                    detectedProfile = '40';
                }

                // 如果检测到标准配置且与当前选择不一致，则更新下拉列表
                // 注意：为了避免触发 change 事件导致死循环，这里只修改 value 而不触发事件
                if (detectedProfile && profileSelect.value !== detectedProfile) {
                    console.log(`Auto-updating dropdown to match detected grid size: ${detectedProfile}`);
                    profileSelect.value = detectedProfile;
                }
            }

            // 分辨率处理
            // 兼容 voxel_size 字段
            let resolution = data['resolution']?.data?.[0] || data['voxel_size']?.data?.[0];
            
            // 如果没有分辨率，根据 grid_size[2] (Z轴) 智能匹配，或使用默认值
            if (!resolution) {
                if (gridSize[2] === 16) {
                    resolution = 0.5;
                } else if (gridSize[2] === 40) {
                    resolution = 0.2;
                } else {
                    resolution = defaultRes;
                }
                console.log('Using inferred/default resolution:', resolution);
            }

            const xRange = Array.from(data['x_range']?.data || [-50, 50]);

            this.currentData = {
                occupancy: occupancy,
                actor_ids: actor_ids,
                mask: mask,
                x_range: xRange,
                y_range: Array.from(data['y_range']?.data || [-50, 50]),
                z_range: Array.from(data['z_range']?.data || [-4, 4]),
                resolution: resolution,
                grid_size: gridSize
            };

            this.renderVoxels();
            this.updateStats();

            const frameList = document.getElementById('frameList');
            Array.from(frameList.children).forEach((item, i) => {
                item.classList.toggle('active', i === index);
            });

            loading.style.display = 'none';
        } catch (error) {
            console.error('Error loading frame:', error);
            if (!this.isPlaying) {
                alert('加载帧失败: ' + error.message);
            }
            loading.style.display = 'none';
        }
    }

    async parseNPZ(arrayBuffer) {
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
        const view = new DataView(data.buffer);

        let offset = 8;
        const headerLen = view.getUint16(offset, true);
        offset += 2;

        const headerBytes = data.slice(offset, offset + headerLen);
        const headerStr = new TextDecoder().decode(headerBytes).trim();
        offset += headerLen;

        const shapeMatch = headerStr.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/);
        const dtypeMatch = headerStr.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/);

        if (!shapeMatch || !dtypeMatch) {
            throw new Error(`无法解析 NPY header: ${headerStr}`);
        }

        const shapeStr = shapeMatch[1].trim();
        let shape;
        if (shapeStr === '' || shapeStr === ',') {
            shape = [];
        } else {
            shape = shapeStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        }

        const dtype = dtypeMatch[1];

        const dataBytes = data.slice(offset);
        let array;

        if (dtype.includes('u1')) {
            array = new Uint8Array(dataBytes);
        } else if (dtype.includes('u4') || dtype.includes('<u4')) {
            array = new Uint32Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 4);
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
        while (this.voxelGroup.children.length > 0) {
            const child = this.voxelGroup.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
            this.voxelGroup.remove(child);
        }

        const { occupancy, actor_ids, grid_size, resolution } = this.currentData;
        const [gridX, gridY, gridZ] = grid_size;

        const voxelSize = resolution;
        const geometry = new THREE.BoxGeometry(voxelSize, voxelSize, voxelSize);

        const instancesByClass = {};

        for (let label = 1; label < OCCUPANCY_COLORS.length; label++) {
            instancesByClass[label] = [];
        }

        let totalVoxels = 0;
        for (let x = 0; x < gridX; x++) {
            for (let y = 0; y < gridY; y++) {
                for (let z = 0; z < gridZ; z++) {
                    const idx = x * (gridY * gridZ) + y * gridZ + z;
                    const label = occupancy.data[idx];

                    // const actorId = actor_ids ? actor_ids.data[idx] : 1;
                    // const isVisible = !actor_ids || (actorId !== 0);
                    // 用户请求：去除可见性检查，完全基于 label 渲染
                    if (label > 0 && this.classVisibility[label]) {
                        const worldX = (x - gridX/2) * voxelSize;
                        const worldY = (y - gridY/2) * voxelSize;
                        const worldZ = (z - gridZ/2) * voxelSize;

                        if (label < OCCUPANCY_COLORS.length) {
                            instancesByClass[label].push({
                                position: new THREE.Vector3(worldX, worldZ, worldY)
                            });
                        }

                        totalVoxels++;
                    }
                }
            }
        }

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
        }

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
            case 'top':
                this.camera.position.set(0, distance, 0);
                this.camera.lookAt(0, 0, 0);
                this.controls.target.set(0, 0, 0);
                break;
            case 'front':
                this.camera.position.set(0, distance * 0.3, distance);
                this.camera.lookAt(0, 0, 0);
                this.controls.target.set(0, 0, 0);
                break;
            case 'side':
                this.camera.position.set(distance, distance * 0.3, 0);
                this.camera.lookAt(0, 0, 0);
                this.controls.target.set(0, 0, 0);
                break;
            case 'free':
                this.camera.position.set(80, 80, 80);
                this.camera.lookAt(0, 0, 0);
                this.controls.target.set(0, 0, 0);
                break;
        }

        this.controls.update();
    }
}

const viewer = new OccupancyViewer();

document.getElementById('folderInput').addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        viewer.loadDataset(files);
    }
});

console.log('✓ Occupancy Viewer loaded (Updated 2025-12-29)');
