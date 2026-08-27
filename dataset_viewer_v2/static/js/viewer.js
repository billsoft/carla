// ============================================================
// 语义类别定义 (18类, 对齐 occnetv3_data_generator/config/occupancy_config.py)
// ============================================================
const OCCUPANCY_COLORS = [
    0x000000,  // 0: free
    0xC8C8C8,  // 1: barrier
    0xFFD700,  // 2: bicycle
    0xFF6347,  // 3: bus
    0xFF8C00,  // 4: car
    0xFFA500,  // 5: construction_vehicle
    0xFF1493,  // 6: motorcycle
    0xFF0000,  // 7: pedestrian
    0xFFFF00,  // 8: traffic_cone
    0x4169E1,  // 9: trailer
    0x0000FF,  // 10: truck
    0x505050,  // 11: driveable_surface
    0x787878,  // 12: other_flat
    0xA0A0A0,  // 13: sidewalk
    0x8B4513,  // 14: terrain
    0xDCDCDC,  // 15: manmade
    0x228B22,  // 16: vegetation
    0xFF00FF,  // 17: general_object
];

const OCCUPANCY_NAMES = [
    'Free', 'Barrier', 'Bicycle', 'Bus', 'Car', 'Constr. Vehicle',
    'Motorcycle', 'Pedestrian', 'Traffic Cone', 'Trailer', 'Truck',
    'Driveable', 'Other Flat', 'Sidewalk', 'Terrain',
    'Manmade', 'Vegetation', 'General Object'
];

// 体素空间定义 (occupancy_config.py)
const VOXEL_RES = 0.2;
const VOXEL_X_MIN = -40.0, VOXEL_Y_MIN = -40.0, VOXEL_Z_MIN = -1.0;

// 相机示意图: 固定锚点 (车顶俯视图容器的百分比坐标)，车头朝上。用相机朝向 (yaw) 而不是
// 物理安装坐标归类——B柱和翼子板摄像头物理位置其实很接近 (甚至翼子板比B柱更靠前)，但
// 功能朝向上 B柱看前方两侧、翼子板看后方两侧，这才是 Tesla 官方示意图摆放 前左/前右
// (B柱) 、后左/后右 (翼子板) 的依据。同一方位有多颗相机 (比如前视三目) 时竖直排成一组，
// 不左右并排，避免互相遮挡。
const SECTOR_ORDER = ['front', 'frontright', 'right', 'rearright', 'rear', 'rearleft', 'left', 'frontleft'];
const SECTOR_ANCHORS = {
    front:      { x: 50, y: 18 },
    frontright: { x: 84, y: 40 },
    right:      { x: 92, y: 58 },
    rearright:  { x: 84, y: 74 },
    rear:       { x: 50, y: 83 },
    rearleft:   { x: 16, y: 74 },
    left:       { x: 8,  y: 58 },
    frontleft:  { x: 16, y: 40 },
};

function computeSector(yawDeg) {
    let s = Math.round(yawDeg / 45) % 8;
    if (s < 0) s += 8;
    return s;
}

// 中文语义命名，按方位 + (同方位内按 FOV 从窄到宽) 排序命名——不依赖固定的 cam 序号，
// 换一批标定数据 (哪怕相机顺序不同) 也能按 FOV 正确分出"前长焦/前/前广角"。
const SECTOR_LABELS = {
    front: '前', frontright: '右前', right: '右', rearright: '右后',
    rear: '后', rearleft: '左后', left: '左', frontleft: '左前',
};

function labelForCamera(sectorName, entry, groupItems) {
    if (sectorName === 'front' && groupItems.length > 1) {
        const sorted = [...groupItems].sort((a, b) => (a.cam && a.cam.fov || 0) - (b.cam && b.cam.fov || 0));
        const rank = sorted.indexOf(entry);
        if (groupItems.length === 3) return ['前长焦', '前', '前广角'][rank] || SECTOR_LABELS.front;
        if (groupItems.length === 2) return ['前长焦', '前广角'][rank] || SECTOR_LABELS.front;
    }
    if (sectorName === 'rear' && entry.cam && entry.cam.fov >= 100) return '后广角';
    return SECTOR_LABELS[sectorName] || `Cam ${entry.idx}`;
}

// Diff 模式的固定三色 (miss / false_positive / confusion)，和 style.css 的 .diff-* 保持一致
const DIFF_COLORS = { 1: 0xffa94d, 2: 0xcc5de8, 0: 0xff6b6b };
const DIFF_NAMES = { 0: 'Confusion (类别错)', 1: 'Miss (漏检)', 2: 'False Positive (误检)' };

function heightColor(z01) {
    const stops = [
        [0.0, [0.13, 0.13, 0.60]],
        [0.25, [0.00, 0.60, 0.90]],
        [0.50, [0.10, 0.80, 0.20]],
        [0.75, [0.95, 0.85, 0.10]],
        [1.0, [0.90, 0.15, 0.10]],
    ];
    for (let i = 0; i < stops.length - 1; i++) {
        const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
        if (z01 >= t0 && z01 <= t1) {
            const f = (z01 - t0) / (t1 - t0);
            return new THREE.Color(
                c0[0] + (c1[0] - c0[0]) * f,
                c0[1] + (c1[1] - c0[1]) * f,
                c0[2] + (c1[2] - c0[2]) * f
            );
        }
    }
    return new THREE.Color(1, 1, 1);
}

function hexColor(n) { return '#' + n.toString(16).padStart(6, '0'); }

class Viewer {
    constructor() {
        this.frames = [];
        this.currentFrameIdx = 0;
        this.isPlaying = false;
        this.playTimer = null;

        this.abortController = null;
        this.activeTimeouts = [];

        // 数据集/预测目录状态
        this.calibration = null;      // {cameras: {...}}
        this.hasDepth = false;
        this.hasEgoPose = false;
        this.predictionCount = 0;

        // 显示状态
        this.viewMode = 'gt';         // gt | pred | diff
        this.colorMode = 'semantic';  // semantic | height
        this.hiddenClasses = new Set();
        this.hiddenDiffCats = new Set();
        this.showFree = false;
        this.zMin = 0; this.zMax = 31;
        this.showTrajectory = false;
        this.showFrustum = false;
        this.trajectoryPoints = null;
        this._reloadTimer = null;

        // 当前帧数据缓存 (客户端类别过滤不重新请求)
        this.currentVoxels = null;
        this.currentDiff = null;

        // Three.js
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.voxelMesh = null;
        this.trajectoryLine = null;
        this.frustumGroup = null;

        this.initThree();
        this.initUI();
        this.loadDatasetInfo();
    }

    // ================================================================
    // Three.js 基础场景
    // ================================================================
    initThree() {
        const container = document.getElementById('three-container');

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0b0b0d);

        this.camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
        this.camera.position.set(-80, 0, 80);
        this.camera.up.set(0, 0, 1);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(this.renderer.domElement);

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambient);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(50, 50, 100);
        this.scene.add(dirLight);

        const grid = new THREE.GridHelper(100, 20, 0x444444, 0x222222);
        grid.rotation.x = Math.PI / 2;
        this.scene.add(grid);

        const heroGeo = new THREE.BoxGeometry(4.8, 2.2, 1.6);
        const heroMat = new THREE.MeshBasicMaterial({ color: 0x00FFFF, wireframe: true, transparent: true, opacity: 0.8 });
        const heroMesh = new THREE.Mesh(heroGeo, heroMat);
        heroMesh.position.set(0, 0, 0.8);
        this.scene.add(heroMesh);
        heroMesh.add(new THREE.AxesHelper(3));
        this.scene.add(new THREE.AxesHelper(5));

        window.addEventListener('resize', () => {
            this.camera.aspect = container.clientWidth / container.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(container.clientWidth, container.clientHeight);
        });

        // 点击拾取体素 (区分拖拽和点击: 位移超过阈值不算点击)
        let downPos = null;
        this.renderer.domElement.addEventListener('pointerdown', (e) => { downPos = [e.clientX, e.clientY]; });
        this.renderer.domElement.addEventListener('pointerup', (e) => {
            if (!downPos) return;
            const dx = e.clientX - downPos[0], dy = e.clientY - downPos[1];
            if (Math.sqrt(dx * dx + dy * dy) < 4) this.onVoxelClick(e);
            downPos = null;
        });
        this.renderer.domElement.addEventListener('mousemove', () => this.hideVoxelTooltip());

        this.animate();
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onVoxelClick(e) {
        if (!this.voxelMesh) return;
        const rect = this.renderer.domElement.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -((e.clientY - rect.top) / rect.height) * 2 + 1
        );
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        const hits = raycaster.intersectObject(this.voxelMesh);
        if (!hits.length) return;

        const instanceId = hits[0].instanceId;
        const meta = this.voxelMesh.userData;
        const srcIdx = meta.sourceIndices ? meta.sourceIndices[instanceId] : instanceId;
        const data = meta.voxelData;

        let html;
        if (meta.isDiff) {
            html = `x=${data.x[srcIdx]} y=${data.y[srcIdx]} z=${data.z[srcIdx]}<br>`
                + `GT: ${OCCUPANCY_NAMES[data.gt[srcIdx]] || '?'} | Pred: ${OCCUPANCY_NAMES[data.pred[srcIdx]] || '?'}<br>`
                + `${DIFF_NAMES[data.cat[srcIdx]]}`;
        } else {
            const wx = (VOXEL_X_MIN + data.x[srcIdx] * VOXEL_RES + VOXEL_RES / 2).toFixed(1);
            const wy = (VOXEL_Y_MIN + data.y[srcIdx] * VOXEL_RES + VOXEL_RES / 2).toFixed(1);
            const wz = (VOXEL_Z_MIN + data.z[srcIdx] * VOXEL_RES + VOXEL_RES / 2).toFixed(1);
            html = `voxel (${data.x[srcIdx]}, ${data.y[srcIdx]}, ${data.z[srcIdx]})<br>`
                + `world (${wx}m, ${wy}m, ${wz}m)<br>`
                + `<b>${OCCUPANCY_NAMES[data.label[srcIdx]] || '?'}</b>`;
        }

        const tip = document.getElementById('voxel-tooltip');
        tip.innerHTML = html;
        tip.style.left = (e.clientX - rect.left + 12) + 'px';
        tip.style.top = (e.clientY - rect.top + 12) + 'px';
        tip.classList.remove('hidden');
    }

    hideVoxelTooltip() {
        document.getElementById('voxel-tooltip').classList.add('hidden');
    }

    // ================================================================
    // UI 事件绑定
    // ================================================================
    initUI() {
        document.getElementById('load-dataset-btn').onclick = () => {
            const path = document.getElementById('dataset-path').value;
            fetch('/api/set_dataset', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path })
            }).then(res => res.json()).then(data => {
                if (data.success) this.loadDatasetInfo();
                else alert('Error: ' + data.message);
            });
        };

        document.getElementById('load-prediction-btn').onclick = () => {
            const path = document.getElementById('prediction-path').value;
            this.setPrediction(path);
        };
        document.getElementById('clear-prediction-btn').onclick = () => {
            document.getElementById('prediction-path').value = '';
            this.setPrediction('');
        };

        document.getElementById('prev-btn').onclick = () => this.prevFrame();
        document.getElementById('next-btn').onclick = () => this.nextFrame();
        document.getElementById('play-btn').onclick = () => this.togglePlay();

        const slider = document.getElementById('frame-slider');
        slider.oninput = (e) => { this.currentFrameIdx = parseInt(e.target.value); this.updateFrame(this.currentFrameIdx); };

        // 模式切换 (GT/Pred/Diff)
        document.querySelectorAll('#mode-switch .seg-btn').forEach(btn => {
            btn.onclick = () => {
                if (btn.disabled) return;
                document.querySelectorAll('#mode-switch .seg-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.viewMode = btn.dataset.mode;
                this.reloadVoxels();
            };
        });

        // 颜色模式切换
        document.querySelectorAll('#color-switch .seg-btn').forEach(btn => {
            btn.onclick = () => {
                document.querySelectorAll('#color-switch .seg-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.colorMode = btn.dataset.color;
                this.rebuildCurrentMesh();
            };
        });

        // 体素显示开关
        document.getElementById('toggle-free').onchange = (e) => { this.showFree = e.target.checked; this.reloadVoxels(); };
        document.getElementById('toggle-trajectory').onchange = (e) => { this.showTrajectory = e.target.checked; this.updateTrajectory(); };
        document.getElementById('toggle-frustum').onchange = (e) => { this.showFrustum = e.target.checked; this.updateFrustum(); };

        // Z 层裁切 (双滑块, 防止 min>max, 防抖重新请求)
        const zMinEl = document.getElementById('z-min-slider');
        const zMaxEl = document.getElementById('z-max-slider');
        const zLabel = document.getElementById('z-range-label');
        const onZChange = () => {
            let zmin = parseInt(zMinEl.value), zmax = parseInt(zMaxEl.value);
            if (zmin > zmax) { [zmin, zmax] = [zmax, zmin]; }
            this.zMin = zmin; this.zMax = zmax;
            zLabel.innerText = `${zmin} - ${zmax}`;
            clearTimeout(this._reloadTimer);
            this._reloadTimer = setTimeout(() => this.reloadVoxels(), 180);
        };
        zMinEl.oninput = onZChange;
        zMaxEl.oninput = onZChange;

        // 图例批量操作
        document.getElementById('legend-show-all').onclick = () => {
            if (this.viewMode === 'diff') this.hiddenDiffCats.clear(); else this.hiddenClasses.clear();
            this.rebuildCurrentMesh(); this.renderLegend();
        };
        document.getElementById('legend-hide-all').onclick = () => {
            if (this.viewMode === 'diff') { this.hiddenDiffCats = new Set([0, 1, 2]); }
            else { this.hiddenClasses = new Set(OCCUPANCY_NAMES.map((_, i) => i)); }
            this.rebuildCurrentMesh(); this.renderLegend();
        };

        // Lightbox
        document.getElementById('lightbox-close').onclick = () => this.closeLightbox();
        document.getElementById('lightbox').onclick = (e) => { if (e.target.id === 'lightbox') this.closeLightbox(); };
        document.querySelectorAll('#lightbox-mode-switch .seg-btn').forEach(btn => {
            btn.onclick = () => {
                document.querySelectorAll('#lightbox-mode-switch .seg-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.setLightboxView(btn.dataset.view);
            };
        });

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (e.key === 'ArrowLeft') this.prevFrame();
            else if (e.key === 'ArrowRight') this.nextFrame();
            else if (e.code === 'Space') { e.preventDefault(); this.togglePlay(); }
            else if (e.key === 'Escape') this.closeLightbox();
        });
    }

    // ================================================================
    // 数据集 / 推理结果目录
    // ================================================================
    loadDatasetInfo() {
        fetch('/api/dataset_info').then(res => res.json()).then(data => {
            this.frames = data.frames;
            this.hasDepth = data.has_depth;
            this.hasEgoPose = data.has_ego_pose;
            this.predictionCount = data.prediction_count || 0;

            document.getElementById('frame-slider').max = Math.max(0, this.frames.length - 1);
            document.getElementById('dataset-path').value = data.path;
            document.getElementById('dataset-summary').innerText = `${data.count} 帧`;

            if (data.prediction_path) {
                document.getElementById('prediction-path').value = data.prediction_path;
            }
            document.getElementById('prediction-summary').innerText = this.predictionCount > 0 ? `${this.predictionCount} 帧预测` : '';
            this.updatePredictionModeAvailability();

            this.rawTrajectoryPoints = null;
            if (this.hasEgoPose) {
                fetch('/api/trajectory').then(r => r.json()).then(d => { this.rawTrajectoryPoints = d.points || null; }).catch(() => {});
            }

            const loadCalib = data.has_calibration
                ? fetch('/api/calibration').then(r => r.ok ? r.json() : null).catch(() => null)
                : Promise.resolve(null);

            loadCalib.then(calib => {
                this.calibration = calib;
                this.buildCameraLayout();
                if (this.frames.length > 0) {
                    this.currentFrameIdx = 0;
                    this.updateFrame(0);
                }
                this.renderFrameList();
            });
        });
    }

    setPrediction(path) {
        fetch('/api/set_prediction', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path })
        }).then(res => res.json()).then(data => {
            if (!data.success) { alert('Error: ' + data.message); return; }
            this.predictionCount = data.count;
            document.getElementById('prediction-summary').innerText = this.predictionCount > 0 ? `${this.predictionCount} 帧预测` : (path ? '目录中没有 occupancy 结果' : '');
            this.updatePredictionModeAvailability();
            if (this.viewMode !== 'gt' && this.predictionCount === 0) {
                this.viewMode = 'gt';
                document.querySelectorAll('#mode-switch .seg-btn').forEach(b => b.classList.remove('active'));
                document.querySelector('#mode-switch .seg-btn[data-mode="gt"]').classList.add('active');
            }
            this.renderFrameList();
            this.reloadVoxels();
        });
    }

    updatePredictionModeAvailability() {
        const enabled = this.predictionCount > 0;
        document.querySelector('#mode-switch .seg-btn[data-mode="pred"]').disabled = !enabled;
        document.querySelector('#mode-switch .seg-btn[data-mode="diff"]').disabled = !enabled;
    }

    // ================================================================
    // 相机布局: 车顶俯视图 (SVG) + 按朝向落到固定示意图锚点的缩略图。
    // 同一方位有多颗相机 (比如前视三目) 竖直排成一组，不左右并排，避免互相遮挡。
    // ================================================================
    buildCameraLayout() {
        const container = document.getElementById('camera-markers');
        container.innerHTML = '';
        this.camTiles = {};

        let camEntries;
        if (this.calibration && this.calibration.cameras) {
            camEntries = Object.keys(this.calibration.cameras)
                .sort((a, b) => parseInt(a.split('_')[1]) - parseInt(b.split('_')[1]))
                .map(key => {
                    const cam = this.calibration.cameras[key];
                    const idx = parseInt(key.split('_')[1]);
                    const R = (cam.extrinsics && cam.extrinsics.rotation_matrix) || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
                    const fwd = [R[0][0], R[1][0], R[2][0]]; // 局部 +X (前) 转到车辆系下的朝向
                    const yawDeg = Math.atan2(fwd[1], fwd[0]) * 180 / Math.PI;
                    return { idx, sector: computeSector(yawDeg), cam };
                });
        } else {
            // 没有标定信息时的兜底: 按 index 顺序分到 8 个方位，至少能用不报错
            camEntries = Array.from({ length: 8 }, (_, idx) => ({ idx, sector: idx % 8, cam: null }));
        }

        const groups = {};
        camEntries.forEach(e => { (groups[e.sector] = groups[e.sector] || []).push(e); });

        const PITCH = 13; // 组内竖排间距 (容器高度百分比)，大于缩略图自身高度占比，保证不重叠
        Object.keys(groups).forEach(sectorKey => {
            const sectorName = SECTOR_ORDER[Number(sectorKey)];
            const anchor = SECTOR_ANCHORS[sectorName];
            const items = groups[sectorKey];
            items.forEach((entry, i) => {
                const offsetY = (i - (items.length - 1) / 2) * PITCH;
                const label = labelForCamera(sectorName, entry, items);
                this._createCamTile(container, entry.idx, entry.cam, anchor.x, anchor.y + offsetY, label);
            });
        });
    }

    _createCamTile(container, idx, cam, leftPct, topPct, name) {
        const tile = document.createElement('div');
        tile.className = 'cam-tile';
        tile.dataset.cam = idx;
        tile.style.left = `${Math.max(3, Math.min(97, leftPct))}%`;
        tile.style.top = `${Math.max(3, Math.min(97, topPct))}%`;

        const img = document.createElement('img');
        img.alt = name || `Cam ${idx}`;

        const label = document.createElement('div');
        label.className = 'cam-label';
        // 水平 FOV (cam.fov) 是 Tesla 规格表里通常引用的数字，比 fov_vertical 更好辨认
        const fovText = cam && cam.fov ? `${Math.round(cam.fov)}°` : '';
        label.innerHTML = `<span>${name || `Cam ${idx}`}</span><span class="cam-fov">${fovText}</span>`;

        tile.appendChild(img);
        tile.appendChild(label);

        if (this.hasDepth) {
            const toggle = document.createElement('button');
            toggle.className = 'cam-view-toggle';
            toggle.innerText = 'Depth';
            toggle.onclick = (e) => {
                e.stopPropagation();
                const isDepth = toggle.classList.toggle('active');
                toggle.innerText = isDepth ? 'RGB' : 'Depth';
                tile.dataset.view = isDepth ? 'depth' : 'rgb';
                this.loadSingleCamera(idx);
            };
            tile.appendChild(toggle);
        }

        tile.onclick = () => this.openLightbox(idx);

        container.appendChild(tile);
        this.camTiles[idx] = { tile, img, cam };
    }

    camImageUrl(frameId, camIdx, view, hires) {
        const t = Date.now();
        if (view === 'depth') return `/api/depth/${frameId}/${camIdx}?t=${t}`;
        return `/api/image/${frameId}/${camIdx}?t=${t}${hires ? '&hires=1' : ''}`;
    }

    loadSingleCamera(camIdx) {
        const entry = this.camTiles[camIdx];
        if (!entry) return;
        const frameId = this.frames[this.currentFrameIdx];
        const view = entry.tile.dataset.view === 'depth' ? 'depth' : 'rgb';
        const src = this.camImageUrl(frameId, camIdx, view, false);

        entry.img.classList.remove('loaded', 'load-error');
        const preload = new Image();
        preload.onload = () => { entry.img.src = src; entry.img.classList.add('loaded'); };
        preload.onerror = () => { entry.img.classList.add('loaded', 'load-error'); };
        preload.src = src;
    }

    loadCameraImages(frameId, signal) {
        const camIndices = Object.keys(this.camTiles).map(Number);
        const loadNext = (i) => {
            if (i >= camIndices.length || signal.aborted) return;
            this.loadSingleCamera(camIndices[i]);
            const id = setTimeout(() => loadNext(i + 1), 10);
            this.activeTimeouts.push(id);
        };
        loadNext(0);
    }

    // ================================================================
    // Lightbox
    // ================================================================
    openLightbox(camIdx) {
        this._lightboxCam = camIdx;
        document.getElementById('lightbox').classList.remove('hidden');
        const depthBtn = document.querySelector('#lightbox-mode-switch .seg-btn[data-view="depth"]');
        depthBtn.style.display = this.hasDepth ? '' : 'none';
        document.querySelectorAll('#lightbox-mode-switch .seg-btn').forEach(b => b.classList.remove('active'));
        document.querySelector('#lightbox-mode-switch .seg-btn[data-view="rgb"]').classList.add('active');
        this.setLightboxView('rgb');
    }

    setLightboxView(view) {
        const frameId = this.frames[this.currentFrameIdx];
        const camIdx = this._lightboxCam;
        document.getElementById('lightbox-img').src = this.camImageUrl(frameId, camIdx, view, true);

        const entry = this.camTiles[camIdx];
        const cam = entry ? entry.cam : null;
        let info = `相机 ${camIdx} — 帧 ${frameId}`;
        if (cam) {
            const pos = (cam.extrinsics && cam.extrinsics.translation) || [];
            info += `<br>分辨率 ${cam.width}×${cam.height} | FOV(h) ${cam.fov}° / FOV(v) ${cam.fov_vertical}°`;
            info += `<br>fx=${cam.fx ? cam.fx.toFixed(1) : '-'} fy=${cam.fy ? cam.fy.toFixed(1) : '-'} cx=${cam.cx || '-'} cy=${cam.cy || '-'}`;
            if (pos.length === 3) info += `<br>安装位置 (车辆系) x=${pos[0].toFixed(2)}m y=${pos[1].toFixed(2)}m z=${pos[2].toFixed(2)}m`;
        }
        document.getElementById('lightbox-info').innerHTML = info;
    }

    closeLightbox() {
        document.getElementById('lightbox').classList.add('hidden');
    }

    // ================================================================
    // 帧切换主流程
    // ================================================================
    updateFrame(idx) {
        if (idx < 0 || idx >= this.frames.length) return Promise.resolve();

        if (this.abortController) this.abortController.abort();
        this.abortController = new AbortController();
        const signal = this.abortController.signal;

        this.activeTimeouts.forEach(id => clearTimeout(id));
        this.activeTimeouts = [];

        const frameId = this.frames[idx];

        document.getElementById('frame-slider').value = idx;
        document.getElementById('frame-counter').innerText = `${idx + 1} / ${this.frames.length}`;
        document.getElementById('frame-name').innerText = `ID: ${frameId}`;

        this.highlightCurrentFrame();
        this.loadCameraImages(frameId, signal);
        this.loadEgoInfo(frameId);
        if (this.showTrajectory) this.updateTrajectory();

        return this.reloadVoxels();
    }

    loadEgoInfo(frameId) {
        const el = document.getElementById('frame-ego-pos');
        if (!this.hasEgoPose || !this.rawTrajectoryPoints) { el.innerText = ''; return; }
        const pt = this.rawTrajectoryPoints[this.currentFrameIdx];
        if (pt) el.innerText = `自车世界坐标: x=${pt[0].toFixed(1)}m, y=${pt[1].toFixed(1)}m`;
    }

    // ================================================================
    // 体素加载 / 渲染 (GT / Pred / Diff)
    // ================================================================
    // 服务端按列拼接 (见 server.py::_pack_voxels 的注释)，这里直接在同一块 ArrayBuffer 上开
    // TypedArray 视图，零拷贝、不用逐体素跑循环——几百万体素时这个差异是"卡一下"和"不卡"的区别。
    decodeVoxelBuffer(buf, count) {
        let off = 0;
        const x = new Uint16Array(buf, off, count); off += count * 2;
        const y = new Uint16Array(buf, off, count); off += count * 2;
        const z = new Uint8Array(buf, off, count); off += count;
        const label = new Uint8Array(buf, off, count);
        return { count, x, y, z, label };
    }

    decodeDiffBuffer(buf, count) {
        let off = 0;
        const x = new Uint16Array(buf, off, count); off += count * 2;
        const y = new Uint16Array(buf, off, count); off += count * 2;
        const z = new Uint8Array(buf, off, count); off += count;
        const gt = new Uint8Array(buf, off, count); off += count;
        const pred = new Uint8Array(buf, off, count); off += count;
        const cat = new Uint8Array(buf, off, count);
        return { count, x, y, z, gt, pred, cat };
    }

    reloadVoxels() {
        if (!this.frames.length) return Promise.resolve();
        const frameId = this.frames[this.currentFrameIdx];
        if (this.abortController) {
            // 复用当前帧的 abort signal (不打断图像加载)，如果没有就新建一个仅用于体素请求
        }
        const signal = this.abortController ? this.abortController.signal : undefined;

        const loadingOverlay = document.getElementById('loading-overlay');
        loadingOverlay.style.display = 'block';
        document.getElementById('diff-summary').innerHTML = '';

        const params = new URLSearchParams({ z_min: this.zMin, z_max: this.zMax });

        if (this.viewMode === 'diff') {
            return Promise.all([
                fetch(`/api/occupancy_diff/${frameId}?${params}`, { signal }),
                fetch(`/api/occupancy_diff_summary/${frameId}`, { signal }),
            ]).then(async ([diffRes, sumRes]) => {
                if (signal && signal.aborted) return;
                if (!diffRes.ok) throw new Error(`diff HTTP ${diffRes.status}`);
                const diffN = parseInt(diffRes.headers.get('X-Voxel-Count') || '0');
                const diffData = this.decodeDiffBuffer(await diffRes.arrayBuffer(), diffN);
                this.currentDiff = diffData;
                this.currentVoxels = null;
                this.buildDiffMesh(diffData);
                this.renderLegend();
                document.getElementById('voxel-stats').innerText = this.voxelCountLabel('不一致体素', diffData.count, diffRes);

                if (sumRes.ok) {
                    const s = await sumRes.json();
                    const accPct = (s.accuracy * 100).toFixed(2);
                    const iouPct = (s.occupancy_iou * 100).toFixed(2);
                    document.getElementById('diff-summary').innerHTML =
                        `整体准确率 <span class="stat-good">${accPct}%</span> | 占用IoU <span class="stat-good">${iouPct}%</span> `
                        + `| 漏检 ${s.miss_count} 误检 ${s.false_positive_count} 类别错 ${s.confusion_count}`;
                }
                loadingOverlay.style.display = 'none';
            }).catch(err => this.handleVoxelError(err, loadingOverlay));
        }

        params.set('source', this.viewMode);
        params.set('include_free', this.showFree ? '1' : '0');

        return fetch(`/api/occupancy/${frameId}?${params}`, { signal })
            .then(async (res) => {
                if (signal && signal.aborted) return;
                if (res.status === 404) {
                    this.currentVoxels = { count: 0, x: new Uint16Array(0), y: new Uint16Array(0), z: new Uint8Array(0), label: new Uint8Array(0) };
                    this.clearVoxelMesh();
                    this.renderLegend();
                    document.getElementById('voxel-stats').innerText = this.viewMode === 'pred' ? '该帧无预测结果' : 'Voxels: 0';
                    loadingOverlay.style.display = 'none';
                    return;
                }
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const voxelN = parseInt(res.headers.get('X-Voxel-Count') || '0');
                const data = this.decodeVoxelBuffer(await res.arrayBuffer(), voxelN);
                if (signal && signal.aborted) return;
                this.currentVoxels = data;
                this.currentDiff = null;
                this.buildInstancedMesh(data);
                this.renderLegend();
                document.getElementById('voxel-stats').innerText = this.voxelCountLabel('Voxels', data.count, res);
                loadingOverlay.style.display = 'none';
            })
            .catch(err => this.handleVoxelError(err, loadingOverlay));
    }

    voxelCountLabel(prefix, sentCount, res) {
        const totalHeader = res.headers.get('X-Voxel-Total');
        const total = totalHeader !== null ? parseInt(totalHeader) : sentCount;
        if (total > sentCount) {
            return `${prefix}: ${sentCount.toLocaleString()} / ${total.toLocaleString()} (已降采样)`;
        }
        return `${prefix}: ${sentCount.toLocaleString()}`;
    }

    handleVoxelError(err, loadingOverlay) {
        if (err.name === 'AbortError') return;
        console.error('[Viewer] Error loading voxels:', err);
        loadingOverlay.style.display = 'none';
        document.getElementById('voxel-stats').innerText = `Error: ${err.message}`;
    }

    rebuildCurrentMesh() {
        if (this.viewMode === 'diff' && this.currentDiff) this.buildDiffMesh(this.currentDiff);
        else if (this.currentVoxels) this.buildInstancedMesh(this.currentVoxels);
    }

    clearVoxelMesh() {
        if (this.voxelMesh) this.voxelMesh.count = 0;
    }

    // 复用同一个 InstancedMesh (靠 .count 控制实际绘制的实例数)，避免每帧
    // dispose+new 造成的 GC 抖动——这是"连续播放不卡"的关键之一。容量不够时才重建。
    _ensureMeshPool(minCapacity) {
        if (this.voxelMesh && this.voxelMesh.__cap >= minCapacity) return this.voxelMesh;

        if (this.voxelMesh) {
            this.scene.remove(this.voxelMesh);
            this.voxelMesh.geometry.dispose();
            this.voxelMesh.material.dispose();
        }
        const cap = Math.max(minCapacity, Viewer.MESH_POOL_CAPACITY);
        const geometry = new THREE.BoxGeometry(VOXEL_RES, VOXEL_RES, VOXEL_RES);
        const material = new THREE.MeshLambertMaterial({ color: 0xffffff });
        const mesh = new THREE.InstancedMesh(geometry, material, cap);
        mesh.__cap = cap;
        mesh.setColorAt(0, new THREE.Color(0xffffff)); // 触发 instanceColor 分配
        mesh.count = 0;
        this.scene.add(mesh);
        this.voxelMesh = mesh;
        return mesh;
    }

    // 直接写 mesh.instanceMatrix.array / instanceColor.array（16/3 个 float 一组），跳过
    // Object3D.updateMatrix() 的四元数换算——单帧几百万体素时这一步是真正的性能瓶颈，
    // 比 Object3D+setMatrixAt 的写法快一个数量级以上。矩阵只有平移 (体素不旋转不缩放，
    // BoxGeometry 已经按 VOXEL_RES 建好了)，所以是固定的单位矩阵 + 平移列。
    _writeInstance(matArray, colorArray, k, gx, gy, gz, r, g, b) {
        const x = VOXEL_X_MIN + gx * VOXEL_RES + VOXEL_RES / 2;
        const y = -(VOXEL_Y_MIN + gy * VOXEL_RES + VOXEL_RES / 2);
        const z = VOXEL_Z_MIN + gz * VOXEL_RES + VOXEL_RES / 2;
        const o = k * 16;
        matArray[o] = 1; matArray[o + 1] = 0; matArray[o + 2] = 0; matArray[o + 3] = 0;
        matArray[o + 4] = 0; matArray[o + 5] = 1; matArray[o + 6] = 0; matArray[o + 7] = 0;
        matArray[o + 8] = 0; matArray[o + 9] = 0; matArray[o + 10] = 1; matArray[o + 11] = 0;
        matArray[o + 12] = x; matArray[o + 13] = y; matArray[o + 14] = z; matArray[o + 15] = 1;
        const co = k * 3;
        colorArray[co] = r; colorArray[co + 1] = g; colorArray[co + 2] = b;
    }

    buildInstancedMesh(data) {
        const n = data.count;
        // 没有隐藏类别时走全量快速路径 (跳过一次额外的 filter 遍历)
        let visible = null;
        if (this.hiddenClasses.size > 0) {
            visible = [];
            for (let i = 0; i < n; i++) if (!this.hiddenClasses.has(data.label[i])) visible.push(i);
        }
        const count = visible ? visible.length : n;
        if (count === 0) { this.clearVoxelMesh(); return; }

        const mesh = this._ensureMeshPool(count);
        mesh.count = count;
        const matArray = mesh.instanceMatrix.array;
        const colorArray = mesh.instanceColor.array;

        for (let k = 0; k < count; k++) {
            const i = visible ? visible[k] : k;
            let r, g, b;
            if (this.colorMode === 'height') {
                const c = heightColor(data.z[i] / 31);
                r = c.r; g = c.g; b = c.b;
            } else {
                const hex = OCCUPANCY_COLORS[data.label[i]] !== undefined ? OCCUPANCY_COLORS[data.label[i]] : 0xffffff;
                r = ((hex >> 16) & 255) / 255; g = ((hex >> 8) & 255) / 255; b = (hex & 255) / 255;
            }
            this._writeInstance(matArray, colorArray, k, data.x[i], data.y[i], data.z[i], r, g, b);
        }

        mesh.instanceMatrix.needsUpdate = true;
        mesh.instanceColor.needsUpdate = true;
        mesh.userData = { sourceIndices: visible, voxelData: data, isDiff: false };
    }

    buildDiffMesh(data) {
        const n = data.count;
        let visible = null;
        if (this.hiddenDiffCats.size > 0) {
            visible = [];
            for (let i = 0; i < n; i++) if (!this.hiddenDiffCats.has(data.cat[i])) visible.push(i);
        }
        const count = visible ? visible.length : n;
        if (count === 0) { this.clearVoxelMesh(); return; }

        const mesh = this._ensureMeshPool(count);
        mesh.count = count;
        const matArray = mesh.instanceMatrix.array;
        const colorArray = mesh.instanceColor.array;

        for (let k = 0; k < count; k++) {
            const i = visible ? visible[k] : k;
            const hex = DIFF_COLORS[data.cat[i]] || 0xffffff;
            const r = ((hex >> 16) & 255) / 255, g = ((hex >> 8) & 255) / 255, b = (hex & 255) / 255;
            this._writeInstance(matArray, colorArray, k, data.x[i], data.y[i], data.z[i], r, g, b);
        }

        mesh.instanceMatrix.needsUpdate = true;
        mesh.instanceColor.needsUpdate = true;
        mesh.userData = { sourceIndices: visible, voxelData: data, isDiff: true };
    }

    // ================================================================
    // 类别图例
    // ================================================================
    renderLegend() {
        const list = document.getElementById('voxel-list');
        list.innerHTML = '';

        if (this.viewMode === 'diff') {
            if (!this.currentDiff) return;
            const counts = { 0: 0, 1: 0, 2: 0 };
            for (let i = 0; i < this.currentDiff.count; i++) counts[this.currentDiff.cat[i]]++;
            [1, 2, 0].forEach(cat => {
                const div = document.createElement('div');
                div.className = 'legend-item' + (this.hiddenDiffCats.has(cat) ? ' hidden-class' : '');
                div.innerHTML = `<div class="color-box" style="background:${hexColor(DIFF_COLORS[cat])}"></div>`
                    + `<span class="legend-name">${DIFF_NAMES[cat]}</span><span class="legend-count">${counts[cat]}</span>`;
                div.onclick = () => {
                    if (this.hiddenDiffCats.has(cat)) this.hiddenDiffCats.delete(cat); else this.hiddenDiffCats.add(cat);
                    this.buildDiffMesh(this.currentDiff);
                    this.renderLegend();
                };
                list.appendChild(div);
            });
            return;
        }

        if (!this.currentVoxels) return;
        const counts = {};
        for (let i = 0; i < this.currentVoxels.count; i++) {
            const l = this.currentVoxels.label[i];
            counts[l] = (counts[l] || 0) + 1;
        }
        const total = this.currentVoxels.count || 1;
        Object.keys(counts).map(Number).sort((a, b) => a - b).forEach(label => {
            const count = counts[label];
            const pct = (count / total * 100).toFixed(1);
            const div = document.createElement('div');
            div.className = 'legend-item' + (this.hiddenClasses.has(label) ? ' hidden-class' : '');
            div.innerHTML = `<div class="color-box" style="background:${hexColor(OCCUPANCY_COLORS[label] || 0xffffff)}"></div>`
                + `<span class="legend-name">${OCCUPANCY_NAMES[label] || 'Unknown'}</span>`
                + `<span class="legend-count">${count} (${pct}%)</span>`;
            div.onclick = () => {
                if (this.hiddenClasses.has(label)) this.hiddenClasses.delete(label); else this.hiddenClasses.add(label);
                this.buildInstancedMesh(this.currentVoxels);
                this.renderLegend();
            };
            list.appendChild(div);
        });
    }

    // ================================================================
    // 自车轨迹
    // ================================================================
    updateTrajectory() {
        if (this.trajectoryLine) { this.scene.remove(this.trajectoryLine); this.trajectoryLine = null; }
        if (!this.showTrajectory || !this.frames.length) return;

        const frameId = this.frames[this.currentFrameIdx];
        fetch(`/api/trajectory?relative_to=${frameId}`).then(r => r.json()).then(d => {
            if (!d.points || d.points.length < 2) return;
            const pts = d.points.map(([x, y]) => new THREE.Vector3(x, -y, 0.05));
            const geometry = new THREE.BufferGeometry().setFromPoints(pts);
            const material = new THREE.LineBasicMaterial({ color: 0x4dabf7, linewidth: 2 });
            this.trajectoryLine = new THREE.Line(geometry, material);
            this.scene.add(this.trajectoryLine);
        }).catch(() => {});
    }

    // ================================================================
    // 相机视锥 (简化, 非精确等距投影, 仅供肉眼核对摆放)
    // ================================================================
    updateFrustum() {
        if (this.frustumGroup) { this.scene.remove(this.frustumGroup); this.frustumGroup = null; }
        if (!this.showFrustum || !this.calibration) return;

        const group = new THREE.Group();
        const dist = 8.0;

        Object.values(this.calibration.cameras).forEach(cam => {
            const ext = cam.extrinsics;
            if (!ext || !ext.translation || !ext.rotation_matrix) return;
            const R = ext.rotation_matrix;
            const p = ext.translation;

            const toThree = (v) => new THREE.Vector3(v[0], -v[1], v[2]);
            const applyR = (local) => [
                R[0][0] * local[0] + R[0][1] * local[1] + R[0][2] * local[2],
                R[1][0] * local[0] + R[1][1] * local[1] + R[1][2] * local[2],
                R[2][0] * local[0] + R[2][1] * local[1] + R[2][2] * local[2],
            ];

            const fwd = applyR([1, 0, 0]);
            const right = applyR([0, 1, 0]);
            const up = applyR([0, 0, 1]);

            const hfov = (cam.fov || 90) * Math.PI / 180 / 2;
            const vfov = (cam.fov_vertical || 60) * Math.PI / 180 / 2;

            const camPos = toThree(p);
            const corners = [];
            [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([sx, sy]) => {
                const dir = [
                    fwd[0] + right[0] * Math.tan(hfov) * sx + up[0] * Math.tan(vfov) * sy,
                    fwd[1] + right[1] * Math.tan(hfov) * sx + up[1] * Math.tan(vfov) * sy,
                    fwd[2] + right[2] * Math.tan(hfov) * sx + up[2] * Math.tan(vfov) * sy,
                ];
                const world = [p[0] + dir[0] * dist, p[1] + dir[1] * dist, p[2] + dir[2] * dist];
                corners.push(toThree(world));
            });

            const points = [];
            corners.forEach(c => { points.push(camPos.clone(), c); });
            for (let i = 0; i < 4; i++) points.push(corners[i], corners[(i + 1) % 4]);

            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({ color: 0x4dabf7, transparent: true, opacity: 0.5 });
            group.add(new THREE.LineSegments(geometry, material));
        });

        this.frustumGroup = group;
        this.scene.add(group);
    }

    // ================================================================
    // 帧列表 / 播放控制
    // ================================================================
    renderFrameList() {
        const listContainer = document.getElementById('frame-list');
        if (!listContainer) return;
        listContainer.innerHTML = '';

        if (this.frames.length === 0) {
            listContainer.innerHTML = '<div style="padding:10px; color:#888;">No frames found</div>';
            return;
        }

        this.frames.forEach((frameId, idx) => {
            const div = document.createElement('div');
            div.className = 'frame-item';
            div.id = `frame-item-${idx}`;
            div.innerHTML = `<span>${frameId}</span>`;
            if (this.predictionCount > 0) {
                const dot = document.createElement('span');
                dot.className = 'pred-dot';
                div.appendChild(dot);
            }
            div.onclick = () => { this.currentFrameIdx = idx; this.updateFrame(idx); };
            listContainer.appendChild(div);
        });

        this.highlightCurrentFrame();
    }

    highlightCurrentFrame() {
        document.querySelectorAll('.frame-item').forEach(item => item.classList.remove('active'));
        const current = document.getElementById(`frame-item-${this.currentFrameIdx}`);
        if (current) {
            current.classList.add('active');
            current.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    nextFrame() {
        if (this.currentFrameIdx < this.frames.length - 1) {
            this.currentFrameIdx++;
            return this.updateFrame(this.currentFrameIdx);
        } else if (this.isPlaying) {
            this.currentFrameIdx = 0;
            return this.updateFrame(0);
        }
        return Promise.resolve();
    }

    prevFrame() {
        if (this.currentFrameIdx > 0) {
            this.currentFrameIdx--;
            return this.updateFrame(this.currentFrameIdx);
        }
        return Promise.resolve();
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        const btn = document.getElementById('play-btn');
        if (this.isPlaying) {
            btn.innerText = '⏸';
            this._playLoop();
        } else {
            btn.innerText = '▶';
            clearTimeout(this.playTimer);
        }
    }

    // 自调度播放循环: 等上一帧真正加载完 (图像+体素) 再排下一帧，而不是固定 setInterval
    // 盲目按时间戳触发——数据集单帧体素量可能很大，setInterval 在加载没跟上时会把请求
    // 堆起来，播放反而卡顿/错乱。speed 因此是"帧间最小间隔"而不是硬性节拍。
    async _playLoop() {
        if (!this.isPlaying) return;
        const speed = parseInt(document.getElementById('speed-input').value) || 500;
        try {
            await this.nextFrame();
        } catch (e) {
            // 忽略 (例如切换数据集时的 AbortError)，继续播放循环
        }
        if (!this.isPlaying) return;
        this.playTimer = setTimeout(() => this._playLoop(), speed);
    }
}

// 预分配的 InstancedMesh 容量下限，只是个起始值——第一帧数据一来容量不够会自动重建到
// 实际需要的大小 (_ensureMeshPool)，之后同一场景的帧一般不会再超过，不会反复重建。
Viewer.MESH_POOL_CAPACITY = 50000;

window.onload = () => {
    window.viewer = new Viewer();
};
