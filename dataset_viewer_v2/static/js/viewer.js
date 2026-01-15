// Colors matching existing viewer
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

class Viewer {
    constructor() {
        this.frames = [];
        this.currentFrameIdx = 0;
        this.isPlaying = false;
        this.playTimer = null;
        
        // Three.js
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.voxelMesh = null;
        
        this.initThree();
        this.initUI();
        this.loadDatasetInfo();
    }
    
    initThree() {
        const container = document.getElementById('three-container');
        
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111);
        
        this.camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
        this.camera.position.set(-80, 0, 80); // Rear-left view
        this.camera.up.set(0, 0, 1); // Z-up
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(this.renderer.domElement);
        
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Lights
        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambient);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(50, 50, 100);
        this.scene.add(dirLight);
        
        // Helpers
        const grid = new THREE.GridHelper(100, 20, 0x444444, 0x222222);
        grid.rotation.x = Math.PI / 2; // Rotate to XY plane
        this.scene.add(grid);
        
        const axes = new THREE.AxesHelper(5);
        this.scene.add(axes);
        
        // Resize
        window.addEventListener('resize', () => {
            this.camera.aspect = container.clientWidth / container.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(container.clientWidth, container.clientHeight);
        });
        
        this.animate();
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    initUI() {
        // Legend
        const legend = document.getElementById('legend');
        OCCUPANCY_NAMES.forEach((name, i) => {
            const div = document.createElement('div');
            div.className = 'legend-item';
            const color = '#' + OCCUPANCY_COLORS[i].toString(16).padStart(6, '0');
            div.innerHTML = `<div class="color-box" style="background:${color}"></div>${name}`;
            legend.appendChild(div);
        });
        
        // Controls
        document.getElementById('load-dataset-btn').onclick = () => {
            const path = document.getElementById('dataset-path').value;
            fetch('/api/set_dataset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path})
            }).then(res => res.json()).then(data => {
                if(data.success) {
                    this.loadDatasetInfo();
                } else {
                    alert('Error: ' + data.message);
                }
            });
        };
        
        document.getElementById('prev-btn').onclick = () => this.prevFrame();
        document.getElementById('next-btn').onclick = () => this.nextFrame();
        document.getElementById('play-btn').onclick = () => this.togglePlay();
        
        const slider = document.getElementById('frame-slider');
        slider.oninput = (e) => {
            this.currentFrameIdx = parseInt(e.target.value);
            this.updateFrame(this.currentFrameIdx);
        };
    }
    
    loadDatasetInfo() {
        fetch('/api/dataset_info')
            .then(res => res.json())
            .then(data => {
                this.frames = data.frames;
                document.getElementById('frame-slider').max = Math.max(0, this.frames.length - 1);
                document.getElementById('dataset-path').value = data.path;
                
                if (this.frames.length > 0) {
                    this.currentFrameIdx = 0;
                    this.updateFrame(0);
                }
            });
    }
    
    updateFrame(idx) {
        if (idx < 0 || idx >= this.frames.length) return Promise.resolve();
        
        const frameId = this.frames[idx];
        
        // Update Slider UI
        document.getElementById('frame-slider').value = idx;
        document.getElementById('frame-counter').innerText = `${idx + 1} / ${this.frames.length}`;
        document.getElementById('frame-name').innerText = `ID: ${frameId}`;
        
        // 1. Load Images (Staggered to prevent blocking)
        // 错峰加载: 每 50ms 加载一张图片，避免瞬间 8 个请求阻塞体素加载
        const loadCamera = (idx) => {
            if (idx >= 8) return;
            
            const img = document.getElementById(`cam_${idx}`);
            // Use timestamp to prevent caching issues
            const src = `/api/image/${frameId}/${idx}?t=${Date.now()}`;
            
            img.onload = () => {
                // Success
                img.style.opacity = 1.0;
            };
            
            img.onerror = () => {
                console.warn(`Failed to load image: ${src}`);
                // Stop retrying to avoid flickering
                img.style.opacity = 0.2;
                // Optional: set a placeholder if needed
            };
            
            img.style.opacity = 0.5; // Dim while loading
            img.src = src;
            
            // Next one
            setTimeout(() => loadCamera(idx + 1), 50);
        };
        
        // Start loading cameras
        loadCamera(0);
        
        // 2. Load Occupancy
        const loadingOverlay = document.getElementById('loading-overlay');
        loadingOverlay.style.display = 'block';
        console.log(`[Viewer] Loading occupancy for frame ${frameId}...`);
        
        return fetch(`/api/occupancy/${frameId}`)
            .then(res => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.json();
            })
            .then(data => {
                console.log(`[Viewer] Loaded occupancy: ${data.points ? data.points.length : 0} voxels`);
                this.renderVoxels(data);
                loadingOverlay.style.display = 'none';
                document.getElementById('voxel-stats').innerText = `Voxels: ${data.points ? data.points.length : 0}`;
            })
            .catch(err => {
                console.error("[Viewer] Error loading occupancy:", err);
                loadingOverlay.style.display = 'none';
                // 可选: 显示错误提示给用户
                document.getElementById('voxel-stats').innerText = `Error: ${err.message}`;
            });
    }
    
    renderVoxels(data) {
        if (this.voxelMesh) {
            this.scene.remove(this.voxelMesh);
            this.voxelMesh.geometry.dispose();
            this.voxelMesh.material.dispose();
            this.voxelMesh = null;
        }
        
        const points = data.points;
        if (!points || points.length === 0) return;
        
        const labels = data.labels;
        
        // Update Voxel List (Statistics)
        const counts = {};
        labels.forEach(l => counts[l] = (counts[l] || 0) + 1);
        
        const list = document.getElementById('voxel-list');
        if (list) {
            list.innerHTML = '';
            Object.keys(counts).map(Number).sort((a,b) => a-b).forEach(label => {
                const count = counts[label];
                const name = OCCUPANCY_NAMES[label] || 'Unknown';
                const colorHex = OCCUPANCY_COLORS[label] || 0xFFFFFF;
                const color = '#' + colorHex.toString(16).padStart(6, '0');
                
                const div = document.createElement('div');
                div.className = 'legend-item';
                div.style.padding = '2px 0';
                div.innerHTML = `<div class="color-box" style="background:${color}"></div><span style="flex:1">${name}</span><span>${count}</span>`;
                list.appendChild(div);
            });
        }
        
        // Use InstancedMesh for high performance
        const geometry = new THREE.BoxGeometry(0.2, 0.2, 0.2); // Resolution 0.2m
        const material = new THREE.MeshLambertMaterial({ color: 0xffffff });
        const mesh = new THREE.InstancedMesh(geometry, material, points.length);
        
        const dummy = new THREE.Object3D();
        
        // Center offset: 512*0.2 / 2 = 51.2
        const offsetX = 51.2;
        const offsetY = 51.2;
        const offsetZ = 2.0; // Assuming z starts from -2.0m, index 0 is -2.0m.
        // Actually, we should map indices to world coordinates.
        // Voxel Generator:
        // x_range = [-51.2, 51.2], y_range = [-51.2, 51.2], z_range = [-2.0, 6.0]
        // resolution = 0.2
        // index 0 -> -51.2 + 0.1 (center)
        
        const res = 0.2;
        const xMin = -40.0;
        const yMin = -40.0;
        const zMin = -1.0; // Fixed: Match occupancy_config.py Z_RANGE [-4.0, 4.0]
        
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const label = labels[i];
            
            // Calculate world position
            const x = xMin + p[0] * res + res/2;
            const y = yMin + p[1] * res + res/2;
            const z = zMin + p[2] * res + res/2;
            
            dummy.position.set(x, y, z);
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);
            
            // Set color
            const colorHex = OCCUPANCY_COLORS[label] || 0xFFFFFF;
            mesh.setColorAt(i, new THREE.Color(colorHex));
        }
        
        mesh.instanceMatrix.needsUpdate = true;
        mesh.instanceColor.needsUpdate = true;
        
        this.scene.add(mesh);
        this.voxelMesh = mesh;
    }
    
    nextFrame() {
        if (this.currentFrameIdx < this.frames.length - 1) {
            this.currentFrameIdx++;
            this.updateFrame(this.currentFrameIdx);
        } else if (this.isPlaying) {
            this.currentFrameIdx = 0; // Loop
            this.updateFrame(0);
        }
    }
    
    prevFrame() {
        if (this.currentFrameIdx > 0) {
            this.currentFrameIdx--;
            this.updateFrame(this.currentFrameIdx);
        }
    }
    
    togglePlay() {
        this.isPlaying = !this.isPlaying;
        const btn = document.getElementById('play-btn');
        if (this.isPlaying) {
            btn.innerText = '⏸';
            const speed = parseInt(document.getElementById('speed-input').value) || 500;
            this.playTimer = setInterval(() => this.nextFrame(), speed);
        } else {
            btn.innerText = '▶';
            clearInterval(this.playTimer);
        }
    }
}

// Initialize
window.onload = () => {
    window.viewer = new Viewer();
};
