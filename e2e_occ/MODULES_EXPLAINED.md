# e2e_occ 模块详解

> 逐模块讲解 `e2e_occ` 网络：每个模块**做什么**、**输入输出**、**怎么算**、**为什么这样设计**。
> 2026-10 重构后版本。结构速查见 [`ARCHITECTURE.md`](./ARCHITECTURE.md)，每层精确形状见
> [`shape_report.md`](./shape_report.md)，训练细节见 [`TRAINING.md`](./TRAINING.md)。代码与本文不一致时以代码为准。
>
> 本次重构的来龙去脉见第 11 节：先修了 3 个让几何先验整体失效的坐标系错误（P0），再分三组加入
> 优化（P1~P3），每组都有配置开关，可以和基线做消融对比。

---

## 0. 全局视角

**任务**：8 个等距鱼眼相机同一时刻的 Bayer RAW → 车辆周围 80m×80m×6.4m 空间里每个 0.2m 体素的
18 类语义，外加每个体素的运动（占用流）。

**核心难点**：图像是 2D 的，答案是 3D 的，中间缺的是深度。

**思路（BEVFormer / Tesla Occupancy 一脉）**：在 3D 空间撒查询点，每个点按相机几何主动投影到
各相机图像上取特征。点知道自己在哪、相机参数已知，就能算出"应该"出现在哪个像素。深度不显式估计，
只作为训练辅助监督让图像特征更懂几何。

```
8 路 RAW ─► 2D 图像侧（多尺度特征 + 射线方向）─► 3D 查询侧（粗 → 时序 → 稀疏细）─► 体素头
            PatchEmbed / ImageEncoder              OccupancyDecoder / TemporalFusion    VoxelHead
```

| 模块 | 文件 | 一句话作用 |
|---|---|---|
| geometry | `geometry.py` | 统一的坐标约定与等距投影公式（全网唯一一份） |
| MultiCameraPatchEmbed | `raw_embed.py` | RAW → stride 8 / 16 两级特征 |
| ImageEncoder | `image_encoder.py` | 注入射线方向、窗口注意力、生成 stride 32、深度头 |
| RayDirectionEncoding | `position_encoding.py` | 像素 → 车辆系射线方向 → 编码 |
| DeformableCrossAttention | `deformable_attention.py` | 3D 点 → 8 相机 × 3 层级采样特征 |
| OccupancyDecoder | `occ_decoder.py` | 粗查询 → 时序 → 稀疏细查询 |
| TemporalFusionModule | `temporal_fusion.py` | 对齐并融合上一帧记忆 |
| VoxelHead / ImplicitVoxelHead | `voxel_head.py` | 细体素特征 → 400×400×32 语义 + 流场 |
| E2ELoss | `loss.py` | 语义 + 深度 + 流场 + 粗占用 |
| layers | `layers.py` | RMSNorm / SwiGLU / 3D RoPE 自注意力 |

---

## 1. 坐标系与几何（`geometry.py`）

### 作用
定义全网唯一的坐标约定和投影公式。射线编码（反投影）和可变形注意力（正投影）都调用这里，
数据加载里的深度目标换算也调用这里——三处不可能再"各写一套、约定不一致"。

### 约定
- **车辆/体素系**：CARLA 原生 X 前、Y 右、Z 上（左手系）。体素中心 `vmin + (i+0.5)·cell`。
- **相机系**：OpenCV x 右、y 下、z 前（光轴）。
- **外参**：Camera(OpenCV)→Vehicle，静态安装标定，**不含**车辆在世界里的位置。
- **像素**：像素中心在整数坐标，stride-s 特征像素 i 对应原图 `(i+0.5)s − 0.5`。

### 算法
- **UE → OpenCV 相机轴**：`p_ue = M·p_cv`，`M = [[0,0,1],[1,0,0],[0,-1,0]]`
  （像素向右 = UE +Y，向下 = UE −Z，光轴 = UE +X，依据 `WideAngleLens.usf`）。`det(M) = −1`
  是因为它把右手系映射进 CARLA 左手系，是正确的。
- **CARLA 旋转矩阵**（`carla_rotation_matrix`）：与 UE `FRotationMatrix` 逐项一致（pitch 正值 = 抬头），
  即引擎实际渲染的朝向。**不是** `carla.Transform.get_matrix()`——上游 #9751 翻转了客户端 pitch/roll
  符号，二者在 pitch/roll≠0 时相反（见第 11 节第 4 条）。
- **等距投影**：反投影 `θ = |((u−cx)/fx, (v−cy)/fy)|`，方向 `(sinθcosφ, sinθsinφ, cosθ)`；
  正投影 `θ = atan2(√(x²+y²), z)`，`u = cx + fx·θ·cosφ`。允许 fx≠fy、主点偏心。
  （不用 `acos(z/|P|)`：光轴附近 float32 精度不足，实测窄视场相机在光轴附近误差 0.55 像素；atan2 版本
  往返误差 ~2e-4 像素，`verify_network.py` 2.1 节专门在主点附近加了测试点。fx≠fy 时往返误差同量级。）
- **归一化**：图像像素 → `grid_sample(align_corners=False)` 坐标 `2(u+0.5)/W − 1`，这个值与特征层级
  无关，所以一个参考点可以直接用于所有层级。

### 为什么重要
重构前恰恰是这里出了三个错（第 11 节），而且全网自检用的都是单位矩阵外参，测不出来。
现在 `verify_network.py` 用真实标定测试"车前 10m 地面点落在 front_main 下半部中央"等语义，
并用旧约定做反例，确认测试有区分度。

---

## 2. MultiCameraPatchEmbed（`raw_embed.py`）

### 输入输出
`[B, 8, 1, 960, 1280]` → `{8: [B,8,256,120,160], 16: [B,8,256,60,80]}`

### 算法
```
rggb_conv  Conv2d(1→4,  k2,s2, 无 bias)        → 480×640
stage1     Conv2d(4→64,  k3,s2)+BN+GELU         → 240×320   (stride 4)
stage2     Conv2d(64→128,k3,s2)+BN+GELU         → 120×160   (stride 8)  ──► 1×1 投影到 256 + LayerNorm
stage3     Conv2d(128→256,k3,s2)+BN+GELU
           Conv2d(256→256,k3,s1)+BN              →  60×80    (stride 16) ──► LayerNorm
```

### 为什么
- **直接吃 RAW**：没有 ISP 的去马赛克/降噪/色调映射，保留线性亮度和更高位深，让网络学"ISP"。
- **`rggb_conv`**：2×2/stride-2 每次恰好覆盖一个 RGGB 单元，是可学习的去马赛克；无 bias 保持对亮度线性。
- **多输出一个 stride 8**：原来只有 stride 16（60×80），对远处行人、锥桶这类小目标太粗。stride 8 是
  stem 中间层，几乎不增加计算。

---

## 3. ImageEncoder（`image_encoder.py`）

### 输入输出
PatchEmbed 输出 + 内外参 → `{'feats': {8,16,32}, 'depth_logits': [B,8,64,120,160]}`

### 算法
```
f16 = f16 + Ray(16)
逐相机: 2 × EncoderBlock（Pre-Norm RMSNorm，窗口注意力 7×7 → SwiGLU FFN）
        第 2 个块循环移位 3 像素（Swin 移位窗口），用注意力掩码屏蔽拼接到一起的不相邻区域和 pad
f8  = f8 + Ray(8)
f32 = Conv(k3,s2)(f16) + Ray(32)
depth_logits = DepthHead(f8)          # Conv3×3 → BN → GELU → Conv1×1(→64 bins)
```

### 为什么
- **射线编码**：卷积平移不变，"图左上角的车"和"图中央的车"卷积输出一样，但 3D 位置完全不同，
  鱼眼边缘还有强畸变。加上射线方向后特征同时知道"看到什么"和"朝哪看"。
- **移位窗口**：不移位时窗口之间完全不交换信息；交替移位让相邻窗口在第二层就能交流（Swin 的核心技巧），
  计算量不变。
- **stride 32 从编码后的 stride 16 下采样**：带上窗口注意力的上下文，给远处大结构和大物体提供更大感受野。
- **深度头（BEVDepth 式辅助监督）**：本网络不做显式深度 lifting，但让 stride-8 特征去预测深度，
  会逼着图像特征编码几何信息，下游交叉注意力取到的特征更"懂"距离。训练时用，推理无开销。

---

## 4. RayDirectionEncoding（`position_encoding.py`）

### 输入输出
内外参 + 层级尺寸/stride → `[B, 8, 256, H, W]`

### 算法
```
像素中心 (i+0.5)s−0.5 → 等距反投影得相机系方向 → R·dir 得车辆系方向
enc = [d, sin(2^k π d), cos(2^k π d)]_{k=0..9}   # 63 维
out = Linear(63→256) → LayerNorm → ReLU → Linear(256→256)   # 各层级共享
```

### 为什么
- **NeRF 式正弦编码**：MLP 有频谱偏置，直接吃 3 维方向分不清细微差别；多频展开后相近方向在高频分量上差异明显。
- **车辆系**（而不是相机系）：不同相机看同一方向得到相同编码；重构前用的是含车辆朝向的"世界系"，
  同一相机的编码会随车转向而变，现在已改为车辆系。

---

## 5. DeformableCrossAttention（`deformable_attention.py`）

整个网络**最核心的模块**——2D 到 3D 的桥梁。

### 输入输出
| | 形状 |
|---|---|
| `query` | `[B, Q, 256]`（Q = 5,000 或稀疏细阶段的 25,600） |
| `points` | `[B, Q, 3]` 查询点车辆系坐标（米） |
| `image_feats` | `{8,16,32: [B,8,256,H,W]}` |
| 输出 | `[B, Q, 256]` |

### 算法
1. **参考点**（`get_reference_points`）：`p_cam = Rᵀ(p − t)`（外参是 Camera→Vehicle）→ 等距正投影 →
   归一化坐标；同时得到每个相机的可见性（投影是否落在图像内）。
2. **偏移与权重**（由查询特征预测）：每个 (相机, 头, 层级) 4 个采样点，偏移 `tanh(·)×4` **个该层级像素**，
   所以粗层级天然看得更远；同一 (相机, 头) 内对 3 层级 × 4 点共 12 个采样做 softmax。
3. **采样**：逐相机、逐层级、按 32,768 个查询分块，8 个头一次性 `grid_sample`，加权求和。
4. **可见相机归一化**：投影在图像外的相机贡献乘 0，最后除以命中相机数。

`DeformableDecoderLayer` = [3D-RoPE 自注意力] → 交叉注意力 → SwiGLU FFN，全部 Pre-Norm(RMSNorm) 残差。

### 为什么
- **可变形**：普通交叉注意力每个查询看全部像素，O(Q·HW)；这里只看参考点附近 K 个点，O(Q·K)。
  偏移可学习，用来补偿标定误差、物体体积等"投影点不一定是最佳采样点"的情况。
- **参考点 = 几何先验**：网络不用从零学"3D 点该看图像哪里"，只学残差偏移——这是数据效率的关键，
  也是为什么坐标系错误（第 11 节）会让网络几乎学不到东西。
- **偏移初始化**：权重为 0，bias 让各头各点初始散成一个半径 0.5~2 像素的小环（Deformable DETR 做法），
  训练初期就在参考点附近采样且互不重合。
- **多尺度**：近处大目标在 stride 32 上采样更稳，远处小目标需要 stride 8 的分辨率，网络通过 12 路
  softmax 自己选。
- **可见相机归一化**：被 1 个相机看到和被 3 个相机看到的体素，累加后特征幅值差 3 倍，网络本来要
  额外学会消除这种尺度差；取平均后各位置尺度一致（BEVFormer 做法）。
- **3D RoPE 自注意力**（粗阶段）：查询之间的注意力分数只依赖相对位置（米），比相加式位置编码更直接地
  表达"离我 3m 的那个体素"，外推也更好。

---

## 6. OccupancyDecoder（`occ_decoder.py`）

### 输入输出
图像特征 + 相机参数 (+ 记忆、ego_motion、时间差) →
`fine_feats [B,80,80,16,256]`、`memory [B,5000,256]`、`coarse_occ_logits [B,25,25,8]`、`fine_selected [B,K]`

### 算法
```
粗：query = 可学习 [5000,256] + 3D 正弦位置编码；点 = 25×25×8 体素中心（米）
    2 × DecoderLayer（带自注意力）
时序：TemporalFusion（第 7 节）
粗占用：coarse_occ_head(query) → [25,25,8] logits（BCE 监督：该粗格是否含任何非空体素）
细初值：三线性上采样到 80×80×16 → Norm(x + FFN(Norm(x))) → + 3D 正弦位置编码
稀疏：score = 上采样(粗占用 logits)，取 top 25%（25,600 个）细体素
      只对它们做 2 × DecoderLayer（无自注意力，梯度检查点），结果 scatter 回稠密网格
空间一致性：vol + GELU(BN(DepthwiseConv3d 3×3×3(vol)))
```

### 为什么
- **先粗后细**：5,000 个查询可以做全局自注意力（理解整体布局、补全遮挡），102,400 个不行。
- **查询点 = 体素中心（米）**：重构前是 `linspace(0,1)` 的边界网格，首尾点落在 ±40m 边界上，
  与真值体素中心差半格；且范围硬编码，改 `voxel_range` 不会同步。
- **稀疏细化**：80m×80m 空间九成以上是空气，对空气体素做精细交叉注意力是浪费。粗阶段先判断
  "哪里可能有东西"，只精修那 25%（SparseOcc/OPUS 一类方法的思路）；没被选中的体素保留粗特征
  上采样的初值，后面的体素头仍能把它们判成空气。省下的算力可以换更深的层或打开细阶段自注意力。
- **Depthwise Conv3D**：细阶段没有自注意力，相邻体素之间靠它保持一致（避免车和路面交错），
  参数只有约 7K。

---

## 7. TemporalFusionModule（`temporal_fusion.py`）

### 输入输出
`current [B,5000,256]`、`memory [B,5000,256] 或 None`、`ego_motion [B,4,4]`、`time_delta [B]` →
`fused [B,5000,256]`、`new_memory [B,5000,256]`

### 算法
1. **运动对齐**（`align_memory`）：对当前帧每个粗体素中心 `p_t`，`p_{t−1} = inv(ego_motion)·p_t`，
   在上一帧记忆体积里三线性采样；同时采样一个全 1 通道得到"有效掩码"（上一帧感知范围外 = 0）。
   体积排成 `[B,C,Z,Y,X]`，使 `grid_sample` 的 (x,y,z) 分量正好对应车辆 X/Y/Z。
2. **时间嵌入**：帧间隔 dt 的正弦编码 → MLP，加到记忆上。
3. **局部可变形时序注意力**（默认）：每个查询在自身格子附近采样 8 头 × 8 点（偏移 ±2 格），softmax 加权；
   结果乘有效掩码后残差相加，再过 FFN。（`temporal_attn_type='global'` 时为 5000×5000 全局注意力。）
4. **GRU 门控**：`z, r` 门 → 候选 → `new_memory = (1−z)⊙memory + z⊙h̃`。
5. **记忆回注**：`fused += mem_out(new_memory)`（`mem_out` 零初始化）。

### 为什么
- **必须运动对齐**：车在动，上一帧"前方 10m 的车"当前应在"前方 9m"，不对齐就会留拖影。
  `verify_network.py` 测了前进、右移、原地右转三种情况的平移/旋转方向。
- **局部而不是全局**：记忆已经对齐，每个位置只需要看邻域。全局注意力是 O(Q²)，还可能被远处无关特征干扰；
  局部采样 O(Q·K)，偏移可学习，能容忍其他车辆自身运动和位姿误差。
- **有效掩码**：刚进入视野的区域对齐出来是 0，不是真实历史，不应该当成"这里以前是空的"。
- **GRU**：静态场景（路、建筑）长期累积，动态物体快速更新，由门控逐通道学会保留多少旧信息。
- **记忆回注**：训练默认流式（每帧 detach 记忆）。如果 GRU 输出只在下一帧被用，而下一帧又被 detach，
  GRU 永远没有梯度。回注后 GRU 能从当前帧损失学习；零初始化保证起步时行为不变。

---

## 8. 体素输出头（`voxel_head.py`）

### 稠密头 VoxelHead（默认）
```
[B,256,80,80,16] → Conv3d 256→128→64（3×3×3，BN，GELU）
  → 三线性到 200×200×32 → refine1(64→32, 3×3×3) + skip1(1×1)         ──► flow 分支 32→16→3，上采样到 400
  → 三线性到 400×400×32 → refine2(32→18, 3×3×3, BN) + skip2(1×1)  → logits [B,18,400,400,32]
```
- **分类推迟到最高分辨率**：先在 80×80 分类再插值，类别边界会被插值糊掉；每次上采样后都在高分辨率用
  3×3 卷积重新判断边界。
- **训练时梯度检查点**：400×400×32 上的插值结果、BN 输出等中间激活单帧数 GB，不保存、反向时重算。
- **输出层近零初始化**：refine2 的 BN gamma=0、skip2 权重很小，初始 logits ≈ 0 → 初始 CE ≈ ln18。
  重构前初始 CE ≈ 7，训练初期要先花大量步数压掉错误的高置信度。

### 隐式头 ImplicitVoxelHead（`head_type='implicit'`）
```
[B,256,80,80,16] → Conv3d 256→128→64 → 对任意点 (x,y,z) 三线性插值 64 维特征
  ‖ 坐标傅里叶编码（6 频，39 维） → MLP(103→128→128→21) → 18 类 logits + 3 维流场
```
- Tesla 公开分享中的"可查询"占用解码器思路：训练时只对采样点算损失（一半非空体素 + 一半随机），
  不在 400×400×32 上做稠密 3D 卷积；推理时分块查询整个网格（也可以按需查询任意分辨率）。

---

## 9. 损失（`loss.py`）

```
total = CE + 0.5·Lovász  +  深度 CE  +  流场 SmoothL1  +  粗占用 BCE
```

| 项 | 作用 |
|---|---|
| CE | 逐体素分类，让训练"启动" |
| Lovász-Softmax | IoU 的可微凸替代，按类别平均，小类与大类同等重要，缓解空气占绝大多数的不平衡。超过 100 万体素时子采样（全保留非空体素） |
| 深度 CE | 64 个对数均匀 bin（1~60m）；目标由针孔深度图换算成沿鱼眼射线的距离（见 TRAINING.md） |
| 流场 SmoothL1 | 只在 `flow_mask`（动态物体）内；目标是相对自车的位移 |
| 粗占用 BCE | 稀疏细化的选择依据要准，否则会把真有东西的体素漏掉 |

验证时计算 **mIoU**（不含 0:free）。

---

## 10. 借鉴 LLM 的构件（`layers.py`）

| 构件 | 用在哪 | 作用 |
|---|---|---|
| RMSNorm | 所有 Transformer 块 | 去掉均值中心化，更省、更稳（LLaMA 系标配） |
| SwiGLU | 所有 FFN | 门控 FFN，同参数量下表达力强于 GELU-MLP；隐层取 2/3·4·dim 保持参数量 |
| 3D RoPE | 粗阶段自注意力 | head_dim 32 拆成 10/10/12 分给 x/y/z，以米为单位旋转 Q/K |
| SDPA | 全部注意力 | 自动选 FlashAttention / Memory-Efficient kernel |
| MAE 自监督预训练 | `pretrain_mae.py` | 遮 60% 的 16×16 RAW 块、只在被遮块上算重建损失（SimMIM 式），用无标注数据预训练骨干 |

---

## 11. 本次重构修复的问题（2026-10）

### P0：坐标系错误（无开关）
1. **相机轴约定**：网络假设 OpenCV 轴，数据给的是 UE 轴，网络把相机"正上方"当光轴。
   → `dataset.py` 统一转换为 OpenCV 轴。
2. **外参含世界绝对位姿**：查询点被当成世界原点附近的点，车离开原点后几乎全部投影到图像外。
   → 外参改为 Camera→Vehicle，世界位姿只用于 ego_motion。
3. **时序对齐**：ego_motion 用 0 号相机位姿（相机系而非车辆系），且 `align_memory` 的 X/Y 对调。
   → 由车辆位姿计算，体积按 `[Z,Y,X]` 排列。
4. 采集侧 `get_extrinsics` 的旋转公式 pitch/roll 符号与 UE 渲染相反（rear 相机 pitch=8° 被算反）。
   旧公式其实与当前 LibCarla 客户端 `carla.Transform.get_matrix()` 一致——上游 #9751 翻转了客户端
   pitch/roll 符号，但 `Rotation→FRotator` 原样传递、UE 端未改，所以客户端矩阵与渲染不符。
   → 采集代码改为 UE FRotationMatrix 公式，并在 `extrinsics.json` 写入 `rotation_convention: ue_frotator`
   标记；旧数据加载时自动修正。
5. 小问题：体素网格改为中心对齐；`voxel_range` 不再硬编码；特征图半像素约定统一；
   时序序列不再跨场景拼接；推理脚本不再硬编码 `/4095`。

### 发现但没有改动的事项
- rear 相机俯仰：2026-08 曾按 #9751 翻转后的客户端公式把 pitch 改成 +8°，在 UE 里实际是**朝上**看 8°。
  2026-10 已按特斯拉实车（牌照上方、略向下俯）改回 **−8°**（UE pitch 负值=低头，光轴车辆系 z=−0.139）。
  用 +8° 采集的旧数据 rear 是朝上拍的（标定文件如实记录了 +8°，网络加载不会错），但与新数据视角不同，
  不建议混用。
- **真值生成受上游 #9751 影响**：`ground_truth_voxel_generator.py`（4 处）和 `main_collection.py`
  的 `ego_pose` 用 `transform.get_matrix()`，它的 pitch/roll 符号与渲染相反。平地（Town10HD）几乎
  无影响；坡道/倾斜路面上朝向误差为 2θ，2° 坡、40m 处竖直偏差约 2.8m（约 14 格），俯仰车辆的
  包围盒也会歪。2026-10 已修复：5 处 `get_matrix()` 和 3 处 `get_world_vertices()` 统一改用
  `processing/transform_utils.py`（UE 公式，已与客户端在仅 yaw 时逐项对齐），但尚未在模拟器坡道场景实测；
  修复前采集的坡道数据需要重采。
- `generate_flow` 的流场是"相对自车速度×dt"且只在动态物体内非零，路面等静止背景的流场为 0
  而不是 −v_ego·dt，这是数据定义问题，网络侧按 `flow_mask` 处理。
- 真值生成器（`ground_truth_voxel_generator.py`）的体素网格与网络约定一致（原点 = 车辆 transform，
  [X,Y,Z] 索引，0 号体素在 −40m，无翻转），但**地面层**的 Z 索引用 `round` 而其余物体用 `floor`，
  地面高度相对体素中心约定低约半格（0.1m）；且地面高度只减去车辆 z 平移，不考虑车身俯仰。
  修改需要重新采集，未改动。
- 数据里没有逐帧时间戳，时间嵌入目前接收的是常数 `frame_dt`，实际只相当于一个可学习偏置；
  若以后采集变步长数据，把真实帧间隔传进 `time_delta` 即可生效。
