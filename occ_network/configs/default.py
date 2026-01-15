import math
class Config:
    image_size = (960, 1280)
    patch_size = 16
    num_cameras = 8
    in_channels = 1
    voxel_size = (400, 400, 32)
    voxel_resolution = 0.2
    pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    num_classes = 18
    class_names = ['empty', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free']
    class_weights = [0.1, 3.0, 12.0, 5.0, 3.0, 8.0, 12.0, 15.0, 10.0, 5.0, 5.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.5]
    embed_dim = 192
    num_heads = 6
    num_encoder_layers = 4
    num_decoder_layers = 3
    window_size = 8
    mlp_ratio = 4.0
    drop_rate = 0.1
    attn_drop_rate = 0.0
    bev_size = (128, 128)
    bev_embed_dim = 192
    num_points = 4
    num_frames = 2  # Already enabled for temporal fusion
    temporal_embed_dim = 192
    use_uncertainty = False
    use_flow = True
    use_sparse = True
    use_coarse_to_fine = True
    coarse_voxel_size = (100, 100, 8)
    chunk_size_z = 10
    sparsity_threshold = 0.1
    use_fp16_input = True
    focal_gamma = 2.0
    focal_alpha = 0.25
    flow_loss_weight = 0.5
    coarse_loss_weight = 0.3
    cameras = {
        'front_main': {'id': 0, 'fov': 50.0, 'position': [1.5, 0.0, 1.5], 'rotation': [0.0, 0.0, 0.0]},
        'front_wide': {'id': 1, 'fov': 120.0, 'position': [1.5, 0.0, 1.5], 'rotation': [0.0, 0.0, 0.0]},
        'front_narrow': {'id': 2, 'fov': 35.0, 'position': [1.5, 0.0, 1.5], 'rotation': [0.0, 0.0, 0.0]},
        'left_pillar': {'id': 3, 'fov': 80.0, 'position': [0.5, 0.9, 1.3], 'rotation': [0.0, 0.0, 55.0]},
        'right_pillar': {'id': 4, 'fov': 80.0, 'position': [0.5, -0.9, 1.3], 'rotation': [0.0, 0.0, -55.0]},
        'left_repeater': {'id': 5, 'fov': 80.0, 'position': [1.0, 1.0, 0.8], 'rotation': [0.0, 0.0, 135.0]},
        'right_repeater': {'id': 6, 'fov': 80.0, 'position': [1.0, -1.0, 0.8], 'rotation': [0.0, 0.0, -135.0]},
        'rear': {'id': 7, 'fov': 80.0, 'position': [-1.5, 0.0, 1.2], 'rotation': [0.0, 0.0, 180.0]},
    }
    batch_size = 1
    num_workers = 0
    max_epochs = 100
    lr = 1e-4
    weight_decay = 0.01
    warmup_epochs = 5
    grad_clip = 1.0
    use_amp = True
    use_checkpoint = True
    save_dir = './checkpoints'
    # Logging
    log_interval = 10
    save_interval = 1
    eval_interval = 1

    # Optimization
    use_mc_dropout = False
    mc_samples = 10
    mc_temperature = 1.0
    sparse_backend = 'auto'  # 'auto' / 'spconv' / 'torchsparse' / 'dense'
config = Config()
