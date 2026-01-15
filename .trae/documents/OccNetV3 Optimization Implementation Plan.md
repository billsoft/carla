# OccNetV3 优化实施计划：MC Dropout 不确定性 + Torchsparse 稀疏后端

根据您的详细评估与重构指南，我制定了以下实施计划。本计划将严格按照您提供的"最完整重构计划"执行，优先实现 MC Dropout，随后集成 Torchsparse，并确保所有改动均通过配置开关控制，保证系统稳定性。

## 1. 配置更新 (`configs/default.py`)

*   **目标**: 添加控制新功能的配置项。
*   **改动**: 在 `Config` 类中添加以下参数：
    *   `use_mc_dropout = False`: 推理不确定性总开关
    *   `mc_samples = 10`: MC Dropout 采样次数
    *   `mc_temperature = 1.0`: Softmax 温度系数
    *   `sparse_backend = 'auto'`: 稀疏后端选择 ('auto' / 'spconv' / 'torchsparse' / 'dense')

## 2. MC Dropout 不确定性实现 (`inference.py`)

*   **目标**: 实现轻量级不确定性估计，无需重新训练。
*   **核心逻辑**:
    *   新增 `inference_with_uncertainty(model, images, num_samples, temperature)` 函数。
    *   使用 `model.train()` 强制激活 Dropout。
    *   执行多次前向传播，收集 Logits。
    *   应用温度 Scaling 并计算 Softmax 概率。
    *   计算 **均值概率** (用于最终预测) 和 **方差/熵** (用于不确定性)。
*   **接口**: 更新 `main` 函数和 `argparse`，添加 `--uncertainty`, `--mc-samples`, `--mc-temp` 命令行参数。

## 3. Torchsparse 稀疏后端集成 (`models/sparse_modules.py` & `occ_net.py`)

*   **目标**: 解除对 spconv 的强依赖，提供更易安装的替代方案。
*   **`models/sparse_modules.py` 改动**:
    *   **导入逻辑**: 尝试导入 `spconv`，失败则尝试 `torchsparse`，最后回退到 `dense`。
    *   **`SparseConvBlock`**:
        *   根据 `SPCONV_AVAILABLE` 和 `TORCHSPARSE_AVAILABLE` 以及 `config.sparse_backend` 选择后端。
        *   实现 `torchsparse` 分支：使用 `tsnn.Conv3d` 和 `tsnn.BatchNorm`。
    *   **`DenseToSparse` / `SparseToDense`**:
        *   适配 `torchsparse.SparseTensor` 的构造方式（注意坐标格式差异）。
*   **`models/occ_net.py` 改动**:
    *   更新 `self.use_sparse` 判断逻辑，使其在 `TORCHSPARSE_AVAILABLE` 为真时也能启用稀疏模式。

## 4. 验证与测试

*   **MC Dropout**: 运行推理脚本带 `--uncertainty` 参数，检查输出结果中是否包含方差图，并验证推理速度变化。
*   **Torchsparse**: 在无 `spconv` 环境（或通过配置强制指定）下运行，验证模型是否能正常加载并进行前向传播。

此计划将分步执行，首先完成配置和 MC Dropout（低风险、高收益），然后进行 Torchsparse 的底层适配。
