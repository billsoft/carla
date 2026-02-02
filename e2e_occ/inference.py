import torch
import numpy as np
import argparse
import time
from config import E2EOccConfig
from e2e_occ_net import build_model

class OccInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.config = ckpt.get('config', E2EOccConfig())
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, images):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        if images.dim() == 4:
            images = images.unsqueeze(0)
        images = images.to(self.device)
        outputs = self.model(images)
        logits = outputs['semantic']
        pred = logits.argmax(dim=1)
        return pred.cpu().numpy()
    
    def benchmark(self, num_runs=100, warmup=10):
        dummy = torch.randn(1, self.config.num_cameras, 1, *self.config.image_size).to(self.device)
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(dummy)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model(dummy)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        fps = num_runs / elapsed
        latency = 1000 / fps
        print(f'FPS: {fps:.2f}, Latency: {latency:.2f}ms')
        return fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='output.npz')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    engine = OccInference(args.checkpoint)
    if args.benchmark:
        engine.benchmark()
        return
    if args.input:
        data = np.load(args.input)
        images = data['images']
    else:
        config = E2EOccConfig()
        images = np.random.randn(config.num_cameras, 1, *config.image_size).astype(np.float32)
    pred = engine.predict(images)
    np.savez(args.output, voxels=pred)
    print(f'Saved to {args.output}, shape: {pred.shape}')

if __name__ == '__main__':
    main()
