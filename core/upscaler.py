"""Real-ESRGAN SRVGGNetCompact, native 4x inference with padded tiles.
Architecture follows https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause).
No synthetic or resize-only fallback is used.
"""
import hashlib
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
MODEL_SHA256 = '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292'


class Cancelled(Exception):
    pass


class Upscaler:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = None
        self.device = self.detect_device()
        self.download_progress = 0
        self.lock = threading.Lock()

    @staticmethod
    def detect_device():
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def ensure_model(self, progress=lambda p: None, checkpoint=lambda: None):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        target = self.model_dir / 'realesr-general-x4v3.pth'
        if target.exists() and self._hash(target) == MODEL_SHA256:
            self.download_progress = 100
            return target
        part = target.with_suffix('.part')
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response, part.open('wb') as stream:
                total = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                while True:
                    checkpoint()
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    downloaded += len(block)
                    self.download_progress = min(99, int(downloaded * 100 / total)) if total else 0
                    progress(self.download_progress)
            if self._hash(part) != MODEL_SHA256:
                raise RuntimeError('Model checksum mismatch. Download rejected.')
            os.replace(part, target)
            self.download_progress = 100
            return target
        finally:
            part.unlink(missing_ok=True)

    @staticmethod
    def _hash(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def load(self, progress=lambda p: None, checkpoint=lambda: None):
        if self.model is not None:
            return
        import torch
        from torch import nn
        checkpoint()
        path = self.ensure_model(progress, checkpoint)
        # The published compact model: 64 channels, 32 internal convolutions, PReLU.
        class SRVGGNetCompact(nn.Module):
            def __init__(self):
                super().__init__()
                layers = [nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(num_parameters=64)]
                for _ in range(32):
                    layers.extend([nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(num_parameters=64)])
                layers.append(nn.Conv2d(64, 3 * 16, 3, 1, 1))
                self.body = nn.ModuleList(layers)
                self.upsampler = nn.PixelShuffle(4)

            def forward(self, x):
                out = x
                for layer in self.body:
                    out = layer(out)
                return self.upsampler(out) + nn.functional.interpolate(x, scale_factor=4, mode='nearest')

        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.model = SRVGGNetCompact()
        weights = torch.load(path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights.get('params_ema', weights.get('params', weights)), strict=True)
        self.model.eval().to(self.device)

    def tile_size(self):
        if self.device == 'cuda':
            import torch
            free, _ = torch.cuda.mem_get_info()
            return 384 if free > 6 * 1024**3 else 256 if free > 3 * 1024**3 else 128
        return 128 if self.device == 'cpu' else 192

    def upscale(self, image, scale=2, quality='high', progress=lambda p: None, checkpoint=lambda: None):
        import torch
        if scale not in (2, 3, 4) or quality not in ('fast', 'balanced', 'high'):
            raise ValueError('Invalid upscale settings')
        with self.lock:
            self.load(lambda p: progress(p * .05), checkpoint)
            width, height = image.size
            if width * height * 16 > 100_000_000:
                raise ValueError('Native 4x output exceeds 100 megapixels; split this image first.')
            rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
            tile = self.tile_size()
            while True:
                try:
                    result = self._tiles(rgb, tile, quality, progress, checkpoint)
                    break
                except (torch.OutOfMemoryError, MemoryError):
                    if self.device == 'cpu':
                        if tile <= 32:
                            raise RuntimeError('Insufficient RAM for this image')
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    if tile > 32:
                        tile //= 2
                    elif self.device != 'cpu':
                        self.device = 'cpu'
                        self.model.to('cpu')
                        tile = 64
            output = Image.fromarray(result)
            # Native neural 4x output is reduced for exact 2x/3x dimensions.
            if scale != 4:
                output = output.resize((width * scale, height * scale), Image.Resampling.LANCZOS)
            if 'A' in image.getbands() or 'transparency' in image.info:
                # Alpha is coverage, not texture; deterministic resampling avoids AI hallucinations.
                alpha = image.convert('RGBA').getchannel('A').resize(output.size, Image.Resampling.LANCZOS)
                output.putalpha(alpha)
            if image.info.get('icc_profile'):
                output.info['icc_profile'] = image.info['icc_profile']
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            return output

    def _tiles(self, rgb, tile, quality, progress, checkpoint):
        import torch
        height, width = rgb.shape[:2]
        result = np.empty((height * 4, width * 4, 3), dtype=np.uint8)
        count = ((height + tile - 1) // tile) * ((width + tile - 1) // tile)
        passes = [(False, False)]
        if quality != 'fast':
            passes.append((True, False))
        if quality == 'high':
            passes += [(False, True), (True, True)]
        done = 0
        # Padding exceeds the network's 34-pixel receptive radius.
        pad = 40
        with torch.inference_mode():
            for y in range(0, height, tile):
                for x in range(0, width, tile):
                    checkpoint()
                    y1, x1 = min(y + tile, height), min(x + tile, width)
                    py, px = max(0, y - pad), max(0, x - pad)
                    patch = rgb[py:min(height, y1 + pad), px:min(width, x1 + pad)]
                    tensor = torch.from_numpy(patch.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    accum = None
                    for horizontal, vertical in passes:
                        checkpoint()
                        dims = ([3] if horizontal else []) + ([2] if vertical else [])
                        prediction = self.model(torch.flip(tensor, dims) if dims else tensor)
                        if dims:
                            prediction = torch.flip(prediction, dims)
                        prediction = prediction[:, :, (y-py)*4:(y1-py)*4, (x-px)*4:(x1-px)*4]
                        accum = prediction.clone() if accum is None else accum.add_(prediction)
                        del prediction
                    pixels = accum.div_(len(passes)).clamp_(0, 1).squeeze(0).permute(1, 2, 0)
                    result[y*4:y1*4, x*4:x1*4] = pixels.mul(255).round().byte().cpu().numpy()
                    del tensor, accum, pixels
                    done += 1
                    progress(5 + 85 * done / count)
        return result
