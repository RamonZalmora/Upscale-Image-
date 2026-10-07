"""Small FSRCNN neural model, CPU-only with bounded threads and padded tiles."""
import os
import numpy as np


class LightUpscaler:
    def __init__(self):
        self.model = None

    def load(self, path):
        if self.model is not None:
            return
        import cv2
        cv2.setNumThreads(min(2, os.cpu_count() or 1))
        model = cv2.dnn_superres.DnnSuperResImpl_create()
        model.readModel(str(path))
        model.setModel('fsrcnn', 2)
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model = model

    def upscale(self, rgb, progress, checkpoint, tile=192):
        height, width = rgb.shape[:2]
        output = np.empty((height * 2, width * 2, 3), dtype=np.uint8)
        total = ((height + tile - 1) // tile) * ((width + tile - 1) // tile)
        done = 0
        pad = 16
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                checkpoint()
                y1, x1 = min(height, y + tile), min(width, x + tile)
                py, px = max(0, y - pad), max(0, x - pad)
                patch = rgb[py:min(height, y1 + pad), px:min(width, x1 + pad)]
                # OpenCV super-resolution expects BGR, with a neural luminance pipeline.
                bgr = np.ascontiguousarray(patch[:, :, ::-1])
                result = self.model.upsample(bgr)
                output[y*2:y1*2, x*2:x1*2] = result[(y-py)*2:(y1-py)*2, (x-px)*2:(x1-px)*2, ::-1]
                done += 1
                progress(5 + 85 * done / total)
        return output
