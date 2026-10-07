import io
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage

from app import ROOT, create_app
from core.compressor import encode_image
from core.image_utils import reserve_output, inspect_image
from core.queue_manager import QueueManager
from core.upscaler import Upscaler, Cancelled


def sample(mode='RGB', size=(24, 20)):
    y, x = np.indices((size[1], size[0]))
    data = np.stack([(x * 11) % 256, (y * 13) % 256, ((x + y) * 7) % 256], axis=2).astype('uint8')
    image = Image.fromarray(data)
    if mode == 'RGBA':
        alpha = np.where(x < size[0] // 2, 0, 200).astype('uint8')
        image.putalpha(Image.fromarray(alpha))
    return image


def upload(image=None, name='asset.png', raw=None):
    stream = io.BytesIO()
    if raw is not None:
        stream.write(raw)
    else:
        (image or sample()).save(stream, 'JPEG' if name.endswith('.jpg') else 'PNG')
    stream.seek(0)
    return FileStorage(stream, filename=name)


def wait_until(predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.03)
    raise AssertionError('Timed out waiting for worker')


class RealInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Upscaler(ROOT / 'models')
        cls.engine.load()
        # Keep numerical and tile-count comparisons deterministic on GPU-equipped hosts.
        cls.engine.device = 'cpu'
        cls.engine.model.to('cpu')

    def test_real_neural_inference_all_scales_and_modes(self):
        image = sample()
        for scale in (2, 3, 4):
            for quality in ('superfast', 'fast', 'balanced', 'high'):
                with self.subTest(scale=scale, quality=quality):
                    result = self.engine.upscale(image, scale, quality)
                    self.assertEqual(result.size, (24 * scale, 20 * scale))
                    plain = image.resize(result.size, Image.Resampling.LANCZOS)
                    self.assertGreater(np.abs(np.array(result).astype(float)-np.array(plain)).mean(), 0.5)
                    self.assertEqual(result.mode, 'RGB')

    def test_alpha_preserved(self):
        image = sample('RGBA')
        result = self.engine.upscale(image, 3, 'high')
        expected = image.getchannel('A').resize(result.size, Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(np.asarray(result.getchannel('A')), np.asarray(expected))
        self.assertEqual(result.getpixel((0, 0))[3], 0)

    def test_tiles_match_full_inference(self):
        image = sample(size=(101, 95))
        pixels = np.asarray(image, dtype=np.float32)/255
        for quality in ('fast', 'high'):
            full = self.engine._tiles(pixels, 256, quality, lambda p: None, lambda: None)
            tiled = self.engine._tiles(pixels, 32, quality, lambda p: None, lambda: None)
            delta = np.abs(full.astype(int)-tiled.astype(int))
            self.assertLessEqual(delta.max(), 1, f'Tile boundary error: {delta.max()}')

    def test_superfast_reduces_tile_work_without_losing_detail(self):
        image = sample(size=(160, 160))
        counts = []
        outputs = []
        for quality in ('fast', 'superfast'):
            calls = []
            hook = self.engine.model.register_forward_hook(lambda *args: calls.append(1))
            try:
                outputs.append(self.engine.upscale(image, 2, quality))
                counts.append(len(calls))
            finally:
                hook.remove()
        self.assertEqual(counts, [4, 1])
        delta = np.abs(np.asarray(outputs[0]).astype(int) - np.asarray(outputs[1]).astype(int))
        self.assertLessEqual(delta.max(), 1)

    def test_superfast_retries_smaller_tiles_on_oom(self):
        import torch
        original = self.engine._tiles
        calls = []
        def simulate_oom(rgb, tile, quality, progress, checkpoint):
            calls.append(tile)
            if len(calls) == 1:
                raise torch.OutOfMemoryError('Test allocation failure')
            return original(rgb, tile, quality, progress, checkpoint)
        with patch.object(self.engine, '_tiles', side_effect=simulate_oom):
            result = self.engine.upscale(sample(), 3, 'superfast')
        self.assertEqual(calls, [256, 128])
        self.assertEqual(result.size, (72, 60))

    def test_cancel_checkpoint(self):
        def cancel():
            raise Cancelled()
        with self.assertRaises(Cancelled):
            self.engine.upscale(sample(), checkpoint=cancel)

    def test_official_weight_checksum(self):
        self.assertEqual(self.engine._hash(ROOT / 'models/realesr-general-x4v3.pth'),
                         '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292')


    def test_model_cache_and_bad_download(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('Cache should skip download')):
            self.engine.ensure_model()
        with tempfile.TemporaryDirectory() as folder:
            engine = Upscaler(folder)
            response = io.BytesIO(b'corrupted weights')
            response.headers = {'Content-Length':'17'}
            with patch('urllib.request.urlopen', return_value=response):
                with self.assertRaisesRegex(RuntimeError, 'checksum'):
                    engine.ensure_model()
            self.assertFalse(list(Path(folder).glob('*.part')))
            self.assertFalse(list(Path(folder).glob('*.pth')))


class ExportTests(unittest.TestCase):
    def test_formats_and_dpi(self):
        for fmt in ('JPEG', 'PNG', 'WEBP'):
            with self.subTest(format=fmt):
                data, _ = encode_image(sample('RGBA') if fmt != 'JPEG' else sample(), fmt, dpi=True)
                with Image.open(io.BytesIO(data)) as result:
                    self.assertEqual(result.format, fmt)
                    self.assertEqual(result.size, (24, 20))
                    if fmt == 'WEBP':
                        self.assertEqual(result.getexif()[282], 300)
                        self.assertEqual(result.getexif()[296], 2)
                    else:
                        self.assertAlmostEqual(result.info['dpi'][0], 300, delta=.1)
                    if fmt != 'JPEG':
                        self.assertEqual(result.mode, 'RGBA')
                        self.assertEqual(result.getpixel((0, 0))[3], 0)

    def test_lossless_png_and_target_warning(self):
        image = sample('RGBA', (80, 80))
        data, warning = encode_image(image, 'PNG', compress=True, target_mb=.00001)
        with Image.open(io.BytesIO(data)) as decoded:
            np.testing.assert_array_equal(np.asarray(image), np.asarray(decoded))
        self.assertTrue(warning)

    def test_smart_jpeg_respects_target_and_quality_floor(self):
        rng = np.random.default_rng(17)
        image = Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8))
        high, _ = encode_image(image, 'JPEG', compress=True, level='balanced')
        floor, _ = encode_image(image, 'JPEG', compress=True, level='balanced', target_mb=.00001)
        target = (len(high)+len(floor))/2
        result, warning = encode_image(image, 'JPEG', compress=True, target_mb=target/1024**2)
        self.assertLessEqual(len(result), target)
        self.assertFalse(warning)
        self.assertGreater(len(floor), 0)
        _, warning = encode_image(image, 'JPEG', compress=True, target_mb=.00001)
        self.assertTrue(warning)

    def test_transparent_jpeg_rejected(self):
        with self.assertRaises(ValueError):
            encode_image(sample('RGBA'), 'JPEG')

    def test_exif_orientation_and_reduced_thumbnail(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'rotated.jpg'
            thumb = Path(folder) / 'thumb.png'
            exif = Image.Exif()
            exif[274] = 6
            sample(size=(400, 300)).save(source, exif=exif)
            width, height, fmt, alpha = inspect_image(source, thumb)
            self.assertEqual((width, height, fmt, alpha), (300, 400, 'JPEG', False))
            with Image.open(thumb) as preview:
                self.assertLessEqual(preview.width, 144)
                self.assertLessEqual(preview.height, 100)

    def test_collision_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            first = reserve_output(folder, 'asset', 'png')
            first.write_bytes(b'original')
            second = reserve_output(folder, 'asset', 'png')
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b'original')


class ApiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Upscaler(ROOT / 'models')
        cls.engine.load()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app(self.temp.name, self.engine)
        self.client = self.app.test_client()
        self.headers = {'X-App-Token':self.app.extensions['csrf_token']}
        self.manager = self.app.extensions['queue']

    def tearDown(self):
        self.manager.close()
        import logging
        for handler in list(logging.getLogger().handlers):
            if getattr(handler, 'baseFilename', '').startswith(self.temp.name):
                logging.getLogger().removeHandler(handler)
                handler.close()
        self.temp.cleanup()

    def add(self, image, name, options):
        file = upload(image, name)
        import json
        response = self.client.post('/api/upload', data={'images':(file.stream, name), 'options':json.dumps(options)}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.data)
        return response.json['ids'][0]

    def test_end_to_end_batch_download_zip_recovery(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        png = self.add(sample('RGBA'), 'clipart.png', {'scale':2, 'dpi':True, 'compress':True, 'quality':'high'})
        jpg = self.add(sample(), 'printable.jpg', {'scale':3, 'dpi':True, 'quality':'balanced', 'base_name':'collection'})
        webp = self.add(sample('RGBA'), 'element.png', {'scale':4, 'format':'WEBP', 'dpi':True, 'quality':'superfast'})
        invalid = self.manager.add(upload(name='corrupt.png', raw=b'not an image'), {})
        rejected = self.add(sample('RGBA'), 'transparent.png', {'format':'JPEG'})
        after_error = self.add(sample(), 'after-error.png', {'quality':'fast'})
        response = self.client.post('/api/control', json={'action':'start'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        wait_until(lambda:not self.manager.snapshot()['running'])
        jobs = {j['id']:j for j in self.manager.snapshot()['jobs']}
        self.assertEqual(jobs[invalid]['status'], 'Failed')
        self.assertEqual(jobs[rejected]['status'], 'Failed')
        self.assertEqual(jobs[after_error]['status'], 'Completed')
        for key, shape, fmt in [(png,(48,40),'PNG'),(jpg,(72,60),'JPEG'),(webp,(96,80),'WEBP')]:
            self.assertEqual(jobs[key]['status'], 'Completed', jobs[key]['error'])
            response = self.client.get(f'/api/download/{key}')
            self.assertEqual(response.status_code, 200)
            with Image.open(io.BytesIO(response.data)) as image:
                self.assertEqual(image.size, shape)
                self.assertEqual(image.format, fmt)
                if fmt == 'PNG':
                    self.assertAlmostEqual(image.info['dpi'][0], 300, delta=.1)
                    self.assertEqual(image.getpixel((0,0))[3], 0)
            response.close()
        thumbnail_response = self.client.get(f'/api/thumbnail/{png}')
        self.assertEqual(thumbnail_response.status_code, 200)
        thumbnail_response.close()
        self.assertEqual(self.client.post('/api/zip', json={}, headers=self.headers).status_code, 200)
        wait_until(lambda:self.manager.zip_state['status'] != 'Processing')
        response = self.client.get('/api/zip/download')
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            self.assertEqual(len(archive.namelist()), 4)
            self.assertTrue(all(Path(n).suffix in ('.png','.jpg','.webp') for n in archive.namelist()))
            self.assertIn('collection-002.jpg',archive.namelist())
        response.close()
        self.manager.close()
        self.manager = QueueManager(self.temp.name, self.engine)
        self.assertEqual(self.manager.snapshot()['completed'], 4)
        path=self.manager.job_path(png)
        self.manager.control('clear_completed')
        self.assertTrue(path.exists(), 'Clear Completed must preserve production output')

    def test_local_action_protection_and_host_check(self):
        response = self.client.post('/api/control', json={'action':'start'})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get('/',headers={'Host':'evil.example'}).status_code,400)
        self.assertEqual(self.client.get('/api/download/missing').status_code, 400)
        self.assertEqual(self.client.post('/api/upload',headers=self.headers).status_code, 400)

    def test_more_than_200_images_real_worker(self):
        ids = [self.manager.add(upload(sample(size=(8, 8)), f'asset-{i}.png'), {'quality':'fast'}) for i in range(205)]
        self.manager.control('start')
        wait_until(lambda:not self.manager.snapshot()['running'], timeout=120)
        self.assertEqual(self.manager.snapshot()['completed'], 205)
        self.assertTrue(all(self.manager.jobs[key]['status'] == 'Completed' for key in ids))

    def test_settings_validation(self):
        file=upload()
        response=self.client.post('/api/upload',data={'images':(file.stream,'a.png'),'options':'{"scale":9}'},headers=self.headers)
        self.assertEqual(response.status_code,400)
        self.assertEqual(self.manager.snapshot()['total'],0)

    def test_delete_only_selected_output(self):
        key=self.add(sample(), 'asset.png',{'quality':'fast'})
        self.manager.control('start')
        wait_until(lambda:not self.manager.snapshot()['running'])
        path=self.manager.job_path(key)
        response=self.client.post(f'/api/delete/{key}',json={},headers=self.headers)
        self.assertEqual(response.status_code,200)
        self.assertFalse(path.exists())
        self.assertEqual(self.manager.jobs[key]['status'],'Cancelled')


class ControlledEngine:
    """A controllable worker fixture for queue concurrency, not an AI validation substitute."""
    device='cpu'
    download_progress=100
    def __init__(self):
        self.entered=threading.Event()
        self.release=threading.Event()
    def upscale(self,image,scale,quality,progress,checkpoint):
        self.entered.set()
        while not self.release.wait(.01):
            checkpoint()
            progress(20)
        checkpoint()
        return image.copy()


class QueueControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.engine=ControlledEngine()
        self.manager=QueueManager(self.temp.name,self.engine)
    def tearDown(self):
        self.engine.release.set()
        self.manager.close()
        self.temp.cleanup()
    def add(self,name='a.png'):
        return self.manager.add(upload(name=name),{})
    def test_pause_resume_cancel_and_add_during_work(self):
        first=self.add()
        self.manager.control('start')
        self.assertTrue(self.engine.entered.wait(5))
        self.manager.control('pause')
        second=self.add('b.png')
        self.assertEqual(self.manager.jobs[second]['status'],'Waiting')
        self.manager.control('cancel_current')
        wait_until(lambda:self.manager.jobs[first]['status']=='Cancelled')
        self.assertTrue(self.manager.snapshot()['paused'])
        self.engine.release.set()
        self.manager.control('resume')
        wait_until(lambda:self.manager.jobs[second]['status']=='Completed')
        self.manager.control('retry',[first])
        self.manager.control('start')
        wait_until(lambda:self.manager.jobs[first]['status']=='Completed')
    def test_apply_remove_cancel_selected_and_persistent_sequence(self):
        first=self.add()
        second=self.add()
        self.manager.control('apply',[first],{'scale':3,'base_name':'new'})
        self.assertEqual(self.manager.jobs[first]['options']['scale'],3)
        self.manager.control('cancel',[first])
        self.assertEqual(self.manager.jobs[first]['status'],'Cancelled')
        self.manager.control('remove',[second])
        self.assertNotIn(second,self.manager.jobs)
        self.manager.control('clear')
        self.manager.close()
        self.manager=QueueManager(self.temp.name,self.engine)
        third=self.add()
        self.assertEqual(self.manager.jobs[third]['sequence'],3)
    def test_restart_recovers_interrupted_job(self):
        key=self.add()
        self.manager._update(self.manager.jobs[key],status='Upscaling',progress=50)
        self.manager.close()
        self.manager=QueueManager(self.temp.name,self.engine)
        self.assertEqual(self.manager.jobs[key]['status'],'Waiting')
        self.assertEqual(self.manager.jobs[key]['progress'],0)
    def test_remove_active_cancels_before_deleting_source(self):
        key=self.add()
        self.manager.control('start')
        self.assertTrue(self.engine.entered.wait(5))
        self.manager.control('remove',[key])
        wait_until(lambda:key not in self.manager.jobs)
        self.assertIsNone(self.manager.active)


if __name__ == '__main__':
    unittest.main()
