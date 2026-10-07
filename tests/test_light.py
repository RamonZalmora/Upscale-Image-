import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import ROOT
from core.compressor import encode_image
from core.image_utils import SCALES, check_dimensions, output_dimensions
from core.queue_manager import DEFAULTS, effective_size_target, validate_options
from core.upscaler import Upscaler, Cancelled, LIGHT_MODEL_SHA256
from test_app import sample, upload, wait_until, ApiWorkflowTests


class LightInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Upscaler(ROOT / 'models')

    def test_real_light_neural_model_all_scales_odd_dimensions(self):
        image = sample(size=(31, 27))
        expected_sizes = [(62,54), (78,68), (93,81), (109,95), (124,108), (140,122)]
        for scale, dimensions in zip(SCALES, expected_sizes):
            with self.subTest(scale=scale):
                result = self.engine.upscale(image, scale, model_name='light')
                self.assertEqual(result.size, dimensions)
                plain = image.resize(result.size, Image.Resampling.LANCZOS)
                self.assertGreater(np.abs(np.asarray(result).astype(float) - np.asarray(plain)).mean(), .2)
        self.assertIsNone(self.engine.model, 'Light model must not load the heavy Real-ESRGAN network')
        self.assertEqual(self.engine.device, 'cpu')

    def test_light_alpha_and_icc_preserved(self):
        image = sample('RGBA', size=(31, 27))
        image.info['icc_profile'] = b'profile-test'
        result = self.engine.upscale(image, 4.5, model_name='light')
        expected = image.getchannel('A').resize(result.size, Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(np.asarray(result.getchannel('A')), np.asarray(expected))
        self.assertEqual(result.info['icc_profile'], b'profile-test')

    def test_light_tiles_match_and_threads_bounded(self):
        import cv2
        self.engine.upscale(sample(), model_name='light')
        rgb = np.asarray(sample(size=(97, 85)))
        full = self.engine.light.upscale(rgb, lambda p: None, lambda: None, tile=192)
        tiled = self.engine.light.upscale(rgb, lambda p: None, lambda: None, tile=24)
        self.assertLessEqual(np.abs(full.astype(int) - tiled.astype(int)).max(), 1)
        self.assertLessEqual(cv2.getNumThreads(), 2)

    def test_light_cancel(self):
        def cancel():
            raise Cancelled()
        with self.assertRaises(Cancelled):
            self.engine.upscale(sample(), model_name='light', checkpoint=cancel)

    def test_light_cache_verified_and_corrupt_download_rejected(self):
        self.assertEqual(self.engine._hash(ROOT / 'models/FSRCNN_x2.pb'), LIGHT_MODEL_SHA256)
        with patch('urllib.request.urlopen', side_effect=AssertionError('Do not redownload cached model')):
            self.engine.ensure_model(model_name='light')
        with tempfile.TemporaryDirectory() as folder:
            response = io.BytesIO(b'bad model')
            response.headers = {'Content-Length': '9'}
            with patch('urllib.request.urlopen', return_value=response):
                with self.assertRaisesRegex(RuntimeError, 'checksum'):
                    Upscaler(folder).ensure_model(model_name='light')
            self.assertFalse(list(Path(folder).glob('*.pb')))
            self.assertFalse(list(Path(folder).glob('*.part')))

    def test_switch_between_models(self):
        engine = Upscaler(ROOT / 'models')
        for name in ('detail', 'light', 'detail'):
            result = engine.upscale(sample(), 2.5, 'fast', model_name=name)
            self.assertEqual(result.size, (60, 50))
        self.assertEqual(engine.device, next(engine.model.parameters()).device.type)

    def test_scale_limits_account_for_final_and_native_resolution(self):
        self.assertEqual(output_dimensions(25, 21, 2.5), (63, 53))
        self.assertEqual(check_dimensions(4000, 4000, 2, 'light'), (8000, 8000))
        with self.assertRaises(ValueError):
            check_dimensions(4000, 4000, 2, 'detail')
        with self.assertRaises(ValueError):
            check_dimensions(2400, 2400, 4.5, 'light')
        for scale in (True, 1, 2.2, float('nan')):
            with self.assertRaises(ValueError):
                validate_options({'scale': scale})


class EfficientExportTests(unittest.TestCase):
    def test_large_jpeg_3_5x_keeps_pixel_scale_and_small_file(self):
        rng = np.random.default_rng(23)
        pixels = rng.integers(0, 256, (1000, 1250, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, 'JPEG', quality=95)
        source = buf.getvalue()
        self.assertGreater(len(source), 1.3 * 1024**2)
        self.assertLess(len(source), 1.6 * 1024**2)
        options = validate_options({'model':'light','scale':3.5,'compress':True,'auto_size':True})
        with Image.open(io.BytesIO(source)) as image:
            result = Upscaler(ROOT / 'models').upscale(image, 3.5, model_name='light')
        data, warning = encode_image(result, 'JPEG', compress=True,
                                     target_mb=effective_size_target(options, len(source)), profile='efficient')
        self.assertLessEqual(len(data), 4 * 1024**2)
        with Image.open(io.BytesIO(data)) as output:
            self.assertEqual(output.size, (4375, 3500))
        if len(data) > effective_size_target(options, len(source)) * 1024**2:
            self.assertTrue(warning)

    def test_auto_target_and_manual_override(self):
        options = validate_options({'model': 'light', 'compress': True, 'auto_size': True})
        self.assertEqual(effective_size_target(options, int(1.5 * 1024**2)), 3.75)
        self.assertEqual(effective_size_target(options, 8 * 1024**2), 4)
        options['target_mb'] = 2
        self.assertEqual(effective_size_target(options, 8 * 1024**2), 2)
        options['compress'] = False
        self.assertIsNone(effective_size_target(options, 8 * 1024**2))

    def test_efficient_jpeg_smaller_and_highest_quality_below_target(self):
        rng = np.random.default_rng(9)
        image = Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8))
        standard, _ = encode_image(image, 'JPEG', compress=True)
        efficient, _ = encode_image(image, 'JPEG', compress=True, profile='efficient')
        self.assertLess(len(efficient), len(standard))
        floor, _ = encode_image(image, 'JPEG', compress=True, profile='efficient', target_mb=.000001)
        target = (len(efficient) + len(floor)) // 2
        result, warning = encode_image(image, 'JPEG', dpi=True, compress=True, profile='efficient', target_mb=target/1024**2)
        self.assertLessEqual(len(result), target)
        self.assertFalse(warning)
        with Image.open(io.BytesIO(result)) as decoded:
            self.assertEqual(decoded.size, image.size)
            self.assertEqual(decoded.info['dpi'], (300, 300))
            from PIL.JpegImagePlugin import get_sampling
            self.assertEqual(get_sampling(decoded), 2)

    def test_efficient_png_lossless_and_warning_when_cap_impossible(self):
        image = sample('RGBA')
        data, warning = encode_image(image, 'PNG', compress=True, profile='efficient', target_mb=.000001)
        with Image.open(io.BytesIO(data)) as decoded:
            np.testing.assert_array_equal(np.asarray(decoded), np.asarray(image))
        self.assertTrue(warning)


# Reuse API fixtures without importing its TestCase into this module's discovery namespace.
class LightApiWorkflowTests(unittest.TestCase):
    setUpClass = classmethod(ApiWorkflowTests.setUpClass.__func__)
    setUp = ApiWorkflowTests.setUp
    tearDown = ApiWorkflowTests.tearDown
    add = ApiWorkflowTests.add

    def test_light_fractional_queue_zip_and_original_size_target(self):
        key = self.add(sample(), 'light.jpg', {'model': 'light', 'scale': 3.5, 'compress': True, 'auto_size': True})
        alpha = self.add(sample('RGBA'), 'alpha.png', {'model': 'light', 'scale': 4.5, 'dpi': True})
        self.manager.control('start')
        wait_until(lambda: not self.manager.snapshot()['running'])
        jobs = {j['id']: j for j in self.manager.snapshot()['jobs']}
        self.assertEqual(jobs[key]['status'], 'Completed', jobs[key]['error'])
        self.assertEqual((jobs[key]['output_width'], jobs[key]['output_height']), (84, 70))
        self.assertEqual(jobs[alpha]['status'], 'Completed', jobs[alpha]['error'])
        with Image.open(self.manager.job_path(key)) as result:
            self.assertEqual(result.size, (84, 70))
        self.manager.close()
        from core.queue_manager import QueueManager
        self.manager = QueueManager(self.temp.name, self.engine)
        self.assertEqual(self.manager.jobs[key]['options']['model'], 'light')
        self.assertEqual(self.manager.jobs[key]['options']['scale'], 3.5)

    def test_legacy_queue_receives_new_defaults(self):
        key = self.add(sample(), 'legacy.jpg', {'quality': 'fast'})
        legacy_options = {k: v for k, v in self.manager.jobs[key]['options'].items() if k not in ('model', 'auto_size')}
        self.manager._update(self.manager.jobs[key], options=legacy_options)
        self.manager.close()
        from core.queue_manager import QueueManager
        self.manager = QueueManager(self.temp.name, self.engine)
        self.assertEqual(self.manager.jobs[key]['options']['model'], 'detail')
        self.assertFalse(self.manager.jobs[key]['options']['auto_size'])
        self.manager.control('start')
        wait_until(lambda: not self.manager.snapshot()['running'])
        self.assertEqual(self.manager.jobs[key]['status'], 'Completed')


# unittest discovers imported TestCases too; only expose this module's own classes.
del ApiWorkflowTests
