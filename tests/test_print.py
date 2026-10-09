import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from core.print_utils import print_guidance
from core.queue_manager import validate_options
from test_app import ApiWorkflowTests, sample, wait_until


class PrintGuidanceTests(unittest.TestCase):
    def test_letter_portrait_and_landscape_recommendations(self):
        portrait = print_guidance(1275,1650,'letter',2)
        self.assertEqual(portrait['recommended_scale'],2)
        self.assertEqual((portrait['target_width'],portrait['target_height']),(2550,3300))
        self.assertEqual(portrait['effective_ppi'],300)
        self.assertTrue(portrait['sufficient'])
        landscape = print_guidance(1650,1275,'letter',2)
        self.assertEqual((landscape['target_width'],landscape['target_height']),(3300,2550))
        self.assertFalse(landscape['aspect_mismatch'])

    def test_a4_minimum_scale_and_already_sufficient(self):
        self.assertEqual(print_guidance(1700,2400,'a4')['recommended_scale'],1.5)
        self.assertEqual(print_guidance(2480,3508,'a4')['recommended_scale'],1)
        self.assertTrue(print_guidance(2480,3508,'a4')['sufficient'])
        self.assertIsNone(print_guidance(100,100,'a4')['recommended_scale'])
        self.assertTrue(print_guidance(1000,1000,'a4')['aspect_mismatch'])

    def test_invalid_paper_and_operation(self):
        for options in ({'print_paper':'fake'},{'operation':'delete'},{'small_files':'yes'}):
            with self.assertRaises(ValueError):
                validate_options(options)


class NewWorkflowTests(unittest.TestCase):
    setUpClass = classmethod(ApiWorkflowTests.setUpClass.__func__)
    setUp = ApiWorkflowTests.setUp
    tearDown = ApiWorkflowTests.tearDown
    add = ApiWorkflowTests.add

    def test_compress_only_does_not_invoke_ai_or_change_pixels(self):
        rng=np.random.default_rng(42)
        image=Image.fromarray(rng.integers(0,256,(160,200,3),dtype=np.uint8))
        key=self.add(image,'photo.jpg',{'operation':'compress','small_files':True,'compression':'maximum'})
        with patch.object(self.engine,'upscale',side_effect=AssertionError('AI must not run')):
            self.manager.control('start')
            wait_until(lambda:not self.manager.snapshot()['running'])
        job=self.manager.jobs[key]
        self.assertEqual(job['status'],'Completed',job['error'])
        self.assertLessEqual(job['output_size'],job['original_size'])
        with Image.open(self.manager.job_path(key)) as output:
            self.assertEqual(output.size,(200,160))
        snapshot=self.manager.snapshot()['jobs'][0]
        self.assertEqual((snapshot['output_width'],snapshot['output_height']),(200,160))

    def test_print_recommendations_apply_individually_and_keep_completed(self):
        first=self.add(Image.new('RGB',(1275,1650)),'half-letter.jpg',{})
        second=self.add(Image.new('RGB',(2550,3300)),'full-letter.jpg',{})
        self.manager.control('recommend_print',[first,second],{'print_paper':'letter','model':'light','small_files':True})
        self.assertEqual(self.manager.jobs[first]['options']['scale'],2)
        self.assertEqual(self.manager.jobs[first]['options']['operation'],'upscale')
        self.assertEqual(self.manager.jobs[second]['options']['operation'],'compress')
        self.assertTrue(self.manager.jobs[second]['options']['dpi'])
        self.manager.control('start')
        wait_until(lambda:not self.manager.snapshot()['running'])
        for key in (first,second):
            self.assertEqual(self.manager.jobs[key]['status'],'Completed',self.manager.jobs[key]['error'])
            with Image.open(self.manager.job_path(key)) as image:
                self.assertEqual(image.size,(2550,3300))
                self.assertEqual(image.info['dpi'],(300,300))
        self.assertTrue(all(j['print']['sufficient'] for j in self.manager.snapshot()['jobs']))

    def test_recommendation_failure_does_not_partially_change_queue(self):
        good=self.add(Image.new('RGB',(1275,1650)),'letter.jpg',{})
        tiny=self.add(sample(),'tiny.png',{})
        previous=dict(self.manager.jobs[good]['options'])
        with self.assertRaises(ValueError):
            self.manager.control('recommend_print',[good,tiny],{'print_paper':'letter'})
        self.assertEqual(self.manager.jobs[good]['options'],previous)

    def test_new_scales_api_and_actual_neural_output(self):
        ids=[self.add(sample('RGBA'),'scale.png',{'model':'light','scale':scale}) for scale in (1.5,5,5.5)]
        self.manager.control('start')
        wait_until(lambda:not self.manager.snapshot()['running'])
        for key,size in zip(ids,((36,30),(120,100),(132,110))):
            self.assertEqual(self.manager.jobs[key]['status'],'Completed')
            with Image.open(self.manager.job_path(key)) as image:
                self.assertEqual(image.size,size)
                self.assertEqual(image.getpixel((0,0))[3],0)


del ApiWorkflowTests
