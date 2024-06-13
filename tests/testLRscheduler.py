import asyncio
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
import pandas as pd
import random
from unittest.mock import MagicMock

class TestLearningRateScheduler(unittest.TestCase):
    
    def setUp(self):
        mode = getattr(self, 'mode', 'cpu')  # Default to 'cpu' if 'mode' is not set
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
            
    def test_constant_scheduler(self):
        scheduler = gnn.LearningRateScheduler(mode=gnn.LearningRateScheduler.CONSTANT, alpha=0.1)
        alpha = scheduler.alpha_scheduler(i=10, iterations=100)
        self.assertEqual(alpha, 0.1)
        
    def test_progressive_scheduler(self):
        scheduler = gnn.LearningRateScheduler(mode=gnn.LearningRateScheduler.PROGRESIVE, alpha=0.1, steepness=0.2)
        alpha = scheduler.alpha_scheduler(i=10, iterations=100)
        expected = 0.1 * (11 / 22)
        self.assertAlmostEqual(alpha, expected) 
        
    def test_progressive_scheduler_min_max_PROGRESIVE(self):
        alpha_max = 0.1
        scheduler = gnn.LearningRateScheduler(mode=gnn.LearningRateScheduler.PROGRESIVE, alpha=alpha_max, steepness=0.2)
        values = []
        for i in range(0, 100):
            values.append(scheduler.alpha_scheduler(i, 100))
        self.assertTrue(max(values) == alpha_max) 
        self.assertTrue(min(values) <= alpha_max * 0.1) 
        self.assertTrue(min(values) >= 0) 
        
    def test_progressive_scheduler_min_max_PROGRESIVE_PARABOIDAL(self):
        alpha_max = 0.1
        scheduler = gnn.LearningRateScheduler(mode=gnn.LearningRateScheduler.PROGRESIVE_PARABOIDAL, alpha=alpha_max, steepness=0.2)
        values = []
        for i in range(0, 100):
            values.append(scheduler.alpha_scheduler(i, 100))
        print("values: ", values)
        print("min(values): ", min(values))
        print("max(values): ", max(values))
        self.assertTrue(max(values) == alpha_max) 
        self.assertTrue(min(values) <= alpha_max * 0.1) 
        self.assertTrue(min(values) >= 0) 



if __name__ == '__main__':
    unittest.main()
