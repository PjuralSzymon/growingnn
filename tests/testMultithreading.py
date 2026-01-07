"""
Unit tests for multithreading configuration in forward propagation.
"""
import sys
import os
import unittest
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import growingnn as gnn
from growingnn.config import config


class TestMultithreading(unittest.TestCase):
    """Tests for multithreading behavior with different MAX_THREADS settings."""

    @classmethod
    def setUpClass(cls):
        """Load the model once for all tests."""
        model_path = os.path.join(os.path.dirname(__file__), 
                                  'resources', 
                                  'GNN_model_0beforeep50_gen20epoch_10save.json')
        cls.model = gnn.Storage.loadModel(model_path)
        cls.model.show_connection_table()
        
        # Save model graph
        result_dir = os.path.join(os.path.dirname(__file__), 'result')
        os.makedirs(result_dir, exist_ok=True)
        gnn.draw(cls.model, os.path.join(result_dir, 'multithreading_test_model.html'))
        
        # Create CIFAR-like test data
        cls.batch_size = 640
        cls.X = np.random.rand(cls.batch_size, 32, 32, 3).astype(np.float64)
        cls.Y = np.random.randint(0, 10, size=(cls.batch_size,))

    def setUp(self):
        self.original_max_threads = config.THREADING_MAX_THREADS

    def tearDown(self):
        config.THREADING_MAX_THREADS = self.original_max_threads

    def _benchmark(self, num_runs=5):
        """Run forward prop and return mean time in ms."""
        times = []
        lr_scheduler = gnn.LearningRateScheduler(gnn.LearningRateScheduler.CONSTANT, 0.01)
        model_copy = self.model.deepcopy()
        for _ in range(num_runs):
            start_time = time.perf_counter()
            model_copy.gradient_descent(self.X, self.Y, 10, lr_scheduler, one_hot_needed=True, quiet=True)
            times.append(time.perf_counter() - start_time)
        return np.mean(times) * 1000, 0

    def test_single_vs_multi_thread(self):
        """Compare MAX_THREADS=1 vs MAX_THREADS=3."""
        repeat_count = 5

        config.update(THREADING_MAX_THREADS = 1)
        print("--------------------------------")
        print("THREADING_MAX_THREADS: ", config.THREADING_MAX_THREADS)
        time_1, result_1 = self._benchmark(repeat_count)
    
        config.update(THREADING_MAX_THREADS = 2)
        print("--------------------------------")
        print("THREADING_MAX_THREADS: ", config.THREADING_MAX_THREADS)
        time_3, result_3 = self._benchmark(repeat_count)
        
        print(f"\nMAX_THREADS=1: {time_1:.1f}ms | MAX_THREADS={config.THREADING_MAX_THREADS}: {time_3:.1f}ms | Speedup: {time_1/time_3:.2f}x")


if __name__ == '__main__':
    unittest.main(verbosity=2)
