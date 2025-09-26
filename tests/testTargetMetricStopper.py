import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import unittest
import tempfile
import os
from testSuite import mode
from testDataGenerator import TestDataGenerator

class TestTargetMetricStopper(unittest.TestCase):
    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
            
        # Create small synthetic datasets for testing
        self.datasize = 20
        self.datadimensionality = 5
        self.classes = 3
        self.x_train = TestDataGenerator.generate_x_data(self.datadimensionality, self.datasize)
        self.y_train = TestDataGenerator.generate_y_data(self.datasize, self.classes)
        self.x_test = TestDataGenerator.generate_x_data(self.datadimensionality, int(self.datasize / 2))
        self.y_test = TestDataGenerator.generate_y_data(int(self.datasize / 2), self.classes)
        self.labels = range(0, self.classes)
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_target_metric_stopper_initialization(self):
        """Test TargetMetricStopper initialization with default parameters"""
        stopper = gnn.TargetMetricStopper()
        self.assertEqual(stopper.target_value, 1.0)
        self.assertEqual(stopper.metric_name, "accuracy")
        self.assertTrue(stopper.greater_is_better)
        self.assertFalse(stopper.should_stop)
        
    def test_target_metric_stopper_custom_initialization(self):
        """Test TargetMetricStopper initialization with custom parameters"""
        stopper = gnn.TargetMetricStopper(target_value=0.85, metric_name="f1_score", greater_is_better=False)
        self.assertEqual(stopper.target_value, 0.85)
        self.assertEqual(stopper.metric_name, "f1_score")
        self.assertFalse(stopper.greater_is_better)
        self.assertFalse(stopper.should_stop)
        
    def test_check_accuracy_greater_is_better_true(self):
        """Test check method when greater_is_better=True and target is reached"""
        stopper = gnn.TargetMetricStopper(target_value=0.9, metric_name="accuracy", greater_is_better=True)
        
        # Test when target is not reached
        result = stopper.check(0.8)
        self.assertFalse(result)
        self.assertFalse(stopper.should_stop)
        
        # Test when target is reached
        result = stopper.check(0.95)
        self.assertTrue(result)
        self.assertTrue(stopper.should_stop)
        
        # Test when target is exactly met
        stopper2 = gnn.TargetMetricStopper(target_value=0.9, metric_name="accuracy", greater_is_better=True)
        result = stopper2.check(0.9)
        self.assertTrue(result)
        self.assertTrue(stopper2.should_stop)
        
    def test_check_accuracy_greater_is_better_false(self):
        """Test check method when greater_is_better=False and target is reached"""
        stopper = gnn.TargetMetricStopper(target_value=0.1, metric_name="loss", greater_is_better=False)
        
        # Test when target is not reached (loss too high)
        result = stopper.check(0.5)
        self.assertFalse(result)
        self.assertFalse(stopper.should_stop)
        
        # Test when target is reached (loss low enough)
        result = stopper.check(0.05)
        self.assertTrue(result)
        self.assertTrue(stopper.should_stop)
        
        # Test when target is exactly met
        stopper2 = gnn.TargetMetricStopper(target_value=0.1, metric_name="loss", greater_is_better=False)
        result = stopper2.check(0.1)
        self.assertTrue(result)
        self.assertTrue(stopper2.should_stop)
        
    def test_check_with_epoch_parameter(self):
        """Test check method with epoch parameter for logging"""
        stopper = gnn.TargetMetricStopper(target_value=0.9, metric_name="accuracy", greater_is_better=True)
        
        # Test with epoch parameter
        result = stopper.check(0.95, epoch=5)
        self.assertTrue(result)
        self.assertTrue(stopper.should_stop)
        
    def test_check_multiple_calls(self):
        """Test that once should_stop is True, it remains True"""
        stopper = gnn.TargetMetricStopper(target_value=0.9, metric_name="accuracy", greater_is_better=True)
        
        # First call - target reached
        result1 = stopper.check(0.95)
        self.assertTrue(result1)
        self.assertTrue(stopper.should_stop)
        
        # Second call - should still be True
        result2 = stopper.check(0.8)
        self.assertTrue(result2)
        self.assertTrue(stopper.should_stop)
        
    def test_check_edge_cases(self):
        """Test edge cases for the check method"""
        # Test with very high target value
        stopper = gnn.TargetMetricStopper(target_value=0.999, metric_name="accuracy", greater_is_better=True)
        result = stopper.check(0.998)
        self.assertFalse(result)
        self.assertFalse(stopper.should_stop)
        
        # Test with very low target value
        stopper2 = gnn.TargetMetricStopper(target_value=0.001, metric_name="loss", greater_is_better=False)
        result = stopper2.check(0.002)
        self.assertFalse(result)
        self.assertFalse(stopper.should_stop)

    def test_integration_with_trainer_no_stopping(self):
        """Test TargetMetricStopper integration when target is not reached"""
        try:
            # Create a stopper with very high target that won't be reached
            stopper = gnn.TargetMetricStopper(target_value=0.999, metric_name="accuracy", greater_is_better=True)
            
            model = gnn.trainer.train(
                x_train=self.x_train,
                y_train=self.y_train,
                x_test=self.x_test,
                y_test=self.y_test,
                labels=self.labels,
                input_paths=1,
                path=self.temp_dir,
                model_name="test_no_stopping",
                epochs=2,
                generations=3,  # Low number since we won't stop early
                input_size=self.datadimensionality,
                hidden_size=self.datadimensionality,
                output_size=self.classes,
                input_shape=None,
                kernel_size=None,
                batch_size=1,
                simulation_scheduler=gnn.SimulationScheduler(
                    gnn.SimulationScheduler.CONSTANT, 
                    simulation_time=1, 
                    simulation_epochs=1
                ),
                deepth=None,
                simulation_alg=gnn.montecarlo_alg,
                optimizer=gnn.SGDOptimizer(),
                stopper=stopper
            )
            self.assertIsNotNone(model)
            # The stopper should not have been triggered
            self.assertFalse(stopper.should_stop)
        except Exception as e:
            self.fail(f"Training without stopping failed with exception: {e}")
            
    def test_integration_with_convolutional_training(self):
        """Test TargetMetricStopper integration with convolutional training"""
        try:
            # Create convolutional data
            x_conv_train = TestDataGenerator.generate_conv_x_data(self.datasize, self.datadimensionality)
            y_conv_train = TestDataGenerator.generate_y_data(self.datasize, self.classes)
            x_conv_test = TestDataGenerator.generate_conv_x_data(int(self.datasize / 2), self.datadimensionality)
            y_conv_test = TestDataGenerator.generate_y_data(int(self.datasize / 2), self.classes)
            
            stopper = gnn.TargetMetricStopper(target_value=0.3, metric_name="accuracy", greater_is_better=True)
            
            model = gnn.trainer.train(
                x_train=x_conv_train,
                y_train=y_conv_train,
                x_test=x_conv_test,
                y_test=y_conv_test,
                labels=self.labels,
                input_paths=1,
                path=self.temp_dir,
                model_name="test_conv_stopper",
                epochs=2,
                generations=5,
                input_size=self.datadimensionality,
                hidden_size=self.datadimensionality,
                output_size=self.classes,
                input_shape=(self.datadimensionality, self.datadimensionality, 1),
                kernel_size=2,
                batch_size=1,
                simulation_scheduler=gnn.SimulationScheduler(
                    gnn.SimulationScheduler.CONSTANT, 
                    simulation_time=1, 
                    simulation_epochs=1
                ),
                deepth=1,
                simulation_alg=gnn.montecarlo_alg,
                optimizer=gnn.SGDOptimizer(),
                stopper=stopper
            )
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Convolutional training with stopper failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
