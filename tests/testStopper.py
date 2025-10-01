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

class TestStoppers(unittest.TestCase):
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
        
        # Create a simple model for testing
        self.test_model = gnn.Model(
            input_size=self.datadimensionality,
            hidden_size=self.datadimensionality,
            output_size=self.classes
        )
        
    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_stopper_initialization(self):
        stopper = gnn.EmptyStopper()
        self.assertFalse(stopper.should_stop)
    
    def test_empty_stopper_reset(self):
        stopper = gnn.EmptyStopper()
        stopper.should_stop = True
        stopper.reset()
        self.assertFalse(stopper.should_stop)
    
    # AccuracyStopper Tests
    def test_accuracy_stopper_initialization(self):
        """Test AccuracyStopper initialization with default parameters"""
        stopper = gnn.AccuracyStopper()
        self.assertEqual(stopper.target_accuracy, 0.9)
        self.assertEqual(stopper.metric_name, "accuracy")
        self.assertFalse(stopper.should_stop)
        
    def test_accuracy_stopper_custom_initialization(self):
        """Test AccuracyStopper initialization with custom parameters"""
        stopper = gnn.AccuracyStopper(target_accuracy=0.95, metric_name="f1_score")
        self.assertEqual(stopper.target_accuracy, 0.95)
        self.assertEqual(stopper.metric_name, "f1_score")
        self.assertFalse(stopper.should_stop)
        
    def test_accuracy_stopper_check_below_target(self):
        """Test AccuracyStopper when accuracy is below target"""
        stopper = gnn.AccuracyStopper(target_accuracy=0.9)
        
        # Mock model.evaluate to return low accuracy
        original_evaluate = self.test_model.evaluate
        self.test_model.evaluate = lambda x, y: 0.8
        
        result = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertFalse(result)
        self.assertFalse(stopper.should_stop)
        
        # Restore original method
        self.test_model.evaluate = original_evaluate
        
    def test_accuracy_stopper_check_above_target(self):
        """Test AccuracyStopper when accuracy is above target"""
        stopper = gnn.AccuracyStopper(target_accuracy=0.9)
        
        # Mock model.evaluate to return high accuracy
        original_evaluate = self.test_model.evaluate
        self.test_model.evaluate = lambda x, y: 0.95
        
        result = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertTrue(result)
        self.assertTrue(stopper.should_stop)
        
        # Restore original method
        self.test_model.evaluate = original_evaluate
        
    def test_accuracy_stopper_check_exact_target(self):
        """Test AccuracyStopper when accuracy exactly matches target"""
        stopper = gnn.AccuracyStopper(target_accuracy=0.9)
        
        # Mock model.evaluate to return exact target accuracy
        original_evaluate = self.test_model.evaluate
        self.test_model.evaluate = lambda x, y: 0.9
        
        result = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertTrue(result)
        self.assertTrue(stopper.should_stop)
        
        # Restore original method
        self.test_model.evaluate = original_evaluate
    
    # ParameterCountStopper Tests
    def test_parameter_count_stopper_initialization(self):
        """Test ParameterCountStopper initialization with default parameters"""
        stopper = gnn.ParameterCountStopper()
        self.assertEqual(stopper.decrease_threshold, 0.5)
        self.assertEqual(stopper.metric_name, "parameter_count")
        self.assertIsNone(stopper.initial_parameter_count)
        self.assertFalse(stopper.should_stop)
        
    def test_parameter_count_stopper_custom_initialization(self):
        """Test ParameterCountStopper initialization with custom parameters"""
        stopper = gnn.ParameterCountStopper(decrease_threshold=0.3, metric_name="param_reduction")
        self.assertEqual(stopper.decrease_threshold, 0.3)
        self.assertEqual(stopper.metric_name, "param_reduction")
        self.assertIsNone(stopper.initial_parameter_count)
        self.assertFalse(stopper.should_stop)
        
    def test_parameter_count_stopper_initialization_phase(self):
        """Test ParameterCountStopper during initialization phase"""
        stopper = gnn.ParameterCountStopper(decrease_threshold=0.5)
        
        # Mock model.get_parametr_count to return initial count
        original_get_count = self.test_model.get_parametr_count
        self.test_model.get_parametr_count = lambda: 1000
        
        result = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertFalse(result)
        self.assertFalse(stopper.should_stop)
        self.assertEqual(stopper.initial_parameter_count, 1000)
        
        # Restore original method
        self.test_model.get_parametr_count = original_get_count
        
    def test_parameter_count_stopper_insufficient_reduction(self):
        """Test ParameterCountStopper when reduction is insufficient"""
        stopper = gnn.ParameterCountStopper(decrease_threshold=0.5)
        
        # Mock model.get_parametr_count to return counts
        original_get_count = self.test_model.get_parametr_count
        call_count = 0
        def mock_get_count():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1000  # Initial count
            else:
                return 800   # 20% reduction (insufficient)
        
        self.test_model.get_parametr_count = mock_get_count
        
        # First call - initialization
        result1 = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertFalse(result1)
        
        # Second call - insufficient reduction
        result2 = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertFalse(result2)
        self.assertFalse(stopper.should_stop)
        
        # Restore original method
        self.test_model.get_parametr_count = original_get_count
        
    def test_parameter_count_stopper_sufficient_reduction(self):
        """Test ParameterCountStopper when reduction is sufficient"""
        stopper = gnn.ParameterCountStopper(decrease_threshold=0.5)
        
        # Mock model.get_parametr_count to return counts
        original_get_count = self.test_model.get_parametr_count
        call_count = 0
        def mock_get_count():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1000  # Initial count
            else:
                return 400   # 60% reduction (sufficient)
        
        self.test_model.get_parametr_count = mock_get_count
        
        # First call - initialization
        result1 = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertFalse(result1)
        
        # Second call - sufficient reduction
        result2 = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertTrue(result2)
        self.assertTrue(stopper.should_stop)
        
        # Restore original method
        self.test_model.get_parametr_count = original_get_count
    
    # AccuracyAndReductionStopper Tests
    def test_accuracy_and_reduction_stopper_initialization(self):
        """Test AccuracyAndReductionStopper initialization with default parameters"""
        stopper = gnn.AccuracyAndReductionStopper()
        self.assertEqual(stopper.accuracy_stopper.target_accuracy, 0.9)
        self.assertEqual(stopper.parameter_stopper.decrease_threshold, 0.5)
        self.assertFalse(stopper.accuracy_reached)
        self.assertFalse(stopper.parameter_reduced)
        self.assertFalse(stopper.should_stop)
        
    def test_accuracy_and_reduction_stopper_custom_initialization(self):
        """Test AccuracyAndReductionStopper initialization with custom parameters"""
        stopper = gnn.AccuracyAndReductionStopper(target_accuracy=0.95, parameter_decrease_threshold=0.3)
        self.assertEqual(stopper.accuracy_stopper.target_accuracy, 0.95)
        self.assertEqual(stopper.parameter_stopper.decrease_threshold, 0.3)
        self.assertFalse(stopper.accuracy_reached)
        self.assertFalse(stopper.parameter_reduced)
        self.assertFalse(stopper.should_stop)
        
    def test_accuracy_and_reduction_stopper_only_accuracy_met(self):
        """Test AccuracyAndReductionStopper when only accuracy condition is met"""
        stopper = gnn.AccuracyAndReductionStopper(target_accuracy=0.9, parameter_decrease_threshold=0.5)
        
        # Mock methods to return high accuracy but no parameter reduction
        original_evaluate = self.test_model.evaluate
        original_get_count = self.test_model.get_parametr_count
        
        call_count = 0
        def mock_evaluate(x, y):
            return 0.95  # High accuracy
        
        def mock_get_count():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1000  # Initial count
            else:
                return 1000  # No reduction
        
        self.test_model.evaluate = mock_evaluate
        self.test_model.get_parametr_count = mock_get_count
        
        # First call - initialization
        result1 = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertFalse(result1)
        
        # Second call - only accuracy met
        result2 = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertFalse(result2)
        self.assertTrue(stopper.accuracy_reached)
        self.assertFalse(stopper.parameter_reduced)
        self.assertFalse(stopper.should_stop)
        
        # Restore original methods
        self.test_model.evaluate = original_evaluate
        self.test_model.get_parametr_count = original_get_count
        
    def test_accuracy_and_reduction_stopper_only_parameter_met(self):
        """Test AccuracyAndReductionStopper when only parameter condition is met"""
        stopper = gnn.AccuracyAndReductionStopper(target_accuracy=0.9, parameter_decrease_threshold=0.5)
        
        # Mock methods to return low accuracy but parameter reduction
        original_evaluate = self.test_model.evaluate
        original_get_count = self.test_model.get_parametr_count
        
        call_count = 0
        def mock_evaluate(x, y):
            return 0.8  # Low accuracy
        
        def mock_get_count():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1000  # Initial count
            else:
                return 400   # 60% reduction
        
        self.test_model.evaluate = mock_evaluate
        self.test_model.get_parametr_count = mock_get_count
        
        # First call - initialization
        result1 = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertFalse(result1)
        
        # Second call - only parameter reduction met
        result2 = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertFalse(result2)
        self.assertFalse(stopper.accuracy_reached)
        self.assertTrue(stopper.parameter_reduced)
        self.assertFalse(stopper.should_stop)
        
        # Restore original methods
        self.test_model.evaluate = original_evaluate
        self.test_model.get_parametr_count = original_get_count
        
    def test_accuracy_and_reduction_stopper_both_conditions_met(self):
        """Test AccuracyAndReductionStopper when both conditions are met"""
        stopper = gnn.AccuracyAndReductionStopper(target_accuracy=0.9, parameter_decrease_threshold=0.5)
        
        # Mock methods to return high accuracy and parameter reduction
        original_evaluate = self.test_model.evaluate
        original_get_count = self.test_model.get_parametr_count
        
        call_count = 0
        def mock_evaluate(x, y):
            return 0.95  # High accuracy
        
        def mock_get_count():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 1000  # Initial count
            else:
                return 400   # 60% reduction
        
        self.test_model.evaluate = mock_evaluate
        self.test_model.get_parametr_count = mock_get_count
        
        # First call - initialization
        result1 = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertFalse(result1)
        
        # Second call - both conditions met
        result2 = stopper.check(self.test_model, self.x_train, self.y_train)
        self.assertTrue(result2)
        self.assertTrue(stopper.accuracy_reached)
        self.assertTrue(stopper.parameter_reduced)
        self.assertTrue(stopper.should_stop)
        
        # Restore original methods
        self.test_model.evaluate = original_evaluate
        self.test_model.get_parametr_count = original_get_count
        
    def test_accuracy_and_reduction_stopper_reset(self):
        """Test AccuracyAndReductionStopper reset functionality"""
        stopper = gnn.AccuracyAndReductionStopper()
        stopper.accuracy_reached = True
        stopper.parameter_reduced = True
        stopper.should_stop = True
        
        stopper.reset()
        self.assertFalse(stopper.accuracy_reached)
        self.assertFalse(stopper.parameter_reduced)
        self.assertFalse(stopper.should_stop)
        self.assertFalse(stopper.accuracy_stopper.should_stop)
        self.assertFalse(stopper.parameter_stopper.should_stop)
    
    # Integration Tests
    def test_integration_with_trainer_accuracy_stopper(self):
        """Test AccuracyStopper integration with trainer.train function"""
        try:
            stopper = gnn.AccuracyStopper(target_accuracy=0.5)  # Low threshold for testing
            
            model = gnn.trainer.train(
                x_train=self.x_train,
                y_train=self.y_train,
                x_test=self.x_test,
                y_test=self.y_test,
                labels=self.labels,
                input_paths=1,
                path=self.temp_dir,
                model_name="test_accuracy_stopper",
                epochs=2,
                generations=5,
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
        except Exception as e:
            self.fail(f"Training with accuracy stopper failed with exception: {e}")
            
    def test_integration_with_trainer_parameter_stopper(self):
        """Test ParameterCountStopper integration with trainer.train function"""
        try:
            stopper = gnn.ParameterCountStopper(decrease_threshold=0.1)  # Low threshold for testing
            
            model = gnn.trainer.train(
                x_train=self.x_train,
                y_train=self.y_train,
                x_test=self.x_test,
                y_test=self.y_test,
                labels=self.labels,
                input_paths=1,
                path=self.temp_dir,
                model_name="test_parameter_stopper",
                epochs=2,
                generations=5,
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
        except Exception as e:
            self.fail(f"Training with parameter stopper failed with exception: {e}")
            
    def test_integration_with_trainer_combined_stopper(self):
        """Test AccuracyAndReductionStopper integration with trainer.train function"""
        try:
            stopper = gnn.AccuracyAndReductionStopper(target_accuracy=0.5, parameter_decrease_threshold=0.1)
            
            model = gnn.trainer.train(
                x_train=self.x_train,
                y_train=self.y_train,
                x_test=self.x_test,
                y_test=self.y_test,
                labels=self.labels,
                input_paths=1,
                path=self.temp_dir,
                model_name="test_combined_stopper",
                epochs=2,
                generations=5,
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
        except Exception as e:
            self.fail(f"Training with combined stopper failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()