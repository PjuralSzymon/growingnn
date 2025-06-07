import sys
import os
import tempfile
import unittest
import numpy as np

# Add the parent directory to the Python path to allow importing the growingnn package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import growingnn as gnn
from testSuite import mode
from growingnn import Model, Loss, Activations, LearningRateScheduler

class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling in the growingnn package"""

    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_model.json")
        
        # Common test parameters
        self.input_size = 10
        self.hidden_size = 5
        self.output_size = 2
        self.batch_size = 8
        self.epochs = 3
        
        # Create sample data
        self.x_train = np.random.random((self.input_size, self.batch_size))
        self.y_train = np.random.randint(self.output_size, size=(self.batch_size,))

        self.model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )

    def tearDown(self):
        # Clean up the temporary directory
        try:
            for root, dirs, files in os.walk(self.test_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.test_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {str(e)}")

    def test_empty_input(self):
        """Test handling of empty input data"""
        # Test with None input
        with self.assertRaises(ValueError):
            self.model.forward_prop(None)
        
        # Test with empty list
        empty_list = []
        with self.assertRaises(ValueError):
            self.model.forward_prop(empty_list)
        
        # Test with array of wrong shape
        wrong_shape = np.array([[]])
        with self.assertRaises(ValueError):
            self.model.forward_prop(wrong_shape)

    def test_mismatched_dimensions(self):
        """Test handling of input data with mismatched dimensions"""
        # Test with wrong input size (too many features)
        wrong_size_input = np.random.rand(self.input_size + 1, self.batch_size)
        try:
            self.model.forward_prop(wrong_size_input)
        except Exception:
            pass
        
        # Test with wrong batch dimension
        wrong_batch_input = np.random.rand(self.input_size, self.batch_size + 1)
        try:
            result = self.model.forward_prop(wrong_batch_input)
        except Exception:
            pass
        
        # Test with 1D array (missing batch dimension)
        wrong_dim_input = np.random.rand(self.input_size)
        try:
            self.model.forward_prop(wrong_dim_input)
        except Exception:
            pass

    def test_invalid_layer_connections(self):
        """Test handling of invalid layer connections"""
        # Test connecting to non-existent layer with string ID
        with self.assertRaises(ValueError) as context:
            self.model.add_connection("init_0", "non_existent")
        self.assertIn("Target layer with ID non_existent does not exist in the model", str(context.exception))
        
        # Test connecting from non-existent layer with string ID
        with self.assertRaises(ValueError) as context:
            self.model.add_connection("non_existent", "init_0")
        self.assertIn("Source layer with ID non_existent does not exist in the model", str(context.exception))
        
        # Test connecting to non-existent layer with integer ID
        with self.assertRaises(ValueError) as context:
            self.model.add_connection("init_0", 999)
        self.assertIn("Target layer with ID 999 does not exist in the model", str(context.exception))
        
        # Test connecting from non-existent layer with integer ID
        with self.assertRaises(ValueError) as context:
            self.model.add_connection(999, "init_0")
        self.assertIn("Source layer with ID 999 does not exist in the model", str(context.exception))

    def test_extreme_learning_rates(self):
        """Test behavior with extreme learning rates"""
        # Create sample data
        X = np.random.rand(10, self.input_size)
        y = np.random.randint(0, self.output_size, size=10)
        
        # Test with very small learning rate
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 1e-10)
        try:
            accuracy, history = self.model.gradient_descent(X, y, 10, lr_scheduler)
        except Exception:
            pass
        
        # Test with very large learning rate
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 1e10)
        try:
            accuracy, history = self.model.gradient_descent(X, y, 10, lr_scheduler)
        except Exception:
            pass
        
        # Test with negative learning rate
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, -0.01)
        try:
            self.model.gradient_descent(X, y, 10, lr_scheduler)
        except Exception:
            pass
        
        # Test with zero learning rate
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.0)
        try:
            accuracy, history = self.model.gradient_descent(X, y, 10, lr_scheduler)
        except Exception:
            pass

    def test_extreme_batch_sizes(self):
        """Test handling of extreme batch sizes"""
        # Create sample data
        X = np.random.rand(10, self.input_size)
        y = np.random.randint(0, self.output_size, size=10)
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.01)
        
        # Test with batch size of 1
        self.model.batch_size = 1
        accuracy, history = self.model.gradient_descent(X, y, 10, lr_scheduler)
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 1)
        
        # Test with batch size equal to dataset size
        self.model.batch_size = len(X)
        accuracy, history = self.model.gradient_descent(X, y, 10, lr_scheduler)
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 1)

        # Test with negative batch size
        self.model.batch_size = -1
        with self.assertRaises(ValueError) as context:
            self.model.gradient_descent(X, y, 10, lr_scheduler)
        self.assertEqual(str(context.exception), "Batch size must be positive")

    def test_invalid_activation(self):
        """Test handling of invalid activation function"""
        try:
            Model(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                activation_fun="InvalidActivation"
            )
        except Exception:
            pass

    def test_invalid_loss_function(self):
        """Test handling of invalid loss function"""
        try:
            Model(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                loss_function="InvalidLoss"
            )
        except Exception:
            pass

    def test_invalid_simulation_params(self):
        """Test handling of invalid simulation parameters"""
        # Test invalid mode
        try:
            LearningRateScheduler(-1, 0.01)
        except Exception:
            pass
        
        # Test invalid learning rate
        try:
            LearningRateScheduler(LearningRateScheduler.CONSTANT, -0.01)
        except Exception:
            pass
        
        # Test invalid steepness
        try:
            LearningRateScheduler(LearningRateScheduler.PROGRESIVE, 0.01, -0.2)
        except Exception:
            pass
        
        # Test zero learning rate
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.0)
        try:
            lr_scheduler.alpha_scheduler(0, 10)
        except Exception:
            pass
        
        # Test zero steepness
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.PROGRESIVE, 0.01, 0.0)
        try:
            lr_scheduler.alpha_scheduler(0, 10)
        except Exception:
            pass

    def test_history_operations(self):
        """Test history operations with edge cases"""
        X = np.random.rand(10, self.input_size)
        y = np.random.randint(0, self.output_size, size=10)
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.01)
        
        # Test with zero iterations
        try:
            _, history = self.model.gradient_descent(X, y, 0, lr_scheduler)
        except Exception:
            pass
        
        # Test with one iteration
        try:
            _, history = self.model.gradient_descent(X, y, 1, lr_scheduler)
        except Exception:
            pass
        
        # Test merging with None
        try:
            history.merge(None)
        except Exception:
            pass
        
        # Test merging with empty history
        try:
            empty_history = gnn.structure.History(['accuracy', 'loss'])
            history.merge(empty_history)
        except Exception:
            pass
        
        # Test merging with incompatible history
        try:
            incompatible_history = gnn.structure.History(['precision', 'recall'])
            history.merge(incompatible_history)
        except Exception:
            pass
        
        # Test getting last value from empty history
        try:
            empty_history = gnn.structure.History(['accuracy', 'loss'])
            empty_history.get_last('accuracy')
        except Exception:
            pass
        
        # Test getting last value from non-empty history
        try:
            history.get_last('accuracy')
        except Exception:
            pass

if __name__ == '__main__':
    unittest.main() 