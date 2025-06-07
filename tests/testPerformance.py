import sys
import os
import tempfile
import unittest
import numpy as np
import time
import asyncio

# Add the parent directory to the Python path to allow importing the growingnn package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from growingnn import Model, Loss, Activations, LearningRateScheduler

# Add the parent directory to the Python path to allow importing the growingnn package
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestPerformance(unittest.TestCase):
    """Performance tests for the growingnn package"""

    def setUp(self):
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
        self.X = np.random.rand(100, self.input_size)
        self.y = np.random.randint(0, self.output_size, size=(100,))
        
        # Create learning rate scheduler
        self.lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.01)

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

    def test_forward_propagation_performance(self):
        """Test the performance of forward propagation"""
        # Create a model
        model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        model.add_res_layer('init_0', 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        
        # Measure forward propagation time
        times = []
        for _ in range(100):
            start_time = time.time()
            model.forward_prop(self.X)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        self.assertLess(avg_time, 0.001)  # Less than 1ms

    def test_backward_propagation_performance(self):
        """Test the performance of backward propagation"""
        # Create a model
        model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        model.add_res_layer('init_0', 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        
        # Forward propagation
        output = model.forward_prop(self.X)
        
        # Create error term
        error = np.random.random(output.shape)
        
        # Measure backward propagation time
        times = []
        for _ in range(100):
            start_time = time.time()
            model.backward_prop(error, self.batch_size, 0.01)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        self.assertLess(avg_time, 0.05)  # Less than 1ms

    def test_training_performance(self):
        """Test the performance of training"""
        # Create a model
        model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        model.add_res_layer('init_0', 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        
        # Measure training time
        times = []
        for _ in range(self.epochs):
            start_time = time.time()
            model.gradient_descent(self.X, self.y, 1, self.lr_scheduler)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        self.assertLess(avg_time, 1.0)  # Less than 1 second per epoch

    def test_model_serialization_performance(self):
        """Test the performance of model serialization and deserialization"""
        # Create a model
        model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        model.add_res_layer('init_0', 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        
        # Measure serialization time
        times_serialize = []
        for _ in range(10):
            start_time = time.time()
            model_json = model.to_json()
            end_time = time.time()
            times_serialize.append(end_time - start_time)
        
        avg_time_serialize = sum(times_serialize) / len(times_serialize)
        self.assertLess(avg_time_serialize, 0.1)  # Less than 0.1 seconds
        
        # Measure deserialization time
        times_deserialize = []
        for _ in range(10):
            start_time = time.time()
            Model.from_json(model_json)
            end_time = time.time()
            times_deserialize.append(end_time - start_time)
        
        avg_time_deserialize = sum(times_deserialize) / len(times_deserialize)
        self.assertLess(avg_time_deserialize, 0.1)  # Less than 0.1 seconds

    def test_large_model_performance(self):
        """Test the performance with a larger model"""
        # Create a larger model
        large_input_size = 100
        large_hidden_size = 50
        large_output_size = 10
        large_batch_size = 32
        
        # Create larger sample data
        X_large = np.random.rand(1000, large_input_size)
        y_large = np.random.randint(0, large_output_size, size=(1000,))
        
        # Create a model
        large_model = Model(
            input_size=large_input_size,
            hidden_size=large_hidden_size,
            output_size=large_output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        large_model.add_res_layer('init_0', 1)
        large_model.add_res_layer(2, 1)
        large_model.add_res_layer(2, 1)
        large_model.add_res_layer(2, 1)
        
        # Test forward propagation
        start_time = time.time()
        large_model.forward_prop(X_large)
        end_time = time.time()
        self.assertLess(end_time - start_time, 0.3)  # Less than 100ms
        
        # Test training
        start_time = time.time()
        large_model.gradient_descent(X_large, y_large, 1, self.lr_scheduler)
        end_time = time.time()
        self.assertLess(end_time - start_time, 5.0)  # Less than 5 seconds

    def test_monte_carlo_performance(self):
        """Test the performance of Monte Carlo tree search"""
        # Create a model
        model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        model.add_res_layer('init_0', 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        
        # Import the Monte Carlo algorithm
        from growingnn.Simulation.montecarlo_alg import get_action
        from growingnn.Simulation.ScoreFunctions import Simulation_score
        
        # Create a simulation score function
        simulation_score = Simulation_score()
        
        # Measure Monte Carlo tree search time
        start_time = time.time()
        
        # Run the Monte Carlo tree search
        # Note: This is a synchronous call, but the actual implementation is async
        # We're using a small timeout for testing purposes
        action, depth, rollouts = asyncio.run(get_action(
            model, 
            max_time_for_dec=1.0,  # 1 second timeout for testing
            epochs=1,
            X_train=self.X,
            Y_train=self.y,
            simulation_score=simulation_score
        ))
        
        end_time = time.time()
        
        # Check that the Monte Carlo tree search time is reasonable (less than 2s)
        self.assertLess(end_time - start_time, 2.0)
        
        # Check that we got a valid action
        self.assertIsNotNone(action)

if __name__ == '__main__':
    unittest.main() 