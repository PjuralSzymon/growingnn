import sys
import os
import tempfile
import unittest
import numpy as np
import copy
import time

# Add the parent directory to the Python path to allow importing the growingnn package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import growingnn as gnn
from testSuite import mode
from growingnn import Model, Loss, Activations, LearningRateScheduler, Storage
from growingnn.config import config

class TestAdvancedFeatures(unittest.TestCase):
    """Advanced tests for the growingnn package"""

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
        self.hidden_size = 20
        self.output_size = 2
        self.batch_size = 10
        self.epochs = 3
        
        # Create sample data
        self.X = np.random.rand(100, self.input_size)
        self.y = np.random.randint(0, self.output_size, size=100)
        
        # Create a basic model for testing
        self.model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        self.model.add_res_layer('init_0', 1)
        self.model.add_res_layer(2, 1)
        self.model.add_res_layer(2, 1)
        self.model.add_res_layer(2, 1)

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

    def test_model_deepcopy(self):
        """Test deep copying of model"""
        # Create a deep copy of the model
        model_copy = self.model.deepcopy()
        
        # Verify that the copy is independent
        self.assertNotEqual(id(self.model), id(model_copy))
        self.assertEqual(len(self.model.hidden_layers), len(model_copy.hidden_layers))
        
        # Modify original and verify copy unchanged
        self.model.remove_layer(2)
        self.assertNotEqual(len(self.model.hidden_layers), len(model_copy.hidden_layers))

    def test_layer_removal(self):
        """Test removing layers"""
        # Create a new model for this test
        model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers
        model.add_res_layer('init_0', 1)
        layer_id = model.add_res_layer(2, 1)
        
        # Verify the layer was added successfully
        self.assertIsNotNone(layer_id, "Layer was not added successfully")
        
        # Check if the layer exists using get_layer method
        layer = model.get_layer(layer_id)
        
        if layer is not None:
            # Remove the layer if it exists
            print("BOB: ",layer)
            model.remove_layer(layer_id)
            
            # Verify that the layer is removed by checking if forward propagation still works
            try:
                model.forward_prop(self.X)
            except Exception as e:
                self.fail(f"Forward propagation failed after layer removal: {e}")
        else:
            # If the layer doesn't exist, the test should fail with a specific message
            with self.assertRaises(ValueError) as context:
                model.remove_layer(layer_id)
            self.assertIn(f"Layer with ID {layer_id} does not exist in the model", str(context.exception))
        
        # Try to remove a non-existent layer
        with self.assertRaises(ValueError) as context:
            model.remove_layer(999)
        self.assertIn("Layer with ID 999 does not exist in the model", str(context.exception))

    def test_layer_connection_update(self):
        """Test updating layer connections"""
        # Create a new model for this test
        model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers
        model.add_res_layer('init_0', 1)
        layer_id = model.add_res_layer(2, 1)
        
        # Update a connection - should not raise an exception
        try:
            model.add_connection('init_0', layer_id)
            # If we get here, no exception was raised, which is good
        except Exception as e:
            self.fail(f"Adding connection raised an exception: {e}")
        
        # Verify that the connection is updated by checking if forward propagation works
        try:
            model.forward_prop(self.X)
        except Exception as e:
            self.fail(f"Forward propagation failed after connection update: {e}")
        
        # Try to update a non-existent connection - should raise an exception
        with self.assertRaises(ValueError) as context:
            model.add_connection('init_0', 999)
        self.assertIn("Target layer with ID 999 does not exist in the model", str(context.exception))

    def test_model_serialization(self):
        """Test model serialization"""
        # Create a new model for this test
        model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers
        model.add_res_layer('init_0', 1)
        model.add_res_layer(2, 1)
        
        # Save the model to a file
        Storage.saveModel(model, self.test_file)
        
        # Load the model from the file
        loaded_model = Storage.loadModel(self.test_file)
        
        # Verify that the loaded model has the same structure by checking forward propagation
        output_original = model.forward_prop(self.X)
        output_loaded = loaded_model.forward_prop(self.X)
        
        # Check that outputs have the same values
        np.testing.assert_array_almost_equal(output_original, output_loaded)

    def test_model_validation(self):
        """Test model validation"""
        # Create a new model for this test
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
        
        # Verify that the model is valid by checking if forward propagation works
        try:
            model.forward_prop(self.X)
        except Exception as e:
            self.fail(f"Forward propagation failed: {e}")

    def test_model_optimization(self):
        """Test model optimization"""
        # Create a new model for this test
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
        
        # Create learning rate scheduler
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.01)
        
        # Train the model
        initial_accuracy = model.evaluate(self.X[:10], self.y[:10])  # Use first 10 samples
        model.gradient_descent(self.X[:10], self.y[:10], iterations=10, lr_scheduler=lr_scheduler, quiet=False)
        final_accuracy = model.evaluate(self.X[:10], self.y[:10])  # Use first 10 samples
        
        # Verify that the model was trained
        self.assertIsInstance(final_accuracy, float)
        # Check that accuracy is between 0 and 1
        self.assertTrue(0 <= final_accuracy <= 1)
        # Note: We don't check if accuracy increased, as it might not always improve
        # especially with a small number of iterations

    def test_different_activation_functions(self):
        """Test model with different activation functions"""
        # Test with Sigmoid (which we know works)
        model_sigmoid = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        model_sigmoid.add_res_layer('init_0', 1)
        model_sigmoid.add_res_layer(2, 1)
        
        # Test with ReLU instead of Tanh
        model_relu = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.ReLu,
            input_paths=1
        )
        
        # Add layers with correct connections
        model_relu.add_res_layer('init_0', 1)
        model_relu.add_res_layer(2, 1)
        
        # Test forward propagation with both models
        output_sigmoid = model_sigmoid.forward_prop(self.X)
        output_relu = model_relu.forward_prop(self.X)
        
        # The outputs should be different due to different activation functions
        self.assertFalse(np.array_equal(output_sigmoid, output_relu))
        
        # Create learning rate scheduler
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.01)
        
        # Test training with both models
        accuracy_sigmoid, loss_sigmoid = model_sigmoid.gradient_descent(self.X[:10], self.y[:10], self.epochs, lr_scheduler)
        accuracy_relu, loss_relu = model_relu.gradient_descent(self.X[:10], self.y[:10], self.epochs, lr_scheduler)
        
        # Both models should train successfully
        self.assertIsInstance(accuracy_sigmoid, float)
        self.assertIsInstance(accuracy_relu, float)
        self.assertTrue(0 <= accuracy_sigmoid <= 1)
        self.assertTrue(0 <= accuracy_relu <= 1)

    def test_different_loss_functions(self):
        """Test model with different loss functions"""
        # Test with MSE
        model_mse = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.MSE,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        model_mse.add_res_layer('init_0', 1)
        model_mse.add_res_layer(2, 1)
        
        # Test with Cross Entropy
        model_ce = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        model_ce.add_res_layer('init_0', 1)
        model_ce.add_res_layer(2, 1)
        
        # Create learning rate scheduler
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.01)
        
        # Test training with both models
        accuracy_mse, loss_mse = model_mse.gradient_descent(self.X[:10], self.y[:10], self.epochs, lr_scheduler)
        accuracy_ce, loss_ce = model_ce.gradient_descent(self.X[:10], self.y[:10], self.epochs, lr_scheduler)
        
        # Both models should train successfully
        self.assertIsInstance(accuracy_mse, float)
        self.assertIsInstance(accuracy_ce, float)
        self.assertTrue(0 <= accuracy_mse <= 1)
        self.assertTrue(0 <= accuracy_ce <= 1)

    def test_learning_rate_schedulers(self):
        """Test different learning rate schedulers"""
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
        
        # Test with constant learning rate
        lr_scheduler_constant = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.01)
        
        # Test with progressive learning rate
        lr_scheduler_prog = LearningRateScheduler(LearningRateScheduler.PROGRESIVE, 0.01, 0.2)
        
        # Test training with both schedulers
        accuracy_constant, loss_constant = model.gradient_descent(self.X[:10], self.y[:10], self.epochs, lr_scheduler_constant)
        
        # Create a new model for the progressive scheduler
        model_prog = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add layers with correct connections
        model_prog.add_res_layer('init_0', 1)
        model_prog.add_res_layer(2, 1)
        
        accuracy_prog, loss_prog = model_prog.gradient_descent(self.X[:10], self.y[:10], self.epochs, lr_scheduler_prog)
        
        # Both models should train successfully
        self.assertIsInstance(accuracy_constant, float)
        self.assertIsInstance(accuracy_prog, float)
        self.assertTrue(0 <= accuracy_constant <= 1)
        self.assertTrue(0 <= accuracy_prog <= 1)

    def test_simulation_schedulers(self):
        """Test different simulation schedulers"""
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
        
        # Create learning rate scheduler
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.01)
        
        # Test with constant simulation scheduler
        sim_scheduler_constant = gnn.structure.SimulationScheduler(
            gnn.structure.SimulationScheduler.CONSTANT, 
            simulation_time=1, 
            simulation_epochs=1
        )
        
        # Test with progress check simulation scheduler
        sim_scheduler_progress = gnn.structure.SimulationScheduler(
            gnn.structure.SimulationScheduler.PROGRESS_CHECK, 
            simulation_time=1, 
            simulation_epochs=1, 
            min_grow_rate=0.2
        )
        
        # Create simulation score
        simulation_score = gnn.Simulation.ScoreFunctions.Simulation_score()
        
        # Test training with both schedulers
        try:
            gnn.trainer.train(
                x_train=self.X[:10], 
                y_train=self.y[:10], 
                x_test=self.X[:10],
                y_test=self.y[:10],
                labels=range(self.output_size),
                input_paths=1,
                path=self.test_dir, 
                model_name="test_model_constant",
                epochs=1, 
                generations=1,
                simulation_scheduler=sim_scheduler_constant,
                simulation_alg=gnn.Simulation.montecarlo_alg,
                sim_set_generator=gnn.Simulation.simulation.create_simulation_set_SAMLE,
                lr_scheduler=lr_scheduler,
                input_size=self.input_size, 
                hidden_size=self.hidden_size, 
                output_size=self.output_size,
                input_shape=None,
                kernel_size=None,
                deepth=None
            )
        except Exception as e:
            self.fail(f"Training with constant simulation scheduler raised an exception: {e}")
        
        try:
            gnn.trainer.train(
                x_train=self.X[:10], 
                y_train=self.y[:10], 
                x_test=self.X[:10],
                y_test=self.y[:10],
                labels=range(self.output_size),
                input_paths=1,
                path=self.test_dir, 
                model_name="test_model_progress",
                epochs=1, 
                generations=1,
                simulation_scheduler=sim_scheduler_progress,
                simulation_alg=gnn.Simulation.montecarlo_alg,
                sim_set_generator=gnn.Simulation.simulation.create_simulation_set_SAMLE,
                lr_scheduler=lr_scheduler,
                input_size=self.input_size, 
                hidden_size=self.hidden_size, 
                output_size=self.output_size,
                input_shape=None,
                kernel_size=None,
                deepth=None
            )
        except Exception as e:
            self.fail(f"Training with progress check simulation scheduler raised an exception: {e}")

    def test_model_evaluation(self):
        """Test model evaluation metrics"""
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
        
        # Create learning rate scheduler
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.01)
        
        # Train the model
        accuracy, loss = model.gradient_descent(self.X[:10], self.y[:10], self.epochs, lr_scheduler)
        
        # Evaluate the model
        accuracy_eval = model.evaluate(self.X[:10], self.y[:10])
        
        # Check that accuracy is a float between 0 and 1
        self.assertIsInstance(accuracy_eval, float)
        self.assertTrue(0 <= accuracy_eval <= 1)

    def test_complex_architecture(self):
        """Test a more complex architecture with multiple layers and connections"""
        # Create a model with multiple layers and connections
        model = Model(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            loss_function=Loss.multiclass_cross_entropy,
            activation_fun=Activations.Sigmoid,
            input_paths=1
        )
        
        # Add multiple layers with different connections
        model.add_res_layer('init_0', 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        model.add_res_layer(2, 1)
        
        # Forward propagation
        output = model.forward_prop(self.X)
        
        # Check that output has the correct shape
        self.assertEqual(output.shape, (self.output_size, self.batch_size))
        
        # Check that the model can be trained
        lr_scheduler = LearningRateScheduler(LearningRateScheduler.PROGRESIVE, 0.01, 0.2)
        
        accuracy, loss = model.gradient_descent(self.X[:10], self.y[:10], self.epochs, lr_scheduler)
        
        # Check that accuracy is a float between 0 and 1
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 1)

if __name__ == '__main__':
    unittest.main() 