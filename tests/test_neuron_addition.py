import unittest
import numpy as np
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
from growingnn.structure import Layer, LearningRateScheduler, Model, Activations, Layer_Type
from growingnn.action import Add_neurons

class TestNeuronAddition(unittest.TestCase):
    def setUp(self):
        # Define test configurations
        self.test_configs = [
            {
                'input_size': 10,
                'hidden_size': 20,
                'output_size': 5,
                'addition_ratio': 1.2,  # Add 20%
                'num_samples': 10,
                'iterations': 10,
                'learning_rate': 0.01
            },
            {
                'input_size': 15,
                'hidden_size': 50,
                'output_size': 8,
                'addition_ratio': 1.5,  # Add 50%
                'num_samples': 15,
                'iterations': 15,
                'learning_rate': 0.005
            },
            {
                'input_size': 30,
                'hidden_size': 100,
                'output_size': 10,
                'addition_ratio': 1.1,  # Add 10%
                'num_samples': 8,
                'iterations': 8,
                'learning_rate': 0.02
            }
        ]

    def test_add_neurons_action(self):
        """Test the Add_neurons action class"""
        # Create a simple model with one hidden layer
        model = Model(10, 20, 5, activation_fun=Activations.ReLu)
        layer_id = model.add_res_layer('init_0', 1)
        
        # Get initial neuron count
        layer = model.get_layer(layer_id)
        initial_neurons = layer.neurons
        
        # Execute Add_neurons action
        actions = Add_neurons.generate_all_actions(model, 1.2)
        for action in actions:
            action.execute(model)
        
        # Verify neurons were added
        self.assertGreater(layer.neurons, initial_neurons)

    def test_neuron_addition_similar_inputs(self):
        for config in self.test_configs:
            with self.subTest(config=config):
                # Create model and layer
                model = Model(
                    config['input_size'], 
                    config['hidden_size'], 
                    config['output_size'], 
                    activation_fun=Activations.ReLu
                )
                layer = model.input_layers[0]
                
                # Create test input
                input = np.random.uniform(-1, 1, (config['input_size'], 1))
                input = np.ascontiguousarray(input, dtype=gnn.config.FLOAT_TYPE)
                
                # Get outputs and weights before addition
                layer.forward_prop(input, -1)
                output1 = layer.A
                W_before_mean = np.mean(layer.W)    
                B_before_mean = np.mean(layer.B)
                
                # Add neurons
                layer.scale_neurons(config['addition_ratio'])
                
                # Get outputs after addition
                layer.forward_prop(input, -1)
                output1_added = layer.A
                W_after_mean = np.mean(layer.W)
                B_after_mean = np.mean(layer.B)
                
                # Calculate statistics
                mean_diff = abs(np.mean(output1) - np.mean(output1_added))
                
                # Verify statistical measures - should be similar but not identical
                self.assertLess(abs(W_before_mean - W_after_mean), 0.3)
                self.assertLess(abs(B_before_mean - B_after_mean), 0.3)
                self.assertLess(mean_diff, 0.5)

    def test_neuron_addition_multi_layer(self):
        for config in self.test_configs:
            with self.subTest(config=config):
                # Create model with multiple layers
                model = Model(
                    config['input_size'], 
                    config['hidden_size'], 
                    config['output_size'], 
                    activation_fun=Activations.ReLu
                )
                
                # Add hidden layers
                model.add_res_layer('init_0', 1)  # First hidden layer
                model.add_res_layer(2, 1)         # Second hidden layer
                
                # Create test data
                X = np.random.uniform(-1, 1, (config['input_size'], config['num_samples']))
                X = np.ascontiguousarray(X, dtype=gnn.config.FLOAT_TYPE)
                y = np.random.randint(0, config['output_size'], (config['num_samples'],))
                y[0] = config['output_size'] - 1

                # Train the model
                lr_scheduler = LearningRateScheduler(
                    LearningRateScheduler.CONSTANT, 
                    config['learning_rate']
                )
                model.gradient_descent(X, y, iterations=config['iterations'], lr_scheduler=lr_scheduler, quiet=True)
                
                # Get output before addition
                output_before = model.forward_prop(X)
                
                # Add neurons in the first hidden layer
                target_layer = model.hidden_layers[0]
                neurons_before = target_layer.neurons
                W_before_mean = np.mean(target_layer.W)
                B_before_mean = np.mean(target_layer.B)
                
                # Add neurons
                target_layer.scale_neurons(config['addition_ratio'])
                
                # Get output after addition
                output_after = model.forward_prop(X)
                
                # Calculate statistics
                mean_diff = abs(np.mean(output_before) - np.mean(output_after))
                median_diff = abs(np.median(output_before) - np.median(output_after))

                # Verify the addition worked
                self.assertGreater(target_layer.neurons, neurons_before)
                self.assertLess(mean_diff, 0.3)
                self.assertLess(median_diff, 0.3)

    def test_add_neurons_generate_all_actions(self):
        """Test the generate_all_actions method for Add_neurons"""
        # Create a model with multiple layers
        model = Model(10, 20, 5, activation_fun=Activations.ReLu)
        model.add_res_layer('init_0', 1)
        model.add_res_layer(2, 1)
        
        # Test different scale ratios
        for ratio in [1.1, 1.5, 2.0]:
            with self.subTest(ratio=ratio):
                actions = Add_neurons.generate_all_actions(model, ratio)
                
                # Should generate actions for applicable layers
                self.assertGreater(len(actions), 0)
                
                # Each action should be an Add_neurons instance
                for action in actions:
                    self.assertIsInstance(action, Add_neurons)
                    self.assertEqual(len(action.params), 2)
                    self.assertEqual(action.params[1], ratio)

    def test_add_neurons_maximum_size_limit(self):
        """Test that Add_neurons respects maximum size limits"""
        # Create a model with a large layer
        model = Model(10, 500, 5, activation_fun=Activations.ReLu)
        
        # Try to add neurons with a large ratio that would exceed limits
        actions = Add_neurons.generate_all_actions(model, 3.0)  # 3x increase
        
        # Should respect the maximum size limit
        for action in actions:
            layer = model.get_layer(action.params[0])
            new_neurons = int(layer.neurons * action.params[1])
            self.assertLessEqual(new_neurons, gnn.config.MAXIMUM_MATRIX_NEURONS_SIZE_FOR_NEURONS_ADDITION)


if __name__ == '__main__':
    unittest.main()
