import unittest
import numpy as np
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
from growingnn.structure import Layer, LearningRateScheduler, Model, Activations, Layer_Type
from growingnn.config import FLOAT_TYPE
from growingnn.action import Del_neurons

class TestNeuronReduction(unittest.TestCase):
    def setUp(self):
        # Define test configurations
        self.test_configs = [
            {
                'input_size': 10,
                'hidden_size': 20,
                'output_size': 5,
                'reduction_ratio': 0.8,  # Reduce by 20%
                'num_samples': 10,
                'iterations': 10,
                'learning_rate': 0.01
            },
            {
                'input_size': 15,
                'hidden_size': 50,
                'output_size': 8,
                'reduction_ratio': 0.7,  # Reduce by 30%
                'num_samples': 15,
                'iterations': 15,
                'learning_rate': 0.005
            },
            {
                'input_size': 30,
                'hidden_size': 100,
                'output_size': 10,
                'reduction_ratio': 0.6,  # Reduce by 10%
                'num_samples': 8,
                'iterations': 8,
                'learning_rate': 0.02
            }
        ]

    def test_del_neurons_action(self):
        """Test the Del_neurons action class"""
        # Create a simple model with one hidden layer
        model = Model(10, 20, 5, activation_fun=Activations.ReLu)
        layer_id = model.add_res_layer('init_0', 1)
        
        # Get initial neuron count
        layer = model.get_layer(layer_id)
        initial_neurons = layer.neurons
        
        # Execute Del_neurons action
        actions = Del_neurons.generate_all_actions(model)
        for action in actions:
            action.execute(model)
        
        # Verify neurons were reduced
        self.assertLess(layer.neurons, initial_neurons)

    def test_neuron_reduction_similar_inputs(self):
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
                input = np.ascontiguousarray(input, dtype=FLOAT_TYPE)
                
                # Get outputs and weights before reduction
                layer.forward_prop(input, -1)
                output1 = layer.A
                W_before_mean = np.mean(layer.W)    
                B_before_mean = np.mean(layer.B)
                
                # Reduce neurons
                layer.remove_neurons(config['reduction_ratio'])
                
                # Get outputs after reduction
                layer.forward_prop(input, -1)
                output1_reduced = layer.A
                W_after_mean = np.mean(layer.W)
                B_after_mean = np.mean(layer.B)
                
                # Calculate statistics
                mean_diff = abs(np.mean(output1) - np.mean(output1_reduced))
                
                # Verify statistical measures
                self.assertLess(abs(W_before_mean - W_after_mean), 0.2)
                self.assertLess(abs(B_before_mean - B_after_mean), 0.2)
                self.assertLess(mean_diff, 0.5)

    def test_neuron_reduction_multi_layer(self):
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
                X = np.ascontiguousarray(X, dtype=FLOAT_TYPE)
                y = np.random.randint(0, config['output_size'], (config['num_samples'],))
                y[0] = config['output_size'] - 1

                # Train the model
                lr_scheduler = LearningRateScheduler(
                    LearningRateScheduler.CONSTANT, 
                    config['learning_rate']
                )
                model.gradient_descent(X, y, iterations=config['iterations'], lr_scheduler=lr_scheduler, quiet=True)
                
                # Get output before reduction
                output_before = model.forward_prop(X)
                
                # Reduce neurons in the first hidden layer
                target_layer = model.hidden_layers[0]
                neurons_before = target_layer.neurons
                W_before_mean = np.mean(target_layer.W)
                B_before_mean = np.mean(target_layer.B)
                
                # Reduce neurons
                target_layer.remove_neurons(config['reduction_ratio'])
                
                # Get output after reduction
                output_after = model.forward_prop(X)
                
                # Calculate statistics
                mean_diff = abs(np.mean(output_before) - np.mean(output_after))
                median_diff = abs(np.median(output_before) - np.median(output_after))

                # Verify the reduction worked
                self.assertLess(target_layer.neurons, neurons_before)
                self.assertLess(mean_diff, 0.2)
                self.assertLess(median_diff, 0.2)


if __name__ == '__main__':
    unittest.main() 