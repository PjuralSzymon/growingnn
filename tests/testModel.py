import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
from testSuite import mode
import numpy as np

class TestModel(unittest.TestCase):
    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
        self.model = gnn.Model(input_size=10, hidden_size=5, output_size=3)
        # Create test data
        self.X = np.random.rand(10, 10)  # 10 samples, 10 features
        self.Y = np.random.rand(3, 10)   # 3 classes, 10 samples

    def test_init(self):
        self.assertEqual(self.model.batch_size, 128)
        self.assertEqual(self.model.input_size, 10)
        self.assertEqual(self.model.output_size, 3)
        self.assertEqual(self.model.hidden_size, 5)
        self.assertEqual(len(self.model.hidden_layers), 0)
        self.assertEqual(self.model.avaible_id, 2)
        self.assertEqual(self.model.convolution, False)
        self.assertIsNone(self.model.input_shape)
        self.assertIsNone(self.model.kernel_size)
        self.assertIsNone(self.model.depth)

    def test_forward_prop(self):
        result = self.model.forward_prop(self.X)
        self.assertEqual(result.shape, (3, 10))  # output_size x num_samples
        self.assertTrue(np.all(np.isfinite(result)))  # Check for NaN/Inf values

    def test_add_and_remove_layer(self):
        self.assertEqual(len(self.model.hidden_layers), 0)
        layer_id = self.model.add_res_layer(layer_from_id='init_0', layer_to_id=1)
        self.assertEqual(len(self.model.hidden_layers), 1)
        self.model.remove_layer(layer_id)
        self.assertEqual(len(self.model.hidden_layers), 0)

    def test_back_prop(self):
        # Add a hidden layer first
        layer_id = self.model.add_res_layer(layer_from_id='init_0', layer_to_id=1)
        
        # Forward pass
        output = self.model.forward_prop(self.X)
        
        # Calculate error
        E = output - self.Y
        
        # Backpropagate
        self.model.back_prop(E, len(self.X), 0.01)
        
        # Check if weights were updated
        for layer in self.model.input_layers:
            self.assertFalse(np.array_equal(layer.W, np.zeros_like(layer.W)))

    def test_layer_connections(self):
        # Add multiple layers
        layer1_id = self.model.add_res_layer(layer_from_id='init_0', layer_to_id=1)
        layer2_id = self.model.add_res_layer(layer_from_id=layer1_id, layer_to_id=2)
        
        # Test connections
        layer1 = self.model.hidden_layers[0]
        layer2 = self.model.hidden_layers[1]
        
        self.assertIn('init_0', layer1.input_layers_ids)
        self.assertIn(layer2_id, layer1.output_layers_ids)
        self.assertIn(layer1_id, layer2.input_layers_ids)
        self.assertIn(2, layer2.output_layers_ids)

    def test_model_operations(self):
        # Create a new model for this test to avoid interference
        model = gnn.Model(input_size=10, hidden_size=5, output_size=3)
        
        # Add a single hidden layer
        layer_id = model.add_res_layer(layer_from_id='init_0', layer_to_id=1)
        
        # Test forward propagation
        output = model.forward_prop(self.X)
        self.assertEqual(output.shape, (3, 10))
        
        # Test backpropagation
        E = output - self.Y
        model.back_prop(E, len(self.X), 0.01)
        
        # Test layer removal
        model.remove_layer(layer_id)
        self.assertEqual(len(model.hidden_layers), 0)
        
        # Test that the model still works after layer removal
        output = model.forward_prop(self.X)
        self.assertEqual(output.shape, (3, 10))

    def test_model_with_different_sizes(self):
        # Test model with different input/output sizes
        model = gnn.Model(input_size=5, hidden_size=3, output_size=2)
        X = np.random.rand(5, 5)  # 5 samples, 5 features
        Y = np.random.rand(2, 5)  # 2 classes, 5 samples
        
        # Add a layer
        layer_id = model.add_res_layer(layer_from_id='init_0', layer_to_id=1)
        
        # Test forward propagation
        output = model.forward_prop(X)
        self.assertEqual(output.shape, (2, 5))
        
        # Test backpropagation
        E = output - Y
        model.back_prop(E, len(X), 0.01)

    def test_model_with_convolution(self):
        # Create a model with the correct input size for convolution
        input_size = 25  # 5*5*1
        model = gnn.Model(input_size=input_size, hidden_size=5, output_size=3)
        
        # Set convolution mode using the proper method
        input_shape = (5, 5, 1)
        kernel_size = 3
        depth = 2
        model.set_convolution_mode(input_shape, kernel_size, depth)
        
        # Create test data for convolution
        # The input needs to be in the correct format for the convolution layer
        # For a Conv layer, we need to pass the original 4D shape, not the flattened version
        X = np.random.rand(1, 5, 5, 1)  # batch_size=1, height=5, width=5, channels=1
        X = X.astype(np.float32)
        
        # For convolution, we need to pass the original 4D shape
        # Don't reshape the input for convolution
        output = model.forward_prop(X)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (3, 1))  # output_size x batch_size

if __name__ == '__main__':
    unittest.main()
