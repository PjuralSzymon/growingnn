import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
from testSuite import mode
from unittest.mock import MagicMock
import numpy as np

class TestLayer(unittest.TestCase):

    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
        self.layer = gnn.Layer(1, None, 10, 5, None)
        self.layer.act_fun = gnn.structure.Activations.ReLu
        self.layer.set_as_ending()

    def test_init(self):
        self.assertEqual(self.layer.id, 1)
        self.assertIsNone(self.layer.model)
        self.assertEqual(self.layer.input_size, 10)
        self.assertEqual(self.layer.neurons, 5)
        self.assertEqual(self.layer.input_layers_ids, [])
        self.assertEqual(self.layer.output_layers_ids, [])
        self.assertEqual(self.layer.f_input, [])
        self.assertEqual(self.layer.b_input, [])
        self.assertEqual(self.layer.reshspers, {})

    def test_connect_input(self):
        self.layer.model = MagicMock()
        self.layer.model.get_layer.return_value = MagicMock(output_layers_ids=[])
        self.layer.connect_input(2)
        self.assertIn(2, self.layer.input_layers_ids)
        self.layer.model.get_layer.assert_called_with(2)
        self.layer.model.get_layer.return_value.connect_output.assert_called_with(1)

    def test_connect_output(self):
        self.layer.model = MagicMock()
        self.layer.model.get_layer.return_value = MagicMock(input_layers_ids=[])
        self.layer.connect_output(2)
        self.assertIn(2, self.layer.output_layers_ids)
        self.layer.model.get_layer.assert_called_with(2)
        self.layer.model.get_layer.return_value.connect_input.assert_called_with(1)

    def test_disconnect(self):
        self.layer.input_layers_ids = [2, 3]
        self.layer.output_layers_ids = [4, 5]
        self.layer.disconnect(3)
        self.assertNotIn(3, self.layer.input_layers_ids)
        self.assertEqual(self.layer.output_layers_ids, [4, 5])
        self.layer.disconnect(5)
        self.assertEqual(self.layer.output_layers_ids, [4])

    def test_forward_prop(self):
        self.layer.input_layers_ids = [2, 3]
        self.layer.model = MagicMock()
        self.layer.forward_prop(gnn.np.ones((10, 1)), -1)
        self.layer.forward_prop(gnn.np.ones((10, 1)), -1)
        result2 = self.layer.A
        self.assertIsNotNone(result2)
        self.assertEqual(len(self.layer.f_input), 0)

    def test_back_prop(self):
        self.layer.input_layers_ids = [2, 3]
        self.layer.model = MagicMock()
        self.layer.forward_prop(gnn.np.ones((10, 1)), -1)
        self.layer.forward_prop(gnn.np.ones((10, 1)), -1)
        self.layer.back_prop(gnn.np.ones((5, 1)), 1, 0.01)
        self.assertEqual(len(self.layer.b_input), 0)
        self.assertTrue(hasattr(self.layer, 'dW'))
        self.assertTrue(hasattr(self.layer, 'dB'))
        
    def test_activation_functions(self):
        # Test different activation functions
        layer = gnn.Layer(1, None, 10, 5, None)
        
        # Test ReLU activation
        layer.act_fun = gnn.structure.Activations.ReLu
        x = np.array([-2, -1, 0, 1, 2])
        result = layer.act_fun.exe(x)
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)
        
        # Test Sigmoid activation
        layer.act_fun = gnn.structure.Activations.Sigmoid
        result = layer.act_fun.exe(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test Tanh activation
        layer.act_fun = gnn.structure.Activations.Tanh
        result = layer.act_fun.exe(x)
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_weight_initialization(self):
        # Test weight initialization
        layer = gnn.Layer(1, None, 10, 5, None)
        
        # Check weight dimensions
        self.assertEqual(layer.W.shape, (5, 10))
        self.assertEqual(layer.B.shape, (5, 1))
        
        # Check that weights are initialized with reasonable values
        # They should not be all zeros or all the same value
        self.assertFalse(np.all(layer.W == 0))
        self.assertFalse(np.all(layer.W == layer.W[0, 0]))
        
        # Check that biases are initialized
        self.assertFalse(np.all(layer.B == 0))
        
    def test_gradient_calculation(self):
        # Test gradient calculation
        layer = gnn.Layer(1, None, 10, 5, None)
        layer.act_fun = gnn.structure.Activations.ReLu
        
        # Set up input and perform forward pass
        x = np.ones((10, 1))
        layer.forward_prop(x, -1)
        
        # Calculate gradients
        error = np.ones((5, 1))
        layer.back_prop(error, 1, 0.01)
        
        # Check that gradients have the correct shape
        self.assertEqual(layer.dW.shape, (5, 10))
        self.assertEqual(layer.dB.shape, (5, 1))
        
        # Check that gradients are not all zeros
        self.assertFalse(np.all(layer.dW == 0))
        self.assertFalse(np.all(layer.dB == 0))
        
        # Test gradient update
        original_W = layer.W.copy()
        original_B = layer.B.copy()
        
        # Update weights and biases
        layer.W -= 0.01 * layer.dW
        layer.B -= 0.01 * layer.dB
        
        # Check that weights and biases were updated
        self.assertFalse(np.array_equal(layer.W, original_W))
        self.assertFalse(np.array_equal(layer.B, original_B))

if __name__ == '__main__':
    unittest.main()
