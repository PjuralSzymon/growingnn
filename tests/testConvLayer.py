import unittest
import numpy as np
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
from growingnn.structure import Conv, Layer_Type
from growingnn.optimizers import SGDOptimizer, AdamOptimizer

class TestConvLayer(unittest.TestCase):
    def setUp(self):
        self.input_shape = (32, 32, 3)  # height, width, channels
        self.kernel_size = 3
        self.depth = 16
        self.activation = gnn.structure.Activations.ReLu
        self.optimizer = SGDOptimizer()
        
    def test_conv_layer_initialization(self):
        # Test basic initialization
        conv_layer = Conv(1, None, self.input_shape, self.kernel_size, self.depth, self.activation, self.optimizer)
        
        # Check output shape
        expected_output_shape = (self.input_shape[0] - self.kernel_size + 1,
                               self.input_shape[1] - self.kernel_size + 1,
                               self.depth)
        self.assertEqual(conv_layer.output_shape, expected_output_shape)
        
        # Check kernel shape
        expected_kernel_shape = (self.kernel_size, self.kernel_size, self.input_shape[2], self.depth)
        self.assertEqual(conv_layer.kernels.shape, expected_kernel_shape)
        
        # Check bias shape
        self.assertEqual(conv_layer.biases.shape, (self.depth,))
        
    def test_forward_propagation(self):
        conv_layer = Conv(1, None, self.input_shape, self.kernel_size, self.depth, self.activation, self.optimizer)
        
        # Create test input
        batch_size = 4
        X = np.random.randn(batch_size, *self.input_shape)
        
        # Forward pass
        output = conv_layer.forward_prop(X, None)
        
        # Check output shape
        expected_shape = (batch_size, *conv_layer.output_shape)
        self.assertEqual(output.shape, expected_shape)
        
        # Check for NaN/Inf
        self.assertFalse(np.any(np.isnan(output)))
        self.assertFalse(np.any(np.isinf(output)))
        
    def test_backward_propagation(self):
        conv_layer = Conv(1, None, self.input_shape, self.kernel_size, self.depth, self.activation, self.optimizer)
        
        # Create test input and error
        batch_size = 4
        X = np.random.randn(batch_size, *self.input_shape)
        E = np.random.randn(batch_size, *conv_layer.output_shape)
        
        # Forward pass
        _ = conv_layer.forward_prop(X, None)
        
        # Backward pass
        conv_layer.back_prop(E, batch_size, 0.01)
        
        # Check gradients
        self.assertIsNotNone(conv_layer.kernel_grads)
        self.assertIsNotNone(conv_layer.bias_grads)
        
        # Check gradient shapes
        self.assertEqual(conv_layer.kernel_grads.shape, conv_layer.kernels.shape)
        self.assertEqual(conv_layer.bias_grads.shape, conv_layer.biases.shape)
        
    def test_weight_update(self):
        conv_layer = Conv(1, None, self.input_shape, self.kernel_size, self.depth, self.activation, self.optimizer)
        
        # Store initial weights
        initial_kernels = conv_layer.kernels.copy()
        initial_biases = conv_layer.biases.copy()
        
        # Create test input and error
        batch_size = 4
        X = np.random.randn(batch_size, *self.input_shape)
        E = np.random.randn(batch_size, *conv_layer.output_shape)
        
        # Forward and backward pass
        _ = conv_layer.forward_prop(X, None)
        conv_layer.back_prop(E, batch_size, 0.01)
        
        # Update weights
        conv_layer.update_params(0.01)
        
        # Check if weights were updated
        self.assertFalse(np.array_equal(conv_layer.kernels, initial_kernels))
        self.assertFalse(np.array_equal(conv_layer.biases, initial_biases))
        
    def test_different_kernel_sizes(self):
        kernel_sizes = [1, 3, 5, 7]
        for kernel_size in kernel_sizes:
            conv_layer = Conv(1, None, self.input_shape, kernel_size, self.depth, self.activation, self.optimizer)
            
            # Check output shape
            expected_output_shape = (self.input_shape[0] - kernel_size + 1,
                                   self.input_shape[1] - kernel_size + 1,
                                   self.depth)
            self.assertEqual(conv_layer.output_shape, expected_output_shape)
            
    def test_different_depths(self):
        depths = [1, 8, 16, 32]
        for depth in depths:
            conv_layer = Conv(1, None, self.input_shape, self.kernel_size, depth, self.activation, self.optimizer)
            
            # Check output shape
            expected_output_shape = (self.input_shape[0] - self.kernel_size + 1,
                                   self.input_shape[1] - self.kernel_size + 1,
                                   depth)
            self.assertEqual(conv_layer.output_shape, expected_output_shape)
            
    def test_different_activations(self):
        activations = [gnn.structure.Activations.ReLu,
                      gnn.structure.Activations.Sigmoid,
                      gnn.structure.Activations.Tanh]
        
        for activation in activations:
            conv_layer = Conv(1, None, self.input_shape, self.kernel_size, self.depth, activation, self.optimizer)
            
            # Create test input
            batch_size = 4
            X = np.random.randn(batch_size, *self.input_shape)
            
            # Forward pass
            output = conv_layer.forward_prop(X, None)
            
            # Check if activation was applied
            if activation == gnn.structure.Activations.ReLu:
                self.assertTrue(np.all(output >= 0))
            elif activation == gnn.structure.Activations.Sigmoid:
                self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
            elif activation == gnn.structure.Activations.Tanh:
                self.assertTrue(np.all(output >= -1) and np.all(output <= 1))
                
    def test_different_optimizers(self):
        optimizers = [SGDOptimizer(), AdamOptimizer()]
        
        for optimizer in optimizers:
            conv_layer = Conv(1, None, self.input_shape, self.kernel_size, self.depth, self.activation, optimizer)
            
            # Create test input and error
            batch_size = 4
            X = np.random.randn(batch_size, *self.input_shape)
            E = np.random.randn(batch_size, *conv_layer.output_shape)
            
            # Forward and backward pass
            _ = conv_layer.forward_prop(X, None)
            conv_layer.back_prop(E, batch_size, 0.01)
            
            # Store initial weights
            initial_kernels = conv_layer.kernels.copy()
            initial_biases = conv_layer.biases.copy()
            
            # Update weights
            conv_layer.update_params(0.01)
            
            # Check if weights were updated
            self.assertFalse(np.array_equal(conv_layer.kernels, initial_kernels))
            self.assertFalse(np.array_equal(conv_layer.biases, initial_biases))
            
    def test_weight_clipping(self):
        conv_layer = Conv(1, None, self.input_shape, self.kernel_size, self.depth, self.activation, self.optimizer)
        
        # Create test input and error
        batch_size = 4
        X = np.random.randn(batch_size, *self.input_shape)
        E = np.random.randn(batch_size, *conv_layer.output_shape) * 1000  # Large error to test clipping
        
        # Forward and backward pass
        _ = conv_layer.forward_prop(X, None)
        conv_layer.back_prop(E, batch_size, 0.01)
        conv_layer.update_params(0.01)
        
        # Check if weights are within reasonable range
        self.assertTrue(np.all(np.abs(conv_layer.kernels) <= 10))
        self.assertTrue(np.all(np.abs(conv_layer.biases) <= 10))
        
    def test_deepcopy(self):
        conv_layer = Conv(1, None, self.input_shape, self.kernel_size, self.depth, self.activation, self.optimizer)
        conv_layer_copy = conv_layer.deepcopy()
        
        # Check if all attributes are copied correctly
        self.assertEqual(conv_layer_copy.output_shape, conv_layer.output_shape)
        self.assertEqual(conv_layer_copy.kernel_size, conv_layer.kernel_size)
        self.assertEqual(conv_layer_copy.depth, conv_layer.depth)
        self.assertEqual(conv_layer_copy.activation, conv_layer.activation)
        
        # Check if weights are copied (but not the same object)
        self.assertTrue(np.array_equal(conv_layer_copy.kernels, conv_layer.kernels))
        self.assertTrue(np.array_equal(conv_layer_copy.biases, conv_layer.biases))
        self.assertIsNot(conv_layer_copy.kernels, conv_layer.kernels)
        self.assertIsNot(conv_layer_copy.biases, conv_layer.biases)

if __name__ == '__main__':
    unittest.main() 