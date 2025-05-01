import unittest
import numpy as np
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
from growingnn.structure import Activations

class TestActivationFunctions(unittest.TestCase):
    def setUp(self):
        self.test_input = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.test_matrix = np.array([[-2.0, -1.0], [0.0, 1.0]])
        
    def test_relu(self):
        # Test ReLU activation
        output = Activations.ReLu.exe(self.test_input)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(output, expected)
        
        # Test ReLU derivative
        grad = Activations.ReLu.der(self.test_input)
        expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(grad, expected_grad)
        
        # Test matrix input
        matrix_output = Activations.ReLu.exe(self.test_matrix)
        expected_matrix = np.array([[0.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(matrix_output, expected_matrix)
        
    def test_leaky_relu(self):
        # Test LeakyReLU activation
        output = Activations.leaky_ReLu.exe(self.test_input)
        expected = np.array([-0.02, -0.01, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(output, expected)
        
        # Test LeakyReLU derivative
        grad = Activations.leaky_ReLu.der(self.test_input)
        expected_grad = np.array([0.01, 0.01, 0.01, 1.0, 1.0])
        np.testing.assert_array_almost_equal(grad, expected_grad)
        
        # Test matrix input
        matrix_output = Activations.leaky_ReLu.exe(self.test_matrix)
        expected_matrix = np.array([[-0.02, -0.01], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(matrix_output, expected_matrix)
        
    def test_sigmoid(self):
        # Test Sigmoid activation
        output = Activations.Sigmoid.exe(self.test_input)
        expected = 1 / (1 + np.exp(-self.test_input))
        np.testing.assert_array_almost_equal(output, expected)
        
        # Test Sigmoid derivative
        grad = Activations.Sigmoid.der(self.test_input)
        expected_grad = output * (1 - output)
        np.testing.assert_array_almost_equal(grad, expected_grad)
        
        # Test matrix input
        matrix_output = Activations.Sigmoid.exe(self.test_matrix)
        expected_matrix = 1 / (1 + np.exp(-self.test_matrix))
        np.testing.assert_array_almost_equal(matrix_output, expected_matrix)
        
    def test_tanh(self):
        # Test Tanh activation
        output = Activations.Tanh.exe(self.test_input)
        expected = np.tanh(self.test_input)
        np.testing.assert_array_almost_equal(output, expected)
        
        # Test Tanh derivative
        grad = Activations.Tanh.der(self.test_input)
        expected_grad = 1 - np.tanh(self.test_input)**2
        np.testing.assert_array_almost_equal(grad, expected_grad)
        
        # Test matrix input
        matrix_output = Activations.Tanh.exe(self.test_matrix)
        expected_matrix = np.tanh(self.test_matrix)
        np.testing.assert_array_almost_equal(matrix_output, expected_matrix)
        
    def test_softmax(self):
        # Test Softmax activation
        output = Activations.SoftMax.exe(self.test_input)
        expected = np.exp(self.test_input) / np.sum(np.exp(self.test_input))
        np.testing.assert_array_almost_equal(output, expected)
        
        # Test Softmax derivative
        grad = Activations.SoftMax.der(self.test_input)
        expected_grad = output * (1 - output)
        np.testing.assert_array_almost_equal(grad, expected_grad)
        
        # Test matrix input
        matrix_output = Activations.SoftMax.exe(self.test_matrix)
        expected_matrix = np.exp(self.test_matrix) / np.sum(np.exp(self.test_matrix))
        np.testing.assert_array_almost_equal(matrix_output, expected_matrix)
        
    def test_numerical_stability(self):
        # Test with large values
        large_input = np.array([1000, -1000])
        
        # ReLU should handle large values
        relu_output = Activations.ReLu.exe(large_input)
        self.assertTrue(np.all(np.isfinite(relu_output)))
        
        # LeakyReLU should handle large values
        leaky_output = Activations.leaky_ReLu.exe(large_input)
        self.assertTrue(np.all(np.isfinite(leaky_output)))
        
        # Sigmoid should handle large values
        sigmoid_output = Activations.Sigmoid.exe(large_input)
        self.assertTrue(np.all(np.isfinite(sigmoid_output)))
        
        # Tanh should handle large values
        tanh_output = Activations.Tanh.exe(large_input)
        self.assertTrue(np.all(np.isfinite(tanh_output)))
        
        # Softmax should handle large values
        softmax_output = Activations.SoftMax.exe(large_input)
        self.assertTrue(np.all(np.isfinite(softmax_output)))
        
    def test_edge_cases(self):
        # Test with zero input
        zero_input = np.zeros(5)
        
        # ReLU with zero input
        relu_zero = Activations.ReLu.exe(zero_input)
        np.testing.assert_array_equal(relu_zero, zero_input)
        
        # LeakyReLU with zero input
        leaky_zero = Activations.leaky_ReLu.exe(zero_input)
        np.testing.assert_array_equal(leaky_zero, zero_input)
        
        # Sigmoid with zero input
        sigmoid_zero = Activations.Sigmoid.exe(zero_input)
        expected_sigmoid = np.ones(5) * 0.5
        np.testing.assert_array_almost_equal(sigmoid_zero, expected_sigmoid)
        
        # Tanh with zero input
        tanh_zero = Activations.Tanh.exe(zero_input)
        np.testing.assert_array_equal(tanh_zero, zero_input)
        
        # Softmax with zero input
        softmax_zero = Activations.SoftMax.exe(zero_input)
        expected_softmax = np.ones(5) / 5
        np.testing.assert_array_almost_equal(softmax_zero, expected_softmax)
        
    def test_gradient_consistency(self):
        # Test if gradients are consistent with numerical approximation
        epsilon = 1e-5
        
        # Test ReLU gradient
        relu_grad = Activations.ReLu.der(self.test_input)
        relu_numerical = (Activations.ReLu.exe(self.test_input + epsilon) - 
                         Activations.ReLu.exe(self.test_input - epsilon)) / (2 * epsilon)
        np.testing.assert_array_almost_equal(relu_grad, relu_numerical, decimal=4)
        
        # Test LeakyReLU gradient
        leaky_grad = Activations.leaky_ReLu.der(self.test_input)
        leaky_numerical = (Activations.leaky_ReLu.exe(self.test_input + epsilon) - 
                          Activations.leaky_ReLu.exe(self.test_input - epsilon)) / (2 * epsilon)
        np.testing.assert_array_almost_equal(leaky_grad, leaky_numerical, decimal=4)
        
        # Test Sigmoid gradient
        sigmoid_grad = Activations.Sigmoid.der(self.test_input)
        sigmoid_numerical = (Activations.Sigmoid.exe(self.test_input + epsilon) - 
                            Activations.Sigmoid.exe(self.test_input - epsilon)) / (2 * epsilon)
        np.testing.assert_array_almost_equal(sigmoid_grad, sigmoid_numerical, decimal=4)
        
        # Test Tanh gradient
        tanh_grad = Activations.Tanh.der(self.test_input)
        tanh_numerical = (Activations.Tanh.exe(self.test_input + epsilon) - 
                         Activations.Tanh.exe(self.test_input - epsilon)) / (2 * epsilon)
        np.testing.assert_array_almost_equal(tanh_grad, tanh_numerical, decimal=4)

if __name__ == '__main__':
    unittest.main() 