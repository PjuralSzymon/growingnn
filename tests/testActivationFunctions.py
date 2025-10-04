import unittest
import numpy as np
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
from growingnn.utils import Activations

# Set numpy print options for nice formatting
np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

class TestActivationFunctions(unittest.TestCase):
    def setUp(self):
        self.test_input = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.test_matrix = np.array([[-2.0, -1.0], [0.0, 1.0]])
        
    def test_relu(self):
        # Test ReLU activation
        output = Activations.ReLu.exe(self.test_input)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(output, expected, decimal=3)
        
        # Test ReLU derivative
        grad = Activations.ReLu.der(self.test_input)
        expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=3)
        
        # Test matrix input
        matrix_output = Activations.ReLu.exe(self.test_matrix)
        expected_matrix = np.array([[0.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(matrix_output, expected_matrix, decimal=3)
        
    def test_leaky_relu(self):
        # Test LeakyReLU activation
        output = Activations.leaky_ReLu.exe(self.test_input)
        expected = np.array([-0.002, -0.001, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(output, expected, decimal=3)
        
        # Test LeakyReLU derivative
        grad = Activations.leaky_ReLu.der(self.test_input)
        expected_grad = np.array([0.001, 0.001, 0.001, 1.0, 1.0])
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=3)
        
        # Test matrix input
        matrix_output = Activations.leaky_ReLu.exe(self.test_matrix)
        expected_matrix = np.array([[-0.002, -0.001], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(matrix_output, expected_matrix, decimal=3)
        
    def test_sigmoid(self):
        # Test Sigmoid activation
        output = Activations.Sigmoid.exe(self.test_input)
        expected = np.array([0.119, 0.268, 0.500, 0.731, 0.880])
        np.testing.assert_array_almost_equal(output, expected, decimal=3)
        
        # Test Sigmoid derivative
        grad = Activations.Sigmoid.der(self.test_input)
        expected_grad = output * (1 - output)
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=3)
        
        # Test matrix input
        matrix_output = Activations.Sigmoid.exe(self.test_matrix)
        expected_matrix = np.array([[0.119, 0.269], [0.500, 0.731]])
        np.testing.assert_array_almost_equal(matrix_output, expected_matrix, decimal=3)
        
    def test_tanh(self):
        # Test Tanh activation
        output = Activations.Tanh.exe(self.test_input)
        expected = np.array([-0.964, -0.761, 0.000, 0.761, 0.964])
        np.testing.assert_array_almost_equal(output, expected, decimal=3)
        
        # Test Tanh derivative
        grad = Activations.Tanh.der(self.test_input)
        expected_grad = np.array([0.071, 0.420, 1.000, 0.420, 0.071])
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=3)
        
        # Test matrix input
        matrix_output = Activations.Tanh.exe(self.test_matrix)
        expected_matrix = np.array([[-0.964, -0.761], [0.000, 0.761]])
        np.testing.assert_array_almost_equal(matrix_output, expected_matrix, decimal=3)
        
    def test_softmax(self):
        # Test Softmax activation
        output = Activations.SoftMax.exe(self.test_input)
        expected = np.array([0.012, 0.032, 0.086, 0.234, 0.636])
        np.testing.assert_array_almost_equal(output, expected, decimal=3)
        
        # Test Softmax derivative
        grad = Activations.SoftMax.der(self.test_input)
        expected_grad = np.array([1.0])
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=3)
        
        # Test matrix input
        matrix_output = Activations.SoftMax.exe(self.test_matrix)
        expected_matrix = np.array([[0.119, 0.119], [0.881, 0.881]])
        np.testing.assert_array_almost_equal(matrix_output, expected_matrix, decimal=3)
        
    def test_linear(self):
        # Test Linear activation
        output = Activations.Linear.exe(self.test_input)
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(output, expected, decimal=3)
        
        # Test Linear derivative
        grad = Activations.Linear.der(self.test_input)
        expected_grad = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=3)
        
        # Test matrix input
        matrix_output = Activations.Linear.exe(self.test_matrix)
        expected_matrix = np.array([[-2.0, -1.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(matrix_output, expected_matrix, decimal=3)
        
        # Test matrix derivative
        matrix_grad = Activations.Linear.der(self.test_matrix)
        expected_matrix_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_array_almost_equal(matrix_grad, expected_matrix_grad, decimal=3)

if __name__ == '__main__':
    unittest.main() 
