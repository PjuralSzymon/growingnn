import unittest
import numpy as np
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
from growingnn.structure import Loss

class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        # Binary classification test data
        self.y_true_binary = np.array([0, 1, 0, 1])
        self.y_pred_binary = np.array([0.1, 0.9, 0.2, 0.8])
        
        # Multi-class classification test data
        self.y_true_multi = np.array([0, 1, 2, 0])
        self.y_pred_multi = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.8, 0.1, 0.1]
        ])
        
        # Regression test data
        self.y_true_reg = np.array([1.0, 2.0, 3.0, 4.0])
        self.y_pred_reg = np.array([1.1, 1.9, 3.1, 3.9])
        
    def test_mse_loss(self):
        # Test MSE loss calculation
        loss = Loss.MSE.exe(self.y_true_reg, self.y_pred_reg)
        expected_loss = np.mean((self.y_true_reg - self.y_pred_reg)**2)
        self.assertAlmostEqual(loss, expected_loss)
        
        # Test MSE gradient
        grad = Loss.MSE.der(self.y_true_reg, self.y_pred_reg)
        expected_grad = 2 * (self.y_pred_reg - self.y_true_reg) / len(self.y_true_reg)
        np.testing.assert_array_almost_equal(grad, expected_grad)
        
        # Test with single value
        single_loss = Loss.MSE.exe(np.array([1.0]), np.array([1.5]))
        self.assertAlmostEqual(single_loss, 0.25)
        
    def test_multiclass_cross_entropy(self):
        # Test cross-entropy loss calculation
        loss = Loss.multiclass_cross_entropy.exe(self.y_true_multi, self.y_pred_multi)
        
        # Calculate expected loss manually
        expected_loss = 0
        for i in range(len(self.y_true_multi)):
            expected_loss -= np.log(self.y_pred_multi[i, self.y_true_multi[i]])
        expected_loss /= len(self.y_true_multi)
        
        self.assertAlmostEqual(loss, expected_loss)
        
        # Test cross-entropy gradient
        grad = Loss.multiclass_cross_entropy.der(self.y_true_multi, self.y_pred_multi)
        
        # Calculate expected gradient manually
        expected_grad = np.zeros_like(self.y_pred_multi)
        for i in range(len(self.y_true_multi)):
            expected_grad[i, self.y_true_multi[i]] = -1 / (self.y_pred_multi[i, self.y_true_multi[i]] * len(self.y_true_multi))
        
        np.testing.assert_array_almost_equal(grad, expected_grad)
        
    def test_numerical_stability(self):
        # Test with very small predictions
        y_pred_small = np.array([1e-10, 1e-10, 1e-10])
        y_true_small = np.array([1, 0, 0])
        
        # Cross-entropy should handle small values
        loss = Loss.multiclass_cross_entropy.exe(y_true_small, y_pred_small)
        self.assertTrue(np.isfinite(loss))
        
        # Test with very large predictions
        y_pred_large = np.array([1e10, 1e10, 1e10])
        y_true_large = np.array([1, 0, 0])
        
        # Cross-entropy should handle large values
        loss = Loss.multiclass_cross_entropy.exe(y_true_large, y_pred_large)
        self.assertTrue(np.isfinite(loss))
        
    def test_edge_cases(self):
        # Test with perfect predictions
        perfect_pred = np.array([0.0, 1.0, 0.0])
        perfect_true = np.array([0, 1, 0])
        
        # MSE should be 0
        mse_loss = Loss.MSE.exe(perfect_true, perfect_pred)
        self.assertAlmostEqual(mse_loss, 0.0)
        
        # Cross-entropy should be 0
        ce_loss = Loss.multiclass_cross_entropy.exe(perfect_true, perfect_pred.reshape(1, -1))
        self.assertAlmostEqual(ce_loss, 0.0)
        
        # Test with all zeros
        zero_pred = np.zeros(3)
        zero_true = np.zeros(3)
        
        # MSE should be 0
        mse_loss = Loss.MSE.exe(zero_true, zero_pred)
        self.assertAlmostEqual(mse_loss, 0.0)
        
    def test_gradient_consistency(self):
        # Test if gradients are consistent with numerical approximation
        epsilon = 1e-5
        
        # Test MSE gradient
        mse_grad = Loss.MSE.der(self.y_true_reg, self.y_pred_reg)
        mse_numerical = np.zeros_like(self.y_pred_reg)
        for i in range(len(self.y_pred_reg)):
            y_pred_plus = self.y_pred_reg.copy()
            y_pred_minus = self.y_pred_reg.copy()
            y_pred_plus[i] += epsilon
            y_pred_minus[i] -= epsilon
            mse_numerical[i] = (Loss.MSE.exe(self.y_true_reg, y_pred_plus) - 
                               Loss.MSE.exe(self.y_true_reg, y_pred_minus)) / (2 * epsilon)
        np.testing.assert_array_almost_equal(mse_grad, mse_numerical, decimal=4)
        
        # Test cross-entropy gradient
        ce_grad = Loss.multiclass_cross_entropy.der(self.y_true_multi, self.y_pred_multi)
        ce_numerical = np.zeros_like(self.y_pred_multi)
        for i in range(self.y_pred_multi.shape[0]):
            for j in range(self.y_pred_multi.shape[1]):
                y_pred_plus = self.y_pred_multi.copy()
                y_pred_minus = self.y_pred_multi.copy()
                y_pred_plus[i, j] += epsilon
                y_pred_minus[i, j] -= epsilon
                ce_numerical[i, j] = (Loss.multiclass_cross_entropy.exe(self.y_true_multi, y_pred_plus) - 
                                     Loss.multiclass_cross_entropy.exe(self.y_true_multi, y_pred_minus)) / (2 * epsilon)
        np.testing.assert_array_almost_equal(ce_grad, ce_numerical, decimal=4)
        
    def test_batch_processing(self):
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            # Generate random data
            y_true = np.random.randint(0, 3, size=batch_size)
            y_pred = np.random.rand(batch_size, 3)
            y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)
            
            # Calculate loss
            loss = Loss.multiclass_cross_entropy.exe(y_true, y_pred)
            
            # Calculate expected loss
            expected_loss = 0
            for i in range(batch_size):
                expected_loss -= np.log(y_pred[i, y_true[i]])
            expected_loss /= batch_size
            
            self.assertAlmostEqual(loss, expected_loss)
            
    def test_invalid_inputs(self):
        # Test with invalid predictions (not summing to 1)
        invalid_pred = np.array([0.3, 0.3, 0.3])
        with self.assertRaises(ValueError):
            Loss.multiclass_cross_entropy.exe(self.y_true_multi, invalid_pred.reshape(1, -1))
            
        # Test with invalid true values (out of range)
        invalid_true = np.array([3])  # Only 3 classes (0,1,2)
        with self.assertRaises(ValueError):
            Loss.multiclass_cross_entropy.exe(invalid_true, self.y_pred_multi[0:1])
            
        # Test with mismatched shapes
        with self.assertRaises(ValueError):
            Loss.MSE.exe(self.y_true_reg, self.y_pred_reg[:-1])

if __name__ == '__main__':
    unittest.main() 