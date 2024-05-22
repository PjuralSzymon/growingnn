import asyncio
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import unittest
import pandas as pd
import random
from unittest.mock import MagicMock


class TestLoss(unittest.TestCase):
    
    def test_getByName(self):
        mse = gnn.Loss.getByName('MSE')
        self.assertIsInstance(mse, gnn.Loss.MSE)
        
        cross_entropy = gnn.Loss.getByName('multiclass_cross_entropy')
        self.assertIsInstance(cross_entropy, gnn.Loss.multiclass_cross_entropy)
        
    def test_MSE_exe(self):
        Y_true = gnn.np.array([1, 2, 3])
        Y_pred = gnn.np.array([1, 2, 4])
        result = gnn.Loss.MSE.exe(Y_true, Y_pred)
        expected = 1/3
        self.assertAlmostEqual(result, expected)
        
    def test_MSE_der(self):
        Y_true = gnn.np.array([1, 2, 3])
        Y_pred = gnn.np.array([1, 2, 4])
        result = gnn.Loss.MSE.der(Y_true, Y_pred)
        expected = gnn.np.array([0, 0, 1])
        gnn.np.testing.assert_array_equal(result, expected)
        
    def test_multiclass_cross_entropy_exe(self):
        Y_true = gnn.np.array([[1, 0], [0, 1]])
        Y_pred = gnn.np.array([[0.8, 0.2], [0.2, 0.8]])
        result = gnn.Loss.multiclass_cross_entropy.exe(Y_true, Y_pred)
        expected = -0.5 * (gnn.np.log(0.8) + gnn.np.log(0.8))
        self.assertAlmostEqual(result, expected, places=5)
        
    # def test_multiclass_cross_entropy_der(self):
    #     Y_true = gnn.np.array([[1, 0], [0, 1]])
    #     Y_pred = gnn.np.array([[0.8, 0.2], [0.2, 0.8]])
    #     result = gnn.Loss.multiclass_cross_entropy.der(Y_true, Y_pred)
    #     expected = gnn.np.array([[-1.25, 0], [0, -1.25]])
    #     print("result: ", result)
    #     print("expected: ", expected)
    #     gnn.np.testing.assert_array_almost_equal(result, expected, decimal=2)
        
class TestActivations(unittest.TestCase):
    
    def test_getByName(self):
        relu = gnn.Activations.getByName('ReLu')
        self.assertIsInstance(relu, gnn.Activations.ReLu)
        
        leaky_relu = gnn.Activations.getByName('leaky_ReLu')
        self.assertIsInstance(leaky_relu, gnn.Activations.leaky_ReLu)
        
        softmax = gnn.Activations.getByName('SoftMax')
        self.assertIsInstance(softmax, gnn.Activations.SoftMax)
        
        sigmoid = gnn.Activations.getByName('Sigmoid')
        self.assertIsInstance(sigmoid, gnn.Activations.Sigmoid)
        
    def test_ReLu_exe(self):
        X = gnn.np.array([-1, 0, 1])
        result = gnn.Activations.ReLu.exe(X)
        expected = gnn.np.array([0, 0, 1])
        gnn.np.testing.assert_array_equal(result, expected)
        
    def test_leaky_ReLu_exe(self):
        X = gnn.np.array([-1, 0, 1])
        result = gnn.Activations.leaky_ReLu.exe(X)
        expected = gnn.np.array([-0.001, 0, 1])
        gnn.np.testing.assert_array_equal(result, expected)
        
    def test_SoftMax_exe(self):
        X = gnn.np.array([[1, 2, 3], [4, 5, 6]])
        result = gnn.Activations.SoftMax.exe(X)
        exp = gnn.np.exp(X - gnn.np.max(X))
        expected = exp / gnn.np.sum(exp, axis=0)
        gnn.np.testing.assert_array_almost_equal(result, expected, decimal=5)
        
    def test_Sigmoid_exe(self):
        X = gnn.np.array([-1, 0, 1])
        result = gnn.Activations.Sigmoid.exe(X)
        expected = 1 / (1 + gnn.np.exp(-X))
        gnn.np.testing.assert_array_almost_equal(result, expected, decimal=5)

class TestLearningRateScheduler(unittest.TestCase):
    
    def test_constant_scheduler(self):
        scheduler = gnn.LearningRateScheduler(mode=gnn.LearningRateScheduler.CONSTANT, alpha=0.1)
        alpha = scheduler.alpha_scheduler(i=10, iterations=100)
        self.assertEqual(alpha, 0.1)
        
    def test_progressive_scheduler(self):
        scheduler = gnn.LearningRateScheduler(mode=gnn.LearningRateScheduler.PROGRESIVE, alpha=0.1, steepness=0.2)
        alpha = scheduler.alpha_scheduler(i=10, iterations=100)
        expected = 0.1 * (11 / 22)
        self.assertAlmostEqual(alpha, expected)

class TestSimulationScore(unittest.TestCase):
    
    def test_grade_accuracy(self):
        score = gnn.Simulation_score(mode=gnn.Simulation_score.ACCURACY)
        result = score.grade(acc=0.9, history=None)
        self.assertEqual(result, 0.9)
        
    def test_grade_loss(self):
        score = gnn.Simulation_score(mode=gnn.Simulation_score.LOSS)
        global_history = MagicMock()
        global_history.Y = {'loss': [0.5, 0.4, 0.3]}
        score.new_max_loss(global_history)
        history = MagicMock()
        history.get_last.return_value = 0.3
        result = score.grade(acc=None, history=history)
        expected = 0.5 - 0.3
        self.assertAlmostEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
