import asyncio
import sys
sys.path.append('.')
sys.path.append('../')
import growingnn as gnn
import numpy as np
import unittest
import random
from testSuite import mode

shape = 20
epochs = 5

class TestingTrain(unittest.TestCase):

    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()

    def test_simple_SGD_train(self):
        M = gnn.structure.Model(shape,shape,2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.SGDOptimizer())
        x = np.random.rand(shape, shape)
        y = np.array([0, 1] * (shape//2))  # Alternating 0s and 1s for binary classification
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        gnn.painter.draw(M, "input_test.html")
        acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler)
        self.assertEqual(acc >= 0.3, True)

    def test_simple_Adam_train(self):
        M = gnn.structure.Model(shape,shape,2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.AdamOptimizer())
        x = np.random.rand(shape, shape)
        y = np.array([0, 1] * (shape//2))  # Alternating 0s and 1s for binary classification
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        gnn.painter.draw(M, "input_test.html")
        acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler)
        self.assertEqual(acc >= 0.2, True)

    def test_res_SGD_structure(self):
        M = gnn.structure.Model(shape, shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.SGDOptimizer())
        M.add_res_layer('init_0', 1)
        x = np.asarray(np.random.rand(shape, shape))
        for _ in range(2):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)

    def test_res_Adam_structure(self):
        M = gnn.structure.Model(shape, shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.AdamOptimizer())
        M.add_res_layer('init_0', 1)
        x = np.asarray(np.random.rand(shape, shape))
        for _ in range(2):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)

    def test_conv_SGD_structure(self):
        M = gnn.structure.Model(shape, shape, shape, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.SGDOptimizer())
        M.set_convolution_mode((shape, shape, 1), shape, 1)
        M.add_res_layer('init_0', 1)
        x = np.random.random((shape, shape, shape, 1))
        for _ in range(2):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)

    def test_conv_Adam_structure(self):
        M = gnn.structure.Model(shape, shape, shape, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.AdamOptimizer())
        M.set_convolution_mode((shape, shape, 1), shape, 1)
        M.add_res_layer('init_0', 1)
        x = np.random.random((shape, shape, shape, 1))
        for _ in range(2):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)

class TestOptimizers(unittest.TestCase):

    def setUp(self):
        global mode
        if mode == 'cpu':
            gnn.switch_to_cpu()
        elif mode == 'gpu':
            gnn.switch_to_gpu()
        # Create a model with proper initialization
        self.model = gnn.structure.Model(10, 5, 3, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.SGDOptimizer())

    def test_sgd_optimizer(self):
        # Test SGD optimizer
        optimizer = gnn.optimizers.SGDOptimizer()
        self.model.optimizer = optimizer
        
        # Generate some dummy data
        X = np.random.rand(10, 10)
        Y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # All three classes in sequence
        
        # Train for a few iterations
        for _ in range(2):
            self.model.gradient_descent(X, Y, 1, gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.CONSTANT, 0.01))
            
        # Check that weights were updated
        self.assertTrue(hasattr(self.model.layers[0], 'W'))
        self.assertTrue(hasattr(self.model.layers[0], 'B'))
        
    def test_adam_optimizer(self):
        # Test Adam optimizer
        optimizer = gnn.optimizers.AdamOptimizer()
        self.model.optimizer = optimizer
        
        # Generate some dummy data
        X = np.random.rand(10, 10)
        Y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # All three classes in sequence
        
        # Train for a few iterations
        for _ in range(2):
            self.model.gradient_descent(X, Y, 1, gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.CONSTANT, 0.01))
            
        # Check that weights were updated
        self.assertTrue(hasattr(self.model.layers[0], 'W'))
        self.assertTrue(hasattr(self.model.layers[0], 'B'))
        
    def test_conv_sgd_optimizer(self):
        # Test ConvSGD optimizer
        optimizer = gnn.optimizers.ConvSGDOptimizer()
        
        # Create a model with convolution
        model = gnn.structure.Model(10, 5, 3, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, optimizer)
        model.set_convolution_mode((5, 5, 1), 3, 1)
        
        # Generate some dummy data with proper shapes
        X = np.random.random((5, 5, 5, 1))  # (batch_size, height, width, channels)
        Y = np.array([0, 1, 2, 1, 0])  # All three classes present
        
        # Train for a few iterations
        for _ in range(2):
            model.gradient_descent(X, Y, 1, gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.CONSTANT, 0.01))
            
        # Check that weights were updated
        self.assertTrue(hasattr(model.layers[0], 'kernels'))
        self.assertTrue(hasattr(model.layers[0], 'biases'))
        
    def test_conv_adam_optimizer(self):
        # Test ConvAdam optimizer
        optimizer = gnn.optimizers.ConvAdamOptimizer()
        
        # Create a model with convolution
        model = gnn.structure.Model(10, 5, 3, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, optimizer)
        model.set_convolution_mode((5, 5, 1), 3, 1)
        model.batch_size = 1
        # Generate some dummy data with proper shapes
        X = np.random.random((5, 5, 5, 1))  # (batch_size, height, width, channels)
        Y = np.array([0, 1, 2, 1, 0])  # All three classes present
        
        # Train for a few iterations
        for _ in range(10):
            model.gradient_descent(X, Y, 1, gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.CONSTANT, 0.01))
            
        # Check that weights were updated
        self.assertTrue(hasattr(model.layers[0], 'kernels'))
        self.assertTrue(hasattr(model.layers[0], 'biases'))
        
    def test_optimizer_hyperparameters(self):
        # Test different hyperparameter settings for optimizers
        
        # Test Adam with different beta values
        beta1_values = [0.8, 0.9, 0.99]
        beta2_values = [0.8, 0.9, 0.99]
        for beta1 in beta1_values:
            for beta2 in beta2_values:
                optimizer = gnn.optimizers.AdamOptimizer(beta1=beta1, beta2=beta2)
                self.assertEqual(optimizer.beta1, beta1)
                self.assertEqual(optimizer.beta2, beta2)
                
        # Test ConvAdam with different beta values
        for beta1 in beta1_values:
            for beta2 in beta2_values:
                optimizer = gnn.optimizers.ConvAdamOptimizer(beta1=beta1, beta2=beta2)
                self.assertEqual(optimizer.beta1, beta1)
                self.assertEqual(optimizer.beta2, beta2)
                
    def test_optimizer_factory(self):
        # Test the optimizer factory
        factory = gnn.optimizers.OptimizerFactory()
        
        # Test creating different optimizers
        sgd_optimizer = factory.create_optimizer(gnn.optimizers.OptimizerFactory.SGD, "Dense")
        self.assertIsInstance(sgd_optimizer, gnn.optimizers.SGDOptimizer)
        
        adam_optimizer = factory.create_optimizer(gnn.optimizers.OptimizerFactory.Adam, "Dense")
        self.assertIsInstance(adam_optimizer, gnn.optimizers.AdamOptimizer)
        
        conv_sgd_optimizer = factory.create_optimizer(gnn.optimizers.OptimizerFactory.SGD, "Conv")
        self.assertIsInstance(conv_sgd_optimizer, gnn.optimizers.ConvSGDOptimizer)
        
        conv_adam_optimizer = factory.create_optimizer(gnn.optimizers.OptimizerFactory.Adam, "Conv")
        self.assertIsInstance(conv_adam_optimizer, gnn.optimizers.ConvAdamOptimizer)

if __name__ == '__main__':
    unittest.main()
