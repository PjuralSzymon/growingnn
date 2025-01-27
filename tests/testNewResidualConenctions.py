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

    def test_res_onnection_2(self):
        M = gnn.structure.Model(shape, shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        M.add_res_layer('init_0', 1)
        M.add_res_layer('init_0', 1)
        M.add_res_layer('init_0', 1)
        x = np.asarray(np.random.rand(shape, shape))
        for _ in range(1):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)
        gnn.painter.draw(M, "zzzz.html")
        
    def test_res_onnection_3(self):
        M = gnn.structure.Model(shape, shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        M.add_res_layer('init_0', 1)
        M.add_res_layer(2, 1)
        M.add_res_layer(2, 1)
        M.add_res_layer(2, 1)
        x = np.random.rand(shape, shape)
        y = np.random.randint(2, size=(shape,))
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        acc, _ = M.gradient_descent(x, y, epochs, lr_scheduler)
        self.assertEqual(acc.any(), True)
        gnn.painter.draw(M, "zzzz.html")
        
    def test_edgecase(self):
        M = gnn.structure.Model(shape, shape, shape, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.AdamOptimizer())
        M.set_convolution_mode((shape, shape, 1), 2, 1)
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        x = np.random.random((shape, shape, shape, 1))
        y = list(range(0, shape))
        acc, _ = M.gradient_descent(x, y, 20, lr_scheduler)
        M.add_conv_res_layer('init_0', 1)
        acc, _ = M.gradient_descent(x, y, 20, lr_scheduler)
        M.add_conv_res_layer('init_0', 1)
        acc, _ = M.gradient_descent(x, y, 20, lr_scheduler)
        
        M.add_res_layer('init_0', 1)
        
        acc, _ = M.gradient_descent(x, y, 20, lr_scheduler)
        for _ in range(1):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)
        gnn.painter.draw(M, "zzzz.html")
        
    def test_res_conv_onnection_forward(self):
        M = gnn.structure.Model(shape, shape, shape, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.AdamOptimizer())
        M.set_convolution_mode((shape, shape, 1), shape, 1)
        M.add_conv_res_layer('init_0', 1)
        M.add_res_layer(2, 1)
        x = np.random.random((shape, shape, shape, 1))
        for _ in range(1):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)
        gnn.painter.draw(M, "zzzz.html")
        
    def test_res_conv_onnection_train(self):
        M = gnn.structure.Model(shape, shape, shape, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1, gnn.optimizers.AdamOptimizer())
        M.set_convolution_mode((shape, shape, 1), shape, 1)
        M.add_conv_res_layer('init_0', 1)
        M.add_res_layer(2, 1)
        x = np.random.random((shape, shape, shape, 1))
        y = list(range(0, shape))
        print("y: ", y)
        lr_scheduler = gnn.structure.LearningRateScheduler(gnn.structure.LearningRateScheduler.PROGRESIVE, 0.03, 0.8)
        acc, _ = M.gradient_descent(x, y, 2, lr_scheduler)
        M.add_res_layer(2, 1)
        acc, _ = M.gradient_descent(x, y, 2, lr_scheduler)

        self.assertEqual(acc.any(), True)
        gnn.painter.draw(M, "zzzz.html")


    def test_res_onnection_3(self):
        M = gnn.structure.Model(shape, shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        M.add_res_layer('init_0', 1)
        M.add_res_layer('init_0', 1)
        x = np.asarray(np.random.rand(shape, shape))
        for _ in range(10):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)

    def test_res_onnection_4(self):
        M = gnn.structure.Model(shape, shape, 2, gnn.structure.Loss.multiclass_cross_entropy, gnn.structure.Activations.Sigmoid, 1)
        M.add_res_layer('init_0', 1)
        M.add_res_layer('init_0', 1)
        M.add_res_layer('init_0', 1)
        x = np.asarray(np.random.rand(shape, shape))
        for _ in range(10):
            output = M.forward_prop(x * float(np.random.ranf(1)))
        self.assertEqual(output.any(), True)
        
if __name__ == '__main__':
    unittest.main()
